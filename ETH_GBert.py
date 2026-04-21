# =============================================================================
# ETH-GBERT — Paper-faithful implementation
# "Dynamic Feature Fusion: Combining Global Graph Structures and Local
#  Semantics for Blockchain Fraud Detection" (Zhang et al., arXiv 2501.02032)
#
# Architecture (Fig. 1 of the paper):
#   1. VocabGraphConvolution  — Graph branch: captures global account/vocab
#                               relationships via GCN layers.
#   2. ETH_GBertEmbeddings    — Injects GCN outputs into BERT's token
#                               embedding slots, then applies DynamicFusionLayer
#                               (the 3-way gating network G(x) from §III-B4).
#   3. ETH_GBertModel         — Full model: fused embeddings → BertEncoder
#                               (Transformer) → BertPooler ([CLS]) → Linear
#                               classifier → Softmax.
#
# What was removed vs. the original file
# ---------------------------------------
# - SetEncoder / _MAB / _SAB / _PMA (SetTransformer set branch) — not in paper
# - TriModalFusionGate (BERT / SET / weighted 3-way pooled gate) — not in paper
# - GRU branch over BERT hidden states                          — not in paper
# - MLM (masked-language-model) auxiliary head                  — not in paper
# - All other commented-out experimental iterations
#
# Known implementation notes vs. the paper
# -----------------------------------------
# NOTE-1  Graph type: The paper describes an account-level transaction graph
#         where A[i,j] is the weighted transaction connection between account i
#         and j (§III-A2). This implementation uses a *vocabulary-level*
#         adjacency matrix (word co-occurrence style, VGCN-BERT convention) fed
#         via `vocab_adj_list` + `gcn_swop_eye`. The GCN maths are identical;
#         only the graph being convolved differs.  If you want to plug in the
#         account transaction graph from the paper directly, replace
#         `vocab_adj_list` with the account adjacency matrix and remove the
#         `gcn_swop_eye` permutation step.
#
# NOTE-2  Gumbel noise: The paper gates with
#             gi = exp( (log G(x)_i + b_i) / τ ),  b ~ Gumbel(0,1)
#         The `diff_softmax` helper below is a temperature-scaled softmax
#         *without* Gumbel noise sampling (matching the existing codebase
#         convention).  To add proper Gumbel-Softmax, uncomment the noise line
#         in `diff_softmax`.
#
# NOTE-3  Fusion level: The paper diagram shows GCN and BERT as parallel branches
#         fused at the pooled/sequence level.  This code fuses them at the
#         *token embedding* level (inside ETH_GBertEmbeddings), then the fused
#         embeddings flow through the full BERT Transformer encoder — which is
#         consistent with the paper's Eq. H_fusion = Transformer(E_Fused).
# =============================================================================

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from pytorch_pretrained_bert.modeling import (
    BertEmbeddings,
    BertEncoder,
    BertModel,
    BertPooler,
)


# =============================================================================
# Graph Branch — VocabGraphConvolution
# =============================================================================
# Paper §III-B2: "we employ graph-based representation learning, which
# aggregates information from neighboring accounts to capture both direct
# and indirect relationships within the transaction network."
#
# Each graph convolution layer computes:
#   H^(l+1) = σ( D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l) )
# where Ã = A + I (adjacency with self-loops), D̃ its degree matrix.
#
# Here `num_adj` allows multiple adjacency matrices (e.g. in-edges, out-edges)
# whose outputs are summed before the final projection.
# =============================================================================

class VocabGraphConvolution(nn.Module):
    """
    Multi-adjacency GCN operating on sparse adjacency matrices.

    Args:
        voc_dim      : vocabulary / node feature dimension (V)
        num_adj      : number of adjacency matrices to convolve over
        hid_dim      : intermediate hidden dimension
        out_dim      : output embedding dimension per GCN slot
        dropout_rate : dropout applied after each sparse matmul
    """

    def __init__(self, voc_dim: int, num_adj: int, hid_dim: int,
                 out_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.voc_dim = voc_dim
        self.num_adj = num_adj
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        # One learnable weight matrix W_vh per adjacency matrix
        for i in range(self.num_adj):
            setattr(self, f"W{i}_vh", nn.Parameter(torch.randn(voc_dim, hid_dim)))

        # Final projection: hid_dim → out_dim
        self.fc_hc = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self._reset_parameters()

    def _reset_parameters(self):
        """Kaiming uniform init for all W matrices (standard GCN convention)."""
        for name, param in self.named_parameters():
            if name.startswith("W") or name in ("W", "a", "dense"):
                init.kaiming_uniform_(param, a=math.sqrt(5))

    def forward(self, vocab_adj_list: list, x_dv: torch.Tensor,
                add_linear_mapping_term: bool = False) -> torch.Tensor:
        """
        Args:
            vocab_adj_list          : list of `num_adj` sparse tensors [V, V]
            x_dv                    : dense input [B, V, D] (or compatible)
            add_linear_mapping_term : if True, also adds a skip-linear term
        Returns:
            out : [B, D, out_dim]  (or compatible with x_dv layout)
        """
        fused_h = None

        for i in range(self.num_adj):
            adj = vocab_adj_list[i]
            if not isinstance(adj, torch.Tensor) or not adj.is_sparse:
                raise TypeError(
                    f"vocab_adj_list[{i}] must be a PyTorch sparse tensor"
                )

            w_vh = getattr(self, f"W{i}_vh")                       # [V, hid]
            h_vh = torch.sparse.mm(adj.float(), w_vh)              # [V, hid]
            h_vh = self.dropout(h_vh)

            h_dh = x_dv.matmul(h_vh)                              # [B, D, hid]

            if add_linear_mapping_term:
                # Optional residual skip-linear projection
                h_linear = self.dropout(x_dv.matmul(w_vh))
                h_dh = h_dh + h_linear

            fused_h = h_dh if fused_h is None else fused_h + h_dh

        return self.fc_hc(fused_h)                                 # [B, D, out_dim]


# =============================================================================
# Dynamic Fusion Gate — diff_softmax + DynamicFusionLayer
# =============================================================================
# Paper §III-B4 (Dynamic Weight Calculation):
#   gi = exp( (log G(x)_i + b_i) / τ ) / Σ_j exp(...)   b ~ Gumbel(0,1)
#
# Three fusion strategies (paper §III-B4, Fusion Strategy):
#   O1 = E_BERT                          (BERT-only)
#   O2 = E_GCN_Enhanced                 (GCN-enhanced)
#   O3 = α·E_BERT + (1-α)·E_GCN_Enhanced  (weighted combination, α learnable)
#
# Final fused embedding:
#   E_Fused = g1·O1 + g2·O2 + g3·O3
# =============================================================================

def diff_softmax(logits: torch.Tensor, tau: float = 1.0,
                 hard: bool = False, dim: int = -1) -> torch.Tensor:
    """
    Temperature-scaled softmax with optional straight-through hard estimator.

    Paper uses Gumbel-Softmax noise:  gi = exp( (log G(x)_i + b_i) / τ )
    where b_i ~ Gumbel(0, 1).  The noise line is left below for easy
    activation (see NOTE-2 at top of file).

    Args:
        logits : raw gate logits  [..., 3]
        tau    : temperature  (lower → sharper / harder selection)
        hard   : if True, uses straight-through estimator (one-hot forward,
                 soft gradient backward) as described in the paper
        dim    : softmax dimension
    Returns:
        gate weights in the probability simplex, same shape as logits
    """
    # --- Optional Gumbel noise (paper-exact; uncomment to enable) ---
    # gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    # logits = logits + gumbel_noise
    # ----------------------------------------------------------------

    y_soft = (logits / tau).softmax(dim=dim)

    if not hard:
        return y_soft  # Soft fusion (default during training)

    # Straight-through estimator: one-hot forward, gradient flows through y_soft
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(
        logits, memory_format=torch.legacy_contiguous_format
    ).scatter_(dim, index, 1.0)
    return y_hard - y_soft.detach() + y_soft


class DynamicFusionLayer(nn.Module):
    """
    Token-level dynamic fusion gate G(x) from paper §III-B4.

    Input : bert_embeddings        [B, L, D]  — E_BERT  (word embeddings)
            gcn_enhanced_embeddings [B, L, D]  — E_GCN_Enhanced
    Output: fused_embeddings        [B, L, D]  — E_Fused

    The gating MLP takes [E_BERT || E_GCN_Enhanced] and outputs
    3 logits corresponding to the three fusion strategies O1/O2/O3.
    """

    def __init__(self, hidden_dim: int, tau: float = 1.0, hard_gate: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.hard_gate = hard_gate

        # Gating network G(x): MLP with two FC layers + ReLU  (paper §III-B4)
        # Input  : concat of E_BERT and E_GCN_Enhanced → [B, L, 2·D]
        # Output : 3 logits  g = [g1, g2, g3]
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),   # 3 = |{O1, O2, O3}|
        )

        # α: learnable weight for weighted combination O3 (paper: initialized 0.5)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, bert_embeddings: torch.Tensor,
                gcn_enhanced_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bert_embeddings         : [B, L, D]
            gcn_enhanced_embeddings : [B, L, D]
        Returns:
            fused_embeddings        : [B, L, D]
        """
        # Concatenate both modalities for the gating input
        concat = torch.cat([bert_embeddings, gcn_enhanced_embeddings], dim=-1)  # [B,L,2D]

        # Gate logits → soft/hard weights via temperature-scaled (Gumbel) softmax
        gate_logits = self.gate_network(concat)                   # [B, L, 3]
        gate = diff_softmax(gate_logits, tau=self.tau,
                            hard=self.hard_gate, dim=-1)          # [B, L, 3]

        g1 = gate[:, :, 0].unsqueeze(-1)   # weight for O1 (BERT-only)
        g2 = gate[:, :, 1].unsqueeze(-1)   # weight for O2 (GCN-enhanced)
        g3 = gate[:, :, 2].unsqueeze(-1)   # weight for O3 (weighted mix)

        # Three fusion candidates (paper §III-B4, Fusion Strategy)
        o1 = bert_embeddings                                         # E_BERT only
        o2 = gcn_enhanced_embeddings                                 # E_GCN_Enhanced
        o3 = (self.fusion_weight * bert_embeddings                   # E_Fusion
              + (1.0 - self.fusion_weight) * gcn_enhanced_embeddings)

        # E_Fused = g1·O1 + g2·O2 + g3·O3
        return g1 * o1 + g2 * o2 + g3 * o3                         # [B, L, D]


# =============================================================================
# BERT Embedding Layer with GCN Injection
# =============================================================================
# Paper §III-B3 (Text Input and Initial Embeddings):
#   E_BERT = E_word + E_position + E_token_type
#
# The GCN output is injected into dedicated token slots of E_word before
# adding positional and type embeddings (see NOTE-3 at top of file).
# The DynamicFusionLayer then fuses the plain word embeddings with the
# GCN-injected version to produce the final embedding fed to the encoder.
# =============================================================================

class ETH_GBertEmbeddings(BertEmbeddings):
    """
    Extended BERT embeddings that incorporate vocabulary-GCN features.

    Extra parameters vs. vanilla BertEmbeddings:
        gcn_adj_dim      : vocabulary/node dimension  (V)
        gcn_adj_num      : number of adjacency matrices
        gcn_embedding_dim: number of GCN output slots injected per sequence
    """

    def __init__(self, config, gcn_adj_dim: int, gcn_adj_num: int,
                 gcn_embedding_dim: int):
        super().__init__(config)
        assert gcn_embedding_dim >= 0, "gcn_embedding_dim must be non-negative"

        self.gcn_embedding_dim = gcn_embedding_dim

        # Graph branch: VocabGCN → gcn_embedding_dim output channels
        self.vocab_gcn = VocabGraphConvolution(
            voc_dim=gcn_adj_dim,
            num_adj=gcn_adj_num,
            hid_dim=128,
            out_dim=gcn_embedding_dim,
        )

        # Dynamic fusion gate G(x) — fuses plain BERT words with GCN-enhanced words
        self.dynamic_fusion_layer = DynamicFusionLayer(config.hidden_size)

    def forward(self, vocab_adj_list: list, gcn_swop_eye: torch.Tensor,
                input_ids: torch.Tensor, token_type_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            vocab_adj_list  : list of sparse adj tensors for VocabGCN
            gcn_swop_eye    : permutation matrix aligning vocab ↔ token positions
                              [B, V, L] — see NOTE-1 at top of file
            input_ids       : [B, L]
            token_type_ids  : [B, L]  (optional, defaults to zeros)
            attention_mask  : [B, L]  (optional, defaults to ones)
        Returns:
            embeddings : [B, L, hidden_size]  — E_Fused ready for BertEncoder
        """
        # ---- (a) Plain BERT word embeddings: E_word ----
        words_embeddings = self.word_embeddings(input_ids)         # [B, L, D]

        # ---- (b) GCN branch ----
        # Project token embeddings onto vocabulary space, run GCN
        # vocab_input : [B, D, V]  (transpose of gcn_swop_eye·words_embeddings)
        vocab_input = gcn_swop_eye.matmul(words_embeddings).transpose(1, 2)
        gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input) # [B, D, gcn_dim]

        # Inject GCN outputs into the last `gcn_embedding_dim` *valid* token
        # positions (before [SEP]).  This produces E_GCN_Enhanced at token level.
        gcn_words_embeddings = words_embeddings.clone()
        for i in range(self.gcn_embedding_dim):
            # Compute flat indices of the injection slots across the batch
            tmp_pos = (
                attention_mask.sum(-1) - 2 - self.gcn_embedding_dim + 1 + i
                + torch.arange(0, input_ids.shape[0], device=input_ids.device)
                  * input_ids.shape[1]
            )
            gcn_words_embeddings.flatten(0, 1)[tmp_pos, :] = gcn_vocab_out[:, :, i]

        # ---- (c) Dynamic fusion: E_Fused = G(x) weighting of O1/O2/O3 ----
        # paper §III-B4: E_Fused = g1·E_BERT + g2·E_GCN_Enhanced + g3·E_Fusion
        fused_word_embeddings = self.dynamic_fusion_layer(
            words_embeddings, gcn_words_embeddings
        )                                                           # [B, L, D]

        # ---- (d) Add position and token-type embeddings (standard BERT) ----
        # paper §III-B3: E_BERT = E_word + E_position + E_token_type
        seq_length = input_ids.size(1)
        position_ids = (
            torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0).expand_as(input_ids)
        )
        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = fused_word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings                                           # [B, L, D]


# =============================================================================
# Full ETH-GBERT Model
# =============================================================================
# Paper §III-B (ETH-GBERT Model Architecture):
#
#   Embedding layer  →  ETH_GBertEmbeddings (GCN injection + dynamic fusion)
#   Encoding layer   →  BertEncoder (multi-layer Transformer)
#                        H_fusion = Transformer(E_Fused)          [paper eq.]
#   Pooling          →  BertPooler  ([CLS] token representation)
#   Classifier       →  Linear + Softmax
#                        y = Softmax(W_fusion · H_fusion + b_fusion) [paper eq.]
# =============================================================================

class ETH_GBertModel(BertModel):
    """
    ETH-GBERT: dynamic multimodal fusion model for blockchain fraud detection.

    Components (paper §III-B1):
        1. Graph-Based Representation Module  — VocabGraphConvolution
        2. Semantic Feature Extraction Module — BertEncoder (Transformer)
        3. Multimodal Fusion                  — DynamicFusionLayer (3-way gate)
        4. Classifier                         — Linear → Softmax

    Args:
        config            : BertConfig
        gcn_adj_dim       : vocabulary/node dimension for the GCN (V)
        gcn_adj_num       : number of adjacency matrices
        gcn_embedding_dim : number of GCN output channels injected into embeddings
        num_labels        : number of output classes (2 for binary fraud detection)
        output_attentions : whether to return attention weights
        keep_multihead_output: legacy flag (kept for API compatibility)
    """

    def __init__(self, config, gcn_adj_dim: int, gcn_adj_num: int,
                 gcn_embedding_dim: int, num_labels: int,
                 output_attentions: bool = False,
                 keep_multihead_output: bool = False):
        super().__init__(config)

        # Override BERT's default embeddings with GCN-enhanced version
        self.embeddings = ETH_GBertEmbeddings(
            config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim
        )
        self.encoder = BertEncoder(config)    # Multi-layer Transformer encoder
        self.pooler = BertPooler(config)      # Maps [CLS] hidden state → pooled

        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Classifier: H_fusion → logits  (paper §III-B5)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # Flags for optional attention output (not used in standard training)
        self.output_attentions = (
            config.output_attentions
            if hasattr(config, "output_attentions") else output_attentions
        )
        self.keep_multihead_output = (
            config.keep_multihead_output
            if hasattr(config, "keep_multihead_output") else keep_multihead_output
        )

        self.apply(self.init_bert_weights)

    def forward(self, vocab_adj_list: list, gcn_swop_eye: torch.Tensor,
                input_ids: torch.Tensor, token_type_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                output_all_encoded_layers: bool = False,
                head_mask=None):
        """
        Forward pass of ETH-GBERT.

        Args:
            vocab_adj_list            : list of sparse GCN adjacency tensors
            gcn_swop_eye              : vocab–token alignment matrix [B, V, L]
            input_ids                 : token IDs  [B, L]
            token_type_ids            : segment IDs [B, L]  (optional)
            attention_mask            : padding mask [B, L]  (optional)
            output_all_encoded_layers : return all transformer layer outputs
            head_mask                 : per-head masking (optional)

        Returns:
            logits : [B, num_labels]  — unnormalised class scores
            If output_attentions is True: (all_attentions, logits)
        """
        # Default masks
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # ---- Step 1: Fused embedding  E_Fused  [B, L, D] ----
        # Combines GCN graph features and BERT token embeddings via G(x)
        embedding_output = self.embeddings(
            vocab_adj_list, gcn_swop_eye,
            input_ids, token_type_ids, attention_mask,
        )

        # ---- Step 2: BertEncoder attention mask (additive, -10000 for padding) ----
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Head mask: broadcast to [num_layers, batch, heads, seq, seq]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (head_mask.unsqueeze(0).unsqueeze(0)
                             .unsqueeze(-1).unsqueeze(-1))
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Pass head_mask only if the encoder supports it
        encoder_args = {}
        if "head_mask" in inspect.signature(self.encoder.forward).parameters:
            encoder_args["head_mask"] = head_mask

        if self.output_attentions:
            output_all_encoded_layers = True

        # ---- Step 3: H_fusion = Transformer(E_Fused)  [B, L, D] ----
        encoded = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            **encoder_args,
        )

        if self.output_attentions:
            all_attentions, encoded_layers = encoded
        else:
            encoded_layers = encoded

        # Last transformer layer output
        sequence_output = encoded_layers[-1]                       # [B, L, D]

        # ---- Step 4: Pooler  → [CLS] representation  [B, D] ----
        pooled_output = self.pooler(sequence_output)

        # ---- Step 5: Classifier  y = Softmax(W · H_fusion + b) ----
        # (Softmax is applied externally by the loss / inference code)
        logits = self.classifier(self.dropout(pooled_output))      # [B, num_labels]

        if self.output_attentions:
            return all_attentions, logits

        return logits