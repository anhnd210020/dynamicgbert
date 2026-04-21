# =============================================================================
# utils.py — Data utilities for ETH-GBERT
# "Dynamic Feature Fusion: Combining Global Graph Structures and Local
#  Semantics for Blockchain Fraud Detection" (Zhang et al., arXiv 2501.02032)
#
# Contents
# --------
#   Graph utilities  : normalize_adj, sparse_scipy2torch
#   Training support : get_class_count_and_weight
#   Data pipeline    : InputExample, InputFeatures, example2feature,
#                      CorpusDataset (+ pad collate_fn)
#
# What was removed vs. the original file
# ----------------------------------------
# - del_http_user_tokenize  — tweet URL / @mention cleaner; irrelevant to
#                             blockchain transaction text; never called
# - clean_str               — NLP punctuation / contraction normaliser for
#                             social media corpora; never called
# - clean_tweet_tokenize    — NLTK TweetTokenizer wrapper; never called;
#                             removes the nltk dependency
# - _truncate_seq_pair      — two-sequence truncation helper; example2feature
#                             asserts text_b is None so this path is dead
# - `re` import             — only used by the three removed text cleaners
# - `TweetTokenizer` import — only used by clean_tweet_tokenize
# - All commented-out dead blocks (ETH-GBert v1 and ETH-GSetBert iterations,
#   lines 1–491 of original)
# =============================================================================

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


# =============================================================================
# Graph utilities
# =============================================================================

def normalize_adj(adj: sp.spmatrix) -> sp.spmatrix:
    """
    Symmetric normalisation of an adjacency matrix:
        Â = D̃⁻¹/² A D̃⁻¹/²
    where D̃ is the degree matrix of A.

    This is the standard GCN normalisation (Kipf & Welling, 2017) and
    corresponds to the adjacency pre-processing described in paper §III-A2
    and the GCN layer formula in §III-B2:
        H^(l+1) = σ( D̃⁻¹/² Ã D̃⁻¹/² H^(l) W^(l) )

    Isolated nodes (rowsum == 0) are assigned rowsum = 1 to avoid division
    by zero, leaving their normalised degree as 1.

    Args:
        adj : scipy sparse matrix  [N, N]
    Returns:
        Â   : symmetrically normalised scipy sparse matrix  [N, N]
    """
    rowsum = np.array(adj.sum(axis=1), dtype=np.float64).flatten()
    rowsum[rowsum == 0] = 1.0                          # guard: isolated nodes
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def sparse_scipy2torch(coo_sparse: sp.spmatrix) -> torch.Tensor:
    """
    Convert a scipy sparse matrix to a PyTorch sparse COO tensor.

    Called in trainModel.py to convert the pre-computed normalised adjacency
    matrix into the format expected by VocabGraphConvolution.

    Args:
        coo_sparse : any scipy sparse matrix (converted to COO internally)
    Returns:
        sparse PyTorch tensor with dtype float32, same shape as input
    """
    coo = coo_sparse.tocoo()
    indices = torch.tensor(
        np.vstack((coo.row, coo.col)), dtype=torch.long
    )
    values = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=coo.shape)


# =============================================================================
# Training support
# =============================================================================

def get_class_count_and_weight(
    y: np.ndarray, n_classes: int
) -> tuple:
    """
    Compute per-class sample counts and inverse-frequency weights for
    weighted cross-entropy loss (paper §V-C).

    Weight formula:  w_i = N / (n_classes × count_i)
    where N is the total number of samples.  This up-weights rare classes
    (fraud) and down-weights common ones (normal), which is important given
    the imbalanced nature of blockchain fraud datasets (paper §VI-D).

    Args:
        y         : 1-D integer label array  [N]
        n_classes : number of distinct classes
    Returns:
        classes_count : list of per-class sample counts  [n_classes]
        weight        : list of per-class loss weights   [n_classes]
                        (0.0 for any class with zero samples)
    """
    total = len(y)
    classes_count, weight = [], []
    for i in range(n_classes):
        count = int(np.sum(y == i))
        classes_count.append(count)
        weight.append(0.0 if count == 0 else total / (n_classes * count))
    return classes_count, weight


def count_transactions(examples: list) -> int:
    """
    Count the total number of transaction records across a list of examples.

    Each transaction in the pre-processed text begins with the token pair
    "in out" (the in_out direction flag from paper §III-A1).
    Counting "in out" occurrences gives the exact transaction count.

    Args:
        examples : list of InputExample
    Returns:
        total number of transaction records in the split
    """
    return sum(ex.text_a.count("in out") for ex in examples)


# =============================================================================
# Data pipeline
# =============================================================================

class InputExample:
    """
    A single raw training/validation/test example.

    Attributes:
        guid       : unique integer identifier
        text_a     : transaction text for this account (paper §III-A3)
                     e.g. "tag=1, value=5.07, value=0.42, ..."
        text_b     : always None for ETH-GBERT (single-sequence model)
        confidence : soft label probabilities  [n_classes]  (for MSE training;
                     carried through the pipeline but not used in CE loss)
        label      : integer hard label  (0 = normal, 1 = fraud)
    """
    def __init__(self, guid, text_a, text_b=None, confidence=None, label=None):
        self.guid       = guid
        self.text_a     = text_a
        self.text_b     = text_b        # always None
        self.confidence = confidence
        self.label      = label


class InputFeatures:
    """
    Processed feature representation of one InputExample, ready for the model.

    Attributes:
        guid         : matches InputExample.guid
        tokens       : list of string tokens (including [CLS] and [SEP] sentinels)
        input_ids    : BERT token IDs  [seq_len]
        gcn_vocab_ids: account/address indices into the GCN vocabulary for each
                       non-special token; -1 for out-of-vocabulary tokens
        input_mask   : attention mask (1 = real token, 0 = padding)  [seq_len]
        segment_ids  : token type IDs (all 0 for single-sequence input)  [seq_len]
        confidence   : soft labels  [n_classes]
        label_id     : integer hard label
    """
    def __init__(
        self,
        guid,
        tokens,
        input_ids,
        gcn_vocab_ids,
        input_mask,
        segment_ids,
        confidence,
        label_id,
    ):
        self.guid          = guid
        self.tokens        = tokens
        self.input_ids     = input_ids
        self.gcn_vocab_ids = gcn_vocab_ids
        self.input_mask    = input_mask
        self.segment_ids   = segment_ids
        self.confidence    = confidence
        self.label_id      = label_id


def example2feature(
    example: InputExample,
    tokenizer,
    gcn_vocab_map: dict,
    max_seq_len: int,
    gcn_embedding_dim: int,
) -> InputFeatures:
    """
    Convert one InputExample into an InputFeatures object.

    Processing steps (paper §III-A3 — Text Transaction Data Generation):

    1. Whitespace-tokenise text_a (pre-tokenised transaction text).
    2. Truncate to max_seq_len - 1 (CLS) - gcn_embedding_dim (GCN slots).
    3. Map each token to its index in gcn_vocab_map (address→index dict).
       Out-of-vocabulary tokens map to gcn_vocab_map.get("UNK", -1).
    4. Build the BERT token sequence:
           [CLS]  t₁ t₂ … tₙ  [SEP] [SEP] … [SEP]
                                      ↑ gcn_embedding_dim + 1 SEP tokens
       The trailing [SEP] slots are the injection positions where
       ETH_GBertEmbeddings will overwrite word embeddings with GCN outputs
       (see ETH_GBert.py, ETH_GBertEmbeddings.forward, the tmp_pos loop).
    5. All segment_ids = 0 (single sequence, paper §III-B3).

    Args:
        example           : InputExample to convert
        tokenizer         : BertTokenizer instance
        gcn_vocab_map     : dict mapping token strings → GCN vocab indices
        max_seq_len       : maximum allowed sequence length (incl. special tokens)
        gcn_embedding_dim : number of GCN output channels / injection slots
    Returns:
        InputFeatures object ready to be returned by CorpusDataset.__getitem__
    """
    assert example.text_b is None, (
        "ETH-GBERT is a single-sequence model; text_b must be None."
    )

    # Step 1–2: tokenise and truncate
    tokens_a = example.text_a.split()
    max_tokens = max_seq_len - 1 - gcn_embedding_dim   # reserve CLS + GCN slots
    if len(tokens_a) > max_tokens:
        tokens_a = tokens_a[:max_tokens]

    # Step 3: GCN vocabulary lookup for each content token
    gcn_vocab_ids = [
        gcn_vocab_map[w] if w in gcn_vocab_map
        else gcn_vocab_map.get("UNK", -1)
        for w in tokens_a
    ]

    # Step 4: build BERT token sequence with GCN injection slots
    # Layout: [CLS] + content_tokens + (gcn_embedding_dim + 1) × [SEP]
    tokens = (
        ["[CLS]"]
        + tokens_a
        + ["[SEP]"] * (gcn_embedding_dim + 1)
    )
    segment_ids = [0] * len(tokens)                    # single sequence
    input_ids   = tokenizer.convert_tokens_to_ids(tokens)
    input_mask  = [1] * len(input_ids)

    return InputFeatures(
        guid=example.guid,
        tokens=tokens,
        input_ids=input_ids,
        gcn_vocab_ids=gcn_vocab_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        confidence=example.confidence,
        label_id=example.label,
    )


class CorpusDataset(Dataset):
    """
    PyTorch Dataset wrapping a list of InputExample objects.

    Each item is converted on-the-fly by example2feature.
    The `pad` method serves as the DataLoader collate_fn and is responsible
    for building the `gcn_swop_eye` alignment matrix (see pad docstring).

    Args:
        examples          : list of InputExample
        tokenizer         : BertTokenizer
        gcn_vocab_map     : dict  {token_string: int_index}  (address_to_index)
        max_seq_len       : maximum sequence length
        gcn_embedding_dim : number of GCN output slots
    """

    def __init__(
        self,
        examples: list,
        tokenizer,
        gcn_vocab_map: dict,
        max_seq_len: int,
        gcn_embedding_dim: int,
    ):
        self.examples         = examples
        self.tokenizer        = tokenizer
        self.gcn_vocab_map    = gcn_vocab_map
        self.max_seq_len      = max_seq_len
        self.gcn_embedding_dim = gcn_embedding_dim

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns a 6-tuple of plain Python lists/values (not tensors):
            (input_ids, input_mask, segment_ids, confidence,
             label_id, gcn_vocab_ids)
        Tensors are created in the `pad` collate_fn so that padding can be
        applied at the batch level.
        """
        feat = example2feature(
            self.examples[idx],
            self.tokenizer,
            self.gcn_vocab_map,
            self.max_seq_len,
            self.gcn_embedding_dim,
        )
        return (
            feat.input_ids,
            feat.input_mask,
            feat.segment_ids,
            feat.confidence,
            feat.label_id,
            feat.gcn_vocab_ids,
        )

    def pad(self, batch: list) -> tuple:
        """
        Collate function for DataLoader (passed as collate_fn=ds.pad).

        Pads all variable-length sequences to the maximum sequence length in
        the batch, then constructs the `gcn_swop_eye` alignment matrix.

        Padding conventions
        -------------------
        - input_ids, input_mask, segment_ids : right-padded with 0
        - gcn_vocab_ids                      : left-padded with -1, then
                                               right-padded with -1
          The leading -1 accounts for the [CLS] token which has no GCN vocab
          entry; it maps to the (gcn_vocab_size)th row of the augmented eye
          matrix, which is then discarded (see gcn_swop_eye construction below).

        gcn_swop_eye construction
        -------------------------
        Purpose: align each token position in the BERT sequence with its
        corresponding GCN vocabulary embedding, producing the permutation
        matrix used by ETH_GBertEmbeddings to inject GCN features into
        specific token slots (ETH_GBert.py NOTE-1).

        Steps:
          1. Build an augmented identity matrix of size
             (gcn_vocab_size + 1) × (gcn_vocab_size + 1).
             The extra row/column at index gcn_vocab_size is the "null" row
             used for tokens that are OOV or special ([CLS], padding).
          2. Index into it with the flat batch of padded gcn_vocab_ids
             (where -1 wraps to row gcn_vocab_size via Python indexing).
             Result: [B × L, gcn_vocab_size + 1]
          3. Drop the last column (the null column):
             Result: [B × L, gcn_vocab_size]
          4. Reshape to [B, L, gcn_vocab_size] then transpose(1, 2):
             Final:  [B, gcn_vocab_size, L]
             Interpretation: for batch item b, gcn_swop_eye[b, v, l] = 1
             iff token at position l corresponds to GCN vocab entry v.

        Returns a 6-tuple of tensors:
            batch_input_ids    : [B, L]               dtype=long
            batch_input_mask   : [B, L]               dtype=long
            batch_segment_ids  : [B, L]               dtype=long
            batch_confidences  : [B, n_classes]       dtype=float
            batch_label_ids    : [B]                  dtype=long
            batch_gcn_swop_eye : [B, gcn_vocab_size, L]  dtype=float
        """
        gcn_vocab_size = len(self.gcn_vocab_map)
        seq_lens = [len(sample[0]) for sample in batch]
        max_len  = int(np.max(seq_lens))

        # --- helpers ---
        def pad_zeros(field_idx):
            """Right-pad field with 0s to max_len."""
            return [
                sample[field_idx] + [0] * (max_len - len(sample[field_idx]))
                for sample in batch
            ]

        def collect(field_idx):
            """Gather a scalar/list field with no padding."""
            return [sample[field_idx] for sample in batch]

        def pad_gcn_ids(field_idx):
            """
            Build GCN-vocab-ID rows that are:
              [-1]  +  gcn_vocab_ids  +  [-1, …, -1]
            The leading -1 covers the [CLS] token slot (no GCN entry).
            Trailing -1s pad to max_len.
            Total length after padding = max_len.
            """
            return [
                [-1]
                + sample[field_idx]
                + [-1] * (max_len - len(sample[field_idx]) - 1)
                for sample in batch
            ]

        # --- build standard BERT tensors ---
        batch_input_ids    = torch.tensor(pad_zeros(0), dtype=torch.long)
        batch_input_mask   = torch.tensor(pad_zeros(1), dtype=torch.long)
        batch_segment_ids  = torch.tensor(pad_zeros(2), dtype=torch.long)
        batch_confidences  = torch.tensor(collect(3),   dtype=torch.float)
        batch_label_ids    = torch.tensor(collect(4),   dtype=torch.long)

        # --- build gcn_swop_eye  [B, gcn_vocab_size, L] ---
        # Step 1-2: flat index lookup into augmented identity matrix
        flat_gcn_ids = np.array(pad_gcn_ids(5)).reshape(-1)   # [B*L]
        # torch.eye(V+1)[flat_gcn_ids] maps each token to a one-hot row in R^(V+1)
        # -1 indices (OOV / special tokens) map to row V (the "null" row)
        swop = torch.eye(gcn_vocab_size + 1)[flat_gcn_ids]    # [B*L, V+1]

        # Step 3: drop the null column → [B*L, V]
        swop = swop[:, :-1]

        # Step 4: reshape and transpose → [B, V, L]
        batch_gcn_swop_eye = swop.view(
            len(batch), max_len, gcn_vocab_size
        ).transpose(1, 2)

        return (
            batch_input_ids,
            batch_input_mask,
            batch_segment_ids,
            batch_confidences,
            batch_label_ids,
            batch_gcn_swop_eye,
        )