# =============================================================================
# trainModel.py — Paper-faithful training script for ETH-GBERT
# "Dynamic Feature Fusion: Combining Global Graph Structures and Local
#  Semantics for Blockchain Fraud Detection" (Zhang et al., arXiv 2501.02032)
#
# Hyperparameters aligned to paper §V:
#   Learning rate  : 8e-6                    (§V-B)
#   L2 decay       : 0.001                   (§V-B)
#   Batch size     : 32                      (§V-B)
#   Gradient accum : 2  (effective batch=32) (§V-B: "after every 2 mini-batches")
#   Epochs         : 40                      (§V-B)
#   Warmup         : 10% of total steps      (§V-C, AdamW + scheduler)
#   Loss           : Cross-entropy           (§V-C)
#   Optimizer      : AdamW (BertAdam)        (§V-C)
#   Metrics        : Precision, Recall, F1   (§V-D)
#
# What was removed vs. the original file
# ----------------------------------------
# - set_feats / set_mask arguments throughout (SetTransformer branch)
# - 8-tuple batch unpacking (reduced to clean 6-tuple)
# - MSE loss path (not in paper; paper uses cross-entropy only)
# - Commented-out experimental block (lines 565-1241 of original):
#     A2 auxiliary MLM head, A3 multi-modal token encoding,
#     addr_pat / Counter address tokenisation, alternative dataloader
# - classifier_act_func, resample_train_set, do_softmax_before_mse
#   (unused variables / MSE-only helpers)
#
# Bugs fixed
# ----------
# - valid_f1_best_epoch and test_f1_when_valid_best were referenced in the
#   final summary print but could be undefined if validation never improved.
#   Both are now initialised before the training loop.
# - train_dataloader used shuffle_choice=0 (no shuffle); changed to 1 (shuffle)
#   as specified in the paper (§V-A: "shuffling applied during the training").
# =============================================================================

import argparse
import gc
import os
import pickle as pkl
import random
import time
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import classification_report, f1_score

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

from env_config import env_config
from ETH_GBert import ETH_GBertModel
from utils import (
    InputExample,
    CorpusDataset,
    sparse_scipy2torch,
    get_class_count_and_weight,
    count_transactions,
)

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Reproducibility & device
# =============================================================================
random.seed(env_config.GLOBAL_SEED)
np.random.seed(env_config.GLOBAL_SEED)
torch.manual_seed(env_config.GLOBAL_SEED)

cuda_yes = torch.cuda.is_available()
if cuda_yes:
    torch.cuda.manual_seed_all(env_config.GLOBAL_SEED)
device = torch.device("cuda:0" if cuda_yes else "cpu")


# =============================================================================
# Argument parsing & hyper-parameters
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--ds",               type=str,   default="Dataset",
                    help="Dataset name; must match a folder under data/preprocessed/")
parser.add_argument("--load",             type=int,   default=0,
                    help="1 = resume training from checkpoint")
parser.add_argument("--sw",               type=int,   default=0,
                    help="1 = use stop-words filter (affects saved filename only)")
parser.add_argument("--dim",              type=int,   default=16,
                    help="gcn_embedding_dim: number of GCN output slots injected into BERT")
parser.add_argument("--lr",               type=float, default=8e-6,
                    help="Initial learning rate (paper §V-B: 8e-6)")
parser.add_argument("--l2",               type=float, default=0.001,
                    help="L2 regularisation coefficient (paper §V-B: 0.001)")
parser.add_argument("--validate_program", action="store_true",
                    help="Run 1 epoch on 1 example for quick smoke test")
args = parser.parse_args()

# ---- Training hyper-parameters (paper §V) ----
cfg_model_type       = "ETH_GBert"
cfg_stop_words       = (args.sw == 1)
will_resume_ckpt     = (args.load == 1)

gcn_embedding_dim    = args.dim
learning_rate0       = args.lr
l2_decay             = args.l2

# Paper §V-B: batch_size=32, gradient_accumulation=2  →  effective batch = 32
# (paper: "updating the model's parameters after every 2 mini-batches")
batch_size                 = 16        # mini-batch fed to GPU each step
gradient_accumulation_steps = 2        # accumulate 2 steps → effective batch 32

total_train_epochs   = 40             # paper §V-B
warmup_proportion    = 0.1            # 10% of total steps

# MAX_SEQ_LENGTH: 200 base tokens + gcn_embedding_dim extra injection slots
MAX_SEQ_LENGTH = 200 + gcn_embedding_dim

bert_model_scale = "bert-base-uncased"
if env_config.TRANSFORMERS_OFFLINE == 1:
    bert_model_scale = os.path.join(
        env_config.HUGGING_LOCAL_MODEL_FILES_PATH,
        f"hf-maintainers_{bert_model_scale}",
    )

do_lower_case   = True
data_dir        = f"data/preprocessed/{args.ds}"
output_dir      = "./output/"
os.makedirs(output_dir, exist_ok=True)

# Metrics printed during training/evaluation
perform_metrics_str = ["weighted avg", "f1-score"]

# Checkpoint filename
model_file_4save = (
    f"{cfg_model_type}{gcn_embedding_dim}_model_{args.ds}_cle"
    f"_sw{int(cfg_stop_words)}.pt"
)

if args.validate_program:
    total_train_epochs = 1

print(f"{cfg_model_type} Start at: {time.asctime()}")
print(
    "\n----- Configure -----"
    f"\n  Dataset         : {args.ds}"
    f"\n  stop_words      : {cfg_stop_words}"
    f"\n  GCN embed dim   : vocab_size -> 128 -> {gcn_embedding_dim}"
    f"\n  Learning rate   : {learning_rate0}"
    f"\n  L2 decay        : {l2_decay}"
    f"\n  Loss            : cross-entropy (paper §V-C)"
    f"\n  Batch size      : {batch_size} x {gradient_accumulation_steps} accum"
    f"  = {batch_size * gradient_accumulation_steps} effective (paper §V-B)"
    f"\n  Epochs          : {total_train_epochs} (paper §V-B)"
    f"\n  MAX_SEQ_LENGTH  : {MAX_SEQ_LENGTH}"
    f"\n  Checkpoint      : {model_file_4save}"
    f"\n  validate_program: {args.validate_program}"
)


# =============================================================================
# Data loading
# =============================================================================
print("\n----- Prepare dataset -----")

# Field names stored by the preprocessing pipeline
_data_fields = [
    "labels",
    "train_y",
    "train_y_prob",
    "valid_y",
    "valid_y_prob",
    "test_y",
    "test_y_prob",
    "shuffled_clean_docs",
    "address_to_index",
]

_objects = []
for name in _data_fields:
    datafile = f"./{data_dir}/data_{args.ds}.{name}"
    with open(datafile, "rb") as f:
        _objects.append(pkl.load(f, encoding="latin1"))

(
    lables_list,
    train_y,
    train_y_prob,
    valid_y,
    valid_y_prob,
    test_y,
    test_y_prob,
    shuffled_clean_docs,
    address_to_index,
) = tuple(_objects)

label2idx, idx2label = lables_list[0], lables_list[1]
num_classes   = len(label2idx)
gcn_vocab_size = len(address_to_index)

# Merge all split arrays for unified example construction
y       = np.hstack((train_y, valid_y, test_y))
y_prob  = np.vstack((train_y_prob, valid_y_prob, test_y_prob))

examples = [
    InputExample(i, ts.strip(), confidence=y_prob[i], label=y[i])
    for i, ts in enumerate(shuffled_clean_docs)
]

train_size = len(train_y)
valid_size = len(valid_y)
test_size  = len(test_y)

indexs          = np.arange(len(examples))
train_examples  = [examples[i] for i in indexs[:train_size]]
valid_examples  = [examples[i] for i in indexs[train_size: train_size + valid_size]]
test_examples   = [examples[i] for i in indexs[train_size + valid_size:
                                                train_size + valid_size + test_size]]

# Load normalised adjacency matrix (COO format) for the GCN branch
# Shape: [vocab_size, vocab_size]  (account interaction graph in paper §III-A2)
npz_path = f"data/preprocessed/{args.ds}/norm_adj_coo.npz"
if not os.path.exists(npz_path):
    npz_path = "data/preprocessed/Dataset/norm_adj_coo.npz"   # fallback
npz = np.load(npz_path)
A_hat = sp.coo_matrix(
    (npz["data"], (npz["row"], npz["col"])),
    shape=tuple(npz["shape"]),
)
gcn_adj_list = [sparse_scipy2torch(A_hat).to(device)]

gc.collect()

# Class weights for weighted cross-entropy (handles class imbalance; paper §V-D)
train_classes_num, train_classes_weight = get_class_count_and_weight(
    train_y, num_classes
)
loss_weight = torch.tensor(train_classes_weight, dtype=torch.float).to(device)

tokenizer = BertTokenizer.from_pretrained(
    bert_model_scale, do_lower_case=do_lower_case
)


# =============================================================================
# DataLoader factory
# =============================================================================
def get_pytorch_dataloader(
    examples,
    tokenizer,
    batch_size,
    shuffle_choice,
    classes_weight=None,
    total_resample_size=-1,
):
    """
    Build a DataLoader from a list of InputExample objects.

    shuffle_choice:
        0 — no shuffle   (used for validation and test)
        1 — shuffle      (used for training; paper §V-A)
        2 — weighted resampling (optional, not used in default paper setup)
    """
    ds = CorpusDataset(
        examples,
        tokenizer,
        address_to_index,
        MAX_SEQ_LENGTH,
        gcn_embedding_dim,
    )

    if shuffle_choice == 0:
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=ds.pad,
        )

    if shuffle_choice == 1:
        # Paper §V-A: "shuffling applied during the training process"
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=ds.pad,
        )

    if shuffle_choice == 2:
        # Weighted random sampler for severe class imbalance (optional)
        assert classes_weight is not None and total_resample_size > 0
        weights = [
            classes_weight[0] if label == 0
            else classes_weight[1] if label == 1
            else classes_weight[2]
            for _, _, _, _, label in ds
        ]
        sampler = WeightedRandomSampler(
            weights, num_samples=total_resample_size, replacement=True
        )
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            collate_fn=ds.pad,
        )

    raise ValueError(f"Unknown shuffle_choice={shuffle_choice}")


# Smoke-test shortcut
if args.validate_program:
    train_examples = [train_examples[0]]
    valid_examples = [valid_examples[0]]
    test_examples  = [test_examples[0]]

# Paper §V-A: training set is shuffled; val/test are not
train_dataloader = get_pytorch_dataloader(
    train_examples, tokenizer, batch_size, shuffle_choice=1   # shuffled
)
valid_dataloader = get_pytorch_dataloader(
    valid_examples, tokenizer, batch_size, shuffle_choice=0
)
test_dataloader  = get_pytorch_dataloader(
    test_examples,  tokenizer, batch_size, shuffle_choice=0
)

total_train_steps = int(
    len(train_dataloader) / gradient_accumulation_steps * total_train_epochs
)

print(f"  Train class counts : {train_classes_num}")
print(f"  {'Split':<6}  {'Accounts':>9}  {'Transactions':>13}  {'Batches':>8}")
print(f"  {'-----':<6}  {'---------':>9}  {'------------':>13}  {'-------':>8}")
for split_name, split_examples, split_loader in (
    ("Train", train_examples, train_dataloader),
    ("Valid", valid_examples, valid_dataloader),
    ("Test",  test_examples,  test_dataloader),
):
    n_acc = len(split_examples)
    n_tx  = count_transactions(split_examples)
    n_bat = len(split_loader)
    print(f"  {split_name:<6}  {n_acc:>9,}  {n_tx:>13,}  {n_bat:>8,}")
print(f"  Batch size (mini)  : {batch_size}")
print(f"  Gradient accum     : {gradient_accumulation_steps}"
      f"  (effective batch = {batch_size * gradient_accumulation_steps})")
print(f"  Total train steps  : {total_train_steps}")

# ── Class distribution per split ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  DATASET CLASS DISTRIBUTION")
print("=" * 65)
print(f"  {'Split':<8} {'Total':>7} {'Normal(0)':>11} {'Fraud(1)':>10} "
      f"{'Normal%':>9} {'Fraud%':>8} {'Ratio(N:F)':>11}")
print(f"  {'-'*8} {'-'*7} {'-'*11} {'-'*10} {'-'*9} {'-'*8} {'-'*11}")
 
_splits = [
    ("Train",  train_y,  train_examples),
    ("Valid",  valid_y,  valid_examples),
    ("Test",   test_y,   test_examples),
]
 
for _name, _y, _exs in _splits:
    _total   = len(_y)
    _normal  = int(np.sum(_y == 0))
    _fraud   = int(np.sum(_y == 1))
    _pct_n   = 100.0 * _normal / _total if _total > 0 else 0.0
    _pct_f   = 100.0 * _fraud  / _total if _total > 0 else 0.0
    _ratio   = f"{_normal/_fraud:.2f}:1" if _fraud > 0 else "N/A"
    print(f"  {_name:<8} {_total:>7,} {_normal:>11,} {_fraud:>10,} "
          f"{_pct_n:>8.1f}% {_pct_f:>7.1f}% {_ratio:>11}")
 
# Overall totals
_all_y    = np.hstack((train_y, valid_y, test_y))
_t_total  = len(_all_y)
_t_normal = int(np.sum(_all_y == 0))
_t_fraud  = int(np.sum(_all_y == 1))
_t_pct_n  = 100.0 * _t_normal / _t_total if _t_total > 0 else 0.0
_t_pct_f  = 100.0 * _t_fraud  / _t_total if _t_total > 0 else 0.0
_t_ratio  = f"{_t_normal/_t_fraud:.2f}:1" if _t_fraud > 0 else "N/A"
print(f"  {'─'*63}")
print(f"  {'TOTAL':<8} {_t_total:>7,} {_t_normal:>11,} {_t_fraud:>10,} "
      f"{_t_pct_n:>8.1f}% {_t_pct_f:>7.1f}% {_t_ratio:>11}")
print("=" * 65)
 
# ── Transaction breakdown per split ──────────────────────────────────────────
print("\n" + "=" * 65)
print("  TRANSACTION BREAKDOWN PER SPLIT")
print("=" * 65)
print(f"  {'Split':<8} {'Accounts':>9} {'Transactions':>14} "
      f"{'Tx/Acc(avg)':>12}")
print(f"  {'-'*8} {'-'*9} {'-'*14} {'-'*12}")
 
for _name, _y, _exs in _splits:
    _n_acc = len(_exs)
    _n_tx  = count_transactions(_exs)
    _avg   = _n_tx / _n_acc if _n_acc > 0 else 0.0
    print(f"  {_name:<8} {_n_acc:>9,} {_n_tx:>14,} {_avg:>11.1f}")
 
print("=" * 65 + "\n")
 
# =============================================================================
# Batch unpacking (6-tuple only — no SetTransformer fields)
# =============================================================================
def unpack_batch(batch):
    """
    Move batch to device and unpack into named tensors.

    CorpusDataset.pad() returns a 6-tuple:
        (input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_swop_eye)

    The `gcn_swop_eye` matrix aligns vocabulary indices with token positions
    for the VocabGraphConvolution (ETH_GBert.py, NOTE-1).
    """
    batch = tuple(t.to(device) for t in batch)
    if len(batch) != 6:
        raise ValueError(
            f"Expected a 6-element batch tuple, got {len(batch)}. "
            "If the dataset produces 8-tuples (set_feats, set_mask), "
            "update CorpusDataset.pad() to return only 6 fields."
        )
    input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_swop_eye = batch
    return input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_swop_eye


# =============================================================================
# Evaluation (validation & test)
# =============================================================================
@torch.no_grad()
def evaluate(model, gcn_adj_list, dataloader, epoch_th, dataset_name):
    """
    Run one full pass over `dataloader` and report loss, accuracy, and F1.

    Metrics (paper §V-D): Precision, Recall, F1 Score.
    Loss: weighted cross-entropy (paper §V-C).

    Returns:
        ev_loss      : average cross-entropy loss over the split
        ev_acc       : accuracy (correct / total)
        f1_weighted  : weighted-average F1 score
    """
    model.eval()
    predict_out, all_label_ids = [], []
    ev_loss, total, correct = 0.0, 0, 0
    start = time.time()

    for batch in dataloader:
        input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_swop_eye = \
            unpack_batch(batch)

        # Forward pass — no MSE path; paper §V-C uses cross-entropy only
        logits = model(
            gcn_adj_list,
            gcn_swop_eye,
            input_ids,
            segment_ids,
            input_mask,
        )

        # Weighted cross-entropy loss (paper §V-C)
        loss = F.cross_entropy(
            logits.view(-1, num_classes), label_ids, weight=loss_weight
        )
        ev_loss += loss.item()

        _, predicted = torch.max(logits, dim=-1)
        predict_out.extend(predicted.tolist())
        all_label_ids.extend(label_ids.tolist())

        total   += len(label_ids)
        correct += predicted.eq(label_ids).sum().item()

    all_label_ids_np = np.array(all_label_ids).reshape(-1)
    predict_out_np   = np.array(predict_out).reshape(-1)

    # Weighted F1 (paper §V-D)
    f1_weighted = f1_score(all_label_ids_np, predict_out_np, average="weighted")

    print(
        "Report:\n"
        + classification_report(all_label_ids_np, predict_out_np, digits=4)
    )

    ev_acc = correct / total
    elapsed = (time.time() - start) / 60.0
    print(
        f"Epoch:{epoch_th}  {' '.join(perform_metrics_str)}: "
        f"{100 * f1_weighted:.3f}  Acc:{100.0 * ev_acc:.3f}  "
        f"[{dataset_name}]  Spent:{elapsed:.3f}m"
    )
    print("--------------------------------------------------------------")
    return ev_loss, ev_acc, f1_weighted


# =============================================================================
# Prediction (returns class predictions + confidence scores)
# =============================================================================
@torch.no_grad()
def predict(model, examples, tokenizer, batch_size):
    """
    Run inference and return (predicted_labels, confidence_scores).
    Used for final evaluation after training completes.
    """
    dataloader = get_pytorch_dataloader(
        examples, tokenizer, batch_size, shuffle_choice=0
    )
    predict_out, confidence_out = [], []

    model.eval()
    for batch in dataloader:
        input_ids, input_mask, segment_ids, _, label_ids, gcn_swop_eye = \
            unpack_batch(batch)

        # Clean model forward: no set_feats / set_mask
        score_out = model(
            gcn_adj_list,
            gcn_swop_eye,
            input_ids,
            segment_ids,
            input_mask,
        )

        predict_out.extend(score_out.max(1)[1].tolist())
        confidence_out.extend(score_out.max(1)[0].tolist())

    return (
        np.array(predict_out).reshape(-1),
        np.array(confidence_out).reshape(-1),
    )


# =============================================================================
# Model initialisation
# =============================================================================
print("\n----- Running training -----")

ckpt_path = os.path.join(output_dir, model_file_4save)

if will_resume_ckpt and os.path.exists(ckpt_path):
    # ---- Resume from checkpoint ----
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Determine where to resume (mid-epoch step or next epoch)
    if "step" in checkpoint:
        prev_save_step = checkpoint["step"]
        start_epoch    = checkpoint["epoch"]
    else:
        prev_save_step = -1
        start_epoch    = checkpoint["epoch"] + 1

    valid_acc_prev      = checkpoint["valid_acc"]
    perform_metrics_prev = checkpoint["perform_metrics"]

    # Instantiate model from BERT pretrained weights + saved state
    model = ETH_GBertModel.from_pretrained(
        bert_model_scale,
        state_dict=checkpoint["model_state"],
        gcn_adj_dim=gcn_vocab_size,
        gcn_adj_num=len(gcn_adj_list),
        gcn_embedding_dim=gcn_embedding_dim,
        num_labels=num_classes,
    )

    # Safe partial load: only update keys present in both dicts
    pretrained_dict  = checkpoint["model_state"]
    net_state_dict   = model.state_dict()
    matched = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(matched)
    model.load_state_dict(net_state_dict)

    print(
        f"  Resumed from checkpoint: {model_file_4save}"
        f"  (epoch={checkpoint['epoch']}, step={prev_save_step},"
        f"  valid_acc={valid_acc_prev:.4f},"
        f"  valid_F1={perform_metrics_prev:.4f})"
    )

else:
    # ---- Fresh training from BERT pretrained weights ----
    start_epoch          = 0
    prev_save_step       = -1
    valid_acc_prev       = 0.0
    perform_metrics_prev = 0.0

    model = ETH_GBertModel.from_pretrained(
        bert_model_scale,
        gcn_adj_dim=gcn_vocab_size,
        gcn_adj_num=len(gcn_adj_list),
        gcn_embedding_dim=gcn_embedding_dim,
        num_labels=num_classes,
    )

model.to(device)


# =============================================================================
# Optimiser — AdamW (BertAdam) with linear warmup + L2 decay
# Paper §V-C: AdamW, lr=8e-6, warmup=10%, L2=0.001
# =============================================================================
optimizer = BertAdam(
    model.parameters(),
    lr=learning_rate0,
    warmup=warmup_proportion,
    t_total=total_train_steps,
    weight_decay=l2_decay,
)


# =============================================================================
# Training loop
# =============================================================================
train_start = time.time()
global_step_th = int(
    len(train_examples) / batch_size / gradient_accumulation_steps * start_epoch
)

all_loss_list = {"train": [], "valid": [], "test": []}
all_f1_list   = {"valid": [], "test": []}

# Initialise tracking variables (prevents NameError if valid F1 never improves)
test_f1_best          = 0.0
test_f1_best_epoch    = start_epoch
test_f1_when_valid_best = 0.0
valid_f1_best_epoch   = start_epoch

for epoch in range(start_epoch, total_train_epochs):
    tr_loss = 0.0
    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(train_dataloader):
        # Skip steps already trained when resuming mid-epoch
        if prev_save_step > -1 and step <= prev_save_step:
            continue
        if prev_save_step > -1:
            prev_save_step = -1

        input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_swop_eye = \
            unpack_batch(batch)

        # ---- Forward pass ----
        logits = model(
            gcn_adj_list,
            gcn_swop_eye,
            input_ids,
            segment_ids,    # token_type_ids
            input_mask,     # attention_mask
        )

        # ---- Loss: weighted cross-entropy (paper §V-C) ----
        loss = F.cross_entropy(
            logits.view(-1, num_classes), label_ids, weight=loss_weight
        )

        # ---- Gradient accumulation (paper §V-B: effective batch = 32) ----
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()
        tr_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step_th += 1

        # Progress log every 40 steps
        if step % 40 == 0:
            elapsed = (time.time() - train_start) / 60.0
            print(
                f"  Epoch:{epoch} step:{step}/{len(train_dataloader)}"
                f"  Loss:{loss.item():.6f}  Elapsed:{elapsed:.2f}m"
            )

    # ---- End-of-epoch evaluation ----
    print("--------------------------------------------------------------")
    valid_loss, valid_acc, valid_f1 = evaluate(
        model, gcn_adj_list, valid_dataloader, epoch, "Valid_set"
    )
    test_loss, _, test_f1 = evaluate(
        model, gcn_adj_list, test_dataloader,  epoch, "Test_set"
    )

    # Track best test F1 (absolute, across all epochs)
    if test_f1 > test_f1_best:
        test_f1_best       = test_f1
        test_f1_best_epoch = epoch

    all_loss_list["train"].append(tr_loss)
    all_loss_list["valid"].append(valid_loss)
    all_loss_list["test"].append(test_loss)
    all_f1_list["valid"].append(valid_f1)
    all_f1_list["test"].append(test_f1)

    print(
        f"Epoch:{epoch} done. "
        f"TrainLoss:{tr_loss:.4f}  ValidLoss:{valid_loss:.4f}  "
        f"TestLoss:{test_loss:.4f}  Elapsed:{(time.time()-train_start)/60.0:.2f}m"
    )

    # ---- Save checkpoint when validation F1 improves ----
    if valid_f1 > perform_metrics_prev:
        to_save = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "valid_acc":       valid_acc,
            "lower_case":      do_lower_case,
            "perform_metrics": valid_f1,   # valid weighted-F1
        }
        torch.save(to_save, ckpt_path)
        perform_metrics_prev    = valid_f1
        test_f1_when_valid_best = test_f1
        valid_f1_best_epoch     = epoch
        print(f"  ✓ Checkpoint saved (valid F1 improved to {100*valid_f1:.3f})")


# =============================================================================
# Final summary
# =============================================================================
total_elapsed = (time.time() - train_start) / 60.0
print(f"\n{'='*60}")
print(f"**Optimisation finished!  Total time: {total_elapsed:.2f}m")
print(f"**Valid weighted F1 best : {100*perform_metrics_prev:.3f}  "
      f"at epoch {valid_f1_best_epoch}")
print(f"**Test  weighted F1 (when valid best): "
      f"{100*test_f1_when_valid_best:.3f}")
print(f"**Test  weighted F1 (absolute best)  : "
      f"{100*test_f1_best:.3f}  at epoch {test_f1_best_epoch}")
print(f"{'='*60}")