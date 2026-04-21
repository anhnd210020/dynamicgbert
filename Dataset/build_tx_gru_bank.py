import os
import pickle
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_config import env_config

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PREP_DIR = os.path.join(ROOT_DIR, 'data/preprocessed/Dataset')

ADDR2IDX_PATH = os.path.join(PREP_DIR, 'data_Dataset.address_to_index')

# Bạn có thể đổi candidates theo nơi bạn lưu dict addr -> list(tx)
TX_CANDIDATES = [
    # ưu tiên file đã được balance/adjust (nếu bạn dùng nó để build graph/splits)
    os.path.join(ROOT_DIR, 'Dataset', 'adjusted_transactions4.pkl'),

    # fallback: file gốc
    os.path.join(ROOT_DIR, 'Dataset', 'transactions4.pkl'),

    # các path cũ (giữ lại để linh hoạt)
    os.path.join(ROOT_DIR, 'transactions4.pkl'),
    os.path.join(ROOT_DIR, 'transactions.pkl'),
    os.path.join(ROOT_DIR, 'data/raw/transactions.pkl'),
    os.path.join(PREP_DIR, 'transactions.pkl'),
]


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def try_load_transactions():
    for p in TX_CANDIDATES:
        if os.path.exists(p):
            print(f"[TX] Loaded transactions from: {p}")
            return load_pickle(p)
    print("[TX] No transaction file found -> will save all-zero banks.")
    return {}

def build_tx_features_seq(txs, d_in):
    """
    txs: list of tx dict.
    Output: feats [len(txs), d_in] float32
    Feature schema (giống tinh thần bạn dùng trước đây, đủ chạy baseline GRU):
      0: log1p(amount>=0)
      1: dt1 = t[i]-t[i-1] (sec)
      2: dt2 = t[i]-t[i-2] (sec)
      3: dt3 = t[i]-t[i-3] (sec)
      4: in_out (0/1)
      5: token_id (float or normalized id)
      6..: zero pad
    """
    if not txs:
        return np.zeros((0, d_in), dtype=np.float32)

    txs = sorted(txs, key=lambda x: x.get("timestamp", 0))
    ts = np.array([t.get("timestamp", 0) for t in txs], dtype=np.float64)
    amt = np.array([t.get("amount", 0.0) for t in txs], dtype=np.float64)
    inout = np.array([t.get("in_out", 0) for t in txs], dtype=np.float32)
    token_id = np.array([t.get("token_id", 0) for t in txs], dtype=np.float32)

    amount_log = np.log1p(np.maximum(amt, 0.0)).astype(np.float32)

    def dt_n(ts, n):
        out = np.zeros_like(ts, dtype=np.float32)
        out[n:] = (ts[n:] - ts[:-n]).astype(np.float32)
        return out

    dt1 = dt_n(ts, 1)
    dt2 = dt_n(ts, 2)
    dt3 = dt_n(ts, 3)

    base = np.stack([amount_log, dt1, dt2, dt3, inout, token_id], axis=1).astype(np.float32)

    if d_in > base.shape[1]:
        pad = np.zeros((base.shape[0], d_in - base.shape[1]), dtype=np.float32)
        base = np.concatenate([base, pad], axis=1)
    elif d_in < base.shape[1]:
        base = base[:, :d_in]

    return base

def build_tx_banks(address_to_index, tx_index, Lmax, d_in, save_dir=PREP_DIR):
    N = len(address_to_index)
    tx_seq_bank = np.zeros((N, Lmax, d_in), dtype=np.float32)
    tx_mask_bank = np.zeros((N, Lmax), dtype=bool)

    for addr, idx in address_to_index.items():
        txs = tx_index.get(addr, [])
        feats = build_tx_features_seq(txs, d_in)  # [len, d_in]

        if feats.shape[0] >= Lmax:
            feats_cut = feats[-Lmax:, :]
            mask_cut = np.ones((Lmax,), dtype=bool)
        else:
            pad_len = Lmax - feats.shape[0]
            feats_cut = np.vstack([feats, np.zeros((pad_len, d_in), dtype=np.float32)])
            mask_cut = np.concatenate([
                np.ones((feats.shape[0],), dtype=bool),
                np.zeros((pad_len,), dtype=bool)
            ])

        tx_seq_bank[idx] = feats_cut
        tx_mask_bank[idx] = mask_cut

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'tx_seq_bank.npy'), tx_seq_bank)
    np.save(os.path.join(save_dir, 'tx_mask_bank.npy'), tx_mask_bank)
    print(f"[TX] Saved tx_seq_bank.npy {tx_seq_bank.shape} and tx_mask_bank.npy {tx_mask_bank.shape} -> {save_dir}")

def main():
    if not os.path.exists(ADDR2IDX_PATH):
        raise FileNotFoundError(f"Missing address_to_index: {ADDR2IDX_PATH}")

    address_to_index = load_pickle(ADDR2IDX_PATH)

    # Reuse env_config.SET_* để khỏi phải thêm config mới (ít sửa nhất)
    Lmax = env_config.SET_LMAX
    d_in = env_config.SET_D_IN

    tx_index = try_load_transactions()
    build_tx_banks(address_to_index, tx_index, Lmax, d_in, save_dir=PREP_DIR)

if __name__ == "__main__":
    main()
