# #ETH-GBert
# import pickle
# import random
# import numpy as np

# # Load data from file
# def load_data(filename):
#     with open(filename, 'rb') as file:
#         return pickle.load(file)

# # Save data to file
# def save_data(data, filename):
#     with open(filename, 'wb') as file:
#         pickle.dump(data, file)

# # Classify normal and abnormal accounts and load transaction data
# data_filename = 'transactions4.pkl'
# accounts_data = load_data(data_filename)

# normal_accounts = {}
# abnormal_accounts = {}

# for address, transactions in accounts_data.items():
#     if transactions[0]['tag'] == 0:
#         normal_accounts[address] = transactions
#     elif transactions[0]['tag'] == 1:
#         abnormal_accounts[address] = transactions

# # Get number of abnormal accounts
# num_abnormal = len(abnormal_accounts)

# # Randomly select twice the number of abnormal accounts from normal accounts
# selected_normal_accounts = random.sample(list(normal_accounts.keys()), 2 * num_abnormal)
# adjusted_normal_accounts = {addr: normal_accounts[addr] for addr in selected_normal_accounts}

# # Merge adjusted normal accounts with all abnormal accounts
# adjusted_accounts_data = {**adjusted_normal_accounts, **abnormal_accounts}

# # Save adjusted data
# save_data_filename = 'adjusted_transactions4.pkl'
# save_data(adjusted_accounts_data, save_data_filename)

# print(f"Data has been adjusted and saved to {save_data_filename}")
# print(f"Number of abnormal accounts: {len(abnormal_accounts)}")
# print(f"Number of selected normal accounts: {len(adjusted_normal_accounts)}")

# # Print the first ten transactions of the first ten accounts
# print("\nFirst ten accounts with their first ten transaction records:")
# for address in list(adjusted_accounts_data.keys())[:10]:  # Only display data for the first ten accounts
#     print(f"\nAccount {address} - First ten transactions:")
#     for transaction in adjusted_accounts_data[address][:10]:  # Show the first ten records for each account
#         print(transaction)

# # Define weight calculation function
# def calculate_weight(transaction):
#     weights = []
#     if '2-gram' in transaction:
#         weights.append(transaction['2-gram'] * 0.1)
#     if '3-gram' in transaction:
#         weights.append(transaction['3-gram'] * 0.2)
#     if '4-gram' in transaction:
#         weights.append(transaction['4-gram'] * 0.3)
#     if '5-gram' in transaction:
#         weights.append(transaction['5-gram'] * 0.4)
#     return np.sum(weights) if weights else 0  # Compute average; if the list is empty, return 0

# # Extract all unique account addresses, only include currently remaining accounts
# addresses = set(adjusted_accounts_data.keys())

# # Mapping from address to index
# address_to_index = {addr: idx for idx, addr in enumerate(addresses)}

# # Create adjacency matrix
# n = len(addresses)
# adj_matrix = np.zeros((n, n), dtype=float)  # Use float type to store weights
# # Save mapping from address to index
# save_data(address_to_index, 'data_Dataset.address_to_index')
# # Fill adjacency matrix
# for account, transactions in adjusted_accounts_data.items():
#     for transaction in transactions:
#         from_addr = transaction['from_address']
#         to_addr = transaction['to_address']
#         if from_addr in addresses and to_addr in addresses:
#             from_idx = address_to_index[from_addr]
#             to_idx = address_to_index[to_addr]
#             weight = calculate_weight(transaction)  # Compute weight
#             adj_matrix[from_idx, to_idx] += weight  # Accumulate weight

# # Save adjacency matrix
# save_data(adj_matrix, 'weighted_adjacency_matrix.pkl')

#ETH-GSetBert

# Dataset/adjust_matrix.py
import os
import sys
import pickle
import random
import numpy as np
import scipy.sparse as sp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_config import env_config

# ---------------------------
# Config mặc định (có thể đổi)
# ---------------------------
INPUT_PKL = 'transactions4.pkl'     # dict: addr -> list[tx dict]
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PREP_DIR = os.path.join(ROOT_DIR, 'data/preprocessed/Dataset')
SAVE_DENSE_WEIGHTED = True          # lưu thêm ma trận dense gốc (không bắt buộc)


# ============ utils i/o ============
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


# ============ trọng số cạnh ============
def calculate_weight(transaction):
    w = 0.0
    if '2-gram' in transaction:
        w += float(transaction['2-gram']) * 0.1
    if '3-gram' in transaction:
        w += float(transaction['3-gram']) * 0.2
    if '4-gram' in transaction:
        w += float(transaction['4-gram']) * 0.3
    if '5-gram' in transaction:
        w += float(transaction['5-gram']) * 0.4

    return w

# ============ chuẩn hoá A ============
def normalize_adj_sym_with_self_loop(adj: sp.coo_matrix) -> sp.coo_matrix:
    """
    A_hat = D^{-1/2} (A + I) D^{-1/2}  (chuẩn hoá đối xứng)
    - Chặn degree 0.
    - Trả về COO.
    """
    if not sp.isspmatrix(adj):
        raise TypeError("adj must be a scipy sparse matrix")
    adj = adj.tocsr()
    # add self-loop
    adj = adj + sp.eye(adj.shape[0], dtype=adj.dtype, format='csr')
    # degree
    rowsum = np.array(adj.sum(1)).ravel()
    rowsum[rowsum == 0.0] = 1.0
    d_inv_sqrt = np.power(rowsum, -0.5)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_hat = D_inv_sqrt @ adj @ D_inv_sqrt
    return A_hat.tocoo()


# ============ main build ============
def main():
    # seed để tái lập
    random.seed(env_config.GLOBAL_SEED)
    np.random.seed(env_config.GLOBAL_SEED)

    if not os.path.exists(INPUT_PKL):
        raise FileNotFoundError(f"Không thấy {INPUT_PKL}. Hãy đặt file dict addr->list(tx) tại đây.")

    print(f"[1/5] Load transactions from {INPUT_PKL}")
    accounts_data = load_pickle(INPUT_PKL)  # dict: addr -> list[tx dict]

    # phân loại normal/abnormal
    normal_accounts = {}
    abnormal_accounts = {}
    for addr, txs in accounts_data.items():
        if not txs:
            continue
        tag = txs[0].get('tag', None)
        if tag == 0:
            normal_accounts[addr] = txs
        elif tag == 1:
            abnormal_accounts[addr] = txs
        # nếu None thì bỏ qua (hoặc bạn có thể gán 0)

    num_abnormal = len(abnormal_accounts)
    num_normal   = len(normal_accounts)
    print(f"   abnormal={num_abnormal}, normal={num_normal}")

    # chọn 2x abnormal từ normal (nếu đủ), có seed
    k = min(num_normal, 2 * num_abnormal) if num_abnormal > 0 else num_normal
    selected_normal = random.sample(list(normal_accounts.keys()), k) if k > 0 else []
    adjusted_normal_accounts = {addr: normal_accounts[addr] for addr in selected_normal}
    adjusted_accounts_data = {**adjusted_normal_accounts, **abnormal_accounts}

    print(f"[2/5] Balanced subset: selected_normal={len(adjusted_normal_accounts)}, "
          f"abnormal={len(abnormal_accounts)}, total={len(adjusted_accounts_data)}")

    # mapping ổn định (sorted)
    os.makedirs(PREP_DIR, exist_ok=True)
    addresses = sorted(adjusted_accounts_data.keys())
    address_to_index = {addr: idx for idx, addr in enumerate(addresses)}
    save_pickle(address_to_index, os.path.join(PREP_DIR, 'data_Dataset.address_to_index'))
    print(f"[3/5] Saved address_to_index -> {os.path.join(PREP_DIR, 'data_Dataset.address_to_index')} "
          f"(N={len(address_to_index)})")

    # xây COO weighted (có thể là có hướng)
    print("[4/5] Build weighted adjacency (COO)")
    rows, cols, vals = [], [], []
    for _, txs in adjusted_accounts_data.items():
        for t in txs:
            fa = t.get('from_address')
            ta = t.get('to_address')
            if (fa in address_to_index) and (ta in address_to_index):
                w = calculate_weight(t)
                if w != 0.0:
                    rows.append(address_to_index[fa])
                    cols.append(address_to_index[ta])
                    vals.append(float(w))

    n = len(addresses)
    if len(vals) == 0:
        # tránh ma trận rỗng
        A = sp.coo_matrix((n, n), dtype=np.float32)
        print("   WARNING: all edge weights are zero -> empty graph.")
    else:
        A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)

    # (tuỳ) lưu dense gốc cho debug
    if SAVE_DENSE_WEIGHTED:
        save_pickle(A.toarray(), os.path.join(PREP_DIR, 'weighted_adjacency_matrix.pkl'))

    # chuẩn hoá đối xứng + self-loop
    print("[5/5] Normalize adjacency with self-loop (sym)")
    A_hat = normalize_adj_sym_with_self_loop(A)

    # lưu NPZ gọn nhẹ
    np.savez_compressed(
        os.path.join(PREP_DIR, 'norm_adj_coo.npz'),
        row=A_hat.row, col=A_hat.col, data=A_hat.data, shape=A_hat.shape
    )
    print(f"Done. Saved normalized adjacency to {os.path.join(PREP_DIR, 'norm_adj_coo.npz')}, "
          f"shape={A_hat.shape}, nnz={A_hat.nnz}")

    # in vài info
    print("\n[Summary]")
    print(f"  abnormal: {len(abnormal_accounts)}")
    print(f"  selected normal: {len(adjusted_normal_accounts)}")
    print(f"  total nodes (N): {n}")
    print(f"  edges (nnz weighted): {len(vals)}")


if __name__ == '__main__':
    main()

