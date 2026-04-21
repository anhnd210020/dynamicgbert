# # ETH-GBert
# import pickle
# import random
# import tqdm

# # Load data from file
# def load_data(filename):
#     with open(filename, 'rb') as file:
#         return pickle.load(file)

# # Save data to file
# def save_data(data, filename):
#     with open(filename, 'wb') as file:
#         pickle.dump(data, file)

# # Select and shuffle accounts
# def select_and_shuffle_accounts(accounts):
#     tag1_accounts = [account for account in accounts.items() if account[1][0]['tag'] == 1]
#     tag0_accounts = [account for account in accounts.items() if account[1][0]['tag'] == 0]
    
#     # Randomly select accounts with tag = 0, the number is twice the number of tag = 1 accounts
#     double_tag1_count = random.sample(tag0_accounts, 2 * len(tag1_accounts))
    
#     # Combine and shuffle the order
#     selected_accounts = tag1_accounts + double_tag1_count
#     random.shuffle(selected_accounts)
    
#     # Return shuffled dictionary
#     return dict(selected_accounts)

# # Load data
# accounts_data = load_data('transactions7.pkl')

# # Select and shuffle accounts
# shuffled_accounts_data = select_and_shuffle_accounts(accounts_data)

# # Save data
# save_data(shuffled_accounts_data, 'transactions8.pkl')

# # Print the first ten processed accounts
# print("Print the first ten accounts:")
# for address, transactions in list(shuffled_accounts_data.items())[:10]:  # Only display data for the first ten accounts
#     print(f"Account {address}:")
#     print(transactions)
#     print("\n")

# print("Data has been processed and saved to transactions8.pkl.")

# ETH-GSetBert
# Dataset/dataset8.py
import os
import sys
import pickle
import random
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_config import env_config

INPUT_PKL  = "transactions7.pkl"
OUTPUT_PKL = "transactions8.pkl"
RATIO = 2  # số tài khoản tag=0 lấy = RATIO * số tài khoản tag=1

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def first_tx_tag(txs, default=0):
    """Đảm bảo giao dịch đầu có tag; nếu thiếu, trả default."""
    if not txs:
        return default
    if "tag" not in txs[0]:
        return default
    return int(txs[0]["tag"])

def select_and_shuffle_accounts(accounts, ratio=RATIO, seed=env_config.GLOBAL_SEED):
    """
    Chọn toàn bộ tài khoản tag=1 và ngẫu nhiên RATIO× số lượng tài khoản tag=0.
    Shuffle thứ tự tài khoản (tái lập nhờ seed). Không thay đổi thứ tự giao dịch trong tài khoản.
    """
    rng = random.Random(seed)

    tag1_accounts = []
    tag0_accounts = []

    for addr, txs in accounts.items():
        tag = first_tx_tag(txs, default=0)
        # ép tag vào tx đầu nếu thiếu, để downstream không lỗi
        if txs and "tag" not in txs[0]:
            txs[0]["tag"] = tag

        if tag == 1:
            tag1_accounts.append((addr, txs))
        else:
            tag0_accounts.append((addr, txs))

    n1 = len(tag1_accounts)
    n0 = len(tag0_accounts)

    # Các trường hợp biên
    if n1 == 0 and n0 == 0:
        return {}
    if n1 == 0:
        # Không có tài khoản tag=1 -> chỉ shuffle toàn bộ tag=0
        rng.shuffle(tag0_accounts)
        return dict(tag0_accounts)
    if n0 == 0:
        # Không có tài khoản tag=0 -> chỉ shuffle toàn bộ tag=1
        rng.shuffle(tag1_accounts)
        return dict(tag1_accounts)

    # Chọn ngẫu nhiên min(n0, ratio*n1) tài khoản tag=0
    sample_n0 = min(n0, ratio * n1)
    selected_tag0 = rng.sample(tag0_accounts, sample_n0)

    selected = tag1_accounts + selected_tag0
    rng.shuffle(selected)

    return dict(selected)

def main():
    print(f"[load] {INPUT_PKL}")
    accounts = load_pickle(INPUT_PKL)
    if not isinstance(accounts, dict):
        raise TypeError("transactions7.pkl phải là dict addr -> list[tx]")

    selected_accounts = select_and_shuffle_accounts(accounts, ratio=RATIO)

    print(f"[save] {OUTPUT_PKL}  (accounts={len(selected_accounts)})")
    save_pickle(selected_accounts, OUTPUT_PKL)

    # preview
    print("\n[preview]")
    shown = 0
    for addr, txs in selected_accounts.items():
        print(f"Account {addr}: {len(txs)} txs, tag(first)={first_tx_tag(txs)}")
        for t in txs[:3]:
            keep = {k: t.get(k) for k in ["tag","from_address","to_address","in_out","amount","timestamp","2-gram","3-gram","4-gram","5-gram"] if k in t}
            print("  ", keep)
        print()
        shown += 1
        if shown >= 3:
            break

if __name__ == "__main__":
    main()






