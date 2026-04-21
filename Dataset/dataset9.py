# #ETH-GBert
# import pickle
# import tqdm

# # Load data from file
# def load_data(filename):
#     with open(filename, 'rb') as file:
#         return pickle.load(file)

# # Save data to file
# def save_data(data, filename):
#     with open(filename, 'wb') as file:
#         pickle.dump(data, file)

# # Remove 'tag' field from transactions
# def remove_tag_from_transactions(accounts):
#     for address, transactions in accounts.items():
#         for transaction in transactions:
#             for sub_transaction in transaction['transactions']:
#                 if 'tag' in sub_transaction:
#                     del sub_transaction['tag']

# # Load data
# accounts_data = load_data('transactions8.pkl')

# # Remove 'tag' field
# remove_tag_from_transactions(accounts_data)

# # Save data
# save_data(accounts_data, 'transactions9.pkl')

# # Print the first ten accounts' data
# print("Print the first ten accounts:")
# for address, transactions in list(accounts_data.items())[:10]:  # Only display data for the first ten accounts
#     print(f"Account {address}:")
#     for transaction in transactions:
#         print(transaction)
#     print("\n")

# print("The 'tag' field has been removed and the data has been saved to transactions9.pkl.")

# Dataset/dataset9.py
import os
import pickle
from tqdm import tqdm

INPUT_PKL  = "transactions8.pkl"
OUTPUT_PKL = "transactions9.pkl"
DEFAULT_TAG = 0

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def keep_tag_only_on_first(accounts, default_tag=DEFAULT_TAG):
    """
    Với mỗi account:
      - Đảm bảo tx đầu có 'tag' (nếu thiếu => gán default_tag)
      - Xoá 'tag' khỏi các giao dịch còn lại (nếu có)
    """
    for addr, txs in tqdm(accounts.items(), desc="normalize tag"):
        if not txs:
            continue
        if "tag" not in txs[0]:
            txs[0]["tag"] = default_tag
        for i in range(1, len(txs)):
            if "tag" in txs[i]:
                del txs[i]["tag"]

def main():
    print(f"[load] {INPUT_PKL}")
    accounts = load_pickle(INPUT_PKL)
    if not isinstance(accounts, dict):
        raise TypeError("transactions8.pkl phải là dict addr -> list[tx]")

    keep_tag_only_on_first(accounts, default_tag=DEFAULT_TAG)

    print(f"[save] {OUTPUT_PKL}")
    save_pickle(accounts, OUTPUT_PKL)

    # preview
    print("\n[preview]")
    shown = 0
    for addr, lst in accounts.items():
        print(f"Account {addr}: {len(lst)} txs, tag_first={lst[0].get('tag', None) if lst else None}")
        for t in lst[:3]:
            print("  ", t)
        print()
        shown += 1
        if shown >= 3:
            break

if __name__ == "__main__":
    main()







