# # ETH-GBert
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

# # Remove the 'tag' field from all transactions except the first one
# def remove_tag_except_first(accounts):
#     for address, transactions in accounts.items():
#         for i in range(1, len(transactions)):
#             if 'tag' in transactions[i]:
#                 del transactions[i]['tag']

# # Merge all transactions of each account into a single entry
# def merge_transactions(accounts):
#     for address in accounts.keys():
#         if accounts[address]:
#             first_tag = accounts[address][0]['tag']  # Keep the tag of the first transaction
#             merged_data = {'tag': first_tag, 'transactions': accounts[address]}
#             accounts[address] = [merged_data]

# # Load data
# accounts_data = load_data('transactions6.pkl')

# # Remove tag field
# remove_tag_except_first(accounts_data)

# # Merge transaction data
# merge_transactions(accounts_data)

# # Save data
# save_data(accounts_data, 'transactions7.pkl')

# # Print the first ten processed transactions for each account
# print("Print the first ten processed transactions for each account:")
# for address in list(accounts_data.keys())[:10]:  # Only display data for the first ten accounts
#     print(f"Account {address} - First ten transactions:")
#     for transaction in accounts_data[address][:10]:  # Show the first ten records for each account
#         print(transaction)
#     print("\n")

# print("Transaction data has been processed and saved to transactions7.pkl.")

# ETH-GSetBert
# Dataset/dataset7.py
import os
import pickle
from tqdm import tqdm

INPUT_PKL  = "transactions6.pkl"
OUTPUT_PKL = "transactions7.pkl"

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def ensure_sorted_by_timestamp(accounts):
    """Đảm bảo mỗi account sort tăng dần theo timestamp (phòng upstream đổi)."""
    for addr, txs in accounts.items():
        for t in txs:
            t["timestamp"] = int(t.get("timestamp", 0))
        txs.sort(key=lambda x: x["timestamp"])

def keep_tag_only_on_first(accounts, default_tag=0):
    """
    Giữ 'tag' ở giao dịch đầu tiên (transactions[0]['tag']), xóa ở các giao dịch sau.
    Nếu giao dịch đầu tiên không có 'tag', gán default_tag.
    """
    for addr, txs in accounts.items():
        if not txs:
            continue
        # đảm bảo tx đầu có 'tag'
        if "tag" not in txs[0]:
            txs[0]["tag"] = default_tag
        # xóa 'tag' ở các tx còn lại
        for i in range(1, len(txs)):
            if "tag" in txs[i]:
                del txs[i]["tag"]

def main():
    print(f"[load] {INPUT_PKL}")
    accounts = load_pickle(INPUT_PKL)
    if not isinstance(accounts, dict):
        raise TypeError("transactions6.pkl phải là dict addr -> list[tx]")

    # 1) đảm bảo thứ tự thời gian (an toàn)
    ensure_sorted_by_timestamp(accounts)

    # 2) giữ tag chỉ ở giao dịch đầu
    keep_tag_only_on_first(accounts, default_tag=0)

    print(f"[save] {OUTPUT_PKL}")
    save_pickle(accounts, OUTPUT_PKL)

    # preview
    print("\n[preview]")
    shown = 0
    for addr, lst in accounts.items():
        print(f"Account {addr}: {len(lst)} txs")
        for t in lst[:5]:
            print("  ", t)
        print()
        shown += 1
        if shown >= 3:
            break

if __name__ == "__main__":
    main()
