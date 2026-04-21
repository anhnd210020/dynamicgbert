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

# # Remove specific fields
# def remove_fields(accounts, fields):
#     for address in tqdm.tqdm(accounts.keys(), desc="Removing fields"):
#         for transaction in accounts[address]:
#             for field in fields:
#                 if field in transaction:
#                     del transaction[field]

# # Load data
# accounts_data = load_data('transactions4.pkl')

# # Fields to remove
# fields_to_remove = ['from_address', 'to_address', 'timestamp']

# # Remove fields
# remove_fields(accounts_data, fields_to_remove)

# # Save data
# save_data(accounts_data, 'transactions5.pkl')

# # Print the first ten processed transactions for each account
# print("Print the first ten processed transactions for each account:")
# for address in list(accounts_data.keys())[:10]:  # Only display data for the first ten accounts
#     print(f"Account {address} - First ten transactions:")
#     for transaction in accounts_data[address][:10]:  # Show the first ten records for each account
#         print(transaction)
#     print("\n")

# print("Fields have been removed and data saved to transactions5.pkl.")

#ETH-GSetBert
# Dataset/dataset5.py
import os
import pickle
from tqdm import tqdm

INPUT_PKL  = "transactions4.pkl"
OUTPUT_PKL = "transactions5.pkl"

# Các khóa BẮT BUỘC phải giữ lại để pipeline an toàn
REQUIRED_KEYS = {"from_address", "to_address", "timestamp"}

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def remove_fields_safely(accounts, fields_to_remove):
    # Lọc bỏ những khóa bắt buộc khỏi danh sách xóa
    will_skip = sorted(list(REQUIRED_KEYS.intersection(fields_to_remove)))
    if will_skip:
        print(f"[WARN] Bỏ qua xoá các khóa bắt buộc (đang dùng downstream): {will_skip}")
    fields = [f for f in fields_to_remove if f not in REQUIRED_KEYS]

    if not fields:
        print("[INFO] Không còn trường nào để xoá sau khi lọc an toàn.")
        return accounts

    for addr, txs in tqdm(accounts.items(), desc="Removing fields"):
        for t in txs:
            for f in fields:
                if f in t:
                    del t[f]
    return accounts

def main():
    print(f"[load] {INPUT_PKL}")
    accounts = load_pickle(INPUT_PKL)
    if not isinstance(accounts, dict):
        raise TypeError("transactions4.pkl phải là dict addr -> list[tx]")

    # Tuỳ bạn: chỉnh danh sách dưới đây — 3 khóa bắt buộc sẽ tự động không bị xoá
    fields_to_remove = ['from_address', 'to_address', 'timestamp']  # sẽ bị lọc bỏ an toàn
    accounts = remove_fields_safely(accounts, fields_to_remove)

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






