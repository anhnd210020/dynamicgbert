# #ETH-GBert
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

# # Shuffle the order of transaction data within each account
# def shuffle_transactions(accounts):
#     for address in tqdm.tqdm(accounts.keys(), desc="Shuffling transactions"):
#         random.shuffle(accounts[address])

# # Load data
# accounts_data = load_data('transactions5.pkl')

# # Shuffle transaction data
# shuffle_transactions(accounts_data)

# # Save data
# save_data(accounts_data, 'transactions6.pkl')

# # Print the first five processed transactions for each account
# print("Print the first five processed transactions for each account:")
# for address in list(accounts_data.keys())[:5]:  # Only display data for the first five accounts
#     print(f"Account {address} - First five transactions:")
#     for transaction in accounts_data[address][:5]:  # Show the first five records for each account
#         print(transaction)
#     print("\n")

# print("Transaction data has been shuffled and saved to transactions6.pkl.")

#ETH-GSetBert
# Dataset/dataset6.py
import os
import pickle
import random
from tqdm import tqdm

INPUT_PKL  = "transactions5.pkl"
OUTPUT_PKL = "transactions6.pkl"

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def ensure_sorted_by_timestamp(accounts):
    """Giữ đúng thứ tự thời gian (tăng dần) cho mỗi tài khoản."""
    for addr, txs in tqdm(accounts.items(), desc="ensure_sorted_by_timestamp"):
        for t in txs:
            t["timestamp"] = int(t.get("timestamp", 0))
        txs.sort(key=lambda x: x["timestamp"])
    return accounts

def main():
    print(f"[load] {INPUT_PKL}")
    accounts = load_pickle(INPUT_PKL)
    if not isinstance(accounts, dict):
        raise TypeError("transactions5.pkl phải là dict addr -> list[tx]")
    print(f"[save] {OUTPUT_PKL}")
    save_pickle(accounts, OUTPUT_PKL)

    # preview
    print("\n[preview]")
    shown = 0
    for addr, lst in accounts.items():
        print(f"Account {addr}: {len(lst)} txs")
        for t in lst[:5]:
            keep = {k: t.get(k) for k in ["tag","from_address","to_address","in_out","amount","timestamp","2-gram","3-gram","4-gram","5-gram"] if k in t}
            print("  ", keep)
        print()
        shown += 1
        if shown >= 3:
            break

if __name__ == "__main__":
    main()
