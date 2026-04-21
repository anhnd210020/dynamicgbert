# # ETH-GBert
# import pickle

# # Load data from file
# def load_data(filename):
#     with open(filename, 'rb') as file:
#         return pickle.load(file)

# # Save data to file
# def save_data(data, filename):
#     with open(filename, 'wb') as file:
#         pickle.dump(data, file)

# # Sort each account's transaction data by timestamp
# def sort_transactions_by_timestamp(accounts):
#     sorted_accounts = {}
#     for address, transactions in accounts.items():
#         sorted_accounts[address] = sorted(transactions, key=lambda x: x['timestamp'])
#     return sorted_accounts

# # Load data
# accounts_data = load_data('transactions2.pkl')

# # Sort data
# sorted_accounts_data = sort_transactions_by_timestamp(accounts_data)

# # Print the first ten sorted transactions for each account
# print("Print the first ten sorted transactions for each account:")
# for address in list(sorted_accounts_data.keys())[:10]:  # Only display data for the first ten accounts
#     print(f"Account {address} - First ten transactions:")
#     for transaction in sorted_accounts_data[address][:10]:  # Show the first ten records for each account
#         print(transaction)
#     print("\n")

# # Save data
# save_data(sorted_accounts_data, 'transactions3.pkl')

# print("Data has been sorted by timestamp for each account and saved to transactions3.pkl.")

# ETH-GSetBert
# Dataset/dataset3.py
import os
import pickle
from tqdm import tqdm

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def sort_and_dedup_accounts(accounts, dedup=True):
    """
    accounts: dict addr -> list[tx]
    - Ép timestamp=int (fallback 0)
    - Sort theo timestamp tăng dần
    - (tuỳ chọn) Khử trùng lặp trong từng account
    """
    out = {}
    for addr, tx_list in tqdm(accounts.items(), desc="sort_by_timestamp"):
        # ép timestamp
        for t in tx_list:
            t["timestamp"] = int(t.get("timestamp", 0))

        # sort ổn định (Python sort là stable)
        tx_sorted = sorted(tx_list, key=lambda x: x["timestamp"])

        if dedup:
            seen = set()
            deduped = []
            for t in tx_sorted:
                key = (
                    t.get("from_address"),
                    t.get("to_address"),
                    float(t.get("amount", 0.0)),
                    int(t.get("timestamp", 0)),
                    int(t.get("in_out", 0)),
                    int(t.get("tag", 0)),
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(t)
            tx_sorted = deduped

        out[addr] = tx_sorted
    return out

def main():
    in_file = "transactions2.pkl"
    out_file = "transactions3.pkl"

    print(f"[load] {in_file}")
    accounts = load_pickle(in_file)
    if not isinstance(accounts, dict):
        raise TypeError("transactions2.pkl phải là dict addr -> list[tx]")

    sorted_accounts = sort_and_dedup_accounts(accounts, dedup=True)

    print(f"[save] {out_file}  (accounts={len(sorted_accounts)})")
    save_pickle(sorted_accounts, out_file)

    # preview
    print("\n[preview]")
    shown = 0
    for addr, lst in sorted_accounts.items():
        print(f"Account {addr}: {len(lst)} txs")
        for t in lst[:5]:
            print("  ", {k: t.get(k) for k in ["tag","from_address","to_address","in_out","amount","timestamp"]})
        print()
        shown += 1
        if shown >= 3:
            break

if __name__ == "__main__":
    main()





