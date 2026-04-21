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

# # Add n-gram data to each transaction
# def add_n_grams(accounts):
#     for address, transactions in tqdm.tqdm(accounts.items(), desc="Processing n-gram data"):
#         for n in range(2, 6):  # Process 2-gram to 5-gram
#             gram_key = f"{n}-gram"
#             for i in range(len(transactions)):
#                 if i < n-1:
#                     transactions[i][gram_key] = 0  # Set initial n-1 values to 0
#                 else:
#                     transactions[i][gram_key] = transactions[i]['timestamp'] - transactions[i-n+1]['timestamp']

# # Load data
# accounts_data = load_data('transactions3.pkl')

# # Add n-gram data
# add_n_grams(accounts_data)

# # Save data
# save_data(accounts_data, 'transactions4.pkl')

# # Print the first ten processed transactions for each account
# print("Print the first ten processed transactions for each account:")
# for address in list(accounts_data.keys())[:10]:  # Only display data for the first ten accounts
#     print(f"Account {address} First ten transactions:")
#     for transaction in accounts_data[address][:10]:  # Show the first ten records for each account
#         print(transaction)
#     print("\n")

# print("n-gram calculations have been completed and saved to transactions4.pkl.")

#ETH-GSetBert
# Dataset/dataset4.py
import os
import pickle
from tqdm import tqdm

N_MAX_GRAM = 5  # sẽ tạo 2..5-gram

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def ensure_sorted_by_ts(accounts):
    """Đảm bảo mỗi account đã sort theo timestamp tăng dần và ép ts=int."""
    for addr, lst in accounts.items():
        for t in lst:
            t["timestamp"] = int(t.get("timestamp", 0))
        lst.sort(key=lambda x: x["timestamp"])

def add_time_ngrams(accounts, max_n=N_MAX_GRAM):
    """
    Thêm các trường '2-gram'..'max_n-gram' theo định nghĩa:
      n-gram[i] = t[i] - t[i-(n-1)]  (giây)  nếu i >= n-1, else 0.
    """
    for addr, txs in tqdm(accounts.items(), desc="add n-grams"):
        # đã đảm bảo sort ở ensure_sorted_by_ts
        T = len(txs)
        if T == 0:
            continue
        for n in range(2, max_n + 1):
            key = f"{n}-gram"
            for i in range(T):
                if i < n - 1:
                    txs[i][key] = 0.0
                else:
                    dt = txs[i]["timestamp"] - txs[i - (n - 1)]["timestamp"]
                    # nếu muốn không âm: dt = max(0, dt)
                    txs[i][key] = float(dt)

def main():
    in_file = "transactions3.pkl"
    out_file = "transactions4.pkl"

    print(f"[load] {in_file}")
    accounts = load_pickle(in_file)
    if not isinstance(accounts, dict):
        raise TypeError("transactions3.pkl phải là dict addr -> list[tx]")

    # 1) đảm bảo sort theo thời gian
    ensure_sorted_by_ts(accounts)

    # 2) thêm n-grams thời gian 2..5
    add_time_ngrams(accounts, max_n=5)

    print(f"[save] {out_file}")
    save_pickle(accounts, out_file)

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





