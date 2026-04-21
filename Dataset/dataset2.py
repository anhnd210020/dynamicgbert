#ETH-GBert
# import pickle

# # Read data from file
# def load_data(filename):
#     with open(filename, 'rb') as file:
#         return pickle.load(file)

# # Save data to file
# def save_data(data, filename):
#     with open(filename, 'wb') as file:
#         pickle.dump(data, file)

# # Main processing function
# def process_transactions(transactions):
#     # Create a dictionary to store transactions for each address
#     accounts = {}

#     # Process transaction data
#     for tx in transactions:
#         # Add outgoing transactions
#         from_address = tx['from_address']
#         if from_address not in accounts:
#             accounts[from_address] = []
#         accounts[from_address].append({**tx, 'in_out': 1})  # Add outgoing flag

#         # Add incoming transactions
#         to_address = tx['to_address']
#         if to_address not in accounts:
#             accounts[to_address] = []
#         accounts[to_address].append({**tx, 'in_out': 0})  # Add incoming flag

#     return accounts

# # Load data
# transactions = load_data('transactions1.pkl')

# # Process data
# processed_data = process_transactions(transactions)

# # Save processed data
# save_data(processed_data, 'transactions2.pkl')

# # Print first 10 addresses for inspection
# for address in list(processed_data.keys())[:10]:  # Show only the first 10 accounts
#     print(f"Transactions for account {address}:")
#     for transaction in processed_data[address][:5]:  # Show first 5 transactions per account
#         print(transaction)
#     print("\n")

#ETH-GSetBert
# Dataset/dataset2.py
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

def iter_all_transactions(data):
    """
    Hỗ trợ 2 kiểu:
      - list[tx]
      - dict addr -> list[tx]
    Yield từng giao dịch tx (dict).
    """
    if isinstance(data, dict):
        for _, tx_list in data.items():
            for tx in tx_list:
                yield tx
    elif isinstance(data, list):
        for tx in data:
            yield tx
    else:
        raise TypeError("transactions must be a list or dict")

def process_transactions_make_accounts(transactions_iter):
    """
    Tạo dict: account_addr -> list[tx_with_inout]
    Mỗi tx xuất hiện 2 lần:
      - ở from_address với in_out=1
      - ở to_address   với in_out=0
    """
    accounts = {}
    for tx in tqdm(transactions_iter, desc="build in/out per account"):
        fa = tx.get("from_address")
        ta = tx.get("to_address")
        ts = int(tx.get("timestamp", 0))

        # outgoing (from) - in_out = 1
        if fa is not None:
            tx_out = dict(tx)
            tx_out["in_out"] = 1
            accounts.setdefault(fa, []).append(tx_out)

        # incoming (to) - in_out = 0
        if ta is not None:
            tx_in = dict(tx)
            tx_in["in_out"] = 0
            accounts.setdefault(ta, []).append(tx_in)

    # sort theo thời gian tăng dần cho mỗi account
    for addr, lst in accounts.items():
        lst.sort(key=lambda t: int(t.get("timestamp", 0)))

    return accounts

def main():
    in_file = "transactions1.pkl"     # đầu vào từ dataset1.py (dict addr->[tx,...])
    out_file = "transactions2.pkl"

    print(f"[load] {in_file}")
    raw = load_pickle(in_file)

    tx_iter = iter_all_transactions(raw)
    accounts = process_transactions_make_accounts(tx_iter)

    print(f"[save] {out_file}  (accounts={len(accounts)})")
    save_pickle(accounts, out_file)

    # preview
    print("\n[preview]")
    for i, (addr, txs) in enumerate(accounts.items()):
        if i >= 3:
            break
        print(f"Account {addr}: {len(txs)} txs")
        for t in txs[:3]:
            print("  ", {k: t[k] for k in ["tag","from_address","to_address","in_out","amount","timestamp"] if k in t})

if __name__ == "__main__":
    main()

