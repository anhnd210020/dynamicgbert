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

# # Convert transaction data into descriptive text
# def convert_transactions_to_text(accounts):
#     for address, transactions in accounts.items():
#         for idx, transaction in enumerate(transactions):
#             tag = transaction['tag']
#             transaction_descriptions = []
#             for sub_transaction in transaction['transactions']:
#                 # Construct the description for a single transaction
#                 description = ' '.join([f"{key}: {sub_transaction[key]}" for key in sub_transaction])
#                 transaction_descriptions.append(description)
#             # Update the transaction data into a single line of text description
#             transactions[idx] = f"{tag} {'  '.join(transaction_descriptions)}."

# # Load data
# accounts_data = load_data('transactions9.pkl')

# # Convert transaction data into text descriptions
# convert_transactions_to_text(accounts_data)

# # Save data
# save_data(accounts_data, 'transactions10.pkl')

# # Print the first ten accounts' data
# print("Print the first ten accounts:")
# for address, transactions in list(accounts_data.items())[:10]:  # Only display data for the first ten accounts
#     print(f"Account {address}:")
#     for transaction in transactions:
#         print(transaction)
#     print("\n")

# print("Data has been converted into descriptive text and saved to transactions10.pkl.")

# ETH-GSetBert

# Dataset/dataset10.py
import os
import pickle
from tqdm import tqdm

INPUT_PKL  = "transactions9.pkl"
OUTPUT_PKL = "transactions10.pkl"

# Các trường sẽ đưa vào mô tả (tuỳ biến được)
DESC_FIELDS = ["in_out", "amount", "2-gram", "3-gram", "4-gram", "5-gram"]

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def tx_to_short_desc(tx):
    """Chuyển 1 giao dịch thành chuỗi ngắn gọn theo các field đã chọn."""
    parts = []
    for k in DESC_FIELDS:
        if k in tx:
            parts.append(f"{k}:{tx[k]}")
    return " ".join(parts)

def convert_accounts_to_text(accounts):
    """
    accounts: dict[address] -> list[tx]
    Trả về dict[address] -> [ "tag sentence..." ], mỗi account đúng 1 câu mô tả,
    bắt đầu bằng tag của giao dịch đầu tiên.
    """
    out = {}
    for addr, txs in tqdm(accounts.items(), desc="to_text"):
        if not txs:
            # không có giao dịch -> ghi câu rỗng với tag 0
            out[addr] = ["0 ."]
            continue

        tag = txs[0].get("tag", 0)  # đảm bảo có tag đầu
        descs = [tx_to_short_desc(t) for t in txs]
        # Ghép thành một câu: "<tag> feat1 feat2 ...  feat... ."
        sentence = f"{int(tag)} " + "  ".join([d for d in descs if d]) + "."
        out[addr] = [sentence]
    return out

def main():
    print(f"[load] {INPUT_PKL}")
    accounts = load_pickle(INPUT_PKL)
    if not isinstance(accounts, dict):
        raise TypeError("transactions9.pkl phải là dict addr -> list[tx]")

    text_accounts = convert_accounts_to_text(accounts)

    print(f"[save] {OUTPUT_PKL}")
    save_pickle(text_accounts, OUTPUT_PKL)

    # preview
    print("\n[preview]")
    shown = 0
    for addr, lst in text_accounts.items():
        print(f"Account {addr}:")
        for s in lst:
            print("  ", s[:200] + ("..." if len(s) > 200 else ""))
        print()
        shown += 1
        if shown >= 3:
            break

if __name__ == "__main__":
    main()






