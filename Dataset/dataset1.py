#ETH-GBert
# import pickle
# import networkx as nx
# from tqdm import tqdm
# import pandas as pd
# import functools
# import pickle

# def read_pkl(pkl_file):
#     # Load data from a pkl file
#     print(f'Reading {pkl_file}...')
#     with open(pkl_file, 'rb') as file:
#         data = pickle.load(file)
#     return data

# def save_pkl(data, pkl_file):
#     # Save data to a pkl file
#     print(f'Saving data to {pkl_file}...')
#     with open(pkl_file, 'wb') as file:
#         pickle.dump(data, file)

# def load_and_print_pkl(pkl_file):
#     # Load a pkl file
#     print(f'Loading {pkl_file}...')
#     with open(pkl_file, 'rb') as file:
#         data = pickle.load(file)
    
#     # Print the first ten records of the data
#     for i, transaction in enumerate(data):
#         if i < 10:  # Only print the first ten records
#             print(transaction)
#         else:
#             break

# def extract_transactions(G):
#     # Extract all transaction data from the graph
#     transactions = []
#     for from_address, to_address, key, tnx_info in tqdm(G.edges(keys=True, data=True), desc='accounts_data_generate'):
#         amount = tnx_info['amount']
#         block_timestamp = int(tnx_info['timestamp'])
#         tag = G.nodes[from_address]['isp']
#         transaction = {
#             'tag': tag,
#             'from_address': from_address,
#             'to_address': to_address,
#             'amount': amount,
#             'timestamp': block_timestamp,
#         }
#         transactions.append(transaction)
#     return transactions

# def data_generate():
#     graph_file = '/home/ducanhhh/Dynamic_Feature/Dataset/MulDiGraph.pkl'
#     out_file = 'transactions1.pkl'
    
#     # Read graph data
#     graph = read_pkl(graph_file)
#     # Extract transaction data
#     transactions = extract_transactions(graph)
#     # Save transaction data to a new file
#     save_pkl(transactions, out_file)

# if __name__ == '__main__':
#     data_generate()
#     pkl_file = 'transactions1.pkl'  # Make sure this file path is correct
#     load_and_print_pkl(pkl_file)

#ETH-GSetBert
# Dataset/dataset1.py
import os
import sys
import pickle
from tqdm import tqdm

import numpy as np
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_config import env_config


def read_pkl(pkl_file):
    print(f"[load] {pkl_file}")
    with open(pkl_file, "rb") as f:
        return pickle.load(f, encoding="latin1")


def save_pkl(data, pkl_file):
    os.makedirs(os.path.dirname(pkl_file) or ".", exist_ok=True)
    print(f"[save] {pkl_file}")
    with open(pkl_file, "wb") as f:
        pickle.dump(data, f)


def extract_transactions_by_from_addr(G: nx.MultiDiGraph):
    """
    Trả về dict: from_addr -> list[tx]
    tx = {
        'tag'         : int (0/1), từ node attr 'isp', mặc định 0 nếu thiếu,
        'from_address': str,
        'to_address'  : str,
        'amount'      : float,
        'timestamp'   : int
    }
    """
    tx_index = {}

    # duyệt cạnh; với MultiDiGraph có thể có nhiều cạnh giữa 2 nút
    for u, v, k, data in tqdm(G.edges(keys=True, data=True), desc="extract"):
        amount = float(data.get("amount", 0.0))
        ts = int(data.get("timestamp", 0))
        tag = int(G.nodes[u].get("isp", 0))  # 0=normal, 1=abnormal (tuỳ dataset)

        tx = {
            "tag": tag,
            "from_address": u,
            "to_address": v,
            "amount": amount,
            "timestamp": ts,
        }
        tx_index.setdefault(u, []).append(tx)

    # sort theo thời gian tăng dần mỗi địa chỉ
    for addr, lst in tx_index.items():
        lst.sort(key=lambda t: t["timestamp"])

    return tx_index


def main():
    # đường dẫn input/output (đổi nếu cần)
    graph_file = os.path.join(os.path.dirname(__file__), "MulDiGraph.pkl")
    out_file = "transactions1.pkl"

    G = read_pkl(graph_file)
    if not isinstance(G, (nx.MultiDiGraph, nx.DiGraph)):
        raise TypeError("MulDiGraph.pkl phải là networkx MultiDiGraph/DiGraph")

    tx_index = extract_transactions_by_from_addr(G)

    save_pkl(tx_index, out_file)

    # In thử 3 địa chỉ đầu và 3 giao dịch đầu mỗi địa chỉ
    print("\n[preview]")
    addrs = list(tx_index.keys())[:3]
    for a in addrs:
        print(f"Account {a}: {len(tx_index[a])} txs")
        for t in tx_index[a][:3]:
            print("  ", t)


if __name__ == "__main__":
    # seed cho tái lập (nếu sau này bạn random/filter gì thêm)
    np.random.seed(env_config.GLOBAL_SEED)
    main()
