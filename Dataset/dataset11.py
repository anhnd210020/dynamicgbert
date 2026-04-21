# # ETH-GBert
# import pickle
# from sklearn.model_selection import train_test_split

# # Load data from transactions10.pkl
# with open('transactions10.pkl', 'rb') as file:
#     transactions_dict = pickle.load(file)

# # Convert the dictionary into a list, each element is a string in the format "tag sentence"
# transactions = []
# for key, value_list in transactions_dict.items():
#     for value in value_list:
#         transactions.append(f"{value}")  # Assume key is the tag and value is the description

# # Define data split ratios
# train_size = 0.8
# validation_size = 0.1
# test_size = 0.1

# # First, split the training set and the remaining part
# train_data, temp_data = train_test_split(transactions, train_size=train_size, random_state=42)

# # Then, split the remaining part into validation and test sets
# validation_data, test_data = train_test_split(temp_data, test_size=test_size/(test_size + validation_size), random_state=42)

# # Function to save training and validation data to TSV files
# def save_to_tsv_train_dev(data, filename):
#     with open(filename, 'w', encoding='utf-8') as file:
#         file.write("label\tsentence\n")
#         for line in data:
#             # Assume the tag is at the beginning of the line, and the rest is the sentence
#             tag, sentence = line.split(' ', 1)
#             file.write(f"{tag}\t{sentence}\n")

# # Function to save test data to TSV file
# def save_to_tsv_test(data, filename):
#     with open(filename, 'w', encoding='utf-8') as file:
#         file.write("index\tsentence\n")
#         for idx, line in enumerate(data):
#             # Split the tag and the remaining description
#             tag, sentence = line.split(' ', 1)
#             file.write(f"{idx}\t{sentence}\n")

# # Save training, validation, and test sets
# save_to_tsv_train_dev(train_data, 'train.tsv')
# save_to_tsv_train_dev(validation_data, 'dev.tsv')
# save_to_tsv_test(test_data, 'test.tsv')

# print("Files saved: train.tsv, dev.tsv, test.tsv")

# ETH-GSetBert
# Dataset/dataset11.py
import os
import sys
import pickle
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from env_config import env_config

INPUT_PKL  = "transactions10.pkl"
TRAIN_TSV  = "train.tsv"
DEV_TSV    = "dev.tsv"   # trong pipeline sẽ dùng làm "test có nhãn"
TEST_TSV   = "test.tsv"  # test không nhãn (để mô phỏng)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def parse_tag_sentence(line: str) -> Tuple[int, str]:
    """
    Expect: "<tag> <sentence...>"
    Trả về (tag:int, sentence:str). Nếu lỗi định dạng -> raise ValueError.
    """
    line = line.strip()
    if not line:
        raise ValueError("empty line")
    # tách token đầu là tag, phần sau là sentence
    parts = line.split(" ", 1)
    if len(parts) < 2:
        raise ValueError(f"bad format (no space): {line[:100]}")
    tag_str, sent = parts[0], parts[1].strip()
    tag = int(tag_str)  # sẽ raise nếu không phải số
    return tag, sent

def sanitize_sentence(s: str) -> str:
    # tránh tab/xuống dòng phá TSV
    return s.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()

def dump_train_dev(pairs: List[Tuple[int, str]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("label\tsentence\n")
        for y, s in pairs:
            f.write(f"{y}\t{sanitize_sentence(s)}\n")

def dump_test(sentences: List[str], path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("index\tsentence\n")
        for i, s in enumerate(sentences):
            f.write(f"{i}\t{sanitize_sentence(s)}\n")

def main():
    data = load_pickle(INPUT_PKL)  # dict[address] -> [ "<tag> sentence ..." ]
    # gom tất cả câu (mỗi account là 1-list có 1 câu)
    raw_lines: List[str] = []
    for _, lst in data.items():
        for v in lst:
            raw_lines.append(str(v))

    # tách nhãn/xâu
    pairs: List[Tuple[int, str]] = []
    errors = 0
    for line in raw_lines:
        try:
            y, s = parse_tag_sentence(line)
            pairs.append((y, s))
        except Exception:
            errors += 1
    if errors:
        print(f"[WARN] Bỏ qua {errors} dòng lỗi định dạng.")

    if not pairs:
        raise RuntimeError("Không có mẫu hợp lệ sau khi parse tag/sentence.")

    # tách X,y để stratify
    y_all = [y for y, _ in pairs]
    X_all = [s for _, s in pairs]

    # split: 80/10/10, stratified theo y
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_all, y_all,
        train_size=0.8,
        random_state=env_config.GLOBAL_SEED,
        stratify=y_all
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=0.5,  # của phần còn lại 20% -> 10/10
        random_state=env_config.GLOBAL_SEED,
        stratify=y_tmp
    )

    # ghi file: train/dev có nhãn; test (mô phỏng) không nhãn
    dump_train_dev(list(zip(y_train, X_train)), TRAIN_TSV)
    dump_train_dev(list(zip(y_valid, X_valid)), DEV_TSV)
    dump_test(X_test, TEST_TSV)

    print(f"Saved: {TRAIN_TSV} (n={len(X_train)}), {DEV_TSV} (n={len(X_valid)}), {TEST_TSV} (n={len(X_test)})")

if __name__ == "__main__":
    main()

