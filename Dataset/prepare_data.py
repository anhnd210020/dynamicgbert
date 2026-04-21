# #ETH-GBert
# import subprocess
# import os
# import shutil

# def run_script(script_name):
#     try:
#         print(f"Running {script_name}...")
#         subprocess.run(['python', script_name], check=True)
#         print(f"{script_name} completed successfully.\n")
#     except subprocess.CalledProcessError as e:
#         print(f"Error occurred while running {script_name}: {e}")


# import os
# import shutil

# def move_files_to_preprocessed_folder():
#     # Correct the folder name if it's a typo (change 'preprocesse' to 'preprocessed' if needed)
#     destination_folder = '../data/preprocessed/Dataset'  # Fix typo here if it's 'preprocesse' in your code
    
#     # Create destination folder if it doesn't exist
#     os.makedirs(destination_folder, exist_ok=True)
    
#     files_to_move = [
#         'data_Dataset.address_to_index',
#         'data_Dataset.labels',
#         'data_Dataset.shuffled_clean_docs',
#         'data_Dataset.test_y',
#         'data_Dataset.test_y_prob',
#         'data_Dataset.tfidf_list',
#         'data_Dataset.train_y',
#         'data_Dataset.train_y_prob',
#         'data_Dataset.valid_y',
#         'data_Dataset.valid_y_prob',
#         'data_Dataset.y',
#         'data_Dataset.y_prob',
#         'dev.tsv',   # This is the one causing the error
#         'test.tsv',
#         'train.tsv'  # Add if needed, based on your script
#     ]
    
#     for file_name in files_to_move:
#         if os.path.exists(file_name):
#             dest_path = os.path.join(destination_folder, os.path.basename(file_name))
#             if os.path.exists(dest_path):
#                 print(f"Overwriting existing file: {dest_path}")
#                 os.remove(dest_path)  # Remove existing to allow overwrite
#             shutil.move(file_name, destination_folder)
#             print(f"Moved {file_name} to {destination_folder}")
#         else:
#             print(f"{file_name} does not exist and will not be moved.")
            
# if __name__ == '__main__':
#     for i in range(1, 12):
#         script_name = f"dataset{i}.py"
#         run_script(script_name)
#     run_script("adjust_matrix.py")
#     run_script("BERT_text_data.py")
#     move_files_to_preprocessed_folder()
    
    
#ETH-GSetBert
import subprocess
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        subprocess.run(['python', script_name], check=True)
        print(f"{script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")

def move_files_to_preprocessed_folder():
    # Correct the folder name if it's a typo (change 'preprocesse' to 'preprocessed' if needed)
    destination_folder = '../data/preprocessed/Dataset'  # Fix typo here if it's 'preprocesse' in your code
    
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    files_to_move = [
        # 'data_Dataset.address_to_index',
        'data_Dataset.labels',
        'data_Dataset.shuffled_clean_docs',
        'data_Dataset.test_y',
        'data_Dataset.test_y_prob',
        'data_Dataset.tfidf_list',
        'data_Dataset.train_y',
        'data_Dataset.train_y_prob',
        'data_Dataset.valid_y',
        'data_Dataset.valid_y_prob',
        'data_Dataset.y',
        'data_Dataset.y_prob',
        'dev.tsv',   # This is the one causing the error
        'test.tsv',
        'train.tsv'  # Add if needed, based on your script
    ]
    
    for file_name in files_to_move:
        if os.path.exists(file_name):
            dest_path = os.path.join(destination_folder, os.path.basename(file_name))
            if os.path.exists(dest_path):
                print(f"Overwriting existing file: {dest_path}")
                os.remove(dest_path)  # Remove existing to allow overwrite
            shutil.move(file_name, destination_folder)
            print(f"Moved {file_name} to {destination_folder}")
        else:
            print(f"{file_name} does not exist and will not be moved.")

                
if __name__ == '__main__':
    # chạy các bước cũ
    for i in range(1, 12):
        script_name = f"dataset{i}.py"
        run_script(script_name)
    run_script("adjust_matrix.py")
    run_script("BERT_text_data.py")
    move_files_to_preprocessed_folder()






