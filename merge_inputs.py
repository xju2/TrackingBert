import os
import glob
import math
from tqdm import trange
import numpy as np

def merge_files(file_dir):
    """
    Merge the individual files in the directory into 3 big files for
    training, validating, testing files.
    
    Parameters
    ----------
    file_dir : str
        The directory where all files are saved
    """
    
    full_path = f'{file_dir}/*.npz'
    inputs = glob.glob(full_path)
    os.makedirs(f'{file_dir}/inputs', exist_ok=True)
    
    total_files = len(inputs)
    n_val = n_test = min(math.floor(total_files * 0.1), 1)
    n_train = total_files - n_val - n_test
    print(f">>> Splitting {total_files} into {n_train} trainings, {n_val} validations, and {n_test} testings.")
    
    n_train_seq, n_val_seq, n_test_seq = 0, 0, 0
    all_seq = []
    all_pt = []
    for i in trange(total_files):
        file = np.load(inputs[i], allow_pickle=True)
        seq = file['seq']
        pt = file['Pt']
        all_seq.append(seq)
        all_pt.append(pt)
        if i == n_train - 1:
            saving_seq = np.concatenate(all_seq)
            saving_pt = np.concatenate(all_pt)
            n_train_seq = len(saving_seq)
            np.savez(f'{file_dir}/inputs/train.npz', seq=saving_seq, pt=saving_pt)
            all_seq = []
            all_pt = []
        elif i == n_train + n_val - 1:
            saving_seq = np.concatenate(all_seq)
            saving_pt = np.concatenate(all_pt)
            n_val_seq = len(saving_seq)
            np.savez(f'{file_dir}/inputs/val.npz', seq=saving_seq, pt=saving_pt)
            all_seq = []
            all_pt = []
        elif i == total_files - 1:
            saving_seq = np.concatenate(all_seq)
            saving_pt = np.concatenate(all_pt)
            n_test_seq = len(saving_seq)
            np.savez(f'{file_dir}/inputs/test.npz', seq=saving_seq, pt=saving_pt)
            all_seq = []
            all_pt = []
            
    with open(f'{file_dir}/inputs/stats.txt',"w") as f:
        f.writelines(f"Number of training tracks: {n_train_seq}\n")
        f.writelines(f"Number of validation tracks: {n_val_seq}\n")
        f.writelines(f"Number of testing tracks: {n_test_seq}\n")
            
if __name__ == '__main__':
    
    input_dir = 'train_data' # The directory that saved all the files processed
    merge_files(input_dir)