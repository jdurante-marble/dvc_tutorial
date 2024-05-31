import os
import shutil

def prepare_data(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_dir, root.split('/')[-2], root.split('/')[-1], file)
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            shutil.copyfile(src_file, dest_file)

src_train_dir = 'cd_data/raw/train'
src_val_dir = 'cd_data/raw/val'
dest_train_dir = 'data/raw/train'
dest_val_dir = 'data/raw/val'

prepare_data(src_train_dir, dest_train_dir)
prepare_data(src_val_dir, dest_val_dir)
