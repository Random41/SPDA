import os
from collections import defaultdict


def collect_files_by_directory(data_dir):
    files_by_dir = defaultdict(list)

    for root, dirs, files in os.walk(data_dir):
        npz_files = sorted([f for f in files if f.endswith('.npz')])
        if npz_files:
            files_by_dir[root] = [os.path.join(root, f) for f in npz_files]

    return files_by_dir
