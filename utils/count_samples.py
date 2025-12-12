import os
import sys
from datasets import load_from_disk

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
mixed_dir = os.path.join(project_root, "datasets", "mixed")


def format_num(num):
    """格式化数字，方便阅读"""
    return f"{num:,}"


def stat_all_datasets():
    if not os.path.exists(mixed_dir):
        print(f"Error: Directory not found: {mixed_dir}")
        return

    # 获取所有子文件夹
    dataset_names = [d for d in os.listdir(mixed_dir) if os.path.isdir(os.path.join(mixed_dir, d))]
    dataset_names.sort()

    print(f"\n{'Dataset Name':<20} | {'Train':<10} | {'Valid':<10} | {'Test':<10} | {'Total':<10}")
    print("-" * 70)

    for name in dataset_names:
        path = os.path.join(mixed_dir, name)

        # 简单检查是否为 HF dataset (包含 dataset_dict.json)
        if not os.path.exists(os.path.join(path, "dataset_dict.json")):
            continue

        try:
            ds_dict = load_from_disk(path)

            # 获取各部分大小，防止某些 split 不存在
            n_train = len(ds_dict['train']) if 'train' in ds_dict else 0
            n_val = len(ds_dict['validation']) if 'validation' in ds_dict else 0
            n_test = len(ds_dict['test']) if 'test' in ds_dict else 0

            total = n_train + n_val + n_test

            print(
                f"{name:<20} | {format_num(n_train):<10} | {format_num(n_val):<10} | {format_num(n_test):<10} | {format_num(total):<10}")

        except Exception as e:
            print(f"{name:<20} | Error loading: {e}")

    print("-" * 70)


if __name__ == "__main__":
    print(f"Scanning directory: {mixed_dir}")
    stat_all_datasets()