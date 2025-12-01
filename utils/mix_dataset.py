import os
import sys
from datasets import load_from_disk, concatenate_datasets, DatasetDict

# 路径黑魔法
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
processed_dir = os.path.join(project_root, "datasets", "processed")


def mix_all_datasets():
    # 这里列出你想合并的所有文件夹名字
    dataset_names = [
        "pq_2h_kb",
        "pq_3h_kb",
        "pql_2h_kb",
        "pql_3h_kb"
    ]

    print(f">>> Mixing datasets: {dataset_names}")

    train_list = []
    val_list = []
    test_list = []

    for name in dataset_names:
        path = os.path.join(processed_dir, name)
        if not os.path.exists(path):
            print(f"[Warn] {path} not found, skipping.")
            continue

        print(f"  Loading {name}...")
        ds_dict = load_from_disk(path)

        # 收集各个 split
        train_list.append(ds_dict['train'])
        val_list.append(ds_dict['validation'])
        test_list.append(ds_dict['test'])

    # 合并
    print("  Concatenating...")
    full_train = concatenate_datasets(train_list)
    full_val = concatenate_datasets(val_list)
    full_test = concatenate_datasets(test_list)

    # 再次打乱 (Shuffle) 很重要！让 2h 和 3h 充分混合
    full_train = full_train.shuffle(seed=42)
    full_val = full_val.shuffle(seed=42)
    # test集通常不打乱也行，但打乱也没事

    # 构建新的 DatasetDict
    mixed_dataset = DatasetDict({
        'train': full_train,
        'validation': full_val,
        'test': full_test
    })

    # 保存
    save_path = os.path.join(processed_dir, "mix_all_kb")
    mixed_dataset.save_to_disk(save_path)

    print(f"\n>>> Success! Mixed dataset saved to: {save_path}")
    print(f"    Train size: {len(full_train)}")
    print(f"    Val size:   {len(full_val)}")
    print(f"    Test size:  {len(full_test)}")


if __name__ == "__main__":
    mix_all_datasets()