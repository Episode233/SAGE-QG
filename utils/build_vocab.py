import os
import json
from datasets import load_from_disk

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# 你的 processed 数据存放位置
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")
OUTPUT_PATH = os.path.join(DATA_DIR, "relations.json")

def build_relation_vocab():
    print(f"Scanning datasets in {DATA_DIR}...")

    unique_relations = set()

    # 遍历 processed 下的所有子文件夹 (pq_2h, pql_3h 等)
    for dataset_name in os.listdir(DATA_DIR):
        dataset_path = os.path.join(DATA_DIR, dataset_name)

        # 确保是文件夹且包含 dataset_dict.json (说明是 HF Dataset)
        if os.path.isdir(dataset_path) and "dataset_dict.json" in os.listdir(dataset_path):
            print(f"  Loading {dataset_name}...")
            try:
                ds_dict = load_from_disk(dataset_path)
                # ds_dict.values() 会返回这三个 dataset 对象
                for split_name, dataset in ds_dict.items():
                    print(f"    Scanning split: {split_name}...")

                    # 遍历该 split 下的所有样本
                    for sample in dataset:
                        rels = sample['edge_attr']
                        unique_relations.update(rels)

            except Exception as e:
                print(f"  [Error] Failed to load {dataset_name}: {e}")

    # 排序，确保 ID 顺序固定
    sorted_relations = sorted(list(unique_relations))

    # 建立映射字典
    rel2id = {rel: i for i, rel in enumerate(sorted_relations)}

    print(f"\nFound {len(rel2id)} unique relations.")

    # 保存
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(rel2id, f, indent=2, ensure_ascii=False)

    print(f"Relation vocabulary saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_relation_vocab()