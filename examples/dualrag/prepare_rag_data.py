import random

import datasets
from rllm.data.dataset import DatasetRegistry


DATASETS = {
    "nq": ("RUC-NLPIR/FlashRAG_datasets", "nq"),
    "tq": ("RUC-NLPIR/FlashRAG_datasets", "triviaqa"),
    "hotpotqqa": ("RUC-NLPIR/FlashRAG_datasets", "hotpotqa"),
    "2wikimultihopqa": ("RUC-NLPIR/FlashRAG_datasets", "2wikimultihopqa"),
    "musique": ("RUC-NLPIR/FlashRAG_datasets", "musique"),
}

RAW2NEW = {
    "nq": {"train": "train", "test": "test"},
    "tq": {"train": "train", "test": "test"},
    "hotpotqqa": {"train": "train", "dev": "test"},
    "2wikimultihopqa": {"train": "train", "dev": "test"},
    "musique": {"train": "train", "dev": "test"},
}


TRAIN_SIZE = 2000
TEST_SIZE = 512


def prepare_rag_data():
    all_datasets = {}

    for name, (hf_name, subset) in DATASETS.items():
        print(f"Processing dataset {name}...")
        raw_ds = datasets.load_dataset(hf_name, subset)
        result_datasets = {}

        for raw_split, new_split in RAW2NEW[name].items():
            dataset = raw_ds[raw_split]
            dataset_size = TRAIN_SIZE if new_split == "train" else TEST_SIZE

            # 随机抽样
            if len(dataset) > dataset_size:
                selected_indices = random.sample(range(len(dataset)), dataset_size)
            else:
                selected_indices = list(range(len(dataset)))

            selected = []
            for i, idx in enumerate(selected_indices):
                item = dataset[idx]
                raw_id = item["id"]
                question = item["question"]
                answer = item["golden_answers"]  # ground_truth 必须存在

                selected.append({
                    "id": f"{new_split}_{raw_id}",
                    "question": question,
                    "ground_truth": answer,
                    "data_source": name,
                })

            # 注册 Dataset
            dataset_reg = DatasetRegistry.register_dataset(name, selected, new_split)
            result_datasets[new_split] = dataset_reg
            print(f"{name} {new_split} dataset size: {len(selected)}")

        all_datasets[name] = result_datasets

    return all_datasets


if __name__ == "__main__":
    all_ds = prepare_rag_data()
    print("All datasets processed successfully.")
