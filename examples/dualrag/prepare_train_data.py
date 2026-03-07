import json
import os

from rllm.data.dataset import DatasetRegistry


def prepare_train_data():
    """Prepare and register ASearcher dataset.

    This function loads the ASearcher dataset from JSON file,
    extracts the relevant fields, and registers it using DatasetRegistry.
    """
    # Load the raw JSON data
    file_path = os.path.join(os.path.dirname(__file__), "ASearcher_en_nomath_rejectsample.json")
    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract relevant fields for registration
    processed_data = []
    for item in data:
        extra_info = item.get("extra_info", {})

        processed_item = {
            "id": extra_info.get("id"),
            "question": extra_info.get("question"),
            "ground_truth": extra_info.get("ground_truth"),
            "data_source": extra_info.get("data_source", "asearcher"),
        }
        processed_data.append(processed_item)

    # Register the dataset
    dataset_name = "asearcher"
    split = "train"

    dataset_reg = DatasetRegistry.register_dataset(dataset_name, processed_data, split)
    print(f"{dataset_name} {split} dataset size: {len(processed_data)}")

    return dataset_reg


if __name__ == "__main__":
    dataset = prepare_train_data()
    print("ASearcher dataset processed successfully.")
