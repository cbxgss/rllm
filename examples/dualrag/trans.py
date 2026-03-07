# 递归地将 rllm/data/datasets/ 下的所有 parquet, 生成对应的 jsonl
import os
import json
import pandas as pd


def parquet_to_jsonl(parquet_file: str, jsonl_file: str):
    df = pd.read_parquet(parquet_file)
    df.to_json(jsonl_file, orient='records', lines=True, force_ascii=False)


if __name__ == "__main__":
    for root, _, files in os.walk("rllm/data/datasets/"):
        for file in files:
            if file.endswith(".parquet"):
                parquet_path = os.path.join(root, file)
                jsonl_path = parquet_path.replace(".parquet", ".jsonl")
                parquet_to_jsonl(parquet_path, jsonl_path)
                print(f"Converted {parquet_path} to {jsonl_path}")
