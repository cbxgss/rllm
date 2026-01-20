import os
import pandas as pd
import json
import numpy as np

path = "rllm/data/datasets/hotpotqqa"

def convert_ndarray(obj):
    """
    递归将 ndarray 转为 list
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(v) for v in obj]
    else:
        return obj

def convert_parquet_to_jsonl(parquet_path, jsonl_path):
    df = pd.read_parquet(parquet_path)
    records = df.to_dict(orient="records")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            record = convert_ndarray(record)  # 递归转换
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + "\n")

for filename in os.listdir(path):
    if filename.endswith(".parquet"):
        parquet_file = os.path.join(path, filename)
        jsonl_file = os.path.join(path, filename.replace(".parquet", ".jsonl"))
        convert_parquet_to_jsonl(parquet_file, jsonl_file)
        print(f"Converted {parquet_file} to {jsonl_file}")

print("Conversion completed.")
