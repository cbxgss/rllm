import asyncio
import json
import os
from copy import deepcopy

from rllm.data.dataset import DatasetRegistry
from rllm.engine import OpenAIEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from transformers import AutoTokenizer

from examples.dualrag.prepare_rag_data import prepare_rag_data
from examples.dualrag.metrics import F1_Score
from examples.dualrag.flow.nativerag import NativeRAGSolverWorkflow
from examples.dualrag.flow.dualrag import DualRAGWorkflow


def load_data(n=1):
    """Load RAG datasets (all five datasets) and return a flat list of tasks."""
    tasks = []

    # 遍历注册的数据集
    for dataset_name in ["nq", "tq", "hotpotqqa", "2wikimultihopqa", "musique"]:
        test_dataset = DatasetRegistry.load_dataset(dataset_name, "test")
        if test_dataset is None:
            print(f"Dataset {dataset_name} not found, preparing dataset...")
            prepare_rag_data()  # 会注册所有数据集
            test_dataset = DatasetRegistry.load_dataset(dataset_name, "test")

        for idx, example in enumerate(test_dataset):
            task = process_rag_example(example, idx, dataset_name)
            for _ in range(n):
                tasks.append(deepcopy(task))

    return tasks


def process_rag_example(example, idx, dataset_name):
    """Convert dataset example to RAG task format."""
    question = example["question"]
    ground_truth = example["ground_truth"]  # 可以是 str 或 list
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    task = {
        "question": question,
        "ground_truth": ground_truth,
        "idx": idx,
        "data_source": dataset_name,
    }
    return task


def evaluate_results(results):
    """Evaluate results using F1 score."""
    f1_metric = F1_Score()
    f1_scores = []

    for episode in results:
        response = episode.trajectories[-1].steps[0].action
        golden_answers = episode.task["ground_truth"]
        f1 = f1_metric.cal(response, golden_answers)["f1"]
        f1_scores.append(f1)

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    print(f"Total tasks: {len(f1_scores)}")
    print(f"Average F1 score: {avg_f1:.4f}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # -----------------------------
    # Engine / Workflow 配置
    # -----------------------------
    n_parallel_tasks = 64
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    rollout_engine = OpenAIEngine(
        model=model_name,
        tokenizer=tokenizer,
        max_prompt_length=1024 * 16,
        max_response_length=1024 * 2,
        base_url="http://api.siliconflow.cn/v1",
        api_key="sk-kaxfpubzsimufozugehbumowicjwiocdaiekeahbqrmokius",
        sampling_params={"temperature": 0.6, "top_p": 0.95},
    )

    method = os.getenv("method")
    workflow = {
        "nativerag": NativeRAGSolverWorkflow,
        "dualrag": DualRAGWorkflow,
    }[method]
    engine = AgentWorkflowEngine(
        workflow_cls=workflow,
        workflow_args={
            "retrieve_mode": os.getenv("retrieve_mode"),
            "log_dir": f"{os.getenv('base_dir')}/outputs/tmp/{method}",
        },
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
    )

    # -----------------------------
    # 加载任务
    # -----------------------------
    tasks = load_data(n=1)[:5]
    print(f"Loaded {len(tasks)} RAG tasks")

    # -----------------------------
    # 执行任务
    # -----------------------------
    results = asyncio.run(engine.execute_tasks(tasks))

    # -----------------------------
    # 评估结果
    # -----------------------------
    print("Evaluating results using F1 metric...")
    evaluate_results(results)

    # -----------------------------
    # 保存结果
    # -----------------------------
    with open(f"outputs/tmp/{method}/rag_solver_results.json", "w", encoding="utf-8") as f:
        json.dump([episode.to_dict() for episode in results], f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to outputs/tmp/{method}/rag_solver_results.json")
