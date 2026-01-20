import os
import asyncio
import json
import re

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import Workflow

from examples.dualrag.server.web.api_web import search_entity
from examples.dualrag.server.local.api_local import search
from examples.dualrag.metrics import BaseMetric, ExactMatch, Sub_ExactMatch, F1_Score


class NativeRAGSolver:
    system_prompt = """You are a RAG chatbot that will answer questions based on the relevant documents provided.
Please output the answer in a few words, keeping it as brief as possible.
For some questions, simply respond with "yes" or "no."""

    user_prompt = """

Question:

{query}

Relevant documents:

{corpus_str}
"""

    def __init__(self, rollout_engine: RolloutEngine, retrieve_mode: str, **kwargs):
        self.rollout_engine = rollout_engine
        self.retrieve_mode = retrieve_mode

    async def search(self, query: str) -> str:
        if self.retrieve_mode == "local":
            search_res = await search(query)
            docs = search_res.get("docs")
            corpus = "\n\n".join(docs) if docs else "No relevant documents found."
        elif self.retrieve_mode == "web":
            search_res = await search_entity(query)
            corpus = search_res.get("output", "No relevant documents found.")
        else:
            raise ValueError(f"Unknown retrieve_mode: {self.retrieve_mode}")
        return corpus

    async def generate_answer(self, question: str) -> tuple[Trajectory, str]:
        # Step 1: 拉取相关文档
        corpus = await self.search(question)

        # Step 2: 构造 prompt
        messages = [
            {"role": "system", "content": NativeRAGSolver.system_prompt},
            {"role": "user", "content": NativeRAGSolver.user_prompt.format(query=question, corpus_str=corpus)},
        ]

        # Step 3: 调用模型生成答案
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)

        # Step 4: 构造 Trajectory
        traj = Trajectory(
            name="rag_solver",
            steps=[
                Step(
                    chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=output.content.strip(),
                    model_output=output,
                )
            ]
        )
        return traj, corpus


class NativeRAGSolverWorkflow(Workflow):
    def __init__(self, rollout_engine: RolloutEngine, log_dir: str, retrieve_mode: str, **kwargs):
        super().__init__(rollout_engine, **kwargs)
        self.log_dir = log_dir
        self.retrieve_mode = retrieve_mode
        self.solver = NativeRAGSolver(rollout_engine, retrieve_mode)
        self.metrics: dict[str, BaseMetric] = {
            "EM": ExactMatch(),
            "Sub-EM": Sub_ExactMatch(),
            "F1": F1_Score(),
        }

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        task: dict, 必须包含 "question" 和 "ground_truth" 字段
        uid: 任务 id
        """
        self.reset(task, uid)
        question = task["question"]
        golden_answers = task["ground_truth"] if isinstance(task["ground_truth"], list) else [task["ground_truth"]]

        # 从 kwargs 中获取训练步数信息
        training_step = kwargs.get("training_step", 0)
        training_epoch = kwargs.get("training_epoch", 0)
        training_mode = kwargs.get("training_mode", "train")

        # Step 1: 生成答案
        traj, corpus = await self.solver.generate_answer(question)
        response = traj.steps[0].action

        # Step 2: 使用 F1 评分计算 reward
        f1_score_dict = self.metrics["F1"].cal(response, golden_answers)
        traj.steps[0].reward = f1_score_dict["f1"]
        metrics = {k: v for k, v in f1_score_dict.items()}
        for metric_name, metric in self.metrics.items():
            if metric_name == "F1":
                continue
            score = metric.cal(response, golden_answers)
            metrics[metric_name] = score

        # 保存日志，将 step 信息包含在目录路径中
        uid_dir = os.path.join(self.log_dir, f"epoch_{training_epoch}", f"step_{training_step}", f"{task['idx']}_{uid}")
        if not os.path.exists(uid_dir):
            os.makedirs(uid_dir)
        with open(os.path.join(uid_dir, "retrieval.md"), "w") as f:
            f.write(corpus)
        with open(os.path.join(uid_dir, "model_output.md"), "w") as f:
            f.write(response)
        metadata = {
            "uid": uid,
            "idx": task["idx"],
            "data_source": task["data_source"],
            "question": question,
            "ground_truth": golden_answers,
            "reward": f1_score_dict["f1"],
            "response": response,
        }
        with open(os.path.join(uid_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        # Step 3: 构造 Episode
        return Episode(
            id=uid,
            task=task,
            trajectories=[traj],
            is_correct=f1_score_dict["f1"] >= 0.5,  # RAG 没有二值对错
            metrics=metrics,
        )
