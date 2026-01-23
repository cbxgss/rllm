import os
import json
import yaml
import re
import asyncio

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import Workflow

from examples.dualrag.server.web.api_web import search_entity
from examples.dualrag.server.local.api_local import search
from examples.dualrag.metrics import BaseMetric, ExactMatch, Sub_ExactMatch, F1_Score
from examples.dualrag.utils import save_dict_to_yaml


def extract_blocks(response, block_type="python"):
    pattern_backticks = r"```" + block_type + r"\s*(.*?)\s*```"
    pattern_dashes = r"^-{3,}\s*\n(.*?)\n-{3,}"
    blocks = re.findall(pattern_backticks, response, re.DOTALL)
    blocks.extend(re.findall(pattern_dashes, response, re.DOTALL | re.MULTILINE))
    return blocks


def construct_trajectory(name: str, messages: list[dict], output: ModelOutput) -> Trajectory:
    return Trajectory(
        name=name,
        steps=[
            Step(
                chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                thought=output.reasoning,
                action=output.content.strip(),
                model_output=output,
            )
        ]
    )


class State:
    def __init__(self, question: str, uid_dir: str):
        self.question: str = question
        self.uid_dir: str = uid_dir
        self.turn = 0
        self.knowledge: dict[str, list[str]] = {}
        self.thoughts: list[str] = []
        self.trajs: list[Trajectory] = []

    def get_knowledge_str(self) -> str:
        return "\n\n".join([
            f"#### {entity}\n\n" + "\n\n".join(contents)
            for entity, contents in self.knowledge.items()
        ])

    def get_thoughts_str(self) -> str:
        return "\n".join([
            f"{i+1}. {thought}"
            for i, thought in enumerate(self.thoughts)
        ])

    def step(self):
        self.turn += 1

class DualRAG:
    def __init__(self, rollout_engine: RolloutEngine, retrieve_mode: str, **kwargs):
        self.rollout_engine = rollout_engine
        self.retrieve_mode = retrieve_mode
        with open(os.path.join(os.path.dirname(__file__), "prompt.yaml"), "r") as f:
            prompts: dict = yaml.safe_load(f)["DualRAG"]
        self.reasoner = prompts["Reasoner"]
        self.entity_identifier = prompts["EntityIdentifier"]
        self.knowledge_summarizer = prompts["KnowledgeSummarizer"]
        self.answer = prompts["Answer"]

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

    async def areason(self, state: State) -> tuple[tuple[str, bool], dict]:
        messages = [
            {"role": "user", "content": self.reasoner.format(
                knowledge=state.get_knowledge_str(),
                question=state.question,
                thoughts=state.get_thoughts_str(),
                json_format="""```json
                {
                    "thought": "<Your reasoning process here>",
                    "need_retrieval": <true or false>
                }
                ```
                """,
            )}
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        try:
            blocks = extract_blocks(output.content, block_type="json")
            if len(blocks) == 0:
                raise ValueError("No json block found in reasoner output.")
            reason_json = json.loads(blocks[0])
            reason_content = (reason_json["thought"], reason_json["need_retrieval"])
        except Exception as e:
            reason_content = None
        state.trajs.append(construct_trajectory("reasoner", messages, output))
        return reason_content, {
            "messages": messages,
            "output": output.to_dict(),
        }

    async def adentify_entities(self, state: State) -> tuple[list[dict[str, list[str]]], dict]:
        messages = [
            {"role": "user", "content": self.entity_identifier.format(
                knowledge=state.get_knowledge_str(),
                question=state.question,
                thoughts=state.get_thoughts_str(),
                json_format="""```json
                [
                    {
                        "entity": "<Entity Name>",
                        "queries": ["<Retrieval Query 1>", "<Retrieval Query 2>", "..."]
                    },
                    ...
                ]
                ```
                """,
            )}
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        try:
            blocks = extract_blocks(output.content, block_type="json")
            if len(blocks) == 0:
                raise ValueError("No json block found in entity identifier output.")
            entities = json.loads(blocks[0])
        except Exception as e:
            entities = None
        state.trajs.append(construct_trajectory("entity_identifier", messages, output))
        return entities, {
            "querys": entities,
            "messages": messages,
            "output": output.to_dict(),
        }

    async def asummarizer(self, state: State, entity: str, queries: list[str], docs: list[str]) -> tuple[str, dict]:
        messages = [
            {"role": "user", "content": self.knowledge_summarizer.format(
                question=state.question,
                thoughts=state.get_thoughts_str(),
                entity=entity,
                queries=", ".join(queries),
                docs="\n\n".join(docs),
            )}
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        state.trajs.append(construct_trajectory("knowledge_summarizer", messages, output))
        return output.content.strip(), {
            "messages": messages,
            "output": output.to_dict(),
        }

    async def aanswer(self, state: State) -> tuple[str, dict]:
        messages = [
            {"role": "user", "content": self.answer.format(
                knowledge=state.get_knowledge_str(),
                question=state.question,
                thoughts=state.get_thoughts_str(),
            )}
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        state.trajs.append(construct_trajectory("answer", messages, output))
        return output.content.strip(), {
            "messages": messages,
            "output": output.to_dict(),
        }

    async def generate_answer(self, question: str, uid_dir: str) -> tuple[list[Trajectory], str]:
        trajs: list[Trajectory] = []
        state = State(question, uid_dir)
        log_all = {}
        for i in range(3):  # 最多三轮
            log_all[f"turn_{i+1}"] = {}
            # 1. Reasoner
            reason_content, log = await self.areason(state)
            log_all[f"turn_{i+1}"]["reasoner"] = log
            if not reason_content: break
            state.thoughts.append(reason_content[0])
            if not reason_content[1]: break
            # 2. Entity Identifier
            entities, log = await self.adentify_entities(state)
            log_all[f"turn_{i+1}"]["entity_identifier"] = log
            if not entities: break
            # 3. Retrieval
            log_all[f"turn_{i+1}"]["retrieval"] = {}
            entity2querys: dict[str, list[str]] = {}
            entity2docs: dict[str, list[str]] = {}
            for entity2query in entities:
                entity = entity2query["entity"]
                queries = entity2query["queries"]
                entity2querys[entity] = queries
                tasks = [asyncio.create_task(self.search(query)) for query in queries]
                search_results = await asyncio.gather(*tasks)
                log_all[f"turn_{i+1}"]["retrieval"][entity] = {
                    k: v for k, v in zip(queries, search_results)
                }
                entity2docs[entity] = search_results
            # 4. Knowledge Summarizer
            log_all[f"turn_{i+1}"]["knowledge_summarizer"] = {}
            tasks = [
                asyncio.create_task(self.asummarizer(state, entity, entity2querys[entity], docs))
                for entity, docs in entity2docs.items()
            ]
            summarizer_results = await asyncio.gather(*tasks)
            for (entity, _), (summary, log) in zip(entity2docs.items(), summarizer_results):
                if summary.lower() != "none":
                    if entity not in state.knowledge:
                        state.knowledge[entity] = []
                    state.knowledge[entity].append(summary)
                log_all[f"turn_{i+1}"]["knowledge_summarizer"][entity] = log
            log_all[f"turn_{i+1}"]["state"] = {
                "thoughts": state.thoughts.copy(),
                "knowledge": state.knowledge.copy(),
            }
            state.step()
        # 5. Answer
        answer_output, log = await self.aanswer(state)
        log_all["answer"] = log
        save_dict_to_yaml(log_all, os.path.join(uid_dir, "log.yaml"))
        return state.trajs, answer_output


class DualRAGWorkflow(Workflow):
    def __init__(self, rollout_engine: RolloutEngine, log_dir: str, retrieve_mode: str, **kwargs):
        super().__init__(rollout_engine, **kwargs)
        self.log_dir = log_dir
        self.retrieve_mode = retrieve_mode
        self.rag = DualRAG(rollout_engine, retrieve_mode)
        self.metrics: dict[str, BaseMetric] = {
            "EM": ExactMatch(),
            "ACC": Sub_ExactMatch(),
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
        uid_dir = os.path.join(self.log_dir, f"epoch_{training_epoch}", f"step_{training_step}", f"{uid}")
        if not os.path.exists(uid_dir):
            os.makedirs(uid_dir)

        # Step 1: 生成答案
        trajs, response = await self.rag.generate_answer(question, uid_dir)

        # Step 2: 使用 F1 评分计算 reward
        f1_score_dict = self.metrics["F1"].cal(response, golden_answers)
        for traj in trajs:
            traj.steps[0].reward = f1_score_dict["f1"]
        metrics = {k: v for k, v in f1_score_dict.items()}
        for metric_name, metric in self.metrics.items():
            if metric_name == "F1":
                continue
            score = metric.cal(response, golden_answers)
            metrics[metric_name] = score

        metadata = {
            "uid": uid,
            "data_source": task["data_source"],
            "question": question,
            "golden_answers": golden_answers,
            "reward": f1_score_dict["f1"],
            "response": response,
        }
        with open(os.path.join(uid_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        # Step 3: 构造 Episode
        return Episode(
            id=uid,
            task=task,
            trajectories=[*trajs],
            is_correct=f1_score_dict["f1"] >= 0.5,  # RAG 没有二值对错
            metrics=metrics,
        )
