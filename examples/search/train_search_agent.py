import hydra
import os
from datetime import datetime

from rllm.agents.system_prompts import SEARCH_SYSTEM_PROMPT
from rllm.agents.tool_agent import ToolAgent
from rllm.data import DatasetRegistry
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import search_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer

# from .local_retrieval_tool import LocalRetrievalTool
from .local_api import LocalRetrievalTool


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # train_dataset = DatasetRegistry.load_dataset("hotpotqa", "train")
    train_dataset = DatasetRegistry.load_dataset("asearcher", "train")
    val_dataset = [
        DatasetRegistry.load_dataset("hotpotqa", "test"),
        DatasetRegistry.load_dataset("2wikimultihopqa", "test"),
        DatasetRegistry.load_dataset("musique", "test"),
    ]

    tool_map = {"local_search": LocalRetrievalTool}

    # Setup logging directory
    base_dir = os.getcwd()
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    log_dir = os.path.join(base_dir, "outputs", "trajectory_logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    env_args = {
        "max_steps": 20,
        "tool_map": tool_map,
        "reward_fn": search_reward_fn,
    }

    agent_args = {
        "system_prompt": SEARCH_SYSTEM_PROMPT,
        "tool_map": tool_map,
        "parser_name": "qwen",
        "log_dir": log_dir,
        "enable_logging": True,
        # 5.4.1 Outlier Suppression: Break + 0 reward
        # 触发条件: 工具解析错误、单步工具调用数量超限、重复查询
        # 处理: 立即停止轨迹并给予 0 reward
        "enable_outlier_suppression": True,
        "max_tool_calls_per_step": 10,
        "check_duplicate_queries": True,
        # 其他配置开关存储在 _config_switches 中供 trainer 使用
        "_config_switches": {
            # 5.4.2 Search errors: discard directly
            # 触发条件: 环境错误（超时、连接失败等）
            # 处理: 完全丢弃轨迹，仅记录统计信息
            "discard_search_errors": True,
            # 5.4.3 Exceeding the search step limit: stop + 0 reward
            # 触发条件: 达到 max_steps
            # 处理: 停止 rollout 并给予 0 reward
            "enable_step_limit_handling": True,
            # 5.4.4 Exceeding the token budget: compute advantage, exclude from updates
            # 触发条件: 达到 max_response_length
            # 处理: 仍用于计算优势，但不参与损失计算和反向传播
            "enable_token_budget_handling": True,
        }
    }

    # Use the registry-based approach (comment out the other approach)
    trainer = AgentTrainer(
        agent_class=ToolAgent,
        env_class=ToolEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_args=agent_args,
        env_args=env_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
