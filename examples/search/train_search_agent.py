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
