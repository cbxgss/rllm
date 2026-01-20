import os
import hydra
from hydra.core.hydra_config import HydraConfig

from examples.dualrag.flow.nativerag import NativeRAGSolverWorkflow
from examples.dualrag.flow.dualrag import DualRAGWorkflow
from examples.dualrag.flow.dualrag import *
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("hotpotqqa", "train")
    test_dataset = DatasetRegistry.load_dataset("hotpotqqa", "test")

    hydra_log_dir = HydraConfig.get().runtime.output_dir
    trainer = AgentTrainer(
        workflow_class={
            "nativerag": NativeRAGSolverWorkflow,
            "dualrag": DualRAGWorkflow,
        }[os.getenv("method")],
        workflow_args={
            "retrieve_mode": os.getenv("retrieve_mode"),
            "log_dir": os.path.realpath(hydra_log_dir),
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
