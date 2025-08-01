import hydra

from rllm.agents import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path=".", config_name="grpo", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("big_math", "train")
    test_dataset = DatasetRegistry.load_dataset("big_math", "test")

    agent_args = {"accumulate_thinking": True}
    env_args = {
        "reward_fn": math_reward_fn,
    }

    trainer = AgentTrainer(
        agent_class=MathAgent,
        env_class=SingleTurnEnvironment,
        agent_args=agent_args,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
