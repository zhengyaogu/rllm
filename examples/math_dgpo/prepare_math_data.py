from datasets import load_dataset
from huggingface_hub import login

from rllm.data.dataset import DatasetRegistry

train_dataset_name = "SynthLabsAI/Big-Math-RL-Verified"


def prepare_math_data():
    train_dataset = load_dataset(train_dataset_name, split="train")
    test_dataset = load_dataset(train_dataset_name, split="train")

    def preprocess_big_math(example, idx):
        return {
            "question": example["problem"],
            "ground_truth": example["answer"],
            "data_source": example["source"],
        }

    train_dataset = train_dataset.filter(lambda x: (
        x["source"] in ["big_math", "orca_math", "cn_k12", "olympiads", "math", "aops_forum"]
    ))
    test_dataset = test_dataset.filter(lambda x: (
        x["source"] in ["omnimath"]
    ))
    train_dataset = train_dataset.map(preprocess_big_math, with_indices=True)
    test_dataset = test_dataset.map(preprocess_big_math, with_indices=True)

    train_dataset = DatasetRegistry.register_dataset("big_math", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("omnimath", test_dataset, "train")
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_math_data()
    print(len(train_dataset))
    print(len(test_dataset))
