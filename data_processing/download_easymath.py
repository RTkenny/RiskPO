"""
Preprocess the MATH-lighteval and gsm8k dataset to parquet format
"""

import argparse
import os
import re
import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution_math(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def extract_solution_gsm8k(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--math_local_dir", default="/home/rt/repo/RiskPO/math")
    parser.add_argument("--gsm8k_local_dir", default="/home/rt/repo/RiskPO/gsm8k")

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source_math = "DigitalLearningGmbH/MATH-lighteval"
    data_source_gsm8k = "openai/gsm8k"
    print(f"Loading the {data_source_math} dataset from huggingface...", flush=True)
    math_dataset = datasets.load_dataset(data_source_math)

    print(f"Loading the {data_source_gsm8k} dataset from huggingface...", flush=True)
    gsm8k_dataset = datasets.load_dataset(data_source_gsm8k, "main")

    math_train_dataset = math_dataset["train"]
    math_test_dataset = math_dataset["test"]

    gsm8k_train_dataset = gsm8k_dataset["train"]
    gsm8k_test_dataset = gsm8k_dataset["test"]


    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn_math(split):
        def process_fn(example, idx):
            question = example.pop("problem")

            question = question + " " + instruction_following

            answer = example.pop("solution")
            solution = extract_solution_math(answer)
            data = {
                "data_source": data_source_math,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    # add a row to each data item that represents a unique id
    def make_map_fn_gsm8k(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution_gsm8k(answer_raw)
            data = {
                "data_source": data_source_gsm8k,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    math_train_dataset = math_train_dataset.map(function=make_map_fn_math("train"), with_indices=True)
    math_test_dataset = math_test_dataset.map(function=make_map_fn_math("test"), with_indices=True)
    math_local_dir = args.math_local_dir
    math_train_dataset.to_parquet(os.path.join(math_local_dir, "train.parquet"))
    math_test_dataset.to_parquet(os.path.join(math_local_dir, "test.parquet"))

    gsm8k_train_dataset = gsm8k_train_dataset.map(function=make_map_fn_gsm8k("train"), with_indices=True)
    gsm8k_test_dataset = gsm8k_test_dataset.map(function=make_map_fn_gsm8k("test"), with_indices=True)
    gsm8k_local_dir = args.gsm8k_local_dir
    gsm8k_train_dataset.to_parquet(os.path.join(gsm8k_local_dir, "train.parquet"))
    gsm8k_test_dataset.to_parquet(os.path.join(gsm8k_local_dir, "test.parquet"))
