#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2023-04-19 00:59:24
LastEditTime: 2023-04-19 01:00:44
LastEditors: Kun
Description: 
FilePath: /ChatGLM-LoRA-RLHF-PyTorch/cover_alpaca2jsonl.py
'''

"""
https://github.com/mymusise/ChatGLM-Tuning
"""

import argparse
import json
from tqdm import tqdm


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/alpaca_data.json")
    parser.add_argument("--save_path", type=str, default="data/alpaca_data.jsonl")

    args = parser.parse_args()
    with open(args.data_path) as f:
        examples = json.load(f)

    with open(args.save_path, 'w') as f:
        for example in tqdm(examples, desc="formatting.."):
            f.write(json.dumps(format_example(example)) + '\n')


if __name__ == "__main__":
    main()
