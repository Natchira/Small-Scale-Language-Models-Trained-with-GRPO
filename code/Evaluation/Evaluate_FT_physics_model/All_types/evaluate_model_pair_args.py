import torch
import gc
import argparse
import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import re
from math_verify import LatexExtractionConfig, parse, verify
import numpy as np
import pandas as pd
import pint
from pint import UnitRegistry
from pylatexenc.latex2text import LatexNodes2Text
import math
from math import floor, log10
from sklearn.model_selection import train_test_split
from datasets import Dataset
from our_rewards_detailed_evaluation import physics_accuracy_reward_evaluation, format_reward


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate original and fine-tuned models")
    parser.add_argument("--n", type=int, default=5, help="Number of top training samples")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Maximum prompt length (tokens)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Original model ID")
    parser.add_argument("--fine_tuned_model_path", type=str, default="psamtam/Qwen2.5-3B-GRPO-Physics", help="Fine-tuned model path or Hub ID")
    return parser.parse_args()


def main():
    # Get arguments
    args = parse_args()
    n = args.n
    max_prompt_length = args.max_prompt_length
    max_new_tokens = args.max_new_tokens
    model_id = args.model_id
    fine_tuned_model_path = args.fine_tuned_model_path

    def create_logger(name, log_file):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Prevent duplicate logs if root logger is configured

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(message)s"))

        # Stream handler (console)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))

        logger.addHandler(fh)
        logger.addHandler(sh)

        return logger

    # Create output dir
    os.makedirs("output", exist_ok=True)

    # Create two loggers
    rewards_log_file = os.path.join("output", os.path.basename(fine_tuned_model_path) + ".rewards")
    completions_log_file = os.path.join("output", os.path.basename(fine_tuned_model_path) + ".completions")

    logger_rewards = create_logger("rewards", rewards_log_file)
    logger_completions = create_logger("completions", completions_log_file)

    def format_options(row):
        options = row["options"]
        if len(options) == 0:
            return row
        options_string = "Here are the options: ["
        options_string += ", ".join(options) + "]"
        row['question'] = row["question"] + " " + options_string
        return row

    # Load top n test samples
    try:
        dataset_id = "psamtam/high_school_physics_dataset"
        train_dataset, test_dataset = load_dataset(dataset_id, split=["train", "test"])
        test_dataset = test_dataset.map(format_options)
        test_dataset = test_dataset.to_pandas()
        test_dataset_length = min(n, len(test_dataset))
        test_dataset = test_dataset.sample(min(n, len(test_dataset)), random_state=42)
        test_dataset = Dataset.from_pandas(test_dataset, preserve_index=False)
    except Exception as e:
        logger_rewards.error(f"Failed to load dataset: {str(e)}")




    # Define SYSTEM_PROMPT and make_conversation
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "'<think> reasoning process here </think><answer> answer here </answer>' If the question provides options, use "
        "the exact wordings in the options. Otherwise, answer with a numerical number and units, with a space between them."
    )
    def make_conversation(example):
        question = example["question"]

        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        }

    try:
        test_dataset = test_dataset.map(make_conversation)
    except Exception as e:
        logger_rewards.error(f"Failed to process dataset: {str(e)}")

    def accuracy_reward(completions, **kwargs):
        return physics_accuracy_reward_evaluation(completions, logger=None, **kwargs)

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        logger_rewards.error(f"Failed to load tokenizer for {model_id}: {str(e)}")

    # Generate completions
    def generate_completions(model, dataset, max_prompt_length, max_new_tokens):
        completions = []
        for example in dataset:
            prompt = example["prompt"]
            try:
                input_text = tokenizer.apply_chat_template(prompt, tokenize=False)
                inputs = tokenizer(input_text, return_tensors="pt", max_length=max_prompt_length, truncation=True).to(model.device)
                input_length = inputs.input_ids.shape[1]
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, num_return_sequences=1, do_sample=False)
                completion = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
                # Extract assistant's response
                assistant_response = '<think>' + completion.split('<think>')[-1].strip()
                completions.append([{"content": assistant_response}])
            except Exception as e:
                logger_rewards.error(f"Generation failed for prompt: {str(e)}")
                completions.append([{"content": ""}])  # Append empty completion to maintain alignment
        return completions

    # Load original model
    try:
        original_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
    except Exception as e:
        logger_rewards.error(f"Failed to load original model {model_id}: {str(e)}")

    # Generate completions for original model
    original_completions = generate_completions(original_model, test_dataset, max_prompt_length, max_new_tokens)

    # Free original model memory
    del original_model
    torch.cuda.empty_cache()
    gc.collect()

    # Load fine-tuned model
    try:
        base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        fine_tuned_model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
    except Exception as e:
        logger_rewards.error(f"Failed to load fine-tuned model {fine_tuned_model_path}: {str(e)}")

    # Generate completions for fine-tuned model
    fine_tuned_completions = generate_completions(fine_tuned_model, test_dataset, max_prompt_length, max_new_tokens)

    # Free fine-tuned model memory
    del fine_tuned_model
    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    # Compute rewards
    answers = [example["answer"] for example in test_dataset]
    options = [example["options"] for example in test_dataset]
    original_format_rewards = format_reward(original_completions)
    original_accuracy_rewards = accuracy_reward(original_completions, answer=answers, options=options)
    fine_tuned_format_rewards = format_reward(fine_tuned_completions)
    fine_tuned_accuracy_rewards = accuracy_reward(fine_tuned_completions, answer=answers, options=options)

    def extract_rewards(format_rewards, accuracy_rewards):
        format_rewards_mc = []
        accuracy_rewards_mc = []

        format_rewards_open_end = []
        accuracy_rewards_open_end = []
        
        for format_reward, (accuracy_reward, q_type) in zip(format_rewards, accuracy_rewards):
            if q_type == 'mc':
                format_rewards_mc.append(format_reward)
                accuracy_rewards_mc.append(accuracy_reward)
            elif q_type == 'open_end':
                format_rewards_open_end.append(format_reward)
                accuracy_rewards_open_end.append(accuracy_reward)

        return format_rewards_mc, accuracy_rewards_mc, format_rewards_open_end, accuracy_rewards_open_end


    original_format_rewards_mc, original_accuracy_rewards_mc, original_format_rewards_open_end, original_accuracy_rewards_open_end = extract_rewards(original_format_rewards, original_accuracy_rewards)
    original_accuracy_rewards_overall = original_accuracy_rewards_mc.copy()
    original_accuracy_rewards_open_end_overall = [sum(x) for x in original_accuracy_rewards_open_end]
    original_accuracy_rewards_overall.extend(original_accuracy_rewards_open_end_overall)

    fine_tuned_format_rewards_mc, fine_tuned_accuracy_rewards_mc, fine_tuned_format_rewards_open_end, fine_tuned_accuracy_rewards_open_end = extract_rewards(fine_tuned_format_rewards, fine_tuned_accuracy_rewards)
    fine_tuned_accuracy_rewards_overall = fine_tuned_accuracy_rewards_mc.copy()
    fine_tuned_accuracy_rewards_open_end_overall = [sum(x) for x in fine_tuned_accuracy_rewards_open_end]
    fine_tuned_accuracy_rewards_overall.extend(fine_tuned_accuracy_rewards_open_end_overall)

    # Log results
    logger_rewards.info(f"Fine-tuned model: {fine_tuned_model_path}")
    logger_rewards.info("")
    logger_rewards.info(f"Rewards on Top {test_dataset_length} Test Samples (max_prompt_length={max_prompt_length}, max_new_tokens={max_new_tokens}):")
    logger_rewards.info("\nOriginal Model Rewards:")
    logger_rewards.info(f"Mean Format Reward for open-end questions: {np.mean(original_format_rewards_open_end):.4f}")
    
    n = len(original_accuracy_rewards_open_end)
    avg_accuracy_open_end = tuple(sum(values) / n for values in zip(*original_accuracy_rewards_open_end))

    logger_rewards.info(
        "Mean Accuracy Reward for open-end questions: "
        f"({', '.join(f'{x:.4f}' for x in avg_accuracy_open_end)})"
    )
    
    logger_rewards.info(f"Mean Format Reward for mc questions: {np.mean(original_format_rewards_mc):.4f}")
    logger_rewards.info(f"Mean accuracy Reward for mc questions: {np.mean(original_accuracy_rewards_mc):.4f}")
    logger_rewards.info(f"Mean Format Reward (Overall): {np.mean(original_format_rewards):.4f}")
    logger_rewards.info(f"Mean Accuracy Reward (Overall): {np.mean(original_accuracy_rewards_overall):.4f}")

    logger_rewards.info("\nFine-Tuned Model Rewards:")
    logger_rewards.info(f"Mean Format Reward for open-end questions: {np.mean(fine_tuned_format_rewards_open_end):.4f}")

    n = len(fine_tuned_accuracy_rewards_open_end)
    avg_accuracy_open_end = tuple(sum(values) / n for values in zip(*fine_tuned_accuracy_rewards_open_end))

    logger_rewards.info(
        "Mean Accuracy Reward for open-end questions: "
        f"({', '.join(f'{x:.4f}' for x in avg_accuracy_open_end)})"
    )

    logger_rewards.info(f"Mean Format Reward for mc questions: {np.mean(fine_tuned_format_rewards_mc):.4f}")
    logger_rewards.info(f"Mean accuracy Reward for mc questions: {np.mean(fine_tuned_accuracy_rewards_mc):.4f}")
    logger_rewards.info(f"Mean Format Reward (Overall): {np.mean(fine_tuned_format_rewards):.4f}")
    logger_rewards.info(f"Mean Accuracy Reward (Overall): {np.mean(fine_tuned_accuracy_rewards_overall):.4f}")
    logger_rewards.info("")
    logger_rewards.info(50*'-')


    ### LOGGING COMPLETIONS
    for question, answer, original_completion, fine_tuned_completion, original_accuracy_reward, fine_tuned_accuracy_reward in zip(test_dataset['question'], test_dataset['answer'],original_completions, fine_tuned_completions, original_accuracy_rewards, fine_tuned_accuracy_rewards):
        out_string = f"Question: {question}\n\nCorrect answer: {answer}\n\nOriginal_accuracy_reward: {original_accuracy_reward}\nOriginal Completion:   {original_completion[0]['content']}\n\nFine-tuned_accuracy_reward: {fine_tuned_accuracy_reward}\nFine-tuned Completion: {fine_tuned_completion[0]['content']}\n\n{100*'='}\n"
        logger_completions.info(out_string)

if __name__ == "__main__":
    main()