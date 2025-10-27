import subprocess
import os
import re
import json

# Model configurations
n = 199
model_configs = [
    {
        "n": n,
        "max_prompt_length": 256,
        "max_new_tokens": 512,
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "fine_tuned_model_path": "Ppear/Qwen2.5-3B-GRPO-Physics_expo"  # Update with your model path
    },
    {
        "n": n,
        "max_prompt_length": 256,
        "max_new_tokens": 32768,
        "model_id": "Qwen/Qwen3-4B",
        "fine_tuned_model_path": None  # Update with your model path
    },
    {
        "n": n,
        "max_prompt_length": 256,
        "max_new_tokens": 32768,
        "model_id": "Qwen/Qwen3-8B",
        "fine_tuned_model_path": None  # Update with your model path
    },
    {
        "n": n,
        "max_prompt_length": 256,
        "max_new_tokens": 32768,
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "fine_tuned_model_path": None  # Update with your model path
    },
    {
        "n": n,
        "max_prompt_length": 256,
        "max_new_tokens": 32768,
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "fine_tuned_model_path": None  # Update with your model path
    },
    {
        "n": n,
        "max_prompt_length": 256,
        "max_new_tokens": 32768,
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "fine_tuned_model_path": None  # Update with your model path
    },
    {
        "n": n,
        "max_prompt_length": 256,
        "max_new_tokens": 32768,
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "fine_tuned_model_path": None  # Update with your model path
    },
    {
        "n": n,
        "max_prompt_length": 256,
        "max_new_tokens": 32768,
        "model_id": "HuggingFaceTB/SmolLM3-3B",
        "fine_tuned_model_path": None  # Update with your model path
    },
    {
        "n": n,
        "max_prompt_length": 256,
        "max_new_tokens": 32768,
        "model_id": "microsoft/Phi-4-mini-instruct",
        "fine_tuned_model_path": None  # Update with your model path
    },
    {
        "n": n,
        "max_prompt_length": 256,
        "max_new_tokens": 32768,
        "model_id": "google/gemma-3-4b-it",
        "fine_tuned_model_path": None  # Update with your model path
    },
]



# Aggregate results
results = {}

# Run evaluations
for config in model_configs:
    fine_tuned_model = config["fine_tuned_model_path"]
    print(f"Evaluating {fine_tuned_model}...")

    if fine_tuned_model is not None: 
        python_file = "evaluate_model_pair_args.py"
    else:
        python_file = "evaluate_model_args.py"
    
    # Construct command
    cmd = [
        "python", python_file,
        "--n", str(config["n"]),
        "--max_prompt_length", str(config["max_prompt_length"]),
        "--max_new_tokens", str(config["max_new_tokens"]),
        "--model_id", config["model_id"],
    ]

    if fine_tuned_model is not None: 
        cmd.append("--fine_tuned_model_path")
        cmd.append(config["fine_tuned_model_path"])
    
    # Run evaluation
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Completed evaluation for {fine_tuned_model}.")

        
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {fine_tuned_model}: {e.stderr}")
        results[fine_tuned_model] = {"error": e.stderr}
        continue
    
