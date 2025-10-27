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
        "fine_tuned_model_path": "psamtam/Qwen2.5-3B-GRPO-Physics-2-50000coeff-sigmoid"  # Update with your model path
    },
    {
        "n": n,
        "max_prompt_length": 256,
        "max_new_tokens": 512,
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "fine_tuned_model_path": "psamtam/Qwen2.5-3B-GRPO-Physics-2-100000coeff-sigmoid"  # Update with your model path
    },
    {
        "n": n,
        "max_prompt_length": 256,
        "max_new_tokens": 512,
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "fine_tuned_model_path": "psamtam/Qwen2.5-3B-GRPO-Physics-2-150000coeff-sigmoid"  # Update with your model path
    },
    {
        "n": n,
        "max_prompt_length": 256,
        "max_new_tokens": 512,
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "fine_tuned_model_path": "psamtam/Qwen2.5-3B-GRPO-Physics-2-5000coeff"  # Update with your model path
    },
    {
        "n": n,
        "max_prompt_length": 256,
        "max_new_tokens": 512,
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "fine_tuned_model_path": "psamtam/Qwen2.5-3B-GRPO-Physics-2-10000coeff"  # Update with your model path
    },
    {
        "n": n,
        "max_prompt_length": 256,
        "max_new_tokens": 512,
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "fine_tuned_model_path": "psamtam/Qwen2.5-3B-GRPO-Physics-2-15000coeff"  # Update with your model path
    },

]


# Aggregate results
results = {}

# Run evaluations
for config in model_configs:
    fine_tuned_model = config["fine_tuned_model_path"]
    print(f"Evaluating {fine_tuned_model}...")


    python_file = "evaluate_fine_tuned_model_args.py"
    # if fine_tuned_model is not None: 
    #     python_file = "evaluate_model_pair_args.py"
    # else:
    #     python_file = "evaluate_model_args.py"
    
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