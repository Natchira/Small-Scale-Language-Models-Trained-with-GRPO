# %% [markdown]
# # Fine-tuning Qwen 2.5 3B instruct into high-school physics reasoning model using GRPO

# %% [markdown]
# ### Memory usage

# %% [markdown]
# ```
# +-----------------------------------------------------------------------------------------+
# | Processes:                                                                              |
# |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
# |        ID   ID                                                               Usage      |
# |=========================================================================================|
# |    0   N/A  N/A   1543689      C   ...conda3/envs/dissertation/bin/python       8646MiB |
# |    1   N/A  N/A   1543689      C   ...conda3/envs/dissertation/bin/python       7798MiB |
# +-----------------------------------------------------------------------------------------+
# ```

# %% [markdown]
# ### Install dependencies

# %%
# !pip install -U trl peft math_verify tensorboardx ipykernel bitsandbytes scikit-learn pint mip pylatexenc

# %% [markdown]
# ### Show packages version

# %%
import transformers

print(f"transformers.__version__: {transformers.__version__}")

import trl

print(f"trl.__version__: {trl.__version__}")

import datasets

print(f"datasets.__version__: {datasets.__version__}")

import peft

print(f"peft.__version__: {peft.__version__}")

import accelerate

print(f"accelerate.__version__: {accelerate.__version__}")

import sklearn

print(f"sklearn.__version__: {sklearn.__version__}")

import torch

print("cuda available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)

# %% [markdown]
# ### Import packages

# %%
import os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from datasets import Dataset

import pint
from pint import UnitRegistry

from pylatexenc.latex2text import LatexNodes2Text

import re

import math
from math import floor, log10


import logging
from datetime import datetime

# Generate timestamp for log file
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"physics_{timestamp}.log"

# Setup root logger
logging.basicConfig(
    level=logging.INFO,  # Or DEBUG for more details
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(),          # Logs to notebook cell
        logging.FileHandler(log_filename) # Logs to file with datetime
    ]
)

logger = logging.getLogger(__name__)

# %%
def test(x):
    if x == 0 or not math.isfinite(x):
        return x
    
math.isfinite(-math.inf)

# %% [markdown]
# ### Connect to HuggingFace

# %% [markdown]
# ### Load Dataset

# %%
# dataset = pd.read_json("cleaned_dataset.json", orient="records")
# dataset.sample(5)

from datasets import load_dataset

dataset = pd.read_json("cleaned_dataset.json", orient="records")
dataset.sample(5)

# %% [markdown]
# ### Format the dataset
# As we will convert it into a Dataset, we need to get rid of the list from the `options` column. 

# %%
def format_options(row):
    options = row["options"]
    if len(options) == 0:
        return row["question"]
    options_string = "Here are the options: ["
    options_string += ", ".join(options) + "]"
    return row["question"] + " " + options_string


dataset["question"] = dataset.apply(format_options, axis=1)

for key, value in zip(dataset.iloc[-1].keys(), dataset.iloc[-1]):
    print(f"{key}: {value}")

# %% [markdown]
# ### Train/test split

# %%
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
# display(train_dataset.sample(3))
# display(test_dataset.sample(3))

# %% [markdown]
# ### Create prompts

# %%
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "'<think> reasoning process here </think><answer> answer here </answer>' If the question provides options, use "
    "the exact wordings in the options. Otherwise, answer with a numerical number and units, with a space between them."
)


def make_conversation(example):
    question = example["question"]
    # if example["options"] != "":
    #     question += " " + example["options"]

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    }


train_dataset = Dataset.from_pandas(train_dataset, preserve_index=False)
test_dataset = Dataset.from_pandas(test_dataset, preserve_index=False)

train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)

# %%
print(train_dataset[0]["prompt"])

# %% [markdown]
# ### Cleaning train dataset

# %%
print(f"Before removing columns: \n{train_dataset}")

# train_dataset = train_dataset.remove_columns(['question', 'options', 'CoT'])
train_dataset = train_dataset.remove_columns(["question", "CoT"])
print(f"After removing columns: \n{train_dataset}")

# %% [markdown]
# ### Load the Baseline Model

# %%
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_id = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

# %% [markdown]
# ### Configuring LoRA

# %%
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# %% [markdown]
# ### Reward functions

# %% [markdown]
# #### Format reward

# %%
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]

# %% [markdown]
# #### Solution rewards

# %%
ureg = UnitRegistry()

preferred_units = [
    ureg.m,     # meter (length) - L
    ureg.kg,    # kilogram (mass) - M
    ureg.s,     # second (time) - T
    ureg.degC,  # Degree C (thermodynamic temperature) - Θ
    ureg.A,     # Ampere (electric current) - I
    ureg.mol,   # mole (amount of substance) - N
    ureg.cd,    # candela (luminous intensity) - J
]

ureg.define('@alias au = AU')
ureg.define('@alias J = Joules')
ureg.define('@alias A = amps')

ureg.default_preferred_units = preferred_units

Q_ = ureg.Quantity

# %%
class RewardConfig:
    def __init__(self):
        self.PENALTY_COEFF = 0.05
        self.N_SIG_FIG = 4
        self.UNIT_REWARD = 0.5
        self.NUMERICAL_REWARD = 1 - self.UNIT_REWARD
        self.MC_REWARD = 1
        self.MC_EXIST_REWARD = 0.1

def smooth_penalty(eps, cap=0.5, tau=0.5):
    return cap * (1 - np.exp(-eps / tau))

# %%
def parse_answer(extracted_answer):
    # split the answer into strings of numerical and unit
    answer_split = LatexNodes2Text().latex_to_text(extracted_answer).replace(' x ', '*').replace(' × ', '*').strip().split(' ')
    if len(answer_split) == 0:
        numerical_string, unit_string = '', ''
    elif len(answer_split) == 1:
        numerical_string, unit_string = answer_split[0], ''
    else:
        numerical_string, unit_string = answer_split[0], ' '.join(answer_split[1:])

    # dealing with the unit system
    try:
        parse_unit = ureg.parse_units(unit_string, as_delta=False) # from string to `pint.Unit`
        try:
            unit_quantity = Q_(1*parse_unit).to_preferred() # convert unit to `preferred_units`
        except pint.DimensionalityError:
            unit_quantity = Q_(1*parse_unit)
        unit_multiplier, pint_units = unit_quantity.magnitude, unit_quantity.units
    except Exception as e:
        unit_multiplier, pint_units = 1, ""
        logger.warning(f"Exception {e}:\nError when parsing unit '{unit_string}' in '{extracted_answer}'")

    # dealing with the numerical part
    try: 
        magnitude = Q_(numerical_string).to_base_units().magnitude * unit_multiplier 
    except Exception as e:
        magnitude = 0
        logger.warning(f"Exception {e}:\nError when parsing numerical '{numerical_string}' in '{extracted_answer}'")

    return magnitude, pint_units


def split_num_and_letters(s):
    match = re.match(r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)([a-zA-Z]*)", s)
    if match and match.group(1):  # found a number
        num_part = float(match.group(1))
        char_part = match.group(2)
    else:  # no number, assume 1
        num_part = 1
        char_part = s
    return num_part, char_part


def extract_answer_from_tag(content):
    extracted_answer = re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if len(extracted_answer):
        extracted_answer = extracted_answer[0]
    else:
        extracted_answer = "" 
    return extracted_answer.strip()


def sig_figs(x, precision):
    if x == 0 or not math.isfinite(x):
        return x
    if precision <= 0:
        raise ValueError("precision must be > 0")
    return round(x, -int(floor(log10(abs(x)))) + (precision - 1))


def abs_percentage_error(true, pred):
    if true == 0: true += 1e-32
    return abs((pred - true) / true)


def format_mc_answer(input):
    output = input.lower().strip().strip('.')
    return " ".join(output.split())


def grade_open_end_factual(answer, extracted_answer, reward_config):
    # parsing เหมือนเดิม -------------------------------------------------
    correct_answer = answer.split(' ')[0].strip().lower()
    model_output   = extracted_answer.replace(' ', '').lower()
    correct_number, correct_letter = split_num_and_letters(correct_answer)
    model_number,   model_letter   = split_num_and_letters(model_output)
    reward = 0
    # --------------------------------------------------------------------

    # 1) ให้รางวัลเรื่องหน่วย (ตัวอักษร)
    if correct_letter == model_letter:
        reward += reward_config.UNIT_REWARD

    # 2) ให้รางวัลเรื่องตัวเลข  (ใช้ soft-clip)
    err = abs_percentage_error(correct_number, model_number)
    penalty = smooth_penalty(err, cap=reward_config.NUMERICAL_REWARD,
                            tau=reward_config.PENALTY_COEFF)
    reward += reward_config.NUMERICAL_REWARD - penalty
    return reward


def grade_open_end_numerical(answer, extracted_answer, reward_config):
    # parsing เหมือนเดิม -------------------------------------------------
    correct_mag, correct_unit = parse_answer(answer)
    model_mag,   model_unit   = parse_answer(extracted_answer)
    reward = 0
    # --------------------------------------------------------------------

    # 1) รางวัลเรื่องหน่วย
    if correct_unit == model_unit:
        reward += reward_config.UNIT_REWARD    

    # 2) รางวัลเรื่องตัวเลข
    model_sf   = sig_figs(model_mag,   reward_config.N_SIG_FIG)
    correct_sf = sig_figs(correct_mag, reward_config.N_SIG_FIG)

    if model_sf == correct_sf:                    # **ตรงทุก sig-fig**
        reward += reward_config.NUMERICAL_REWARD
    else:                                         # **ไม่ตรง ⇒ คิดโทษแบบโค้ง**
        err = abs_percentage_error(correct_mag, model_mag)
        penalty = smooth_penalty(err, cap=reward_config.NUMERICAL_REWARD,
                                tau=reward_config.PENALTY_COEFF)
        reward += reward_config.NUMERICAL_REWARD - penalty
    return reward

def grade_mc_question(answer, extracted_answer, option, reward_config):
    formatted_model_answer = format_mc_answer(extracted_answer)
    formatted_answer = format_mc_answer(answer)
    formatted_option = [format_mc_answer(o) for o in option]
    
    if formatted_model_answer == formatted_answer:
        return reward_config.MC_REWARD
    elif formatted_model_answer in formatted_option:
        return reward_config.MC_EXIST_REWARD
    else:
        return 0


def accuracy_reward(completions, **kwargs):
    reward_config = RewardConfig()

    # Read the completions
    completion_contents = [completion[0]["content"] for completion in completions]
    answers = kwargs['answer']
    options = kwargs['options']

    rewards = []

    # Loop through the batch
    for content, answer, option in zip(completion_contents, answers, options):
        # Extract the answer in the answer tag
        extracted_answer = extract_answer_from_tag(content)
        logger.info(f"Correct: '{answer}' Model: '{extracted_answer}'")

        # If it is an open-end questions
        if len(option) == 0:    
            # If the answer is something like [0.3d, ma, 32q, 4F, etc], but not [2.1 cm, 3 m^3, etc]
            if bool(re.search(r"[a-zA-Z]", answer.split(' ')[0])):
                reward = grade_open_end_factual(answer, extracted_answer, reward_config)

            else: # if the answer is in the format {number unit(optional)}
                reward = grade_open_end_numerical(answer, extracted_answer, reward_config)

        # If it is a multiple-choice questions
        else:
            reward = grade_mc_question(answer, extracted_answer, option, reward_config)
        
        rewards.append(reward)

    
    return rewards

# %% [markdown]
# ### Configuring GRPO Training Parameters

# %%
from trl import GRPOConfig

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="Qwen2.5-3B-GRPO-Physics_expo",
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=64,  # default: 16
    num_train_epochs=30,
    bf16=True,
    per_device_train_batch_size=2,  # default: 8
    per_device_eval_batch_size=2,  # default: 8
    # Parameters that control de data preprocessing
    # max_completion_length=64, # default: 256
    max_completion_length=512,  # default: 256
    num_generations=4,  # default: 8
    # max_prompt_length=128, # default: 512
    max_prompt_length=256,  # default: 512
    # Parameters related to reporting and saving
    report_to=["tensorboard"],
    logging_steps=5,
    push_to_hub=True,
    save_strategy="steps",
    save_steps=5,
)

# %% [markdown]
# ### Train the model

# %%
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
)

# %%
trainer.train(resume_from_checkpoint=False)

# %%
trainer.save_model(training_args.output_dir)
trainer.push_to_hub()


