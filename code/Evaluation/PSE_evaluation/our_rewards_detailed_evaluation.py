import pint
from pint import UnitRegistry

from pylatexenc.latex2text import LatexNodes2Text

import re

import math
from math import floor, log10

import logging

from math_verify import LatexExtractionConfig, parse, verify

logger = None

ureg = UnitRegistry()

preferred_units = [
    ureg.m,  # meter (length) - L
    ureg.kg,  # kilogram (mass) - M
    ureg.s,  # second (time) - T
    ureg.degC,  # Degree C (thermodynamic temperature) - Θ
    ureg.A,  # Ampere (electric current) - I
    ureg.mol,  # mole (amount of substance) - N
    ureg.cd,  # candela (luminous intensity) - J
]

ureg.define("@alias au = AU")
ureg.define("@alias J = Joules")
ureg.define("@alias A = amps")

ureg.default_preferred_units = preferred_units

Q_ = ureg.Quantity


class RewardConfig:
    def __init__(
        self,
        PENALTY_COEFF: float = 10000,
        N_SIG_FIG: int = 4,
        UNIT_REWARD: float = 0.5,
        MC_REWARD: float = 1,
        MC_EXIST_REWARD: float = 0.1,
    ):
        self.PENALTY_COEFF = PENALTY_COEFF
        self.N_SIG_FIG = N_SIG_FIG
        self.UNIT_REWARD = UNIT_REWARD
        self.NUMERICAL_REWARD = 1 - UNIT_REWARD  # derived from UNIT_REWARD
        self.MC_REWARD = MC_REWARD
        self.MC_EXIST_REWARD = MC_EXIST_REWARD


def init_logger(passed_logger=None):
    """Initialize the global logger if needed."""
    global logger
    if passed_logger is not None:
        logger = passed_logger
    elif logger is None:
        logger = logging.getLogger("reward")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(handler)


def parse_answer(extracted_answer):
    # split the answer into strings of numerical and unit
    answer_split = (
        LatexNodes2Text()
        .latex_to_text(extracted_answer)
        .replace(" x ", "*")
        .replace(" × ", "*")
        .strip()
        .split(" ")
    )
    if len(answer_split) == 0:
        numerical_string, unit_string = "", ""
    elif len(answer_split) == 1:
        numerical_string, unit_string = answer_split[0], ""
    else:
        numerical_string, unit_string = answer_split[0], " ".join(answer_split[1:])

    # dealing with the unit system
    try:
        parse_unit = ureg.parse_units(
            unit_string, as_delta=False
        )  # from string to `pint.Unit`
        try:
            unit_quantity = Q_(
                1 * parse_unit
            ).to_preferred()  # convert unit to `preferred_units`
        except pint.DimensionalityError:
            unit_quantity = Q_(1 * parse_unit)
        unit_multiplier, pint_units = unit_quantity.magnitude, unit_quantity.units
    except Exception as e:
        unit_multiplier, pint_units = 1, ""
        logger.warning(
            f"Exception {e}:\nError when parsing unit '{unit_string}' in '{extracted_answer}'"
        )

    # dealing with the numerical part
    try:
        magnitude = Q_(numerical_string).to_base_units().magnitude * unit_multiplier
    except Exception as e:
        magnitude = 0
        logger.warning(
            f"Exception {e}:\nError when parsing numerical '{numerical_string}' in '{extracted_answer}'"
        )

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


def squared_percentage_error(true, pred):
    diff = pred - true
    if true == 0:
        true += 1e-32
    error = diff / true
    result = error**2
    return result


def format_mc_answer(input):
    output = input.lower().replace(" ", "").strip(".")
    return output


def grade_open_end_factual(answer, extracted_answer, reward_config):
    # parsing the answers
    correct_answer = answer.split(" ")[0].strip().lower()
    model_output = extracted_answer.replace(" ", "").lower()
    correct_number, correct_letter = split_num_and_letters(correct_answer)
    model_number, model_letter = split_num_and_letters(model_output)

    reward = None

    # Rewarding the letter
    if correct_letter == model_letter:
        unit_reward = reward_config.UNIT_REWARD
    else:
        unit_reward = 0

    # Rewarding the numerical part using squared percentage error
    penalty = reward_config.PENALTY_COEFF * squared_percentage_error(
        correct_number, model_number
    )
    penalty = min(penalty, reward_config.NUMERICAL_REWARD)
    reward = (reward_config.NUMERICAL_REWARD - penalty), unit_reward

    return reward


def grade_open_end_numerical(answer, extracted_answer, reward_config):
    # parsing the answers
    correct_magnitude, correct_pint_units = parse_answer(answer)
    model_magnitude, model_pint_units = parse_answer(extracted_answer)

    reward = None

    # Rewarding the unit (consider km/h as the same as m/s, as they are both units for speed/velocity)
    if correct_pint_units == model_pint_units:
        unit_reward = reward_config.UNIT_REWARD
    else: 
        unit_reward = 0

    # Rewarding the numerical part
    model_magnitude_sf = sig_figs(model_magnitude, reward_config.N_SIG_FIG)
    correct_magnitude_sf = sig_figs(correct_magnitude, reward_config.N_SIG_FIG)

    # If the n-sig-figs are the same, reward NUMERICAL_REWARD (considered asexact match)
    if model_magnitude_sf == correct_magnitude_sf:
        reward = reward_config.NUMERICAL_REWARD, unit_reward
    # Else reward based on squared percentage error
    else:
        penalty = reward_config.PENALTY_COEFF * squared_percentage_error(
            correct_magnitude, model_magnitude
        )
        penalty = min(penalty, reward_config.NUMERICAL_REWARD)
        reward = (reward_config.NUMERICAL_REWARD - penalty), unit_reward

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


def physics_accuracy_reward(completions, passed_logger=None, reward_config=None, **kwargs):
    init_logger(passed_logger)

    if reward_config is None:
        reward_config = RewardConfig()
        
    # Read the completions
    completion_contents = [completion[0]["content"] for completion in completions]
    answers = kwargs["answer"]
    options = kwargs["options"]

    rewards = []

    # Loop through the batch
    for content, answer, option in zip(completion_contents, answers, options):
        # Extract the answer in the answer tag
        extracted_answer = extract_answer_from_tag(content)
        logger.info(f"Correct: '{answer}' Model: '{extracted_answer}'")

        # If it is an open-end questions
        if len(option) == 0:
            # If the answer is something like [0.3d, ma, 32q, 4F, etc], but not [2.1 cm, 3 m^3, etc]
            if bool(re.search(r"[a-zA-Z]", answer.split(" ")[0])):
                reward = grade_open_end_factual(answer, extracted_answer, reward_config)

            else:  # if the answer is in the format {number unit(optional)}
                reward = grade_open_end_numerical(
                    answer, extracted_answer, reward_config
                )

        # If it is a multiple-choice questions
        else:
            reward = grade_mc_question(answer, extracted_answer, option, reward_config)

        rewards.append(reward)

    return rewards


def physics_accuracy_reward_evaluation(completions, passed_logger=None, reward_config=None, **kwargs):
    init_logger(passed_logger)

    if reward_config is None:
        reward_config = RewardConfig()
        
    # Read the completions
    completion_contents = [completion[0]["content"] for completion in completions]
    answers = kwargs["answer"]
    options = kwargs["options"]

    rewards = []

    # Loop through the batch
    for content, answer, option in zip(completion_contents, answers, options):
        # Extract the answer in the answer tag
        extracted_answer = extract_answer_from_tag(content)
        logger.info(f"Correct: '{answer}' Model: '{extracted_answer}'")

        # If it is an open-end questions
        if len(option) == 0:
            # If the answer is something like [0.3d, ma, 32q, 4F, etc], but not [2.1 cm, 3 m^3, etc]
            if bool(re.search(r"[a-zA-Z]", answer.split(" ")[0])):
                reward = grade_open_end_factual(answer, extracted_answer, reward_config)

            else:  # if the answer is in the format {number unit(optional)}
                reward = grade_open_end_numerical(
                    answer, extracted_answer, reward_config
                )

            reward = (reward, "open_end")

        # If it is a multiple-choice questions
        else:
            reward = grade_mc_question(answer, extracted_answer, option, reward_config)
            reward = (reward, "mc")

        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]


def maths_accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs["solution"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(
            solution,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        answer_parsed = parse(
            content,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards
