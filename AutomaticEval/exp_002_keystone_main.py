"""
EXP-002: Alternative Keystone Experiment (RQ: Definition Recall vs. Understanding)
====================================================================================
Tests whether the high Potemkin rates reported in the original paper depend
on the *definition generation* keystone.  Runs three keystone variants for
game-theory concepts (the only auto-gradable domain) and computes Potemkin
rates separately for each.

VARIANT A – Definition-as-Keystone (paper's default)
  Keystone: "Define [concept] in 1-2 sentences."
  Proceed only if judge grades definition as CORRECT.

VARIANT B – Multiple-Choice Recognition
  Keystone: 4-option MCQ "Which best defines [concept]?"
  Proceed only if model selects the correct option (deterministic grade).

VARIANT C – Classification Keystone
  Keystone: Scenario presented; model classifies whether it exhibits [concept].
  Proceed only if classification matches ground truth.

Usage:
    python exp_002_keystone_main.py \
        --model gpt-4o \
        --variant A          # or B or C \
        --num_concepts 15 \
        --num_subquestions 5

Output:
    results/exp-002-keystone/{model}/{variant}/
    └── {concept}_{trial}.json
"""
import argparse
import json
import os
import random
import sys
import re

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
from utils import (
    answer_open_ended_question,
    edit_to_introduce_error,
    generate_inference,
    generate_subquestions,
    grade_open_ended_question,
)

# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------
utils.models_to_developer.update({
    "gpt-4o":                                      "openai",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "together",
    "claude-3-5-sonnet-20241022":                   "claude",
})

FINAL_TAG = "FINAL ANSWER:"

# ---------------------------------------------------------------------------
# Game theory concept definitions (ground-truth, auto-gradable domain)
# ---------------------------------------------------------------------------
GAME_THEORY_CONCEPTS = {
    "Strict Dominance": {
        "definition": (
            "A strategy strictly dominates another if it yields a strictly higher payoff "
            "for a player regardless of what the other players do."
        ),
        "mcq_options": {
            "A": "A strategy strictly dominates another if it yields a strictly higher payoff "
                 "for a player regardless of what the other players do.",   # CORRECT
            "B": "A strategy strictly dominates another if it yields a higher payoff "
                 "only when all other players also use their best strategies.",
            "C": "A strategy strictly dominates another when a player can achieve "
                 "the highest possible total payoff in the game.",
            "D": "Strict dominance means a player has only one rational strategy "
                 "that is consistent with the Nash equilibrium.",
        },
        "correct_mcq": "A",
        "scenario": (
            "In a simultaneous-move game, Alice has two strategies: Up and Down. "
            "Regardless of whether Bob plays Left or Right, Alice earns 10 with Up "
            "but only 5 with Down."
        ),
        "scenario_exhibits": True,
    },
    "Weak Dominance": {
        "definition": (
            "A strategy weakly dominates another if it yields at least as high a payoff "
            "in every situation and strictly higher in at least one situation."
        ),
        "mcq_options": {
            "A": "Weak dominance occurs when a strategy yields a higher payoff in every "
                 "possible outcome, regardless of the other players' choices.",
            "B": "A strategy weakly dominates another if it yields at least as high a payoff "
                 "in every situation and strictly higher in at least one situation.",   # CORRECT
            "C": "Weak dominance means a player is indifferent between two strategies "
                 "because they always produce identical payoffs.",
            "D": "A strategy weakly dominates another if it is chosen in the unique "
                 "Nash equilibrium of the game.",
        },
        "correct_mcq": "B",
        "scenario": (
            "Carol chooses between strategies X and Y. Against Dave's Left, both strategies "
            "give Carol a payoff of 4. Against Dave's Right, strategy X gives Carol 6 "
            "while Y gives 4."
        ),
        "scenario_exhibits": True,
    },
    "Iterated Dominance": {
        "definition": (
            "Iterated elimination of dominated strategies is a process of sequentially "
            "removing dominated strategies, then solving the reduced game, until no "
            "dominated strategies remain."
        ),
        "mcq_options": {
            "A": "Iterated dominance requires all players to have strictly dominant "
                 "strategies that can be identified in a single round of analysis.",
            "B": "Iterated dominance is the process of choosing the strategy with the "
                 "highest average payoff across all opponent strategies.",
            "C": "Iterated elimination of dominated strategies is a process of sequentially "
                 "removing dominated strategies, then solving the reduced game, until no "
                 "dominated strategies remain.",   # CORRECT
            "D": "Iterated dominance means repeatedly applying a mixed-strategy Nash "
                 "equilibrium until payoffs converge.",
        },
        "correct_mcq": "C",
        "scenario": (
            "Eve and Frank play a 3×3 game. Eve's strategy R3 is strictly dominated and "
            "removed. After removal, Frank's strategy C3 becomes dominated and is removed. "
            "The remaining 2×2 game has a unique solution."
        ),
        "scenario_exhibits": True,
    },
    "Pure Strategy Nash Equilibrium": {
        "definition": (
            "A pure strategy Nash equilibrium is a combination of strategies (one per player) "
            "such that no player can gain by unilaterally switching to a different strategy, "
            "given the strategies of all other players."
        ),
        "mcq_options": {
            "A": "A pure strategy Nash equilibrium is a strategy profile that maximises "
                 "total social welfare for all players simultaneously.",
            "B": "A pure strategy Nash equilibrium requires every player to use a mixed "
                 "strategy that randomises over all available options.",
            "C": "A pure strategy Nash equilibrium exists only in zero-sum games where "
                 "one player's gain equals the other player's loss.",
            "D": "A pure strategy Nash equilibrium is a combination of strategies (one per player) "
                 "such that no player can gain by unilaterally switching to a different strategy, "
                 "given the strategies of all other players.",   # CORRECT
        },
        "correct_mcq": "D",
        "scenario": (
            "In a 2×2 game, when Alice plays Up and Bob plays Left, Alice's best response "
            "to Bob's Left is also Up, and Bob's best response to Alice's Up is also Left."
        ),
        "scenario_exhibits": True,
    },
    "Mixed Strategy Nash Equilibrium": {
        "definition": (
            "A mixed strategy Nash equilibrium is a strategy profile in which at least "
            "one player randomises over multiple pure strategies with positive probability, "
            "and no player can increase expected payoff by deviating."
        ),
        "mcq_options": {
            "A": "A mixed strategy Nash equilibrium is a solution where every player "
                 "always plays their dominant strategy with certainty.",
            "B": "A mixed strategy Nash equilibrium is a strategy profile in which at least "
                 "one player randomises over multiple pure strategies with positive probability, "
                 "and no player can increase expected payoff by deviating.",   # CORRECT
            "C": "A mixed strategy Nash equilibrium requires all players to choose "
                 "strategies randomly with equal probability.",
            "D": "Mixed strategy Nash equilibrium is only applicable when a pure strategy "
                 "Nash equilibrium does not exist in the game.",
        },
        "correct_mcq": "B",
        "scenario": (
            "In Matching Pennies, each player randomises 50/50 between Heads and Tails. "
            "Given this mixing by the opponent, each player is indifferent between "
            "their strategies and has no incentive to deviate."
        ),
        "scenario_exhibits": True,
    },
    "Pareto Optimality": {
        "definition": (
            "An outcome is Pareto optimal if there is no alternative outcome that makes "
            "at least one player better off without making any other player worse off."
        ),
        "mcq_options": {
            "A": "Pareto optimality means the total payoff across all players is maximised.",
            "B": "An outcome is Pareto optimal if it is also the Nash equilibrium "
                 "of the game in question.",
            "C": "An outcome is Pareto optimal if there is no alternative outcome that makes "
                 "at least one player better off without making any other player worse off.",   # CORRECT
            "D": "Pareto optimality requires equal payoffs for all players in the game.",
        },
        "correct_mcq": "C",
        "scenario": (
            "Alice gets 8 and Bob gets 8 in outcome X. In outcome Y, Alice gets 9 and "
            "Bob gets 8. Outcome X is not Pareto optimal because moving to Y makes "
            "Alice better off without harming Bob."
        ),
        "scenario_exhibits": False,   # X is NOT Pareto optimal
    },
    "Best Response": {
        "definition": (
            "A best response is a strategy that maximises a player's payoff given the "
            "strategies chosen by all other players."
        ),
        "mcq_options": {
            "A": "A best response is the strategy that always yields the highest payoff "
                 "regardless of what the opponents choose.",
            "B": "A best response is a strategy that minimises the opponent's payoff "
                 "rather than maximising the player's own payoff.",
            "C": "A best response is always unique and corresponds to a dominant strategy.",
            "D": "A best response is a strategy that maximises a player's payoff given the "
                 "strategies chosen by all other players.",   # CORRECT
        },
        "correct_mcq": "D",
        "scenario": (
            "Given that Bob chooses Left, Alice earns 10 with Up and 5 with Down. "
            "Alice therefore plays Up as her best response to Bob's Left."
        ),
        "scenario_exhibits": True,
    },
    "Zero-Sum Game": {
        "definition": (
            "A zero-sum game is a game in which the total payoff across all players is "
            "constant (usually zero), so any gain for one player represents an equal "
            "loss for another."
        ),
        "mcq_options": {
            "A": "A zero-sum game is any game in which all players receive a payoff of zero.",
            "B": "A zero-sum game is a game where players cooperate to achieve a combined "
                 "payoff of zero.",
            "C": "A zero-sum game is a game in which the total payoff across all players is "
                 "constant (usually zero), so any gain for one player represents an equal "
                 "loss for another.",   # CORRECT
            "D": "A zero-sum game requires all strategies to be dominant for every player.",
        },
        "correct_mcq": "C",
        "scenario": (
            "Two traders bet on the price of oil. If oil rises, trader A gains $100 "
            "while trader B loses $100. The sum of payoffs is always zero."
        ),
        "scenario_exhibits": True,
    },
    "Symmetric Game": {
        "definition": (
            "A symmetric game is one in which the payoffs depend only on the strategies "
            "chosen, not on which player chooses them; swapping player identities "
            "does not change the game."
        ),
        "mcq_options": {
            "A": "A symmetric game is one where all players have the same number of "
                 "strategies available to them.",
            "B": "A symmetric game requires all Nash equilibria to be Pareto optimal.",
            "C": "A symmetric game is one in which the payoffs depend only on the strategies "
                 "chosen, not on which player chooses them; swapping player identities "
                 "does not change the game.",   # CORRECT
            "D": "Symmetry in games means every player has the same dominant strategy.",
        },
        "correct_mcq": "C",
        "scenario": (
            "In the Prisoner's Dilemma, both prisoners face identical payoff structures: "
            "if both cooperate they each get 3, if both defect they each get 1, and "
            "if one defects while the other cooperates, the defector gets 5 and the "
            "cooperator gets 0."
        ),
        "scenario_exhibits": True,
    },
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="EXP-002: Keystone Sensitivity")
parser.add_argument("--model",            type=str, default="gpt-4o")
parser.add_argument("--variant",          type=str, default="A",
                    choices=["A", "B", "C"],
                    help="A=Definition, B=MCQ, C=Classification")
parser.add_argument("--num_concepts",     type=int, default=9,
                    help="Number of game-theory concepts to test (max 9)")
parser.add_argument("--num_subquestions", type=int, default=5)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Helper: grade a free-text definition
# ---------------------------------------------------------------------------
def grade_definition(concept: str, definition: str, gold_definition: str,
                     model: str) -> bool:
    prompt = (
        f"A student defined the game-theory concept '{concept}' as follows:\n\n"
        f"Student definition: {definition}\n\n"
        f"Correct definition: {gold_definition}\n\n"
        "Is the student's definition correct?  A definition is correct if it captures "
        "the core meaning accurately, even if the wording differs.  It is incorrect if "
        "it omits a key element or contains a factual error.\n\n"
        f"Reason briefly, then end with `{FINAL_TAG}` followed by exactly 'correct' "
        "or 'incorrect'."
    )
    raw  = generate_inference(prompt, model)
    ans  = utils.extract_final_answer(raw).strip().lower().replace("*", "")
    return ans.startswith("correct"), raw


def generate_classification_scenario(concept: str, exhibits: bool, model: str) -> str:
    """Generate a game-theory scenario that either exhibits or does not exhibit concept."""
    exhibit_str = "clearly exhibits" if exhibits else "does NOT exhibit"
    prompt = (
        f"Write a brief (3-5 sentence) game-theory scenario that {exhibit_str} "
        f"the concept of '{concept}'.  Do not name the concept explicitly.  "
        f"Output only the scenario text, nothing else."
    )
    return generate_inference(prompt, model).strip()


def grade_classification(concept: str, scenario: str, model_says_yes: bool,
                         ground_truth_yes: bool) -> bool:
    """Return True if model's classification matches ground truth."""
    return model_says_yes == ground_truth_yes


def run_subquestion_pipeline(concept: str, model: str, num_subquestions: int,
                             trigger_question: str) -> list[int]:
    """
    Run the standard subquestion pipeline given a trigger question.
    Returns a list of coherence values (0 or 1) for each (subquestion, version) pair.
    """
    # Generate subquestions
    max_attempts = 10
    subquestions = []
    while max_attempts > 0:
        subquestions = generate_subquestions(
            trigger_question, concept, model, num_subquestions
        )
        if len(subquestions) >= num_subquestions:
            subquestions = np.random.choice(
                subquestions, size=num_subquestions, replace=False
            ).tolist()
            break
        max_attempts -= 1

    if not subquestions:
        return []

    coherence_vals = []
    for sq in subquestions:
        extracted = answer_open_ended_question(sq, model)
        _, answer_with_error = edit_to_introduce_error(sq, extracted, model)

        for version_idx, (answer, expected) in enumerate(
            zip([extracted, answer_with_error], ["correct", "incorrect"])
        ):
            judge_ans, _ = grade_open_ended_question(sq, answer, model)
            valid = (
                judge_ans.strip().lower()[:7]  == "correct" or
                judge_ans.strip().lower()[:9]  == "incorrect"
            )
            if valid:
                coherent = 1 if judge_ans.strip().lower()[:len(expected)] == expected else 0
                coherence_vals.append(coherent)

    return coherence_vals


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
concepts = list(GAME_THEORY_CONCEPTS.items())[:args.num_concepts]

model_tag  = args.model.replace("/", "_").replace(":", "_")
save_base  = os.path.join("results", "exp-002-keystone", model_tag, f"variant_{args.variant}")
os.makedirs(save_base, exist_ok=True)

all_potemkin_rates        = []
all_keystone_pass_rates   = []

print(f"\n{'='*65}")
print(f"  EXP-002: Keystone Sensitivity  |  Variant {args.variant}")
print(f"  Model     : {args.model}")
print(f"  Concepts  : {len(concepts)}")
print(f"{'='*65}\n")

bar = tqdm(concepts, desc=f"Variant {args.variant}")

for concept_name, concept_data in bar:
    bar.set_description(f"Var {args.variant} | {concept_name[:25]}")
    gold_def   = concept_data["definition"]
    scenario   = concept_data["scenario"]
    exhibits   = concept_data["scenario_exhibits"]
    mcq        = concept_data["mcq_options"]
    correct_ltr = concept_data["correct_mcq"]

    keystone_passed = False
    keystone_raw    = ""
    trigger_q       = f"What is {concept_name} in game theory?"

    # ------------------------------------------------------------------ #
    # Variant A: Definition keystone                                       #
    # ------------------------------------------------------------------ #
    if args.variant == "A":
        def_prompt = (
            f"Define the game-theory concept '{concept_name}' in 1-2 sentences.  "
            f"You can reason, but end with `{FINAL_TAG}` followed by your definition."
        )
        raw_def = generate_inference(def_prompt, args.model)
        model_def = utils.extract_final_answer(raw_def).strip()
        keystone_raw = model_def
        keystone_passed, _ = grade_definition(concept_name, model_def, gold_def, args.model)

    # ------------------------------------------------------------------ #
    # Variant B: MCQ keystone                                             #
    # ------------------------------------------------------------------ #
    elif args.variant == "B":
        opts_text = "\n".join(f"{ltr}. {txt}" for ltr, txt in mcq.items())
        mcq_prompt = (
            f"Which of the following best defines '{concept_name}' in game theory?\n\n"
            f"{opts_text}\n\n"
            f"Think briefly, then end with `{FINAL_TAG}` followed by the single letter "
            f"(A, B, C, or D) of the best answer."
        )
        raw_mcq = generate_inference(mcq_prompt, args.model)
        model_choice_raw = utils.extract_final_answer(raw_mcq).strip().upper()
        # Accept if first character matches correct letter
        model_choice = model_choice_raw[0] if model_choice_raw else ""
        keystone_raw = model_choice_raw
        keystone_passed = (model_choice == correct_ltr)

    # ------------------------------------------------------------------ #
    # Variant C: Classification keystone                                  #
    # ------------------------------------------------------------------ #
    elif args.variant == "C":
        classify_prompt = (
            f"Consider the following scenario:\n\n{scenario}\n\n"
            f"Does this scenario exhibit the concept of '{concept_name}'?  "
            f"Reason briefly, then end with `{FINAL_TAG}` followed by 'yes' or 'no'."
        )
        raw_cls = generate_inference(classify_prompt, args.model)
        model_cls_raw = utils.extract_final_answer(raw_cls).strip().lower().replace("*", "")
        model_says_yes = model_cls_raw.startswith("yes")
        keystone_raw   = model_cls_raw
        keystone_passed = grade_classification(concept_name, scenario, model_says_yes, exhibits)

    all_keystone_pass_rates.append(1 if keystone_passed else 0)

    # ---- Only proceed to downstream tasks if keystone PASSED ---- #
    coherence_vals = []
    potemkin_rate  = None

    if keystone_passed:
        coherence_vals = run_subquestion_pipeline(
            concept_name, args.model, args.num_subquestions, trigger_q
        )
        if coherence_vals:
            potemkin_rate = 2 * (1 - np.mean(coherence_vals))
            all_potemkin_rates.append(potemkin_rate)

    # Save per-concept result
    result_record = {
        "concept":          concept_name,
        "model":            args.model,
        "variant":          args.variant,
        "keystone_passed":  keystone_passed,
        "keystone_raw":     keystone_raw,
        "gold_definition":  gold_def,
        "coherence_vals":   coherence_vals,
        "potemkin_rate":    potemkin_rate,
        "n_valid":          len(coherence_vals),
    }
    safe_name = concept_name.replace(" ", "_").replace("/", "_")
    with open(os.path.join(save_base, f"{safe_name}.json"), "w") as fh:
        json.dump(result_record, fh, indent=2)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
overall_potemkin = float(np.mean(all_potemkin_rates)) if all_potemkin_rates else float("nan")
overall_keystone = float(np.mean(all_keystone_pass_rates))
stderr = (float(np.std(all_potemkin_rates, ddof=1) / np.sqrt(len(all_potemkin_rates)))
          if len(all_potemkin_rates) > 1 else float("nan"))

summary = {
    "model":                args.model,
    "variant":              args.variant,
    "num_concepts_tested":  len(concepts),
    "keystone_pass_rate":   overall_keystone,
    "potemkin_rate_mean":   overall_potemkin,
    "potemkin_rate_stderr": stderr,
    "n_downstream_trials":  len(all_potemkin_rates),
}

summary_path = os.path.join(save_base, "summary.json")
with open(summary_path, "w") as fh:
    json.dump(summary, fh, indent=2)

print(f"\n{'='*65}")
print(f"  VARIANT {args.variant} RESULTS  |  Model: {args.model}")
print(f"  Keystone pass rate : {overall_keystone:.3f}")
print(f"  Potemkin rate      : {overall_potemkin:.4f} ± {stderr:.4f}")
print(f"  N downstream trials: {len(all_potemkin_rates)}")
print(f"{'='*65}")
print(f"\nResults saved → {save_base}/")
