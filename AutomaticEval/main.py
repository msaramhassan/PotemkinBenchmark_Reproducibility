import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd

from utils import (
  answer_and_grade_benchmark_question, 
  answer_open_ended_question,
  edit_to_introduce_error,
  generate_subquestions,
  grade_open_ended_question,
  relies_on_concept,
  sample_question, 
)
from experiment_logger import ExperimentLogger

parser = argparse.ArgumentParser()
parser.add_argument("--model", 
                    type=str, 
                    default=
                        "gpt-4o", 
                    help="Model to use. Can be an API model (e.g., gpt-4o, claude-3-5-sonnet-20241022) "
                         "or a local HuggingFace model path (e.g., meta-llama/Llama-2-7b-chat-hf). "
                         "Prefix with 'local:' to explicitly use local inference (e.g., local:mistralai/Mistral-7B-Instruct-v0.2)")
parser.add_argument("--benchmark", 
                    type=str, 
                    default="mmlu",
                    choices=["mmlu", "bbh"])
parser.add_argument("--num_subquestions", 
                    type=int, 
                    default=5   ,
                    help="number of subquestions (m) to use from the benchmark (for now assume k =1)")
parser.add_argument("--num_trials", 
                    type=int, 
                    default=10,
                    help="number of concepts to test")
args = parser.parse_args()

# Set up save directory
index_to_category = {0: "initial", 1: "edited_with_error"}
category_to_coherence = defaultdict(list)
overall_coherence = []
score_per_concept = []
bar = tqdm(range(args.num_trials))
score_per_concept_std_err = float("inf")

# Initialize experiment logger
log_dir = f"logs/{args.model.replace('/', '_')}/{args.benchmark}"
logger = ExperimentLogger(output_dir=log_dir, model=args.model, benchmark=args.benchmark)

print("\n==============================")
print("Starting experiment with model:", args.model)
print("Benchmark:", args.benchmark)
print("Number of subquestions:", args.num_subquestions)
print("Number of trials:", args.num_trials)
print("==============================\n")

for trial_index in bar:
    print(f"\n------------------------------")
    print(f"Starting trial {trial_index + 1}/{args.num_trials}")
    print("------------------------------\n")
    # Sample questions until we find one that depends on a concept and is answered correctly.
    while True:
        print("\nSampling question...")
        question, answer, subject = sample_question(
            args.benchmark, 
        )
        print("Sampled question:", question)
        logger.log_sample_question(trial_index, question, subject, answer)
        
        # See if question relies on a concept.
        print("\nChecking if question relies on a concept...")
        concept_classification, concept = relies_on_concept(question, args.model)
        print("Concept classification:", concept_classification, "Concept:", concept)
        logger.log_concept_detection(trial_index, question, subject, concept_classification, concept)
        
        if concept_classification:
            # See if question is answered correctly.
            print("\nChecking if question is answered correctly...")
            correct, full_answer = answer_and_grade_benchmark_question(question, args.model, answer, args.benchmark == "mmlu")
            print("Answer correct:", correct)
            logger.log_initial_answer(trial_index, question, subject, concept, full_answer, correct)
            if correct:
                break
    # Generate subquestions.
    print("\nGenerating subquestions...")
    max_attempts = 10
    while max_attempts > 0:
        subquestions = generate_subquestions(question, concept, args.model, args.num_subquestions)
        print("Generated subquestions:", subquestions)
        logger.log_subquestion_generation(trial_index, question, concept, subquestions, args.num_subquestions)
        if len(subquestions) >= args.num_subquestions:
            subquestions = np.random.choice(subquestions, size=args.num_subquestions, replace=False).tolist()
            break
        else:
            print(f"Failed to generate {args.num_subquestions} subquestions. Generated {len(subquestions)} subquestions. {max_attempts} attempts remaining. Retrying...")
            max_attempts -= 1

    if not subquestions:
        raise ValueError("Failed to generate any subquestions after multiple attempts.")

    print("\nProcessing subquestions...")
    subquestion_bar = tqdm(enumerate(subquestions), total=len(subquestions), desc="Subquestions")
    for index, subquestion in subquestion_bar:
        print(f"\nProcessing subquestion {index + 1}/{len(subquestions)}: {subquestion}")
        extracted_answer = answer_open_ended_question(subquestion, args.model)
        print("Extracted answer:", extracted_answer)
        logger.log_subquestion_answering(trial_index, question, concept, index, subquestion, extracted_answer)
        
        # Modify answer to either introduce or remove errors. 
        print("\nEditing answer to introduce/remove errors...")
        _, answer_with_error = edit_to_introduce_error(subquestion, extracted_answer, args.model)
        print("Answer with error:", answer_with_error)
        logger.log_answer_editing(trial_index, question, concept, index, subquestion, extracted_answer, answer_with_error)
        
        # Self-grading
        print("\nStarting self-grading...")
        expected_answers = ["correct", "incorrect"]
        all_answers = [extracted_answer, answer_with_error]
        for i, answer in enumerate(all_answers):
            print(f"\nGrading answer {i + 1}/{len(all_answers)}: {answer}")
            judge_answer, full_judge_answer = grade_open_ended_question(subquestion, answer, args.model)
            print("Judge answer:", judge_answer)
            expected_answer = expected_answers[i]
            
            # Check if judge output is valid
            valid_grading = judge_answer.strip().lower()[:7] == "correct" or judge_answer.strip().lower()[:9] == "incorrect"
            
            if valid_grading:
                coherent = 1 if judge_answer.strip().lower()[:len(expected_answer)] == expected_answer.strip().lower() else 0
                judge_label = "correct" if judge_answer.strip().lower()[:7] == "correct" else "incorrect"
            else:
                coherent = None
                judge_label = None
            
            logger.log_grading(
                trial_index=trial_index,
                question=question,
                concept=concept,
                subquestion_index=index,
                subquestion=subquestion,
                model_answer=answer,
                judge_answer_raw=full_judge_answer,
                judge_label=judge_label,
                expected_label=expected_answer,
                category=index_to_category[i],
                coherent=coherent,
                valid=valid_grading
            )
            
            # Only add answer if it's not empty. Sometimes R1 doesn't finish.
            if valid_grading:
                category_to_coherence[index_to_category[i]].append(coherent)
                overall_coherence.append(coherent)
                
                logger.log_coherence_scoring(
                    trial_index=trial_index,
                    question=question,
                    concept=concept,
                    coherent=coherent,
                    expected_label=expected_answer,
                    judge_label=judge_label,
                    category=index_to_category[i]
                )
                
                log_dict = {
                    "original_question": question,
                    "concept": concept,
                    "subquestion": subquestion,
                    "answer": answer,
                    "judge_answer": full_judge_answer,
                    "category": index_to_category[i],
                    "expected_answer": expected_answer,
                    "coherent": coherent,
                }
                save_dir = f"results/{args.model}/{args.benchmark}/{trial_index}/{index_to_category[i]}"
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, f"{concept.replace('/', '_').split(' ')[0]}_{index}_{i}.json"), "w") as f:
                    json.dump(log_dict, f)
    # Potemkin rate is normalized incoherence so 1 is random and 0 is perfect.
    print("\nCalculating Potemkin rate...")
    potemkin_rate = 2 * (1 - np.mean(overall_coherence))
    score_per_concept.append(potemkin_rate)
    score_per_concept_std_err = np.std(score_per_concept) / np.sqrt(len(score_per_concept))
    # Log results to bar
    bar.set_description(f"Model: {args.model} "
                        f"Potemkin rate (lower bound): {np.mean(score_per_concept):.2f} ({score_per_concept_std_err:.2f}). ")

print(f"\n==============================")
print(f"Potemkin rate (lower bound): {np.mean(score_per_concept):.2f} ({score_per_concept_std_err:.2f}).")
print("==============================\n")

# Save experiment logs
print("\nSaving experiment logs...")
log_file = logger.save()
log_file_jsonl = logger.save_jsonl()
print(f"Experiment logs saved to: {log_file}")
print(f"Experiment logs (JSONL) saved to: {log_file_jsonl}")
