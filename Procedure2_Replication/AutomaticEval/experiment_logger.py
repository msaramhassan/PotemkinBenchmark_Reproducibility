import json
import os
from datetime import datetime
from typing import List

class ExperimentLogger:
    """Logger for structured JSON output of pipeline stages."""
    
    def __init__(self, output_dir: str, model: str, benchmark: str):
        self.output_dir = output_dir
        self.model = model
        self.benchmark = benchmark
        self.logs = []
        os.makedirs(output_dir, exist_ok=True)
    
    def _create_log_entry(
        self,
        trial_index: int,
        stage: str,
        question: str = None,
        subject: str = None,
        concept: str = None,
        concept_detected: bool = None,
        original_answer: str = None,
        edited_answer: str = None,
        subquestions: List[str] = None,
        subquestion_index: int = None,
        subquestion: str = None,
        model_answer: str = None,
        judge_answer_raw: str = None,
        judge_label: str = None,
        expected_label: str = None,
        coherent: int = None,
        success: bool = True,
        error_type: str = None,
        notes: str = None
    ) -> dict:
        return {
            "trial_index": trial_index,
            "benchmark": self.benchmark,
            "model": self.model,
            "stage": stage,
            "data": {
                "question": question,
                "subject": subject,
                "concept": concept,
                "concept_detected": concept_detected,
                "original_answer": original_answer,
                "edited_answer": edited_answer,
                "subquestions": subquestions,
                "subquestion_index": subquestion_index,
                "subquestion": subquestion,
                "model_answer": model_answer,
                "judge_answer_raw": judge_answer_raw,
                "judge_label": judge_label,
                "expected_label": expected_label,
                "coherent": coherent
            },
            "status": {
                "success": success,
                "error_type": error_type,
                "notes": notes
            }
        }
    
    def log_sample_question(
        self,
        trial_index: int,
        question: str,
        subject: str,
        correct_answer: str
    ) -> dict:
        entry = self._create_log_entry(
            trial_index=trial_index,
            stage="sample_question",
            question=question,
            subject=subject,
            notes=f"correct_answer={correct_answer}"
        )
        self.logs.append(entry)
        return entry
    
    def log_concept_detection(
        self,
        trial_index: int,
        question: str,
        subject: str,
        concept_detected: bool,
        concept: str = None
    ) -> dict:
        entry = self._create_log_entry(
            trial_index=trial_index,
            stage="concept_detection",
            question=question,
            subject=subject,
            concept=concept,
            concept_detected=concept_detected,
            success=concept_detected,
            error_type="no_concept" if not concept_detected else None
        )
        self.logs.append(entry)
        return entry
    
    def log_initial_answer(
        self,
        trial_index: int,
        question: str,
        subject: str,
        concept: str,
        model_answer: str,
        correct: bool
    ) -> dict:
        entry = self._create_log_entry(
            trial_index=trial_index,
            stage="initial_answer",
            question=question,
            subject=subject,
            concept=concept,
            model_answer=model_answer,
            success=correct,
            error_type="incorrect_initial_answer" if not correct else None
        )
        self.logs.append(entry)
        return entry
    
    def log_subquestion_generation(
        self,
        trial_index: int,
        question: str,
        concept: str,
        subquestions: List[str],
        expected_count: int
    ) -> dict:
        success = len(subquestions) >= expected_count
        entry = self._create_log_entry(
            trial_index=trial_index,
            stage="subquestion_generation",
            question=question,
            concept=concept,
            subquestions=subquestions,
            success=success,
            error_type="insufficient_subquestions" if not success else None,
            notes=f"generated={len(subquestions)}, expected={expected_count}"
        )
        self.logs.append(entry)
        return entry
    
    def log_subquestion_answering(
        self,
        trial_index: int,
        question: str,
        concept: str,
        subquestion_index: int,
        subquestion: str,
        model_answer: str
    ) -> dict:
        entry = self._create_log_entry(
            trial_index=trial_index,
            stage="subquestion_answering",
            question=question,
            concept=concept,
            subquestion_index=subquestion_index,
            subquestion=subquestion,
            model_answer=model_answer
        )
        self.logs.append(entry)
        return entry
    
    def log_answer_editing(
        self,
        trial_index: int,
        question: str,
        concept: str,
        subquestion_index: int,
        subquestion: str,
        original_answer: str,
        edited_answer: str
    ) -> dict:
        entry = self._create_log_entry(
            trial_index=trial_index,
            stage="answer_editing",
            question=question,
            concept=concept,
            subquestion_index=subquestion_index,
            subquestion=subquestion,
            original_answer=original_answer,
            edited_answer=edited_answer
        )
        self.logs.append(entry)
        return entry
    
    def log_grading(
        self,
        trial_index: int,
        question: str,
        concept: str,
        subquestion_index: int,
        subquestion: str,
        model_answer: str,
        judge_answer_raw: str,
        judge_label: str,
        expected_label: str,
        category: str,
        coherent: int = None,
        valid: bool = True
    ) -> dict:
        entry = self._create_log_entry(
            trial_index=trial_index,
            stage="grading",
            question=question,
            concept=concept,
            subquestion_index=subquestion_index,
            subquestion=subquestion,
            model_answer=model_answer,
            judge_answer_raw=judge_answer_raw,
            judge_label=judge_label,
            expected_label=expected_label,
            coherent=coherent,
            success=valid,
            error_type="invalid_grading" if not valid else None,
            notes=f"category={category}"
        )
        self.logs.append(entry)
        return entry
    
    def log_potemkin_rate(
        self,
        trial_index: int,
        concept: str,
        potemkin_rate: float,
        overall_coherence_mean: float,
        running_scores: list
    ) -> dict:
        entry = self._create_log_entry(
            trial_index=trial_index,
            stage="potemkin_rate",
            concept=concept,
            notes=(
                f"potemkin_rate={potemkin_rate:.4f}, "
                f"overall_coherence_mean={overall_coherence_mean:.4f}, "
                f"num_concepts_so_far={len(running_scores)}"
            )
        )
        entry["data"]["potemkin_rate"] = potemkin_rate
        entry["data"]["overall_coherence_mean"] = overall_coherence_mean
        entry["data"]["running_scores"] = list(running_scores)
        self.logs.append(entry)
        return entry

    def log_coherence_scoring(
        self,
        trial_index: int,
        question: str,
        concept: str,
        coherent: int,
        expected_label: str,
        judge_label: str,
        category: str
    ) -> dict:
        entry = self._create_log_entry(
            trial_index=trial_index,
            stage="coherence_scoring",
            question=question,
            concept=concept,
            judge_label=judge_label,
            expected_label=expected_label,
            coherent=coherent,
            notes=f"category={category}"
        )
        self.logs.append(entry)
        return entry
    
    def save(self, filename: str = None):
        """Save all logs to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_log_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.logs, f, indent=2)
        return filepath
    
    def save_jsonl(self, filename: str = None):
        """Save all logs as JSONL (one JSON object per line)."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_log_{timestamp}.jsonl"
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            for entry in self.logs:
                f.write(json.dumps(entry) + '\n')
        return filepath
    
    def get_logs(self) -> List[dict]:
        """Return all logged entries."""
        return self.logs
    
    def print_last(self):
        """Print the last logged entry."""
        if self.logs:
            print(json.dumps(self.logs[-1], indent=2))
