import os
import re
import sys
import json
import logging
import pandas as pd
from typing import List, Dict, Any

# Ensure we can import from utils when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import clean_medicine_name

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates medication recovery models by comparing ground-truth medicines 
    with their predicted recovered equivalents, supporting Top-K analysis.
    """

    def __init__(
        self,
        predictions_path: str = os.path.join("results", "baseline_predictions.csv"),
        errors_path: str = os.path.join("results", "error_analysis.csv")
    ) -> None:
        """
        Initializes the Evaluator.

        Args:
            predictions_path (str): Path to the predictions CSV.
            errors_path (str): Path where the error analysis CSV will be saved.
        """
        self.predictions_path = predictions_path
        self.errors_path = errors_path

    @staticmethod
    def is_match(ground_truth_str: str, predicted_name: str) -> bool:
        """
        Checks if the predicted name matches any of the ground truth medicines.
        Handles multiple ground truths separated by ' / '.
        """
        if not predicted_name or pd.isna(predicted_name):
            return False

        # Split ground truth on a SPACE-padded slash only. A bare '/' is part of a
        # combination dosage ("Lancas D 10mg/30mg Capsule") and splitting on it
        # shreds the name into fragments that can never match any catalog entry.
        gt_medicines = [
            clean_medicine_name(m.strip())
            for m in re.split(r'\s+/\s+', str(ground_truth_str))
            if m.strip()
        ]
        
        pred_cleaned = clean_medicine_name(str(predicted_name))
        
        return pred_cleaned in gt_medicines and bool(pred_cleaned)

    def evaluate(self) -> None:
        """
        Runs the full evaluation pipeline, computing Top-K metrics and language 
        breakdowns, and exports the error analysis.
        """
        if not os.path.exists(self.predictions_path):
            logger.error(f"Could not find predictions file at {self.predictions_path}")
            return

        df = pd.read_csv(self.predictions_path)
        total_samples = len(df)
        
        if total_samples == 0:
            logger.error("No samples found to evaluate.")
            return

        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        
        # Track per-language top-1 accuracy
        lang_stats: Dict[str, Dict[str, int]] = {}
        incorrect_rows: List[pd.Series] = []
        
        for _, row in df.iterrows():
            gt_str = str(row.get('medicine_name', ''))
            top_k_json = row.get('top_k_predictions', '[]')
            lang = str(row.get('language', 'unknown'))
            
            if lang not in lang_stats:
                lang_stats[lang] = {'total': 0, 'correct': 0}
                
            lang_stats[lang]['total'] += 1
            
            try:
                top_k_list = json.loads(top_k_json) if pd.notna(top_k_json) else []
            except (json.JSONDecodeError, TypeError):
                top_k_list = []
                
            match_top1, match_top3, match_top5 = False, False, False
            
            for k_idx, pred_dict in enumerate(top_k_list):
                if not isinstance(pred_dict, dict):
                    continue
                    
                pred_med = pred_dict.get("medicine", "")
                if self.is_match(gt_str, pred_med):
                    if k_idx == 0:
                        match_top1 = True
                    if k_idx < 3:
                        match_top3 = True
                    if k_idx < 5:
                        match_top5 = True
                        
            if match_top1:
                top1_correct += 1
                lang_stats[lang]['correct'] += 1
            else:
                incorrect_rows.append(row)
                
            if match_top3:
                top3_correct += 1
                
            if match_top5:
                top5_correct += 1
                
        self._print_summary(
            total_samples=total_samples,
            top1_correct=top1_correct,
            top3_correct=top3_correct,
            top5_correct=top5_correct,
            lang_stats=lang_stats
        )
        
        self._save_error_analysis(incorrect_rows)

    def _print_summary(
        self, 
        total_samples: int, 
        top1_correct: int, 
        top3_correct: int, 
        top5_correct: int, 
        lang_stats: Dict[str, Dict[str, int]]
    ) -> None:
        """
        Prints the evaluation summary to the console.
        """
        top1_acc = (top1_correct / total_samples) * 100 if total_samples else 0.0
        top3_acc = (top3_correct / total_samples) * 100 if total_samples else 0.0
        top5_acc = (top5_correct / total_samples) * 100 if total_samples else 0.0
        
        en_acc = (lang_stats.get('en', {}).get('correct', 0) / max(1, lang_stats.get('en', {}).get('total', 1))) * 100
        hi_acc = (lang_stats.get('hi', {}).get('correct', 0) / max(1, lang_stats.get('hi', {}).get('total', 1))) * 100
        kn_acc = (lang_stats.get('kn', {}).get('correct', 0) / max(1, lang_stats.get('kn', {}).get('total', 1))) * 100
        
        print("\nBaseline Evaluation\n")
        print(f"Overall Top-1 Accuracy : {top1_acc:.2f}%")
        print(f"Overall Top-3 Accuracy : {top3_acc:.2f}%")
        print(f"Overall Top-5 Accuracy : {top5_acc:.2f}%\n")
        print(f"English Accuracy : {en_acc:.2f}%")
        print(f"Hindi Accuracy   : {hi_acc:.2f}%")
        print(f"Kannada Accuracy : {kn_acc:.2f}%\n")
        print(f"Total Samples : {total_samples}")
        print(f"Correct       : {top1_correct}")
        print(f"Incorrect     : {total_samples - top1_correct}\n")

    def _save_error_analysis(self, incorrect_rows: List[pd.Series]) -> None:
        """
        Saves the incorrect predictions to a CSV file.
        """
        if not incorrect_rows:
            logger.info("No incorrect predictions to save.")
            return
            
        error_df = pd.DataFrame(incorrect_rows)
        
        req_cols = [
            'sample_id', 'language', 'medicine_name', 'recovered_medicine',
            'similarity_score', 'matched_phrase', 'phrase_length', 'top_k_predictions'
        ]
        
        # Only keep columns that actually exist
        cols_to_keep = [c for c in req_cols if c in error_df.columns]
        error_df = error_df[cols_to_keep]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.errors_path), exist_ok=True)
        
        error_df.to_csv(self.errors_path, index=False)
        logger.info(f"Saved {len(error_df)} incorrect predictions to {self.errors_path}")


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate()
