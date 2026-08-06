import os
import sys
import json
import logging
from typing import Optional
import pandas as pd
from tqdm import tqdm

# Add the src directory to sys.path so modules can be imported correctly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.data_loader import load_metadata, load_medicine_dataset
from baseline.string_match import StringMatcher

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Main function to run the baseline predictions over the metadata transcripts.
    """
    metadata_path = os.path.join("data", "transcripts", "metadata.csv")
    medicine_path = os.path.join("data", "raw", "updated_indian_medicine_data.csv")
    results_dir = "results"
    output_path = os.path.join(results_dir, "baseline_predictions.csv")

    # Create the results directory if it does not exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.info(f"Created directory: {results_dir}")

    # Load data
    logger.info("Loading medicine dataset...")
    medicines_df = load_medicine_dataset(medicine_path)
    
    logger.info("Loading metadata...")
    metadata_df = load_metadata(metadata_path)

    logger.info("Initializing StringMatcher...")
    matcher = StringMatcher(
        medicines=medicines_df,
        corpus=metadata_df['stt_transcript'].dropna().astype(str).tolist(),
    )
    
    # Initialize columns explicitly with correct types using assignment
    if 'similarity_score' not in metadata_df.columns:
        metadata_df = metadata_df.assign(similarity_score=pd.Series(dtype='float64'))
    
    if 'phrase_length' not in metadata_df.columns:
        metadata_df = metadata_df.assign(phrase_length=pd.Series(dtype='Int64'))
        
    # Force object type to avoid LossySetitemError when assigning strings
    for col in ['recovered_medicine', 'matched_phrase', 'top_k_predictions']:
        if col not in metadata_df.columns:
            metadata_df = metadata_df.assign(**{col: pd.Series(dtype='object')})
        else:
            metadata_df[col] = metadata_df[col].astype(object)
        
    total_processed = 0
    successful_predictions = 0
    skipped_samples = 0
    
    logger.info("Starting baseline predictions...")
    
    # Iterate through every row in the metadata
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Predicting medicines"):
        transcript = row.get('stt_transcript')
        
        # Skip rows with empty transcripts
        if pd.isna(transcript) or not str(transcript).strip():
            skipped_samples += 1
            continue
            
        total_processed += 1
        
        # Call the string matcher to predict the top k medicines
        top_k_predictions = matcher.find_best_match(str(transcript))
        
        # Store predictions
        if top_k_predictions:
            best_match = top_k_predictions[0]
            metadata_df.at[idx, 'recovered_medicine'] = best_match["medicine"]
            metadata_df.at[idx, 'similarity_score'] = best_match["score"]
            metadata_df.at[idx, 'matched_phrase'] = best_match["candidate_phrase"]
            metadata_df.at[idx, 'phrase_length'] = best_match["phrase_length"]
            metadata_df.at[idx, 'top_k_predictions'] = json.dumps(top_k_predictions)
            successful_predictions += 1
            
    # Save the results without overwriting the original metadata.csv
    logger.info(f"Saving predictions to {output_path}...")
    metadata_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "="*45)
    print("Baseline Prediction Summary")
    print("="*45)
    print(f"Total samples processed: {total_processed}")
    print(f"Successful predictions:  {successful_predictions}")
    print(f"Skipped samples:         {skipped_samples}")
    print(f"Output file location:    {output_path}")
    print("="*45 + "\n")

if __name__ == "__main__":
    main()
