import csv
import random
import os

def sample_medicines(input_filepath, output_filepath, sample_size=30):
    try:
        # Read all rows from the CSV
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            rows = list(reader)
        
        # Ensure we don't try to sample more rows than exist
        actual_sample_size = min(sample_size, len(rows))
        sampled_rows = random.sample(rows, actual_sample_size)
        
        # Write the sampled rows to the new CSV
        with open(output_filepath, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(sampled_rows)
            
        print(f"Successfully sampled {actual_sample_size} medicines and saved to '{output_filepath}'.")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{input_filepath}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define file paths
    base_dir = r"c:\medication-recovery"
    input_csv = os.path.join(base_dir, "data", "raw", "updated_indian_medicine_data.csv")
    output_csv = os.path.join(base_dir, "sampled_medicines.csv")
    
    sample_medicines(input_csv, output_csv)
