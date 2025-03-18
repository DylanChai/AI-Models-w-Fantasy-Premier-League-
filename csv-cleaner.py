import pandas as pd
import os

def clean_csv_file(input_file_path, output_file_path=None):
    """
    Clean a problematic CSV file and save the cleaned version.
    
    Args:
        input_file_path (str): Path to the input CSV file
        output_file_path (str, optional): Path to save the cleaned CSV file. 
                                        If None, appends '_cleaned' to original filename.
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Starting to clean file: {input_file_path}")
    
    if not os.path.exists(input_file_path):
        print(f"Error: File not found - {input_file_path}")
        return False
    
    if output_file_path is None:
        # Create a default output path by appending '_cleaned' to the original filename
        filename, ext = os.path.splitext(input_file_path)
        output_file_path = f"{filename}_cleaned{ext}"
    
    try:
        # First, try to read the file with the Python parser
        # This is slower but more flexible for problematic files
        data = pd.read_csv(
            input_file_path,
            engine='python',  # Use the more flexible Python parser
            on_bad_lines='skip',  # Skip problematic rows
            encoding='utf-8',  # Explicitly specify encoding
            low_memory=False   # Avoid dtype guessing on chunks
        )
        
        print(f"Successfully read file with {data.shape[0]} rows and {data.shape[1]} columns")
        print(f"Found columns: {data.columns.tolist()}")
        
        # Count rows before cleaning
        original_row_count = data.shape[0]
        
        # Clean the data
        # 1. Remove duplicate rows
        data = data.drop_duplicates()
        print(f"Removed {original_row_count - data.shape[0]} duplicate rows")
        
        # 2. Handle any special characters in column names
        data.columns = [col.strip().replace('\n', '').replace('\r', '') for col in data.columns]
        
        # 3. Save the cleaned data
        data.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f"Cleaned data saved to: {output_file_path}")
        
        return True
        
    except Exception as e:
        print(f"Error cleaning file: {e}")
        
        # If the above fails, try a more aggressive approach
        try:
            print("Attempting more aggressive cleaning...")
            
            # Read the file as text and manually process it
            with open(input_file_path, 'r', encoding='utf-8', errors='replace') as file:
                lines = file.readlines()
            
            # Identify the header line (first line)
            header = lines[0].strip()
            
            # Count expected columns from header
            expected_columns = header.count(',') + 1
            print(f"Header suggests {expected_columns} columns")
            
            # Process each line to ensure the right number of fields
            cleaned_lines = [header]
            skipped_lines = 0
            
            for i, line in enumerate(lines[1:], 1):
                if i % 10000 == 0:
                    print(f"Processing line {i}...")
                
                # Simple heuristic: if the line has roughly the right number of commas, keep it
                fields = line.count(',')
                if abs(fields - (expected_columns - 1)) <= 1:  # Allow small variation
                    cleaned_lines.append(line.strip())
                else:
                    skipped_lines += 1
            
            print(f"Skipped {skipped_lines} problematic lines")
            
            # Write cleaned content to a temporary file
            temp_file = output_file_path + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as file:
                file.write('\n'.join(cleaned_lines))
            
            # Try to read with pandas again
            data = pd.read_csv(temp_file, low_memory=False)
            
            # Save the properly formatted CSV
            data.to_csv(output_file_path, index=False, encoding='utf-8')
            
            # Clean up the temporary file
            os.remove(temp_file)
            
            print(f"Aggressive cleaning completed. Saved to: {output_file_path}")
            return True
            
        except Exception as e:
            print(f"Failed to clean even with aggressive approach: {e}")
            return False

def main():
    # Clean the merged gameweek file
    merged_gw_path = "data/2024-25/gws/merged_gw.csv"
    
    if not os.path.exists(merged_gw_path):
        print(f"Error: File not found - {merged_gw_path}")
        return
    
    # Clean the file
    success = clean_csv_file(merged_gw_path, "data/2024-25/gws/merged_gw_cleaned.csv")
    
    if success:
        print("\nCleaning process completed successfully!")
        print("You can now update your scripts to use the cleaned CSV file.")
        print("\nExample changes for your scripts:")
        print("  From: data_path = \"data/2024-25/gws/merged_gw.csv\"")
        print("  To:   data_path = \"data/2024-25/gws/merged_gw_cleaned.csv\"")
    else:
        print("\nFailed to clean the CSV file.")
        print("Please consider manually examining and fixing the problematic rows.")

if __name__ == "__main__":
    main()