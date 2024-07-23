import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Converts an Excel file to a CSV file.")
    parser.add_argument("input_file", help="Path to the input Excel file.")
    parser.add_argument("-s", "--sheet", default=0,
                        help="Sheet name or index to convert (default is the first sheet).")
    return parser.parse_args()

def convert_xlsx_to_csv(input_file, sheet):
    # Determine the output file name by replacing the extension
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}.csv"
    
    # Load the Excel file using the specified sheet
    df = pd.read_excel(input_file, sheet_name=sheet, engine='openpyxl')
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"File '{input_file}' has been converted to '{output_file}'.")

if __name__ == "__main__":
    args = parse_args()
    convert_xlsx_to_csv(args.input_file, args.sheet)
