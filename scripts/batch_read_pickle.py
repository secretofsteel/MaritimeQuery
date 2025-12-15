import sys
from pathlib import Path

# 1. Setup path to allow importing from the 'scripts' directory
# This assumes 'scripts' is a subdirectory of the current script's location
#urrent_dir = Path(__file__).parent
#ys.path.append(str(current_dir / "scripts"))

# 2. Import the function
# Replace 'cache_utils' with the actual name of the python file inside /scripts
from read_pickle_cache import read_pickle_cache 

def process_directory(directory_path: Path):
    """
    Iterates through all files in the given directory and runs the cache function.
    """
    # Ensure the directory exists
    if not directory_path.exists():
        print(f"Error: Directory {directory_path} not found.")
        return

    # 3. Iterate through files
    for file_path in directory_path.iterdir():
        if file_path.is_file():
            # Create output filename: original_name_no_ext.txt
            output_file = f"{file_path.stem}.txt"
            
            # Optional: If you want the full output path (not just name), use this:
            # output_path = file_path.parent / output_filename
            
            print(f"Processing: {file_path.name} -> {output_file}")
            
            # 4. Run the function
            # Passing full input path (Path object) and new output filename (str)
            read_pickle_cache(file_path, output_file)

if __name__ == "__main__":
    # Define your data directory here
    target_dir = Path("./data/cache/text_extracts") 
    process_directory(target_dir)