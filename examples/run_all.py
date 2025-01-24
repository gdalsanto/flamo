import os
import argparse
import subprocess

def run_all_examples(print_output=False):
    """
    Runs all Python scripts in the 'examples' folder without displaying figure windows.
    """
    examples_folder = 'examples'
    
    # Check if the examples folder exists
    if not os.path.exists(examples_folder):
        print(f"The folder '{examples_folder}' does not exist.")
        return
    
    # Get all Python files in the examples folder
    example_files = [f for f in os.listdir(examples_folder) if f.endswith('.py')]
    
    # Run each Python file
    for example_file in example_files:
        if example_file == 'run_all.py':
            continue
        file_path = os.path.join(examples_folder, example_file)
        print(f"Running {file_path}...")
        if print_output:
            result = subprocess.run(['python', file_path], capture_output=True, text=True)
        else:
            result = subprocess.run(['python', '-c', f"import os; os.environ['MPLBACKEND'] = 'Agg'; exec(open('{file_path}').read())"], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Error running {file_path}:\n{result.stderr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print_output", type=bool, default=False, help="print output of each example")
    args = parser.parse_args()
    run_all_examples(args.print_output)