import json
import os
from rich import print

def count_processed_files(json_file_path: str):
    """
    Counts the number of unique files processed and stored in the generated QA pairs JSON.
    Also provides a breakdown of QA pairs per file.

    Args:
        json_file_path: The path to the JSON file containing the generated QA pairs.
    """
    if not os.path.exists(json_file_path):
        print(f"[red]Error: File not found at '{json_file_path}'[/red]")
        return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            num_files = len(data)
            print(f"\n[green]📊 Successfully processed {num_files} unique files.[/green]")
            print("[blue]Individual file breakdown:[/blue]")
            # for filename, file_data in data.items():
            #     qa_count = len(file_data.get("qa_pairs", []))
            #     print(f"  - [cyan]{filename}[/cyan]: [yellow]{qa_count}[/yellow] QA pairs generated.")
        else:
            print(f"[red]Error: The JSON file does not contain a top-level dictionary. Expected a dictionary where keys are filenames.[/red]")

    except json.JSONDecodeError:
        print(f"[red]Error: Could not decode JSON from '{json_file_path}'. The file might be corrupted or empty.[/red]")
    except Exception as e:
        print(f"[red]An unexpected error occurred while counting processed files: {e}[/red]")

def count_total_qa_pairs_in_json(file_path: str) -> int:
    """
    Counts the total number of QA pairs across all files in a generated QA JSON.

    Args:
        file_path: The path to the JSON file containing the generated QA pairs.

    Returns:
        The total number of QA pairs found in the JSON file.
    """
    if not os.path.exists(file_path):
        print(f"[red]Error: File not found at '{file_path}'[/red]")
        return 0

    total_qa_pairs = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print("[red]Error: Expected JSON content to be a dictionary.[/red]")
            return 0

        for filename, file_data in data.items():
            if "qa_pairs" in file_data and isinstance(file_data["qa_pairs"], list):
                total_qa_pairs += len(file_data["qa_pairs"])
            else:
                print(f"[yellow]Warning: 'qa_pairs' key not found or not a list for file '{filename}'. Skipping.[/yellow]")

    except json.JSONDecodeError as e:
        print(f"[red]Error decoding JSON from '{file_path}': {e}[/red]")
        return 0
    except Exception as e:
        print(f"[red]An unexpected error occurred while counting total QA pairs: {e}[/red]")
        return 0

    return total_qa_pairs

if __name__ == "__main__":
    json_file_path = "/workspaces/Data_prep/Code/Data/QA/generated_qa_pairs_2025.json"
    
    print(f"[bold magenta]🚀 Starting QA Analysis Tool[/bold magenta]")
    
    # Analyze processed files
    count_processed_files(json_file_path)

    # Analyze total QA pairs
    print(f"\n[blue]Counting total QA pairs in: {json_file_path}[/blue]")
    num_qa = count_total_qa_pairs_in_json(json_file_path)

    if num_qa > 0:
        print(f"\n[green]🎉 Total number of QA pairs generated across all files: {num_qa}[/green]")
    else:
        print("\n[yellow]No QA pairs found or an error occurred during counting.[/yellow]")

    print(f"\n[bold magenta]✅ QA Analysis Complete[/bold magenta]")
    
    
# 📊 Successfully processed 148 unique files.
# Individual file breakdown:

# Counting total QA pairs in: /workspaces/Data_prep/Code/Data/QA/generated_qa_pairs_2023.json

# 🎉 Total number of QA pairs generated across all files: 2068


# Counting total QA pairs in: /workspaces/Data_prep/Code/Data/QA/generated_qa_pairs_2024.json

# 🎉 Total number of QA pairs generated across all files: 4008

# Counting total QA pairs in: /workspaces/Data_prep/Code/Data/QA/generated_qa_pairs_2025.json

# 🎉 Total number of QA pairs generated across all files: 6622




