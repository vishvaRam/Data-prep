import json
import os
from rich import print

def count_qa_pairs_in_json(file_path: str) -> int:
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
        print(f"[red]An unexpected error occurred: {e}[/red]")
        return 0

    return total_qa_pairs

if __name__ == "__main__":
    json_file_path = "/workspaces/Data_prep/Code/Data/QA/generated_qa_pairs_2024_gemini_2.0-flash.json"
    
    print(f"[blue]Counting QA pairs in: {json_file_path}[/blue]")
    num_qa = count_qa_pairs_in_json(json_file_path)

    if num_qa > 0:
        print(f"\n[green]Total number of QA pairs generated: {num_qa}[/green]")
    else:
        print("\n[yellow]No QA pairs found or an error occurred.[/yellow]")