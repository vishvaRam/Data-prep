import json
import os
from datasets import Dataset
import pandas as pd
from rich import print # For colorful console output

def convert_json_folder_to_huggingface_dataset(json_folder_path: str, dataset_name: str = "combined_qa_dataset"):
    """
    Converts all JSON files in a specified folder into a single Hugging Face Dataset.

    Args:
        json_folder_path (str): The path to the folder containing the input JSON files.
        dataset_name (str): The name for the combined Hugging Face dataset.

    Returns:
        datasets.Dataset: A Hugging Face Dataset object, or None if no files are found or processed.
    """
    if not os.path.isdir(json_folder_path):
        print(f"[red]Error: Folder not found at '{json_folder_path}'[/red]")
        return None

    # List to hold flattened data for all QA pairs from all files
    all_flattened_data = []
    processed_files_count = 0
    skipped_files_count = 0

    print(f"[blue]🔍 Scanning folder: '{json_folder_path}' for JSON files...[/blue]")

    for entry in os.listdir(json_folder_path):
        if entry.endswith(".json"):
            json_file_path = os.path.join(json_folder_path, entry)
            print(f"[blue]Processing file: '{entry}'[/blue]")

            try:
                with open(json_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"[yellow]Warning: Could not decode JSON from '{entry}'. Skipping this file.[/yellow]")
                skipped_files_count += 1
                continue
            except Exception as e:
                print(f"[red]Error reading '{entry}': {e}. Skipping this file.[/red]")
                skipped_files_count += 1
                continue

            if not data:
                print(f"[yellow]Warning: JSON file '{entry}' is empty or contains no data. Skipping.[/yellow]")
                skipped_files_count += 1
                continue

            processed_files_count += 1

            for filename_in_json, file_info in data.items():
                document = file_info.get("document")
                model_name = file_info.get("model_name")
                metadata = file_info.get("metadata", {})
                chunks_text = file_info.get("chunks_text")
                is_table = file_info.get("is_table")
                qa_pairs = file_info.get("qa_pairs", [])

                # Extract specific metadata fields
                regulation_area = metadata.get("regulation_area")
                applicable_to = metadata.get("applicable_to")
                issued_on = metadata.get("issued_on")
                key_topics = metadata.get("key_topics")

                for qa_pair in qa_pairs:
                    question = qa_pair.get("question")
                    answer = qa_pair.get("answer")
                    evaluation_criteria = qa_pair.get("evaluation_criteria")
                    category = qa_pair.get("category")
                    estimated_difficulty = qa_pair.get("estimated_difficulty")

                    all_flattened_data.append({
                        "document": document,
                        "filename": filename_in_json, # This is the original filename key from the JSON
                        "model_name": model_name,
                        "regulation_area": regulation_area,
                        "applicable_to": applicable_to,
                        "issued_on": issued_on,
                        "key_topics": key_topics,
                        "chunks_text": chunks_text,
                        "is_table": is_table,
                        "question": question,
                        "answer": answer,
                        "evaluation_criteria": evaluation_criteria,
                        "category": category,
                        "estimated_difficulty": estimated_difficulty,
                    })
        else:
            print(f"[light_blue]Skipping non-JSON file: '{entry}'[/light_blue]")


    if not all_flattened_data:
        print("[red]No QA pairs found across all processed files to create the dataset.[/red]")
        return None

    # Create a pandas DataFrame first, then convert to Hugging Face Dataset
    df = pd.DataFrame(all_flattened_data)
    hf_dataset = Dataset.from_pandas(df)

    print(f"\n[green]✅ Successfully processed {processed_files_count} JSON files and skipped {skipped_files_count}.[/green]")
    print(f"[green]🎉 Created Hugging Face Dataset '{dataset_name}' with {len(hf_dataset)} rows.[/green]")
    print("[blue]Dataset features (columns):[/blue]", hf_dataset.features.keys())

    return hf_dataset

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Set this to the path of your folder containing the JSON files
    input_json_folder = "/workspaces/Data_prep/Code/Data/QA/"
    output_dataset_name = "RBI_Circular_QA_Dataset"

    # --- Conversion ---
    combined_dataset = convert_json_folder_to_huggingface_dataset(input_json_folder, output_dataset_name)

    if combined_dataset:
        # --- Example Usage (Optional) ---
        print("\n--- First 5 rows of the combined dataset ---")
        print(combined_dataset.to_pandas().head())

        # You can save the combined dataset locally
        # It creates a directory with the dataset's name containing its files
        print(f"\n[bold yellow]Saving combined dataset to local disk...[/bold yellow]")
        combined_dataset.save_to_disk(f"./workspaces/Data_prep/Code/Data/Dataset/{output_dataset_name}")
        print(f"[green]💾 Combined Dataset saved to ./{output_dataset_name}[/green]")

        # To load it back later:
        from datasets import load_from_disk
        loaded_dataset = load_from_disk(f"./workspaces/Data_prep/Code/Data/Dataset/{output_dataset_name}")
        print("\n[bold magenta]Loaded dataset example (first row):[/bold magenta]")
        print(loaded_dataset[0])

        # You can also push it to the Hugging Face Hub (requires authentication)
        # Make sure you're logged in: huggingface-cli login
        try:
            from huggingface_hub import login
            login() # You will be prompted to enter your token if not already logged in
            # Replace "your-username/your-repo-name" with your actual Hugging Face Hub details
            combined_dataset.push_to_hub("Vishva007/RBI-Circular-QA-Dataset",)
            print(f"\n[green]🚀 Dataset pushed to Hugging Face Hub under 'Vishva007/RBI-Circular-QA-Dataset'[/green]")
        except Exception as e:
            print(f"[red]Error pushing to Hugging Face Hub: {e}[/red]")