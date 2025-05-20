import os


def is_table_line(line):
    """Check if the line contains a table marker."""
    return '|' in line


def split_markdown_file(input_file, output_folder, chunk_size=5000, min_chunk_size=500,
                        skip_tables=False, skip_text=False):
    """
    Process a single Markdown file:
    - Extract tables to a .txt file (if any)
    - Chunk non-table content into parts of `chunk_size` characters
      and merge last chunk if smaller than `min_chunk_size`

    Args:
        input_file (str): Path to the input markdown file
        output_folder (str): Folder to write output files
        chunk_size (int): Max number of characters per text chunk
        min_chunk_size (int): Minimum acceptable character count for final chunk
        skip_tables (bool): Skip extracting tables
        skip_text (bool): Skip chunking and writing text content
    """
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    table_content = []
    text_content = []
    in_table = False

    for line in lines:
        if not skip_tables and is_table_line(line):
            if not in_table and text_content and text_content[-1] != '\n':
                text_content.append('\n')
            table_content.append(line)
            in_table = True
        else:
            if in_table and not skip_tables:
                if table_content[-1] != '\n':
                    table_content.append('\n')
                in_table = False
            text_content.append(line)

    # Write table content if enabled and present
    if not skip_tables and table_content:
        table_output_path = os.path.join(output_folder, f"{base_name}_tables.txt")
        with open(table_output_path, 'w', encoding='utf-8') as f:
            f.writelines(table_content)

    # Write text chunks if enabled and content exists
    if not skip_text:
        total_text = ''.join(text_content)
        if total_text.strip():
            # Generate initial list of chunks
            chunks = [total_text[i:i + chunk_size] for i in range(0, len(total_text), chunk_size)]

            # Merge small last chunk with previous one if needed
            if len(chunks) > 1 and len(chunks[-1]) < min_chunk_size:
                chunks[-2] += chunks[-1]
                chunks.pop()

            # Write each chunk to file
            for idx, chunk in enumerate(chunks, start=1):
                text_output_path = os.path.join(output_folder, f"{base_name}_text_part{idx}.txt")
                with open(text_output_path, 'w', encoding='utf-8') as f:
                    f.write(chunk)

    print(f"✅ Processed: {input_file}")


def process_all_markdowns(root_folder, output_base_folder, chunk_size=5000, min_chunk_size=500,
                          skip_tables=False, skip_text=False):
    """
    Recursively process all .md files in root_folder.

    Args:
        root_folder (str): Root directory containing markdown files
        output_base_folder (str): Base folder to store output files
        chunk_size (int): Max number of characters per text chunk
        min_chunk_size (int): Minimum acceptable character count for final chunk
        skip_tables (bool): Skip extracting tables
        skip_text (bool): Skip chunking and writing text content
    """
    os.makedirs(output_base_folder, exist_ok=True)

    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(".md"):
                input_path = os.path.join(dirpath, file)
                rel_dir = os.path.relpath(dirpath, root_folder)
                output_subfolder = os.path.join(output_base_folder, rel_dir)
                os.makedirs(output_subfolder, exist_ok=True)
                split_markdown_file(
                    input_path,
                    output_subfolder,
                    chunk_size=chunk_size,
                    min_chunk_size=min_chunk_size,
                    skip_tables=skip_tables,
                    skip_text=skip_text
                )


# -----------------------------
# Run normally here
# -----------------------------
if __name__ == '__main__':
    # Set your paths here manually
    root_folder = r"/workspaces/Data_prep/Code/Data/pdf-markdowns/2025"
    output_folder = r"/workspaces/Data_prep/Code/Data/Chunks/2025"

    # Optional settings
    default_chunk_size = 5000
    default_min_chunk_size = 800
    skip_tables = False  # Set to True to skip extracting tables
    skip_text = False    # Set to True to skip text extraction

    print("🚀 Starting Markdown Splitter...")
    process_all_markdowns(
        root_folder=root_folder,
        output_base_folder=output_folder,
        chunk_size=default_chunk_size,
        min_chunk_size=default_min_chunk_size,
        skip_tables=skip_tables,
        skip_text=skip_text
    )
    print("🏁 All files processed successfully.")