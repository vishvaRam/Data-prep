import os


class MarkdownProcessor:
    """
    Class responsible for processing a single markdown file,
    splitting text into chunks and extracting tables.
    """

    def __init__(self, input_file, output_folder, chunk_size=5000, min_chunk_size=500):
        self.input_file = input_file
        self.output_folder = output_folder
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.base_name = os.path.splitext(os.path.basename(input_file))[0]

    def is_table_line(self, line):
        """Check if the line contains a table marker."""
        return '|' in line

    def process(self, skip_tables=False, skip_text=False):
        """Process the markdown file: extract tables and split text content."""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        table_content = []
        text_content = []
        in_table = False

        for line in lines:
            if not skip_tables and self.is_table_line(line):
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

        # Save extracted tables
        if not skip_tables and table_content:
            self._write_table_content(table_content)

        # Save text chunks
        if not skip_text:
            self._write_text_chunks(text_content)

        print(f"✅ Processed: {self.input_file}")

    def _write_table_content(self, table_lines):
        """Write table content to a separate file."""
        table_path = os.path.join(self.output_folder, f"{self.base_name}_tables.txt")
        with open(table_path, 'w', encoding='utf-8') as f:
            f.writelines(table_lines)

    def _write_text_chunks(self, text_lines):
        """Chunk and write text content."""
        total_text = ''.join(text_lines)
        if not total_text.strip():
            return

        chunks = [total_text[i:i + self.chunk_size] for i in range(0, len(total_text), self.chunk_size)]

        if len(chunks) > 1 and len(chunks[-1]) < self.min_chunk_size:
            chunks[-2] += chunks[-1]
            chunks.pop()

        for idx, chunk in enumerate(chunks, start=1):
            part_path = os.path.join(self.output_folder, f"{self.base_name}_text_part{idx}.txt")
            with open(part_path, 'w', encoding='utf-8') as f:
                f.write(chunk)


class MarkdownSplitter:
    """
    Orchestrates recursive processing of all .md files in a root folder.
    """

    def __init__(self, root_folder, output_base_folder, chunk_size=5000, min_chunk_size=500):
        self.root_folder = root_folder
        self.output_base_folder = output_base_folder
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size

    def process_all(self, skip_tables=False, skip_text=False):
        """Recursively process all markdown files."""
        os.makedirs(self.output_base_folder, exist_ok=True)

        for dirpath, _, filenames in os.walk(self.root_folder):
            for filename in filenames:
                if filename.endswith(".md"):
                    input_path = os.path.join(dirpath, filename)
                    rel_dir = os.path.relpath(dirpath, self.root_folder)
                    output_subfolder = os.path.join(self.output_base_folder, rel_dir)
                    os.makedirs(output_subfolder, exist_ok=True)

                    processor = MarkdownProcessor(
                        input_file=input_path,
                        output_folder=output_subfolder,
                        chunk_size=self.chunk_size,
                        min_chunk_size=self.min_chunk_size
                    )
                    processor.process(skip_tables=skip_tables, skip_text=skip_text)

        print("🏁 All files processed successfully.")


# -----------------------------
# Run normally here
# -----------------------------
if __name__ == '__main__':
    # Set your paths here manually
    root_folder = r"/workspaces/Data_prep/Data/pdf-markdowns/2021"
    output_folder = r"/workspaces/Data_prep/Data/Chunks/2021"

    # Optional settings
    default_chunk_size = 5000
    default_min_chunk_size = 800
    skip_tables = False  # Set to True to skip extracting tables
    skip_text = False    # Set to True to skip text extraction

    print("🚀 Starting Markdown Splitter...")
    splitter = MarkdownSplitter(
        root_folder=root_folder,
        output_base_folder=output_folder,
        chunk_size=default_chunk_size,
        min_chunk_size=default_min_chunk_size
    )
    splitter.process_all(skip_tables=skip_tables, skip_text=skip_text)