from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
import os
import time
from datetime import datetime
from typing import Optional, List, Tuple, Dict
import pypdf # This is necessary for PDF splitting
import shutil # For cleaning up temporary directories

class PDFToMarkdownConverter:
    """
    A class to handle PDF to Markdown conversion using the marker-pdf library,
    with added functionality for splitting large PDFs to avoid memory issues.
    """

    def __init__(self):
        """Initialize the PDFToMarkdownConverter with a marker-pdf converter."""
        # You can add arguments to PdfConverter here if you have specific model paths
        # or device configurations for marker-pdf. E.g., device='cuda:0' for GPU.
        self.converter = PdfConverter(artifact_dict=create_model_dict())

    def _split_pdf(self, pdf_filepath: str, output_dir: str, pages_per_chunk: int) -> List[str]:
        """
        Splits a large PDF into multiple smaller PDF files.

        Args:
            pdf_filepath (str): The full path to the input PDF file.
            output_dir (str): Directory where split PDF files will be saved.
            pages_per_chunk (int): Maximum number of pages per split PDF chunk.

        Returns:
            List[str]: A list of file paths to the newly created split PDF chunks.
        """
        if not os.path.exists(pdf_filepath):
            raise FileNotFoundError(f"PDF file '{pdf_filepath}' not found for splitting.")

        os.makedirs(output_dir, exist_ok=True)
        split_pdf_paths = []
        
        try:
            pdf_reader = pypdf.PdfReader(pdf_filepath)
            total_pages = len(pdf_reader.pages)
        except Exception as e:
            print(f"Error reading PDF {pdf_filepath} with pypdf for splitting: {e}")
            return [] # Return empty list if PDF cannot be read

        base_filename = os.path.splitext(os.path.basename(pdf_filepath))[0]

        for i in range(0, total_pages, pages_per_chunk):
            pdf_writer = pypdf.PdfWriter()
            end_page = min(i + pages_per_chunk, total_pages)
            
            # Add pages to the writer
            for page_num in range(i, end_page):
                try:
                    pdf_writer.add_page(pdf_reader.pages[page_num])
                except Exception as page_add_e:
                    print(f"Warning: Could not add page {page_num + 1} from {pdf_filepath} to chunk: {page_add_e}")
                    # Decide if you want to skip this chunk or stop
                    continue # Skip this problematic page and try next

            chunk_filepath = os.path.join(output_dir, f"{base_filename}_part{i // pages_per_chunk + 1}.pdf")
            try:
                with open(chunk_filepath, 'wb') as output_pdf:
                    pdf_writer.write(output_pdf)
                split_pdf_paths.append(chunk_filepath)
                print(f"Created chunk: {chunk_filepath} (pages {i+1}-{end_page})")
            except Exception as write_e:
                print(f"Error writing chunk {chunk_filepath}: {write_e}")

        return split_pdf_paths

    def convert_file(self, pdf_filepath: str, output_filepath: str, pages_per_chunk: Optional[int] = None) -> Tuple[bool, float, int]:
        """
        Convert a single PDF file to markdown format and save it.
        If pages_per_chunk is provided and the PDF is large, it will be split and reassembled.

        Args:
            pdf_filepath (str): The full path to the input PDF file.
            output_filepath (str): The full path to the desired output markdown file.
            pages_per_chunk (Optional[int]): If set, PDFs larger than this will be split
                                              and processed in chunks.

        Returns:
            Tuple[bool, float, int]: (success status, total conversion time in seconds, total page count)
        """
        if not pdf_filepath or not output_filepath:
            raise ValueError("Both input and output file paths must be provided")

        if not os.path.exists(pdf_filepath):
            raise FileNotFoundError(f"PDF file '{pdf_filepath}' not found")

        start_time = time.time()
        markdown_text_parts = []
        total_pages_processed = 0
        temp_split_dir = None # Initialize to None for cleanup in except block

        try:
            # First, check the total page count of the original PDF
            try:
                pdf_reader_check = pypdf.PdfReader(pdf_filepath)
                original_pdf_page_count = len(pdf_reader_check.pages)
            except Exception as e:
                print(f"Error reading page count for {pdf_filepath} with pypdf: {e}")
                print("Attempting to process as a single file, page count may be inaccurate or conversion may fail.")
                original_pdf_page_count = 0 # Default to 0, which won't trigger splitting

            # Determine if splitting is needed
            if pages_per_chunk and original_pdf_page_count > pages_per_chunk:
                print(f"PDF '{os.path.basename(pdf_filepath)}' has {original_pdf_page_count} pages. Splitting into chunks of {pages_per_chunk} pages.")
                
                # Create a unique temporary directory for this PDF's chunks
                temp_split_dir = os.path.join(os.path.dirname(output_filepath), "temp_pdf_chunks", os.path.basename(os.path.splitext(pdf_filepath)[0]))
                
                # Clean up previous temporary directory for this specific PDF if it exists
                if os.path.exists(temp_split_dir):
                    shutil.rmtree(temp_split_dir)
                os.makedirs(temp_split_dir, exist_ok=True)

                split_pdf_paths = self._split_pdf(pdf_filepath, temp_split_dir, pages_per_chunk)
                
                if not split_pdf_paths: # If splitting failed or resulted in no chunks
                    print(f"Failed to split PDF {pdf_filepath}. Attempting to process as a single file.")
                    # Fallback to single file processing
                    rendered = self.converter(pdf_filepath)
                    total_pages_processed = len(rendered.pages) if hasattr(rendered, 'pages') else 0
                    markdown_text, _, _ = text_from_rendered(rendered)
                    markdown_text_parts.append(markdown_text)
                else:
                    for i, chunk_path in enumerate(split_pdf_paths):
                        print(f"Processing chunk {i+1}/{len(split_pdf_paths)}: {os.path.basename(chunk_path)}")
                        try:
                            rendered = self.converter(chunk_path)
                            # Get page count from 'rendered' object for the chunk
                            chunk_page_count = len(rendered.pages) if hasattr(rendered, 'pages') else 0
                            total_pages_processed += chunk_page_count
                            chunk_markdown_text, _, _ = text_from_rendered(rendered)
                            markdown_text_parts.append(chunk_markdown_text)
                        except Exception as chunk_e:
                            print(f"Error processing chunk {chunk_path}: {chunk_e}")
                            # Decide how to handle a failed chunk: skip, re-raise, or log
                            # For robustness, we'll continue but acknowledge the issue
                            markdown_text_parts.append(f"\n\n\n\n")

            else:
                # Process the PDF as a single file if no splitting is needed or if it's smaller than the chunk size
                print(f"Processing PDF '{os.path.basename(pdf_filepath)}' as a single file ({original_pdf_page_count} pages).")
                rendered = self.converter(pdf_filepath)
                # Get total page count from 'rendered' object for the whole PDF
                total_pages_processed = len(rendered.pages) if hasattr(rendered, 'pages') else 0
                markdown_text, _, _ = text_from_rendered(rendered)
                markdown_text_parts.append(markdown_text)

            # Join all markdown parts
            final_markdown_text = "\n\n".join(markdown_text_parts)

            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(final_markdown_text)

            conversion_time = time.time() - start_time
            print(f"Finished processing {os.path.basename(pdf_filepath)}. Time: {conversion_time:.2f}s, Pages: {total_pages_processed}")
            
            return True, conversion_time, total_pages_processed

        except Exception as e:
            print(f"Error converting {pdf_filepath}: {e}")
            return False, time.time() - start_time, 0
        finally:
            # Ensure cleanup of temporary split files happens regardless of success or failure
            if temp_split_dir and os.path.exists(temp_split_dir):
                try:
                    shutil.rmtree(temp_split_dir)
                    print(f"Cleaned up temporary directory: {temp_split_dir}")
                except Exception as cleanup_e:
                    print(f"Error cleaning up temporary directory {temp_split_dir}: {cleanup_e}")

    def batch_convert_directory(self, input_dir: str, output_dir: str, pages_per_chunk: Optional[int] = None) -> Dict:
        """
        Convert all PDF files in the specified directory to markdown files.

        Args:
            input_dir (str): Directory containing PDF files.
            output_dir (str): Directory where markdown files will be saved.
            pages_per_chunk (Optional[int]): If set, PDFs larger than this will be split
                                              and processed in chunks during conversion.

        Returns:
            Dict: Conversion statistics.
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory '{input_dir}' not found")

        os.makedirs(output_dir, exist_ok=True)
        pdf_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]

        if not pdf_files:
            print(f"No PDF files found in '{input_dir}'")
            return {}

        successful = 0
        total_pages = 0
        total_time = 0

        for pdf_file in pdf_files:
            output_filename = os.path.splitext(os.path.basename(pdf_file))[0] + '.md'
            output_path = os.path.join(output_dir, output_filename)

            # Pass the pages_per_chunk parameter to the convert_file method
            success, conv_time, pages = self.convert_file(pdf_file, output_path, pages_per_chunk=pages_per_chunk)
            total_time += conv_time

            if success:
                successful += 1
                total_pages += pages

        return {
            "total": len(pdf_files),
            "successful": successful,
            "failed": len(pdf_files) - successful,
            "pages": total_pages,
            "time": total_time,
            "speed": total_pages / total_time if total_time > 0 else 0
        }

def main():
    # Adjust paths based on your environment.
    # Using absolute paths or paths relative to the script's execution can be safer.
    input_directory = os.path.abspath("../Data/Raw-pdfs/Book/now")
    output_directory = os.path.abspath("../Data/pdf-markdowns/Book")
    
    # Define the number of pages per chunk. Adjust this value based on your GPU memory constraints.
    # For example, if you find that PDFs with more than 50 pages cause OOM errors, set this to 50.
    # A smaller number (like your 5) is good for testing the splitting logic.
    pages_to_split_into = 5 

    print(f"\nStarting PDF to Markdown conversion at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    converter = PDFToMarkdownConverter()
    # Pass the pages_to_split_into as the pages_per_chunk parameter
    stats = converter.batch_convert_directory(input_directory, output_directory, pages_per_chunk=pages_to_split_into)

    print("\nConversion Summary:")
    print("-" * 40)
    print(f"Total PDFs processed: {stats.get('total', 0)}")
    print(f"Successfully converted: {stats.get('successful', 0)}")
    print(f"Failed conversions: {stats.get('failed', 0)}")
    print(f"Total pages processed: {stats.get('pages', 0)}")
    print(f"Total processing time: {stats.get('time', 0):.2f} seconds")
    print(f"Average speed: {stats.get('speed', 0):.2f} pages/second")
    print("-" * 40)
    print(f"Conversion completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()