from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
import os
import time
from datetime import datetime
from typing import Optional, List, Tuple, Dict

class PDFToMarkdownConverter:
    """
    A class to handle PDF to Markdown conversion using the marker-pdf library.
    """

    def __init__(self):
        """Initialize the PDFToMarkdownConverter with a marker-pdf converter."""
        self.converter = PdfConverter(artifact_dict=create_model_dict())

    def convert_file(self, pdf_filepath: str, output_filepath: str) -> Tuple[bool, float, int]:
        """
        Convert a single PDF file to markdown format and save it.

        Args:
            pdf_filepath (str): The full path to the input PDF file
            output_filepath (str): The full path to the desired output markdown file

        Returns:
            Tuple[bool, float, int]: (success status, conversion time in seconds, page count)
        """
        if not pdf_filepath or not output_filepath:
            raise ValueError("Both input and output file paths must be provided")

        if not os.path.exists(pdf_filepath):
            raise FileNotFoundError(f"PDF file '{pdf_filepath}' not found")

        start_time = time.time()

        try:
            rendered = self.converter(pdf_filepath)
            page_count = len(rendered.pages) if hasattr(rendered, 'pages') else 1
            markdown_text, _, _ = text_from_rendered(rendered)

            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_text)

            conversion_time = time.time() - start_time
            return True, conversion_time, page_count

        except Exception as e:
            print(f"Error converting {pdf_filepath}: {e}")
            return False, time.time() - start_time, 0

    def batch_convert_directory(self, input_dir: str, output_dir: str) -> Dict:
        """
        Convert all PDF files in the specified directory to markdown files.

        Args:
            input_dir (str): Directory containing PDF files
            output_dir (str): Directory where markdown files will be saved

        Returns:
            Dict: Conversion statistics
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

            success, conv_time, pages = self.convert_file(pdf_file, output_path)
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
    input_directory = "/workspaces/Data_prep/Code/Data/Raw-pdfs/2023"
    output_directory = "/workspaces/Data_prep/Code/Data/pdf-markdowns/2023"

    print(f"\nStarting PDF to Markdown conversion at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    converter = PDFToMarkdownConverter()
    stats = converter.batch_convert_directory(input_directory, output_directory)

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