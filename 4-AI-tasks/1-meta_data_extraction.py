from dotenv import load_dotenv
import os
import json
import time
from tqdm import tqdm
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pydantic import BaseModel
from langchain_core.runnables import RunnableSequence


# === Load environment variables once at the top ===
load_dotenv()

# === 1. Pydantic Model for Structured Output ===
class RBIMetadata(BaseModel):
    document_type: str = Field(..., description="Type of document (e.g., KYC Guidelines)")
    regulation_area: str = Field(..., description="Regulatory domain or topic area")
    applicable_to: str = Field(..., description="Entities this applies to")
    issued_on: str = Field(default="unknown", description="Date of issuance in YYYY-MM-DD or 'unknown'")
    key_topics: List[str] = Field(..., description="3–5 key topics or compliance requirements")


class BookMetadata(BaseModel):
    title: str = Field(..., description="The title of the book.")
    author: str = Field(..., description="The author(s) of the book.")
    genre: str = Field(..., description="The primary genre of the book (e.g., 'Science Fiction', 'Mystery', 'Biography').")
    publication_year: str = Field(default="unknown", description="The year the book was first published in YYYY format or 'unknown'.")
    key_themes: List[str] = Field(..., description="3–5 key themes or subjects explored in the book.")
    target_audience: str = Field(..., description="The intended audience for the book (e.g., 'Young Adult', 'Adult', 'Children').")



# === 2. Config Class for Global Settings ===
class AppConfig:
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
    # Markdown dir
    ROOT_DIR = "/workspaces/Data_prep/Data/pdf-markdowns/2021"
    OUTPUT_FILE = "/workspaces/Data_prep/Data/meta-data/metadata_2021.json"
    MODEL_NAME = "gemini-2.0-flash"
    REQUESTS_PER_MINUTE = 2000

RBI_Prompt = ChatPromptTemplate.from_messages([
            ("system", """
                You are an AI assistant tasked with extracting structured metadata from an RBI regulatory circular.
                The extracted fields will be used to guide synthetic question-answer pair generation.

                Instructions:
                1. Identify and extract the following fields based on the input:
                - document_type: Type of document (e.g., "KYC Guidelines", "Cybersecurity Framework")
                - regulation_area: Regulatory domain or topic area (e.g., "Anti-Money Laundering", "Digital Payments")
                - applicable_to: Entities this applies to (e.g., "Banks", "NBFCs", "Payment Service Providers")
                - issued_on: Date of issuance in format YYYY-MM-DD (if not present, respond with "unknown")
                - key_topics: 3–5 key topics or compliance requirements mentioned in the document

                2. Output only the JSON object with no additional explanation.
                3. Do not include any markdown formatting.
                4. If a field cannot be determined, mark it as "unknown"
                """),
            ("human", "Use the given format to extract information from the following input: {input}"),
            ("human", "Tip: Make sure to answer in the correct format"),
        ])

Book_Prompt = ChatPromptTemplate.from_messages([
            ("system", """
                You are an AI assistant tasked with extracting structured metadata from a book description.
                The extracted fields will be used to categorize and recommend books.

                Instructions:
                1. Identify and extract the following fields based on the input:
                - title: The title of the book.
                - author: The author(s) of the book.
                - genre: The primary genre of the book (e.g., "Science Fiction", "Mystery", "Biography").
                - publication_year: The year the book was first published (if not present, respond with "unknown").
                - key_themes: 3–5 key themes or subjects explored in the book.
                - target_audience: The intended audience for the book (e.g., "Young Adult", "Adult", "Children").

                2. Output only the JSON object with no additional explanation.
                3. Do not include any markdown formatting.
                4. If a field cannot be determined, mark it as "unknown"
                """),
            ("human", "Use the given format to extract information from the following input: {input}"),
            ("human", "Tip: Make sure to answer in the correct format"),
        ])

# === 3. Metadata Extractor Class ===
class RBIMetadataExtractor:
    def __init__(self):
        self.llm = self._init_llm()
        self.chain = self._build_chain()

    def _init_llm(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=AppConfig.MODEL_NAME,
            temperature=0.4,
            api_key=AppConfig.GOOGLE_API_KEY
        )


    def _build_prompt(self) -> ChatPromptTemplate:
        return RBI_Prompt

    def _build_chain(self) -> RunnableSequence:
        prompt = self._build_prompt()
        return prompt | self.llm.with_structured_output(RBIMetadata)

    def extract(self, content: str) -> Optional[BaseModel]:
        try:
            return self.chain.invoke({"input": content})
        except Exception as e:
            print(f"Error during extraction: {str(e)}")
            return None


# === 4. File Processor Class ===
class MarkdownFileProcessor:
    def __init__(self, extractor: RBIMetadataExtractor):
        self.extractor = extractor
        self.results = {}

    def collect_files(self, root_dir: str) -> List[str]:
        file_paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".md"):
                    file_paths.append(os.path.join(dirpath, filename))
        return file_paths

    def read_markdown_file(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def process_files(self, file_paths: List[str]):
        delay_between_requests = 60 / AppConfig.REQUESTS_PER_MINUTE  # ~4 seconds

        for file_path in tqdm(file_paths, desc="Processing Files", unit="file"):
            filename = os.path.basename(file_path)
            print(f"Processing: {filename}")

            try:
                content = self.read_markdown_file(file_path)
                result = self.extractor.extract(content)

                if result:
                    key = os.path.splitext(filename)[0]
                    self.results[key] = result.model_dump()
                else:
                    print(f"Failed to extract metadata from {filename}")
            except Exception as e:
                print(f"Unexpected error processing {filename}: {str(e)}")

            time.sleep(delay_between_requests)

    def save_results(self, output_file: str):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
        print(f"\nMetadata saved to: {output_file}")


# === 5. Main Execution Runner Class ===
class MetadataRunner:
    def __init__(self):
        self.extractor = RBIMetadataExtractor()
        self.processor = MarkdownFileProcessor(self.extractor)

    def run(self):
        file_paths = self.processor.collect_files(AppConfig.ROOT_DIR)
        print(f"Found {len(file_paths)} Markdown files to process.")
        self.processor.process_files(file_paths)
        self.processor.save_results(AppConfig.OUTPUT_FILE)


# === Run the Application ===
if __name__ == "__main__":
    runner = MetadataRunner()
    runner.run()