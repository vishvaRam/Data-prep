from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from rich import print
import json
import re
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
import time
from datetime import datetime
from google.api_core.exceptions import ResourceExhausted, InternalServerError, GoogleAPIError # Import specific exceptions, including a more general GoogleAPIError
from langfuse.callback import CallbackHandler

# Load environment variables
load_dotenv()

langfuse_handler = CallbackHandler()

# Tests the SDK connection with the server
langfuse_handler.auth_check()


# Model Definitions
class QAPair(BaseModel):
    question: str = Field(..., description="The question being asked")
    answer: str = Field(..., description="The answer to the question")
    evaluation_criteria: str = Field(..., description="Criteria to evaluate the answer")
    category: Literal['reasoning', 'fact-based'] = Field(..., description="Category of the question")
    estimated_difficulty: int = Field(..., description="Estimated difficulty level")


class QAPairList(BaseModel):
    items: List[QAPair] = Field(..., description="List of Questions and answers")


class QAGenerator:
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.2,
        requests_per_minute: int = 15,
        max_iterations: int = 10,
        max_consecutive_duplicates: int = 3,  # New parameter
        api_key: Optional[str] = None
    ):
        """
        Initialize the QA Generator with configurable settings.

        Args:
            model_name: Name of the LLM to use.
            temperature: Sampling temperature for response diversity.
            requests_per_minute: Rate limit for API calls.
            max_iterations: Max number of QA pairs per file.
            max_consecutive_duplicates: Max consecutive duplicates before moving to next file.
            api_key: Google Gemini API key (defaults to GEMINI_API_KEY env var).
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env file.")

        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            timeout=None,
            api_key=self.api_key
        )

        self.max_iterations = max_iterations
        self.max_consecutive_duplicates = max_consecutive_duplicates
        self.requests_per_minute = requests_per_minute
        self.minute = 60  # seconds
        
        # API Request Monitoring
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.session_start_time = datetime.now()
        self.quota_exceeded_flag = False # New flag to signal global quota exhaustion

        # Prompt Template
        self.chat_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are an expert in generating concise, clear, and contextually rich question-answer pairs based on provided information. Your goal is to create standalone QA pairs that do not explicitly reference their source material. Ensure questions are direct and answers are complete, as if they are general knowledge.
                """
            ),
            (
                "user",
                """
                Content Metadata:
                {metadata}

                Full Content:
                {content}

                Already Generated Questions (avoid these topics and specific questions):
                {qa_history}

                Instructions:
                - Generate a single, new, high-quality question-answer pair.
                - The **question** should be clear and directly answerable from the content. It should not hint that the answer comes from a specific document or chunk (e.g., avoid phrases like "According to the document...", "As per the text...", "In this circular...").
                - The **answer** should be comprehensive and self-contained, providing all necessary information without requiring the reader to refer back to the original content. It should also avoid phrases that indicate it's drawn from a specific source (e.g., "The text states...", "As mentioned in the document...").
                - The **evaluation_criteria** should be specific and varied for each QA pair. Instead of generic "The answer should accurately reflect the...", specify *what* aspects of the answer should be accurate, what key terms should be included, or what logical steps should be followed. Think about unique checks for each answer.
                - Assign a **category** ('reasoning' or 'fact-based').
                - Assign an **estimated_difficulty** as an integer from 1 to 10, where:
                    - 1–3: Very easy (basic factual recall)
                    - 4–6: Medium (requires some interpretation or synthesis)
                    - 7–10: Hard (requires inference, reasoning, or deep understanding)

                - If no more meaningful or distinct questions can be generated from the provided content, respond with the special sentinel JSON:
                {{
                    "question": "NO_MORE_QUESTIONS",
                    "answer": "NO_MORE_QUESTIONS",
                    "evaluation_criteria": "NO_MORE_QUESTIONS",
                    "category": "fact-based",
                    "estimated_difficulty": 1
                }}

                Output format:
                JSON object conforming to the QAPair schema.
                """
            )
        ])

        self.structure_chain = self.chat_prompt | self.model.with_structured_output(QAPair)

    def print_request_stats(self, context: str = ""):
        """Print current API request statistics."""
        elapsed_time = datetime.now() - self.session_start_time
        elapsed_minutes = elapsed_time.total_seconds() / 60
        
        rate_per_minute = self.total_requests / elapsed_minutes if elapsed_minutes > 0 else 0
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        print(f"\n[bold cyan]🔍 API REQUEST STATS {context}[/bold cyan]")
        print(f"[cyan]📊 Total Requests: {self.total_requests}[/cyan]")
        print(f"[green]✅ Successful: {self.successful_requests}[/green]")
        print(f"[red]❌ Failed: {self.failed_requests}[/red]")
        print(f"[blue]⏱️  Session Time: {elapsed_time.total_seconds():.1f}s ({elapsed_minutes:.1f} min)[/blue]")
        print(f"[yellow]📈 Request Rate: {rate_per_minute:.2f} req/min[/yellow]")
        print(f"[magenta]💯 Success Rate: {success_rate:.1f}%[/magenta]")
        print("-" * 50)

    def extract_metadata_from_chunks(self, chunks_folder_path: str, metadata_file_path: str) -> Dict[str, Dict]:
        """Load all text chunks and associate them with JSON metadata."""
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        result = {}

        for filename in os.listdir(chunks_folder_path):
            if filename.endswith('.txt'):
                base_name = os.path.splitext(filename)[0]
                clean_key = re.sub(r'_(?:text|tables)(?:_part\d+)?$', '', base_name)
                if clean_key in metadata:
                    file_path = os.path.join(chunks_folder_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as txt_file:
                        content = txt_file.read()

                    result[filename] = {
                        "metadata": metadata[clean_key],
                        "content": content
                    }
                else:
                    print(f"⚠️ Metadata not found for file: {filename} (key '{clean_key}')")

        return result

    def generate_qa_pairs(self, content: str, metadata: dict, filename: str) -> QAPairList:
        """Generate multiple QA pairs for a given content and metadata."""
        qa_pairs = []
        history_str = "None"
        consecutive_duplicates = 0  # Track consecutive duplicates
        consecutive_errors = 0      # Track consecutive errors

        print(f"\n[blue]🚀 Starting QA generation for: {filename}[/blue]")
        self.print_request_stats(f"- Before processing {filename}")

        for i in range(self.max_iterations):
            # Check the global quota exceeded flag before making an API call
            if self.quota_exceeded_flag:
                print(f"[red]🛑 STOPPING ITERATIONS for {filename}: Global quota limit already hit.[/red]")
                break # Stop generating for the current file

            # Proactive check before starting a new request within the file's iterations
            # This is to catch if we are very close to the limit before the next API call
            if self.total_requests >= 990: 
                print(f"[red]🛑 STOPPING ITERATIONS for {filename}: Approaching global quota limit ({self.total_requests}/1000).[/red]")
                self.quota_exceeded_flag = True # Set the global flag
                break # Stop generating for the current file


            try:
                # Print request count before each API call
                print(f"\n[bright_blue]🔄 About to make API request #{self.total_requests + 1} (iteration {i + 1}/{self.max_iterations})[/bright_blue]")
                
                # Format history
                if qa_pairs:
                    history_str = "\n".join([
                        f"{idx + 1}. {pair.question}" for idx, pair in enumerate(qa_pairs)
                    ])
                else:
                    history_str = "None"

                # Increment total requests counter before making the call
                self.total_requests += 1
                
                # Invoke LLM
                response: QAPair = self.structure_chain.invoke({
                    "metadata": json.dumps(metadata),
                    "content": content,
                    "qa_history": history_str
                }, config={"callbacks":[langfuse_handler]})

                # Increment successful requests counter
                self.successful_requests += 1
                print(f"[bright_green]✅ API Request #{self.total_requests} SUCCESSFUL[/bright_green]")

                # Print stats every 5 requests or at important milestones
                if self.total_requests % 5 == 0 or i == 0:
                    self.print_request_stats(f"- After request #{self.total_requests}")

                # Check for end-of-generation signal
                if (
                    response.question == "NO_MORE_QUESTIONS" and
                    response.answer == "NO_MORE_QUESTIONS" and
                    response.evaluation_criteria == "NO_MORE_QUESTIONS"
                ):
                    print("[green]Model indicated no more questions can be generated.[/green]")
                    break

                # Check for duplicate questions
                if any(pair.question == response.question for pair in qa_pairs):
                    consecutive_duplicates += 1
                    print(f"[yellow]Duplicate question detected ({consecutive_duplicates}/{self.max_consecutive_duplicates}). Skipping.[/yellow]")
                    
                    # If too many consecutive duplicates, break out of the loop
                    if consecutive_duplicates >= self.max_consecutive_duplicates:
                        print(f"[yellow]⚠️ Too many consecutive duplicates ({consecutive_duplicates}). Moving to next file.[/yellow]")
                        break
                    
                    # Sleep before retrying
                    time.sleep(self.minute / self.requests_per_minute)
                    continue

                # Reset consecutive counters on successful generation
                consecutive_duplicates = 0
                consecutive_errors = 0

                # Add to list
                qa_pairs.append(response)
                print(f"[green]✅ Generated QA pair #{len(qa_pairs)} (iteration {i + 1})[/green]")

                # Sleep to avoid hitting Gemini API rate limits
                time.sleep(self.minute / self.requests_per_minute)

            # Catch specific Google API exceptions for better handling
            except ResourceExhausted as e:
                self.failed_requests += 1
                consecutive_errors += 1
                error_str = str(e)
                print(f"[red]❌ API Request #{self.total_requests} FAILED: ResourceExhausted (Quota Issue)[/red]")
                print(f"[red]Error during iteration {i + 1} (consecutive errors: {consecutive_errors}): {error_str}[/red]")
                
                print(f"[red]🚫 QUOTA EXCEEDED ERROR DETECTED![/red]")
                print(f"[yellow]💡 You've likely hit the Gemini API free tier limit (1000 requests/day)[/yellow]")
                print(f"[yellow]⏰ Consider waiting until tomorrow or upgrading your plan[/yellow]")
                
                self_temp = self # Store self to avoid passing self as an argument
                def _handle_quota_exceeded():
                    self_temp.quota_exceeded_flag = True # Set the global flag
                    print(f"[blue]💾 Saving current progress before handling quota error...[/blue]")
                    # No need to explicitly return from here, the break will handle it

                _handle_quota_exceeded() # Call the helper function
                break # Break out of the inner loop immediately

            except InternalServerError as e:
                self.failed_requests += 1
                consecutive_errors += 1
                error_str = str(e)
                print(f"[red]❌ API Request #{self.total_requests} FAILED: Internal Server Error[/red]")
                print(f"[red]Error during iteration {i + 1} (consecutive errors: {consecutive_errors}): {error_str}[/red]")
                if consecutive_errors >= 3:
                    print(f"[red]⚠️ Too many consecutive Internal Server Errors ({consecutive_errors}). Moving to next file.[/red]")
                    break
                time.sleep(self.minute / self.requests_per_minute) # Wait before retrying
                continue # Try the next iteration

            except GoogleAPIError as e: # Catch other Google API specific errors
                self.failed_requests += 1
                consecutive_errors += 1
                error_str = str(e)
                print(f"[red]❌ API Request #{self.total_requests} FAILED: Google API Error[/red]")
                print(f"[red]Error during iteration {i + 1} (consecutive errors: {consecutive_errors}): {error_str}[/red]")
                if consecutive_errors >= 3:
                    print(f"[red]⚠️ Too many consecutive Google API Errors ({consecutive_errors}). Moving to next file.[/red]")
                    break
                time.sleep(self.minute / self.requests_per_minute)
                continue

            except Exception as e:
                # General exception for unexpected errors
                self.failed_requests += 1
                consecutive_errors += 1
                error_str = str(e)
                
                print(f"[red]❌ API Request #{self.total_requests} FAILED: Unexpected Error[/red]")
                print(f"[red]Error during iteration {i + 1} (consecutive errors: {consecutive_errors}): {error_str}[/red]")
                
                self.print_request_stats(f"- After FAILED request #{self.total_requests}")
                
                if consecutive_errors >= 3:
                    print(f"[red]⚠️ Too many consecutive errors ({consecutive_errors}). Moving to next file.[/red]")
                    break
                
                print(f"[yellow]⏱️ Waiting {self.minute / self.requests_per_minute:.1f}s before retry...[/yellow]")
                time.sleep(self.minute / self.requests_per_minute)

        print(f"\n[blue]🏁 Finished processing {filename}[/blue]")
        self.print_request_stats(f"- After completing {filename}")
        
        return QAPairList(items=qa_pairs)

    def run(self, chunks_path: str, metadata_path: str, output_file: str = "generated_qa_pairs.json"):
        """Main function to process files and generate QA pairs — appends results to existing JSON."""
        print("\n[blue]Starting QA generation...[/blue]\n")
        print(f"[blue]🕐 Session started at: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}[/blue]")
        print(f"[yellow]⚠️ Free tier limit: 1000 requests/day for gemini-2.0-flash[/yellow]")
        self.print_request_stats("- SESSION START")

        # Load existing results if file exists
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                try:
                    all_results = json.load(f)
                    print(f"[blue]📁 Loaded existing results with {len(all_results)} files[/blue]")
                except json.JSONDecodeError:
                    print("[yellow]⚠️ Existing JSON file is empty or corrupted. Starting fresh.[/yellow]")
                    all_results = {}
        else:
            all_results = {}

        # Extract metadata and content
        metadata_dict = self.extract_metadata_from_chunks(chunks_path, metadata_path)
        
        print(f"[blue]📊 Found {len(metadata_dict)} files to process[/blue]")
        # `self.quota_exceeded_flag` will now directly control the session termination

        for idx, (filename, data) in enumerate(metadata_dict.items(), 1):
            if self.quota_exceeded_flag: # Check the global flag
                print(f"[red]🛑 SESSION STOPPED: Quota previously exceeded. Not processing {filename}.[/red]")
                break # Exit the main file processing loop

            print(f"\n[blue]📄 Processing file {idx}/{len(metadata_dict)}: {filename}[/blue]")
            
            # Skip if already processed (optional - remove if you want to reprocess)
            if filename in all_results:
                print(f"[yellow]⚠️ File {filename} already processed. Skipping.[/yellow]")
                continue
            
            # Proactive check before starting a new file in the main loop
            if self.total_requests >= 980: # Aggressive stop to ensure we don't hit 1000 within a file
                print(f"[red]🛑 STOPPING ENTIRE SESSION: Very close to quota limit ({self.total_requests}/1000 requests used).[/red]")
                self.quota_exceeded_flag = True # Set the flag
                break # Exit the main file processing loop
            
            content = data["content"]
            meta = data["metadata"]

            # The generate_qa_pairs function will now set `self.quota_exceeded_flag` if needed
            qa_list = self.generate_qa_pairs(content, meta, filename)

            # After `generate_qa_pairs` returns, check the global flag
            if self.quota_exceeded_flag:
                print(f"[red]🛑 SESSION STOPPED: Quota limit hit during processing of {filename}.[/red]")
                # We still want to save what was generated for this file if any
                # This save logic is already present below, so no need to duplicate.
            
            # Get clean key (from metadata key)
            base_name = os.path.splitext(filename)[0]
            clean_key = re.sub(r'_(?:text|tables)(?:_part\d+)?$', '', base_name)

            # Determine if it's a table file
            is_table = '_table' in filename

            # Build result for this file (even if QA list is empty due to quota or other issues)
            file_result = {
                filename: {
                    "document": clean_key,
                    "model_name": self.model.model,
                    "metadata": meta,
                    "chunks_text": content,
                    "is_table": is_table,
                    "qa_pairs": [qap.model_dump() for qap in qa_list.items]
                }
            }

            # Update the overall results (this will overwrite if filename already exists)
            all_results.update(file_result)

            print(f"[green]✔ Completed {len(qa_list.items)} QA pairs for {filename}[/green]")

            # Save after each file (for safety in case of crashes)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4)
                
            print(f"[blue]💾 Progress saved to {output_file}[/blue]")

            # Print comprehensive stats after each file
            print(f"\n[bold green]📈 PROGRESS UPDATE - File {idx}/{len(metadata_dict)} completed[/bold green]")
            self.print_request_stats(f"- After file {idx}/{len(metadata_dict)}")
            
            # If the quota was hit, break the main loop after saving the current file's progress
            if self.quota_exceeded_flag:
                break # This will exit the `for filename, data in metadata_dict.items()` loop

        print(f"\n[green]✅ Final JSON file saved at: {output_file}[/green]")
        print(f"[green]📊 Total files processed: {len(all_results)}[/green]")
        
        if self.quota_exceeded_flag:
            print(f"[red]⚠️ Session ended due to quota limits[/red]")
            print(f"[yellow]💡 Recommendations:[/yellow]")
            print(f"[yellow]   • Wait until tomorrow (quota resets daily)[/yellow]")
            print(f"[yellow]   • Upgrade to paid Gemini API plan for higher limits[/yellow]")
            print(f"[yellow]   • Reduce requests_per_minute or max_iterations[/yellow]")
        
        # Final comprehensive stats
        print(f"\n[bold magenta]🎉 SESSION COMPLETE![/bold magenta]")
        self.print_request_stats("- FINAL SESSION STATS")
        

if __name__ == "__main__":
    generator = QAGenerator(
        model_name="gemini-2.0-flash",
        temperature=0.2,
        requests_per_minute=1800, # Keep this reasonable to avoid hitting per-minute limits too often
        max_iterations=20, # Max Q&A pairs per chunk
        max_consecutive_duplicates=3
    )

    chunks_path = "/workspaces/Data_prep/Code/Data/Chunks/2024"
    metadata_path = "/workspaces/Data_prep/Code/Data/meta-data/metadata_2024.json"
    output_file = "/workspaces/Data_prep/Code/Data/QA/generated_qa_pairs_2024.json"

    generator.run(chunks_path, metadata_path, output_file)