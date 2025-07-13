import json
import re
from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from filter_qa import QAAnalyzerFilter
import os
from dotenv import load_dotenv

load_dotenv()

class ImprovedQAPair(BaseModel):
    question: str
    answer: str
    evaluation_criteria: str
    category: str
    estimated_difficulty: int

class QAPostProcessor:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            api_key=self.api_key
        )
        
        # Create improvement prompt
        self.improvement_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert at transforming context-dependent questions into context-independent, standalone questions suitable for training.

            Your task is to rewrite questions and answers to remove document references while preserving the substantive knowledge being tested.

            TRANSFORMATION RULES:
            1. Remove ALL references to "the document", "the circular", "this text", "the passage", etc.
            2. Transform document-specific questions into domain knowledge questions
            3. Make answers self-contained without source references
            4. Preserve the core knowledge and difficulty level
            5. Ensure the question can stand alone as general knowledge

            EXAMPLES:
            ❌ "What is the circular number?" 
            ✅ "What are the reporting requirements for authorized dealer banks in foreign exchange?"

            ❌ "To whom is this circular addressed?"
            ✅ "Which category of banks must comply with foreign exchange dealer regulations?"

            ❌ "According to the document, what are the compliance requirements?"
            ✅ "What are the key compliance requirements for anti-money laundering in banking?"
            """),
            ("user", """
            Original Question: {question}
            Original Answer: {answer}
            Original Category: {category}
            Original Difficulty: {difficulty}

            INSTRUCTIONS:
            Rewrite this QA pair to be context-independent while preserving the substantive knowledge. 
            If the original question is purely about document metadata (circular numbers, addresses, etc.) and cannot be meaningfully transformed into domain knowledge, respond with "SKIP_THIS_PAIR".

            Output the improved QA pair as JSON matching the ImprovedQAPair schema.
            """)
        ])
        
        self.chain = self.improvement_prompt | self.model.with_structured_output(ImprovedQAPair)
    
    def needs_improvement(self, qa: Dict) -> bool:
        """Check if QA pair needs improvement"""
        context_patterns = [
            r'\bthe document\b', r'\bthis document\b', r'\bthe circular\b', 
            r'\bthis circular\b', r'\baccording to\b', r'\bas per the\b',
            r'\bthe above\b', r'\babove mentioned\b', r'\baddressed to\b'
        ]
        
        text = f"{qa['question']} {qa['answer']}".lower()
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in context_patterns)
    
    def improve_qa_pair(self, qa: Dict) -> Dict:
        """Improve a single QA pair"""
        try:
            response = self.chain.invoke({
                "question": qa['question'],
                "answer": qa['answer'], 
                "category": qa['category'],
                "difficulty": qa['estimated_difficulty']
            })
            
            if hasattr(response, 'question') and response.question != "SKIP_THIS_PAIR":
                return {
                    'question': response.question,
                    'answer': response.answer,
                    'evaluation_criteria': response.evaluation_criteria,
                    'category': response.category,
                    'estimated_difficulty': response.estimated_difficulty,
                    'original_question': qa['question'],  # Keep for reference
                    'improved': True
                }
            else:
                return None  # Skip this pair
                
        except Exception as e:
            print(f"Error improving QA pair: {e}")
            return qa  # Return original if improvement fails
    
    def process_salvageable_pairs(self, input_file: str, output_file: str = None):
        """Process salvageable QA pairs maintaining original JSON structure"""
        if output_file is None:
            output_file = input_file.replace('.json', '_improved.json')
        
        # Load the filtered data (already in original format)
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        improved_data = {}
        total_processed = 0
        total_improved = 0
        total_skipped = 0
        
        for file_key, file_data in data.items():
            qa_pairs = file_data.get('qa_pairs', [])
            improved_pairs = []
            
            for qa in qa_pairs:
                total_processed += 1
                
                if self.needs_improvement(qa):
                    print(f"🔧 Improving: {qa['question'][:60]}...")
                    improved_qa = self.improve_qa_pair(qa)
                    
                    if improved_qa is None:
                        total_skipped += 1
                        print(f"   ⏭️  Skipped (metadata only)")
                    else:
                        # Keep only core QA fields for final output
                        clean_improved_qa = {
                            'question': improved_qa['question'],
                            'answer': improved_qa['answer'],
                            'evaluation_criteria': improved_qa['evaluation_criteria'],
                            'category': improved_qa['category'],
                            'estimated_difficulty': improved_qa['estimated_difficulty']
                        }
                        improved_pairs.append(clean_improved_qa)
                        total_improved += 1
                        print(f"   ✅ Improved")
                else:
                    # Keep as-is if no improvement needed (remove analysis fields)
                    clean_qa = {
                        'question': qa['question'],
                        'answer': qa['answer'],  
                        'evaluation_criteria': qa['evaluation_criteria'],
                        'category': qa['category'],
                        'estimated_difficulty': qa['estimated_difficulty']
                    }
                    improved_pairs.append(clean_qa)
            
            if improved_pairs:  # Only add if there are pairs left
                # Maintain original file structure
                improved_data[file_key] = {
                    "document": file_data.get("document", ""),
                    "model_name": file_data.get("model_name", ""),
                    "metadata": file_data.get("metadata", {}),
                    "chunks_text": file_data.get("chunks_text", ""),
                    "is_table": file_data.get("is_table", False),
                    "qa_pairs": improved_pairs
                }
        
        # Save improved data in original format
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(improved_data, f, indent=4, ensure_ascii=False)
        
        total_final = sum(len(data['qa_pairs']) for data in improved_data.values())
        print(f"\n📊 POST-PROCESSING COMPLETE:")
        print(f"   Total Processed: {total_processed}")
        print(f"   Improved: {total_improved}")
        print(f"   Kept As-Is: {total_processed - total_improved - total_skipped}")
        print(f"   Skipped: {total_skipped}")
        print(f"   Final Count: {total_final}")
        print(f"💾 Saved to: {output_file}")
        
        return output_file


# Rule-based post-processor (faster, no API calls)
class RuleBasedPostProcessor:
    def __init__(self):
        self.context_removals = {
            # Direct replacements
            r'\bthe document states that\b': '',
            r'\baccording to the document,?\s*': '',
            r'\bas per the circular,?\s*': '',
            r'\bthe circular states that\b': '',
            r'\bas mentioned in the document,?\s*': '',
            r'\bthe text mentions that\b': '',
            r'\bin the document,?\s*': '',
            
            # Question transformations
            r'^what does the document say about': 'What are the requirements for',
            r'^what is mentioned in the circular about': 'What are the guidelines for',
            r'^according to the circular, what': 'What',
            r'^as per the document, what': 'What',
        }
        
        # Skip patterns - these indicate metadata-only questions
        self.skip_patterns = [
            r'circular number', r'circular no\.?', r'a\.p\.\s*\(dir',
            r'addressed to', r'to whom.*addressed', r'subject.*circular',
            r'reference.*no', r'notification.*no'
        ]
    
    def should_skip(self, question: str) -> bool:
        """Check if question should be skipped (metadata only)"""
        return any(re.search(pattern, question.lower()) for pattern in self.skip_patterns)
    
    def clean_text(self, text: str) -> str:
        """Remove context references from text"""
        cleaned = text
        for pattern, replacement in self.context_removals.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        # Clean up extra spaces and capitalization
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if cleaned and not cleaned[0].isupper():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned
    
    def process_qa_pairs(self, input_file: str, output_file: str = None):
        """Process QA pairs with rule-based cleaning maintaining original JSON structure"""
        if output_file is None:
            output_file = input_file.replace('.json', '_rule_cleaned.json')
        
        # Load data (could be original format or filtered format)
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cleaned_data = {}
        total_processed = 0
        total_kept = 0
        total_skipped = 0
        
        for file_key, file_data in data.items():
            # Handle both original format and filtered format
            if isinstance(file_data, dict) and 'qa_pairs' in file_data:
                # Original format: file contains metadata + qa_pairs
                qa_pairs = file_data['qa_pairs']
                file_metadata = {
                    "document": file_data.get("document", ""),
                    "model_name": file_data.get("model_name", ""),
                    "metadata": file_data.get("metadata", {}),
                    "chunks_text": file_data.get("chunks_text", ""),
                    "is_table": file_data.get("is_table", False)
                }
            elif isinstance(file_data, list):
                # Simplified format: file_key maps directly to list of qa_pairs
                qa_pairs = file_data
                file_metadata = {
                    "document": file_key,
                    "model_name": "gemini-2.0-flash",
                    "metadata": {},
                    "chunks_text": "",
                    "is_table": False
                }
            else:
                continue
            
            cleaned_pairs = []
            
            for qa in qa_pairs:
                total_processed += 1
                
                if self.should_skip(qa['question']):
                    total_skipped += 1
                    continue
                
                cleaned_qa = {
                    'question': self.clean_text(qa['question']),
                    'answer': self.clean_text(qa['answer']),
                    'evaluation_criteria': qa['evaluation_criteria'],
                    'category': qa['category'],
                    'estimated_difficulty': qa['estimated_difficulty']
                }
                
                cleaned_pairs.append(cleaned_qa)
                total_kept += 1
            
            if cleaned_pairs:
                # Always output in original format
                cleaned_data[file_key] = {
                    **file_metadata,
                    "qa_pairs": cleaned_pairs
                }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
        
        print(f"\n📊 RULE-BASED CLEANING COMPLETE:")
        print(f"   Total Processed: {total_processed}")
        print(f"   Kept & Cleaned: {total_kept}")
        print(f"   Skipped (Metadata): {total_skipped}")
        print(f"💾 Saved to: {output_file}")
        
        return output_file


# Usage example
def main():
    input_file = "/workspaces/Data_prep/Code/Data/QA/generated_qa_pairs_2024.json"
    
    print("🚀 Step 1: Analyzing QA pairs...")
    analyzer = QAAnalyzerFilter(input_file)
    results = analyzer.analyze_context_dependency()
    
    # Save filtered pairs
    filtered_file, _ = analyzer.save_filtered_pairs(results)
    
    print(f"\n🚀 Step 2: Rule-based cleaning...")
    rule_processor = RuleBasedPostProcessor()
    cleaned_file = rule_processor.process_qa_pairs(filtered_file)
    
    print(f"\n✅ PIPELINE COMPLETE!")
    print(f"📁 Final cleaned file: {cleaned_file}")

if __name__ == "__main__":
    main()