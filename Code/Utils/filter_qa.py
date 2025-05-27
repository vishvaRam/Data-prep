import json
import re
from typing import List, Dict, Tuple
from collections import Counter
import pandas as pd

class QAAnalyzerFilter:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.qa_pairs = []
        self.load_qa_pairs()
        
    def load_qa_pairs(self):
        """Load all QA pairs from the JSON file"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract all QA pairs from all files
        for file_key, file_data in data.items():
            for qa in file_data.get('qa_pairs', []):
                qa['source_file'] = file_key
                qa['document'] = file_data.get('document', 'unknown')
                self.qa_pairs.append(qa)
        
        print(f"📊 Loaded {len(self.qa_pairs)} QA pairs from {len(data)} files")
    
    def analyze_context_dependency(self) -> Dict:
        """Analyze and categorize QA pairs by context dependency"""
        
        # Context-dependent indicators
        context_indicators = {
            'document_references': [
                r'\bthe document\b', r'\bthis document\b', r'\bthe text\b', r'\bthis text\b',
                r'\bthe circular\b', r'\bthis circular\b', r'\bthe passage\b', r'\bthis passage\b',
                r'\baccording to the\b', r'\bas per the\b', r'\bas mentioned in\b',
                r'\bthe above\b', r'\babove mentioned\b', r'\bthe given\b'
            ],
            'administrative_metadata': [
                r'\bcircular no\b', r'\bcircular number\b', r'\ba\.p\.\s*\(dir\b',
                r'\baddressed to\b', r'\bto whom\b', r'\bwho is.*addressed\b',
                r'\bsubject.*circular\b', r'\breference.*no\b', r'\bdated\b',
                r'\bnotification.*no\b', r'\bmaster direction.*no\b'
            ],
            'generic_document_questions': [
                r'^what is the.*number\b', r'^what is the.*date\b',
                r'^who.*addressed\b', r'^what.*subject\b',
                r'^what.*title\b', r'^what.*heading\b'
            ]
        }
        
        results = {
            'good_pairs': [],      # Context-independent, substantive
            'salvageable': [],     # Minor context issues, can be fixed
            'problematic': [],     # Heavy context dependency
            'metadata_only': []    # Pure administrative metadata
        }
        
        for qa in self.qa_pairs:
            question = qa['question'].lower()
            answer = qa['answer'].lower()
            
            # Count context indicators
            context_score = 0
            found_indicators = []
            
            for category, patterns in context_indicators.items():
                for pattern in patterns:
                    if re.search(pattern, question, re.IGNORECASE) or re.search(pattern, answer, re.IGNORECASE):
                        context_score += 1
                        found_indicators.append((category, pattern))
            
            # Categorize based on context score and content
            qa['context_score'] = context_score
            qa['found_indicators'] = found_indicators
            
            if context_score == 0 and not self._is_metadata_only(qa):
                results['good_pairs'].append(qa)
            elif context_score <= 2 and not self._is_metadata_only(qa):
                results['salvageable'].append(qa)
            elif self._is_metadata_only(qa):
                results['metadata_only'].append(qa)
            else:
                results['problematic'].append(qa)
        
        return results
    
    def _is_metadata_only(self, qa: Dict) -> bool:
        """Check if QA pair is purely about document metadata"""
        metadata_keywords = [
            'circular no', 'circular number', 'a.p.', 'addressed to', 'subject',
            'reference', 'dated', 'notification no', 'master direction no',
            'dear sir', 'yours faithfully', 'chief general manager'
        ]
        
        question = qa['question'].lower()
        return any(keyword in question for keyword in metadata_keywords)
    
    def generate_analysis_report(self, results: Dict) -> str:
        """Generate detailed analysis report"""
        total = len(self.qa_pairs)
        
        report = f"""
🔍 QA PAIRS ANALYSIS REPORT
{'='*50}

📊 OVERALL STATISTICS:
- Total QA Pairs: {total:,}
- Good (Context-Independent): {len(results['good_pairs']):,} ({len(results['good_pairs'])/total*100:.1f}%)
- Salvageable (Minor Issues): {len(results['salvageable']):,} ({len(results['salvageable'])/total*100:.1f}%)
- Problematic (Heavy Context): {len(results['problematic']):,} ({len(results['problematic'])/total*100:.1f}%)
- Metadata Only: {len(results['metadata_only']):,} ({len(results['metadata_only'])/total*100:.1f}%)

✅ USABLE PAIRS: {len(results['good_pairs']) + len(results['salvageable']):,} ({(len(results['good_pairs']) + len(results['salvageable']))/total*100:.1f}%)

🚫 PROBLEMATIC PAIRS: {len(results['problematic']) + len(results['metadata_only']):,} ({(len(results['problematic']) + len(results['metadata_only']))/total*100:.1f}%)

📈 RECOMMENDATIONS:
1. Keep {len(results['good_pairs']):,} good pairs as-is
2. Post-process {len(results['salvageable']):,} salvageable pairs  
3. Consider regenerating {len(results['problematic']) + len(results['metadata_only']):,} problematic pairs

"""
        return report
    
    def save_filtered_pairs(self, results: Dict, output_file: str = None):
        """Save filtered QA pairs maintaining original JSON structure"""
        if output_file is None:
            base_name = self.input_file.replace('.json', '')
            output_file = f"{base_name}_filtered.json"
        
        # Load original data structure to maintain format
        with open(self.input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        # Combine good and salvageable pairs
        usable_pairs = results['good_pairs'] + results['salvageable']
        
        # Create mapping of source_file to QA pairs
        file_qa_mapping = {}
        for qa in usable_pairs:
            source_file = qa['source_file']
            if source_file not in file_qa_mapping:
                file_qa_mapping[source_file] = []
            
            # Keep only the core QA fields, remove analysis fields
            clean_qa = {
                'question': qa['question'],
                'answer': qa['answer'],
                'evaluation_criteria': qa['evaluation_criteria'],
                'category': qa['category'],
                'estimated_difficulty': qa['estimated_difficulty']
            }
            file_qa_mapping[source_file].append(clean_qa)
        
        # Reconstruct original structure with filtered QA pairs
        filtered_data = {}
        for file_key, file_data in original_data.items():
            if file_key in file_qa_mapping and file_qa_mapping[file_key]:
                # Maintain exact original structure
                filtered_data[file_key] = {
                    "document": file_data.get("document", ""),
                    "model_name": file_data.get("model_name", ""),
                    "metadata": file_data.get("metadata", {}),
                    "chunks_text": file_data.get("chunks_text", ""),
                    "is_table": file_data.get("is_table", False),
                    "qa_pairs": file_qa_mapping[file_key]
                }
        
        # Save filtered data in original format
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=4, ensure_ascii=False)
        
        total_qa_pairs = sum(len(data['qa_pairs']) for data in filtered_data.values())
        print(f"💾 Saved {total_qa_pairs} filtered QA pairs across {len(filtered_data)} files to: {output_file}")
        
        # Also save problematic pairs for analysis (grouped by source file)
        problematic_file = output_file.replace('_filtered.json', '_problematic.json')
        problematic_pairs = results['metadata_only'] + results['problematic']
        
        problematic_file_mapping = {}
        for qa in problematic_pairs:
            source_file = qa['source_file']
            if source_file not in problematic_file_mapping:
                problematic_file_mapping[source_file] = []
            
            problematic_qa = {
                'question': qa['question'],
                'answer': qa['answer'],
                'evaluation_criteria': qa['evaluation_criteria'],
                'category': qa['category'],
                'estimated_difficulty': qa['estimated_difficulty'],
                'context_score': qa.get('context_score', 0),
                'issues': [ind[1] for ind in qa.get('found_indicators', [])]
            }
            problematic_file_mapping[source_file].append(problematic_qa)
        
        # Reconstruct original structure for problematic pairs too
        problematic_data = {}
        for file_key, file_data in original_data.items():
            if file_key in problematic_file_mapping and problematic_file_mapping[file_key]:
                problematic_data[file_key] = {
                    "document": file_data.get("document", ""),
                    "model_name": file_data.get("model_name", ""),
                    "metadata": file_data.get("metadata", {}),
                    "chunks_text": file_data.get("chunks_text", ""),
                    "is_table": file_data.get("is_table", False),
                    "qa_pairs": problematic_file_mapping[file_key]
                }
        
        with open(problematic_file, 'w', encoding='utf-8') as f:
            json.dump(problematic_data, f, indent=4, ensure_ascii=False)
        
        total_problematic = sum(len(data['qa_pairs']) for data in problematic_data.values())
        print(f"🔍 Saved {total_problematic} problematic pairs across {len(problematic_data)} files to: {problematic_file}")
        
        return output_file, problematic_file
    
    def show_examples(self, results: Dict, num_examples: int = 3):
        """Show examples from each category"""
        categories = ['good_pairs', 'salvageable', 'problematic', 'metadata_only']
        
        for category in categories:
            pairs = results[category][:num_examples]
            print(f"\n🔸 {category.upper().replace('_', ' ')} EXAMPLES:")
            print("-" * 40)
            
            for i, qa in enumerate(pairs, 1):
                print(f"\n{i}. Q: {qa['question']}")
                print(f"   A: {qa['answer'][:100]}{'...' if len(qa['answer']) > 100 else ''}")
                if qa.get('found_indicators'):
                    indicators = [ind[1] for ind in qa['found_indicators']]
                    print(f"   Issues: {', '.join(indicators[:3])}")
    
    def get_quality_distribution(self) -> Dict:
        """Analyze quality distribution across different dimensions"""
        categories = Counter(qa['category'] for qa in self.qa_pairs)
        difficulties = Counter(qa['estimated_difficulty'] for qa in self.qa_pairs)
        
        return {
            'categories': dict(categories),
            'difficulties': dict(difficulties),
            'avg_difficulty': sum(qa['estimated_difficulty'] for qa in self.qa_pairs) / len(self.qa_pairs)
        }


def main():
    # Replace with your actual file path
    input_file = "/workspaces/Data_prep/Code/Data/QA/generated_qa_pairs_2024.json"
    
    print("🚀 Starting QA Analysis...")
    analyzer = QAAnalyzerFilter(input_file)
    
    # Analyze context dependency
    print("\n🔍 Analyzing context dependency...")
    results = analyzer.analyze_context_dependency()
    
    # Generate and print report
    report = analyzer.generate_analysis_report(results)
    print(report)
    
    # Show examples
    print("\n📝 EXAMPLES FROM EACH CATEGORY:")
    analyzer.show_examples(results)
    
    # Show quality distribution
    quality = analyzer.get_quality_distribution()
    print(f"\n📊 QUALITY DISTRIBUTION:")
    print(f"Categories: {quality['categories']}")
    print(f"Average Difficulty: {quality['avg_difficulty']:.1f}")
    
    # Save filtered results
    print("\n💾 Saving filtered results...")
    good_file, bad_file = analyzer.save_filtered_pairs(results)
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📁 Good pairs saved to: {good_file}")
    print(f"📁 Problematic pairs saved to: {bad_file}")

if __name__ == "__main__":
    main()