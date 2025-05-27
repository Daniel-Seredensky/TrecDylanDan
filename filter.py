import pandas as pd
import json
DYLANDOCUMENT = 'clueweb22-en0042-56-04016'
DANDOCUMENT = 'clueweb22-en0024-53-03398'
DYLANOUT = 'ContextDylan.jsonl'
DANOUT = 'Context.jsonl'

DOCUMENT = DYLANDOCUMENT

def filter_relevant_documents():
    # Read the qrels file as a space-delimited dataframe
    df = pd.read_csv('clueweb/2024-retrieval-qrels.txt', sep=' ', header=0)
    
    # Filter for rows where Document contains the target document substring and Relevance >= 1
    filtered_df = df[(df['Document'].str.contains(DOCUMENT, na=False)) & (df['Relevance'] > 1)]
    
    # Extract the AssessedDocument column
    assessed_documents = filtered_df['AssessedDocument'].tolist()
    
    print(f"Found {len(assessed_documents)} relevant documents for {DOCUMENT}:")
    for doc in assessed_documents:
        print(f"  {doc}")
    
    return assessed_documents, filtered_df

def create_context_jsonl(assessed_documents, filtered_df, jsonl_file_path):
    # Create a dictionary for quick relevance lookup
    relevance_dict = dict(zip(filtered_df['AssessedDocument'], filtered_df['Relevance']))
    
    # Read the jsonl file and create Context.jsonl
    with open(jsonl_file_path, 'r', encoding='utf-8') as input_file, \
         open(DYLANOUT, 'w', encoding='utf-8') as output_file:
        
        found_documents = set()
        
        for line in input_file:
            try:
                json_entry = json.loads(line.strip())
                clue_web_id = json_entry.get('ClueWeb22-ID')
                
                # Check if this document is in our assessed documents list
                if clue_web_id in assessed_documents:
                    context_entry = {
                        'Document': clue_web_id,
                        'Clean-Text': json_entry.get('Clean-Text', ''),
                        'Relevance': relevance_dict.get(clue_web_id, 0)
                    }
                    
                    output_file.write(json.dumps(context_entry) + '\n')
                    found_documents.add(clue_web_id)
                    
            except json.JSONDecodeError:
                continue
        
        print(f"Created Context.jsonl with {len(found_documents)} entries")
        
        # Report any documents that weren't found
        missing_documents = set(assessed_documents) - found_documents
        if missing_documents:
            print(f"Warning: {len(missing_documents)} documents not found in jsonl file:")
            for doc in missing_documents:
                print(f"  {doc}")

if __name__ == "__main__":
    assessed_documents, filtered_df = filter_relevant_documents()
    
    # Specify the path to your jsonl file here
    jsonl_file_path = 'clueweb/TREC-LR-2024/T2/trec-2024-lateral-reading-task2-baseline-documents.jsonl'  # Update this path
    
    if assessed_documents:
        create_context_jsonl(assessed_documents, filtered_df, jsonl_file_path)
    else:
        print("No relevant documents found to process")
