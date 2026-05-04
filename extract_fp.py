import json
import pandas as pd
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_fp_extraction(gold_file, pred_file, token_csv):
    # 1. Load Token Mapping
    df_tokens = pd.read_csv(token_csv)
    token_lookup = dict(zip(df_tokens['entity_text'].str.lower().str.strip(), df_tokens['biobert_token_count']))

    # 2. Load Datasets
    gold_data = load_json(gold_file)
    pred_data = load_json(pred_file)

    # Map gold entities and text by doc_id for easy lookup
    gold_map = {}
    text_map = {}
    for doc in gold_data:
        doc_id = doc["doc_id"]
        text_map[doc_id] = doc.get("text", "")
        # Store lowercased true entities for this doc
        gold_map[doc_id] = {ent["text"].lower().strip() for ent in doc.get("true_entities", [])}

    # 3. Find 1-Token False Positives
    false_positives = []
    
    for doc in pred_data:
        doc_id = doc["doc_id"]
        true_ents = gold_map.get(doc_id, set())
        original_text = text_map.get(doc_id, "")
        
        for ent in doc.get("predicted_entities", []):
            text = ent["text"].lower().strip()
            label = ent["label"]
            
            # Check if it's a 1-Token word
            if len(text.split()) == 1:
                # If it's NOT in the true entities, it's a False Positive
                if text not in true_ents:
                    # Grab a snippet of text around the word for context
                    start_idx = max(0, ent["start"] - 40)
                    end_idx = min(len(original_text), ent["end"] + 40)
                    snippet = f"...{original_text[start_idx:end_idx]}..."
                    
                    false_positives.append({
                        "doc_id": doc_id,
                        "word": text,
                        "label": label,
                        "context": snippet.replace('\n', ' ')
                    })

    # 4. Output the top 50 worst offenders to a CSV for manual review
    df_fp = pd.DataFrame(false_positives)
    
    if not df_fp.empty:
        # Count which words Gemini hallucinates the most
        top_hallucinations = df_fp['word'].value_counts().head(20)
        print("\n=== TOP 20 MOST HALLUCINATED 1-TOKEN WORDS ===")
        print(top_hallucinations)
        
        df_fp.to_csv("gemini_1_token_errors.csv", index=False)
        print("\nExported full context to 'gemini_1_token_errors.csv'")
    else:
        print("No 1-token false positives found.")

if __name__ == '__main__':
    # Using your Hard subset since that's what you just analyzed
    run_fp_extraction(
        gold_file='project/test_hard.json', 
        pred_file='project/gemini_predictions_hard.json', 
        token_csv='project/token_analysis.csv'
    )