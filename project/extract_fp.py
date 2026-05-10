import json
import pandas as pd
from collections import defaultdict

# JSON Loading Utility
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# Extraction Logic for False Positives
def run_fp_extraction(gold_file, pred_file, token_csv):
    df_tokens = pd.read_csv(token_csv)
    token_lookup = dict(zip(df_tokens['entity_text'].str.lower().str.strip(), df_tokens['biobert_token_count']))

    gold_data = load_json(gold_file)
    pred_data = load_json(pred_file)

    # Data Preparation and Lookup Mapping
    gold_map = {}
    text_map = {}
    for doc in gold_data:
        doc_id = doc["doc_id"]
        text_map[doc_id] = doc.get("text", "")
        gold_map[doc_id] = {ent["text"].lower().strip() for ent in doc.get("true_entities", [])}

    # Identifying 1-Token False Positives with Context
    false_positives = []
    
    for doc in pred_data:
        doc_id = doc["doc_id"]
        true_ents = gold_map.get(doc_id, set())
        original_text = text_map.get(doc_id, "")
        
        for ent in doc.get("predicted_entities", []):
            text = ent["text"].lower().strip()
            label = ent["label"]
            
            if len(text.split()) == 1:
                if text not in true_ents:
                    start_idx = max(0, ent["start"] - 40)
                    end_idx = min(len(original_text), ent["end"] + 40)
                    snippet = f"...{original_text[start_idx:end_idx]}..."
                    
                    false_positives.append({
                        "doc_id": doc_id,
                        "word": text,
                        "label": label,
                        "context": snippet.replace('\n', ' ')
                    })

    # Summary Statistics and CSV Export
    df_fp = pd.DataFrame(false_positives)
    
    if not df_fp.empty:
        top_hallucinations = df_fp['word'].value_counts().head(20)
        print("\n=== TOP 20 MOST HALLUCINATED 1-TOKEN WORDS ===")
        print(top_hallucinations)
        
        df_fp.to_csv("gemini_1_token_errors.csv", index=False)
        print("\nExported full context to 'gemini_1_token_errors.csv'")
    else:
        print("No 1-token false positives found.")

# Main Execution
if __name__ == '__main__':
    run_fp_extraction(
        gold_file='project/test_hard.json', 
        pred_file='project/gemini_predictions_hard.json', 
        token_csv='project/token_analysis.csv'
    )