import json
import pandas as pd
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_entities_by_doc(data):
    """Groups entities by document ID for easier span matching."""
    grouped = defaultdict(list)
    for doc in data:
        doc_id = doc.get("doc_id")
        entities = doc.get("predicted_entities", doc.get("true_entities", []))
        for ent in entities:
            grouped[doc_id].append({
                "start": ent.get("start"),
                "end": ent.get("end"),
                "text": ent.get("text").lower().strip() 
            })
    return grouped

def analyze_fragmentation_dual(gold_file, pred_file, token_csv, model_name):
    # 1. Load the token mapping
    df_tokens = pd.read_csv(token_csv)
    token_lookup = dict(zip(df_tokens['entity_text'].str.lower().str.strip(), df_tokens['biobert_token_count']))

    gold_grouped = get_entities_by_doc(load_json(gold_file))
    pred_grouped = get_entities_by_doc(load_json(pred_file))

    # Trackers for Strict AND Relaxed
    buckets = {
        "1 Token":   {"Strict TP": 0, "Strict FP": 0, "Strict FN": 0, "Relaxed TP": 0, "Relaxed FP": 0, "Relaxed FN": 0},
        "2 Tokens":  {"Strict TP": 0, "Strict FP": 0, "Strict FN": 0, "Relaxed TP": 0, "Relaxed FP": 0, "Relaxed FN": 0},
        "3+ Tokens": {"Strict TP": 0, "Strict FP": 0, "Strict FN": 0, "Relaxed TP": 0, "Relaxed FP": 0, "Relaxed FN": 0}
    }

    def get_bucket_name(text):
        count = token_lookup.get(text, 1) 
        if count == 1: return "1 Token"
        elif count == 2: return "2 Tokens"
        else: return "3+ Tokens"

    all_doc_ids = set(gold_grouped.keys()).union(set(pred_grouped.keys()))

    for doc_id in all_doc_ids:
        gold_ents = gold_grouped[doc_id]
        pred_ents = pred_grouped[doc_id]

        # ==============================
        # A. STRICT EXACT MATCHING
        # ==============================
        gold_spans = {(e["start"], e["end"], e["text"]) for e in gold_ents}
        pred_spans = {(e["start"], e["end"], e["text"]) for e in pred_ents}

        for span in gold_spans.intersection(pred_spans):
            buckets[get_bucket_name(span[2])]["Strict TP"] += 1
        for span in pred_spans - gold_spans:
            buckets[get_bucket_name(span[2])]["Strict FP"] += 1
        for span in gold_spans - pred_spans:
            buckets[get_bucket_name(span[2])]["Strict FN"] += 1

        # ==============================
        # B. RELAXED OVERLAP MATCHING
        # ==============================
        matched_golds = set()
        for p_ent in pred_ents:
            p_start, p_end = p_ent["start"], p_ent["end"]
            
            # Find any gold entities this prediction overlaps with
            overlapped_golds = [g for g in gold_ents if max(p_start, g["start"]) < min(p_end, g["end"])]
            
            if overlapped_golds:
                # It overlapped! Assign TP to the underlying concept's bucket
                bucket = get_bucket_name(overlapped_golds[0]["text"])
                buckets[bucket]["Relaxed TP"] += 1
                for g in overlapped_golds:
                    matched_golds.add((g["start"], g["end"], g["text"]))
            else:
                # No overlap. It's a hallucination. Assign FP to the predicted text's bucket
                bucket = get_bucket_name(p_ent["text"])
                buckets[bucket]["Relaxed FP"] += 1

        # Check for missed gold entities
        for g_ent in gold_ents:
            g_tuple = (g_ent["start"], g_ent["end"], g_ent["text"])
            if g_tuple not in matched_golds:
                bucket = get_bucket_name(g_ent["text"])
                buckets[bucket]["Relaxed FN"] += 1

    # ==============================
    # C. CALCULATE METRICS
    # ==============================
    results = []
    for bucket_name, counts in buckets.items():
        # Strict Math
        s_tp, s_fp, s_fn = counts["Strict TP"], counts["Strict FP"], counts["Strict FN"]
        s_p = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
        s_r = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
        s_f1 = 2 * (s_p * s_r) / (s_p + s_r) if (s_p + s_r) > 0 else 0.0
        
        results.append({
            "Model": model_name,
            "Token Bucket": bucket_name,
            "Match Type": "Strict Match",
            "F1-Score": round(s_f1, 4)
        })

        # Relaxed Math
        r_tp, r_fp, r_fn = counts["Relaxed TP"], counts["Relaxed FP"], counts["Relaxed FN"]
        r_p = r_tp / (r_tp + r_fp) if (r_tp + r_fp) > 0 else 0.0
        r_r = r_tp / (r_tp + r_fn) if (r_tp + r_fn) > 0 else 0.0
        r_f1 = 2 * (r_p * r_r) / (r_p + r_r) if (r_p + r_r) > 0 else 0.0
        
        results.append({
            "Model": model_name,
            "Token Bucket": bucket_name,
            "Match Type": "Relaxed (Overlap)",
            "F1-Score": round(r_f1, 4)
        })

    return pd.DataFrame(results)

if __name__ == '__main__':
    # Analyze Gemini 
    gemini_df = analyze_fragmentation_dual(
        gold_file='project/test_hard.json', 
        pred_file='project/gemini_predictions_hard.json', 
        token_csv='project/token_analysis.csv', 
        model_name='Gemini (Zero-Shot)'
    )
    
    # Analyze BioBERT 
    biobert_df = analyze_fragmentation_dual(
        gold_file='project/test_hard.json', 
        pred_file='project/biobert_predictions_hard.json', 
        token_csv='project/token_analysis.csv', 
        model_name='BioBERT'
    )

    # Analyze Standard BERT
    bert_df = analyze_fragmentation_dual(
        gold_file='project/test_hard.json', 
        pred_file='project/bert_predictions_hard.json', 
        token_csv='project/token_analysis.csv', 
        model_name='Standard BERT'
    )

    # Combine dataframes for all three models
    final_df = pd.concat([bert_df, biobert_df, gemini_df])

    print("=== FINAL DUAL FRAGMENTATION RESULTS ===")
    print(final_df.to_string(index=False))