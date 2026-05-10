import json
from collections import defaultdict

# Grouping utility by Document ID and Entity Label
def get_entities_by_doc_and_label(data):
    grouped = defaultdict(list)
    for doc in data:
        doc_id = doc.get("doc_id")
        entities = doc.get("predicted_entities", doc.get("true_entities", []))
        
        for ent in entities:
            key = (doc_id, ent.get("label"))
            grouped[key].append({
                "start": ent.get("start"),
                "end": ent.get("end"),
                "text": ent.get("text")
            })
            
    return grouped

# Core function for calculating Strict and Relaxed F1 scores
def calculate_dual_metrics(gold_data, pred_data):
    gold_grouped = get_entities_by_doc_and_label(gold_data)
    pred_grouped = get_entities_by_doc_and_label(pred_data)

    strict_tp, strict_fp, strict_fn = 0, 0, 0
    relaxed_tp, relaxed_fp, relaxed_fn = 0, 0, 0

    all_keys = set(gold_grouped.keys()).union(set(pred_grouped.keys()))

    for key in all_keys:
        gold_ents = gold_grouped[key]
        pred_ents = pred_grouped[key]

        # Strict Matching logic (Exact Boundaries)
        gold_spans = {(e["start"], e["end"]) for e in gold_ents}
        pred_spans = {(e["start"], e["end"]) for e in pred_ents}

        strict_tp += len(gold_spans.intersection(pred_spans))
        strict_fp += len(pred_spans - gold_spans)
        strict_fn += len(gold_spans - pred_spans)

        # Relaxed Matching logic (Character Overlap)
        matched_gold_indexes = set()

        for p_ent in pred_ents:
            p_start, p_end = p_ent["start"], p_ent["end"]
            has_overlap = False

            for g_idx, g_ent in enumerate(gold_ents):
                g_start, g_end = g_ent["start"], g_ent["end"]

                if max(p_start, g_start) < min(p_end, g_end):
                    has_overlap = True
                    matched_gold_indexes.add(g_idx)

            if has_overlap:
                relaxed_tp += 1
            else:
                relaxed_fp += 1

        relaxed_fn += (len(gold_ents) - len(matched_gold_indexes))

    # Math utility for Precision, Recall, and F1
    def calc_f1(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return round(precision, 4), round(recall, 4), round(f1, 4)

    s_p, s_r, s_f1 = calc_f1(strict_tp, strict_fp, strict_fn)
    r_p, r_r, r_f1 = calc_f1(relaxed_tp, relaxed_fp, relaxed_fn)

    return {
        "Strict Match": {
            "TP": strict_tp, "FP": strict_fp, "FN": strict_fn,
            "Precision": s_p, "Recall": s_r, "F1-Score": s_f1
        },
        "Relaxed Match (Overlap)": {
            "TP": relaxed_tp, "FP": relaxed_fp, "FN": relaxed_fn,
            "Precision": r_p, "Recall": r_r, "F1-Score": r_f1
        }
    }

# File loading utility
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# Main execution: Loading subsets and printing model performance
if __name__ == '__main__':
    gold_easy = load_json('project/test_easy.json')
    gold_hard = load_json('project/test_hard.json')

    pred_gemini_easy = load_json('project/gemini_predictions_easy.json')
    pred_gemini_hard = load_json('project/gemini_predictions_hard.json')
    
    pred_biobert_easy = load_json('project/biobert_predictions_easy.json')
    pred_biobert_hard = load_json('project/biobert_predictions_hard.json')
    
    pred_bert_easy = load_json('project/bert_predictions_easy.json')
    pred_bert_hard = load_json('project/bert_predictions_hard.json')
    
    print("=== GEMINI: EASY SUBSET ===")
    print(json.dumps(calculate_dual_metrics(gold_easy, pred_gemini_easy), indent=4))
    
    print("\n=== GEMINI: HARD SUBSET ===")
    print(json.dumps(calculate_dual_metrics(gold_hard, pred_gemini_hard), indent=4))

    print("\n=== BIOBERT: EASY SUBSET ===")
    print(json.dumps(calculate_dual_metrics(gold_easy, pred_biobert_easy), indent=4))
    
    print("\n=== BIOBERT: HARD SUBSET ===")
    print(json.dumps(calculate_dual_metrics(gold_hard, pred_biobert_hard), indent=4))
    
    print("\n=== BERT: EASY SUBSET ===")
    print(json.dumps(calculate_dual_metrics(gold_easy, pred_bert_easy), indent=4))

    print("\n=== BERT: HARD SUBSET ===")
    print(json.dumps(calculate_dual_metrics(gold_hard, pred_bert_hard), indent=4))