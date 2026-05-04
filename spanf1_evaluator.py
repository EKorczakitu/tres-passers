import json
from collections import defaultdict

def get_entities_by_doc_and_label(data):
    """
    Groups entities into a dictionary where the key is (doc_id, label)
    and the value is a list of entity dictionaries containing start/end indexes.
    """
    grouped = defaultdict(list)
    for doc in data:
        doc_id = doc.get("doc_id")
        # Handle both the gold standard key and the prediction key
        entities = doc.get("predicted_entities", doc.get("true_entities", []))
        
        for ent in entities:
            # We group by Document ID and Entity Label (e.g., 'Disease' or 'Chemical')
            key = (doc_id, ent.get("label"))
            grouped[key].append({
                "start": ent.get("start"),
                "end": ent.get("end"),
                "text": ent.get("text")
            })
            
    return grouped

def calculate_dual_metrics(gold_data, pred_data):
    """
    Calculates both Strict (Exact Match) and Relaxed (Overlap) F1 scores.
    """
    gold_grouped = get_entities_by_doc_and_label(gold_data)
    pred_grouped = get_entities_by_doc_and_label(pred_data)

    # Trackers
    strict_tp, strict_fp, strict_fn = 0, 0, 0
    relaxed_tp, relaxed_fp, relaxed_fn = 0, 0, 0

    # Get all unique (doc_id, label) combinations from both datasets
    all_keys = set(gold_grouped.keys()).union(set(pred_grouped.keys()))

    for key in all_keys:
        gold_ents = gold_grouped[key]
        pred_ents = pred_grouped[key]

        # ==========================================
        # 1. STRICT EVALUATION (Exact Boundaries)
        # ==========================================
        gold_spans = {(e["start"], e["end"]) for e in gold_ents}
        pred_spans = {(e["start"], e["end"]) for e in pred_ents}

        strict_tp += len(gold_spans.intersection(pred_spans))
        strict_fp += len(pred_spans - gold_spans)
        strict_fn += len(gold_spans - pred_spans)

        # ==========================================
        # 2. RELAXED EVALUATION (Overlapping Spans)
        # ==========================================
        matched_gold_indexes = set()

        for p_ent in pred_ents:
            p_start, p_end = p_ent["start"], p_ent["end"]
            has_overlap = False

            for g_idx, g_ent in enumerate(gold_ents):
                g_start, g_end = g_ent["start"], g_ent["end"]

                # The logic for span overlap: 
                # The highest start value must be lower than the lowest end value.
                if max(p_start, g_start) < min(p_end, g_end):
                    has_overlap = True
                    matched_gold_indexes.add(g_idx)

            if has_overlap:
                # Prediction overlapped with at least one true entity
                relaxed_tp += 1
            else:
                # Prediction touched absolutely nothing
                relaxed_fp += 1

        # Any gold entities that were NEVER touched by a prediction are False Negatives
        relaxed_fn += (len(gold_ents) - len(matched_gold_indexes))

    # Helper function to calculate final F1 maths
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

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == '__main__':
    # Load data (Update these paths if needed)
    gold_easy = load_json('project/test_easy.json')
    gold_hard = load_json('project/test_hard.json')

    pred_gemini_easy = load_json('project/gemini_predictions_easy.json')
    pred_gemini_hard = load_json('project/gemini_predictions_hard.json')
    
    # Optional: uncomment once you have your BioBERT JSONs
    # pred_biobert_easy = load_json('project/biobert_predictions_easy.json')
    # pred_biobert_hard = load_json('project/biobert_predictions_hard.json')

    print("=== GEMINI: EASY SUBSET ===")
    print(json.dumps(calculate_dual_metrics(gold_easy, pred_gemini_easy), indent=4))
    
    print("\n=== GEMINI: HARD SUBSET ===")
    print(json.dumps(calculate_dual_metrics(gold_hard, pred_gemini_hard), indent=4))

    # print("\n=== BIOBERT: EASY SUBSET ===")
    # print(json.dumps(calculate_dual_metrics(gold_easy, pred_biobert_easy), indent=4))
    
    # print("\n=== BIOBERT: HARD SUBSET ===")
    # print(json.dumps(calculate_dual_metrics(gold_hard, pred_biobert_hard), indent=4))
    
    
    """
    How to use this for your report discussion
By running this script, you will get two distinct F1 scores for every model. Because LLMs inherently struggle with precise character-level indexing compared to specialized models, this data gives you a powerful narrative for your analysis:  

If Strict F1 is low, but Relaxed F1 is high: This proves the model successfully understands the medical concepts and extracts the right terms, but it fails at token boundary precision (e.g., extracting "severe pyrexia" instead of just "pyrexia").

If both Strict F1 and Relaxed F1 are low: This indicates a complete knowledge failure or hallucination—the model is completely missing the entities or extracting entirely wrong concepts.
    """