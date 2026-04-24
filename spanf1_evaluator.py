import json

def extract_spans_from_json(data):
    spans = set()
    for doc in data:
        doc_id = doc.get("doc_id")
        entities = doc.get("predicted_entities", doc.get("entities", []))
        
        for ent in entities:
            span = (doc_id, ent.get("label"), ent.get("start"), ent.get("end"))
            spans.add(span)
            
    return spans

def calculate_strict_span_f1(gold_data, pred_data):
    gold_spans = extract_spans_from_json(gold_data)
    pred_spans = extract_spans_from_json(pred_data)
    
    tp = len(gold_spans.intersection(pred_spans)) 
    fp = len(pred_spans - gold_spans)
    fn = len(gold_spans - pred_spans)            
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4)
    }

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == '__main__':
    gold_easy = load_json('project/test_easy.json')
    gold_hard = load_json('project/test_hard.json')

    pred_gemini_easy = load_json('project/gemini_predictions_easy.json')
    pred_gemini_hard = load_json('project/gemini_predictions_hard.json')
    
    pred_biobert_easy = load_json('project/biobert_predictions_easy.json')
    pred_biobert_hard = load_json('project/biobert_predictions_hard.json')

    print("Gemini - Easy Subset:")
    print(json.dumps(calculate_strict_span_f1(gold_easy, pred_gemini_easy), indent=4))
    
    print("\nGemini - Hard Subset:")
    print(json.dumps(calculate_strict_span_f1(gold_hard, pred_gemini_hard), indent=4))

    print("\nBioBERT - Easy Subset:")
    print(json.dumps(calculate_strict_span_f1(gold_easy, pred_biobert_easy), indent=4))
    
    print("\nBioBERT - Hard Subset:")
    print(json.dumps(calculate_strict_span_f1(gold_hard, pred_biobert_hard), indent=4))