import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
                "text": ent.get("text").lower().strip() # Normalize text
            })
    return grouped

def analyze_fragmentation(gold_file, pred_file, token_csv, model_name):
    # 1. Load the token mapping from your CSV
    df_tokens = pd.read_csv(token_csv)
    
    # We will use BioBERT's token count for the bucketing baseline
    token_lookup = dict(zip(df_tokens['entity_text'].str.lower().str.strip(), df_tokens['biobert_token_count']))

    # 2. Load the predictions and gold standard
    gold_grouped = get_entities_by_doc(load_json(gold_file))
    pred_grouped = get_entities_by_doc(load_json(pred_file))

    # Trackers for our buckets
    # Structure: {"1 Token": {"TP": 0, "FP": 0, "FN": 0}, ...}
    buckets = {
        "1 Token": {"TP": 0, "FP": 0, "FN": 0},
        "2 Tokens": {"TP": 0, "FP": 0, "FN": 0},
        "3+ Tokens": {"TP": 0, "FP": 0, "FN": 0}
    }

    def get_bucket_name(text):
        count = token_lookup.get(text, 1) # Default to 1 if somehow missing
        if count == 1: return "1 Token"
        elif count == 2: return "2 Tokens"
        else: return "3+ Tokens"

    # 3. Calculate TP, FP, FN per token bucket (Strict Exact Match)
    all_doc_ids = set(gold_grouped.keys()).union(set(pred_grouped.keys()))

    for doc_id in all_doc_ids:
        gold_ents = gold_grouped[doc_id]
        pred_ents = pred_grouped[doc_id]

        gold_spans = {(e["start"], e["end"], e["text"]) for e in gold_ents}
        pred_spans = {(e["start"], e["end"], e["text"]) for e in pred_ents}

        # True Positives
        for span in gold_spans.intersection(pred_spans):
            bucket = get_bucket_name(span[2])
            buckets[bucket]["TP"] += 1

        # False Positives (Over-generation by the model)
        for span in pred_spans - gold_spans:
            bucket = get_bucket_name(span[2])
            buckets[bucket]["FP"] += 1

        # False Negatives (Missed by the model)
        for span in gold_spans - pred_spans:
            bucket = get_bucket_name(span[2])
            buckets[bucket]["FN"] += 1

    # 4. Calculate F1 for each bucket
    results = []
    for bucket_name, counts in buckets.items():
        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            "Model": model_name,
            "Token Bucket": bucket_name,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Total Entities": tp + fn # How many gold entities actually existed in this bucket
        })

    return pd.DataFrame(results)

if __name__ == '__main__':
    # Analyze Gemini on the Hard subset
    gemini_df = analyze_fragmentation(
        gold_file='project/test_hard.json', 
        pred_file='project/gemini_predictions_hard.json', 
        token_csv='project/token_analysis.csv', 
        model_name='Gemini (Zero-Shot)'
    )
    
    # Optional: Analyze BioBERT on the Hard subset (Uncomment when you have the file)
    # biobert_df = analyze_fragmentation(
    #     gold_file='project/test_hard.json', 
    #     pred_file='project/biobert_predictions_hard.json', 
    #     token_csv='project/token_analysis.csv', 
    #     model_name='BioBERT'
    # )

    # Combine dataframes for plotting (If using both models)
    # final_df = pd.concat([gemini_df, biobert_df])
    final_df = gemini_df # Using just Gemini for now

    print("--- Fragmentation Analysis Results ---")
    print(final_df.to_string(index=False))

    # 5. Generate the Visualization
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # If plotting one model:
    ax = sns.barplot(x="Token Bucket", y="F1-Score", hue="Token Bucket", data=final_df, palette="Blues_d", order=["1 Token", "2 Tokens", "3+ Tokens"], legend=False)
    
    # If plotting BOTH models (uncomment this and comment the line above):
    # ax = sns.barplot(x="Token Bucket", y="F1-Score", hue="Model", data=final_df, palette="muted", order=["1 Token", "2 Tokens", "3+ Tokens"])

    plt.title('Impact of Sub-Word Fragmentation on NER Performance (Hard Subset)', fontsize=14, fontweight='bold')
    plt.ylabel('Strict F1-Score', fontsize=12)
    plt.xlabel('BioBERT Token Count per Entity', fontsize=12)
    plt.ylim(0, 1.0)
    
    # Add the F1 scores on top of the bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

    plt.tight_layout()
    plt.savefig('fragmentation_impact.png', dpi=300)
    print("\nChart successfully saved as 'fragmentation_impact.png'")