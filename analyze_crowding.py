"""
Crowding/Uncrowding Experiment - Results Analysis Script
Calculates accuracy per decoder, per ShapeCode, and visualizes crowding effects.
Place this file in the project root (D:\mindset-vision\) and run:
    python analyze_crowding.py
"""

import pandas as pd
import numpy as np
import os

# ============================================================
# 1. Load predictions
# ============================================================
inside_path = "results/decoder/crowding/eval/inside_vernier/predictions.csv"
outside_path = "results/decoder/crowding/eval/outside_vernier_test/predictions.csv"

df_inside = pd.read_csv(inside_path)
df_outside = pd.read_csv(outside_path)

# Number of decoders
num_decoders = len([c for c in df_inside.columns if c.startswith("prediction_dec_")])

# ============================================================
# 2. Overall accuracy per decoder (outside vs inside)
# ============================================================
print("=" * 60)
print("OVERALL ACCURACY PER DECODER")
print("=" * 60)
print(f"{'Decoder':<12} {'Outside':>10} {'Inside':>10} {'Difference':>12}")
print("-" * 46)

for i in range(num_decoders):
    col = f"prediction_dec_{i}"
    acc_out = (df_outside["label"] == df_outside[col]).mean()
    acc_in = (df_inside["label"] == df_inside[col]).mean()
    diff = acc_out - acc_in
    print(f"Decoder {i}:   {acc_out:>9.1%} {acc_in:>9.1%} {diff:>+11.1%}")

# ============================================================
# 3. Extract ShapeCode from image path
# ============================================================
def extract_shape_code(path):
    """Extract shape code from filename like 'inside/0/11111_0_abc123.png'"""
    filename = os.path.basename(path)
    # Shape code is everything before the first underscore followed by a digit and underscore
    parts = filename.split("_")
    return parts[0]

df_inside["ShapeCode"] = df_inside["image_path"].apply(extract_shape_code)

# ============================================================
# 4. Categorize flanker configurations
# ============================================================
def categorize_flanker(code):
    """Map ShapeCode to a human-readable category"""
    if code == "none":
        return "0_no_flanker"
    
    # Remove 'nl' (newline markers for multi-row patterns) and count elements
    clean = code.replace("nl", "")
    num_elements = len(clean)
    
    # Check if all same shape
    unique_shapes = set(clean)
    
    if num_elements == 1:
        return "1_single_flanker"
    elif num_elements == 3:
        return "2_three_flankers"
    elif num_elements == 5:
        return "3_five_flankers"
    elif num_elements == 7:
        return "4_seven_flankers"
    elif num_elements > 7:
        return "5_multi_row"
    else:
        return "6_other"

df_inside["FlankerCategory"] = df_inside["ShapeCode"].apply(categorize_flanker)

# ============================================================
# 5. Accuracy per FlankerCategory per Decoder
# ============================================================
print("\n" + "=" * 60)
print("ACCURACY BY FLANKER CONFIGURATION (per decoder)")
print("=" * 60)

for i in range(num_decoders):
    col = f"prediction_dec_{i}"
    df_inside[f"correct_{i}"] = (df_inside["label"] == df_inside[col]).astype(int)

print(f"\n{'Category':<22}", end="")
for i in range(num_decoders):
    print(f"{'Dec_' + str(i):>8}", end="")
print(f"{'Count':>8}")
print("-" * (22 + 8 * num_decoders + 8))

for cat in sorted(df_inside["FlankerCategory"].unique()):
    subset = df_inside[df_inside["FlankerCategory"] == cat]
    print(f"{cat:<22}", end="")
    for i in range(num_decoders):
        acc = subset[f"correct_{i}"].mean()
        print(f"{acc:>7.1%}", end="")
    print(f"{len(subset):>8}")

# ============================================================
# 6. Summary
# ============================================================
print("\n" + "=" * 60)
print("INTERPRETATION GUIDE")
print("=" * 60)
print("- Outside acc high, Inside acc low = CROWDING effect")
print("- Single flanker acc < No flanker acc = CROWDING")
print("- Multi flanker acc > Single flanker acc = UNCROWDING")
print("- If no uncrowding: model lacks Gestalt grouping ability")

# ============================================================
# 7. Save detailed results
# ============================================================
output_path = "results/decoder/crowding/eval/analysis_summary.csv"
summary_rows = []
for cat in sorted(df_inside["FlankerCategory"].unique()):
    subset = df_inside[df_inside["FlankerCategory"] == cat]
    row = {"FlankerCategory": cat, "Count": len(subset)}
    for i in range(num_decoders):
        row[f"acc_dec_{i}"] = subset[f"correct_{i}"].mean()
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(output_path, index=False)
print(f"\nDetailed results saved to: {output_path}")