"""Test predict.py on the full dataset and display probabilities."""

import json
import sys
from pathlib import Path

import numpy as np

from config import Config
from predict import predict


def main(model_type: str = "rgcn"):
    cfg = Config()
    cfg.resolve_data_dir()
    data_dir = Path(cfg.data_dir)

    # Load all splits
    all_rows = []
    for split_file, split_name in [
        ("train.json", "train"),
        ("dev.json", "val"),
        ("test.json", "test"),
    ]:
        path = data_dir / split_file
        with open(path) as f:
            rows = json.load(f)
        for r in rows:
            r["_split"] = split_name
        all_rows.extend(rows)

    print(f"Total users: {len(all_rows)}")
    print(f"Model: {model_type}\n")

    # Run prediction
    probs = predict(all_rows, model_type=model_type)  # (N, 2)

    # Display results
    print(f"{'User ID':<20} {'Split':<8} {'Label':<8} {'P(human)':<10} {'P(bot)':<10} {'Pred':<6}")
    print("-" * 70)

    correct = 0
    total = 0
    split_stats = {"train": {"correct": 0, "total": 0},
                   "val": {"correct": 0, "total": 0},
                   "test": {"correct": 0, "total": 0}}

    for i, row in enumerate(all_rows):
        uid = str(row["ID"])
        label = int(row["label"])
        split = row["_split"]
        p_human = probs[i, 0]
        p_bot = probs[i, 1]
        pred = 1 if p_bot >= 0.5 else 0

        label_str = "bot" if label == 1 else "human"
        pred_str = "bot" if pred == 1 else "human"
        marker = "OK" if pred == label else "XX"

        #print(f"{uid:<20} {split:<8} {label_str:<8} {p_human:<10.4f} {p_bot:<10.4f} {pred_str:<6} {marker}")

        total += 1
        split_stats[split]["total"] += 1
        if pred == label:
            correct += 1
            split_stats[split]["correct"] += 1

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Split':<10} {'Accuracy':<12} {'Correct':<10} {'Total':<10}")
    print("-" * 42)
    for split in ("train", "val", "test"):
        s = split_stats[split]
        if s["total"] > 0:
            acc = s["correct"] / s["total"]
            print(f"{split:<10} {acc:<12.4f} {s['correct']:<10} {s['total']:<10}")
    overall_acc = correct / total if total > 0 else 0
    print(f"{'overall':<10} {overall_acc:<12.4f} {correct:<10} {total:<10}")

    # Probability distribution stats
    p_bot_all = probs[:, 1]
    print(f"\nP(bot) statistics:")
    print(f"  Mean:   {np.mean(p_bot_all):.4f}")
    print(f"  Std:    {np.std(p_bot_all):.4f}")
    print(f"  Min:    {np.min(p_bot_all):.4f}")
    print(f"  Max:    {np.max(p_bot_all):.4f}")
    print(f"  Median: {np.median(p_bot_all):.4f}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "rgcn"
    main(model_type=model)