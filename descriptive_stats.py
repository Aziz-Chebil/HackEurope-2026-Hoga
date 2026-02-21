"""Descriptive statistics & visualizations for TwiBot-20 dataset.

Run:  python descriptive_stats.py
Outputs plots to  stats_output/
"""

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "archive"
OUT_DIR = Path(__file__).parent / "stats_output"
OUT_DIR.mkdir(exist_ok=True)

COLLECTION_DATE = datetime(2022, 2, 1)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_int(val, default=0):
    if val is None or str(val).strip() in ("None", ""):
        return default
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return default


def _safe_str(val, default=""):
    if val is None or str(val).strip() in ("None", ""):
        return default
    return str(val).strip()


def _safe_bool(val, default=False):
    if val is None or str(val).strip() in ("None", ""):
        return default
    return str(val).strip().lower() in ("true", "1", "yes")


def _parse_age_days(val):
    s = _safe_str(val)
    if not s:
        return np.nan
    try:
        dt = datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
        age = (COLLECTION_DATE - dt.replace(tzinfo=None)).total_seconds() / 86400
        return max(age, 0.0)
    except (ValueError, TypeError):
        return np.nan


# ── Load data ────────────────────────────────────────────────────────────────

def load_all():
    """Load train/dev/test JSON and return a single DataFrame."""
    records = []
    for fname, split in [("train.json", "train"), ("dev.json", "dev"), ("test.json", "test")]:
        path = DATA_DIR / fname
        print(f"Loading {fname}...")
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            profile = entry.get("profile", {})
            tweets = entry.get("tweet", [])
            neighbor = entry.get("neighbor") or {}

            followers = _safe_int(profile.get("followers_count"))
            friends = _safe_int(profile.get("friends_count"))
            statuses = _safe_int(profile.get("statuses_count"))
            favourites = _safe_int(profile.get("favourites_count"))
            listed = _safe_int(profile.get("listed_count"))
            age_days = _parse_age_days(profile.get("created_at"))

            screen_name = _safe_str(profile.get("screen_name"))
            description = _safe_str(profile.get("description"))

            records.append({
                "id": entry["ID"],
                "label": int(entry["label"]),
                "split": split,
                # Counts
                "followers_count": followers,
                "friends_count": friends,
                "statuses_count": statuses,
                "favourites_count": favourites,
                "listed_count": listed,
                # Ratios
                "follower_friend_ratio": followers / (friends + 1),
                "statuses_per_follower": statuses / (followers + 1),
                "listed_per_follower": listed / (followers + 1),
                "tweets_per_day": statuses / (age_days + 1) if not np.isnan(age_days) else np.nan,
                # Account metadata
                "age_days": age_days,
                "name_length": len(_safe_str(profile.get("name"))),
                "screen_name_length": len(screen_name),
                "description_length": len(description),
                "digits_in_screen_name": sum(c.isdigit() for c in screen_name),
                # Binary
                "verified": _safe_bool(profile.get("verified")),
                "default_profile": _safe_bool(profile.get("default_profile")),
                "default_profile_image": _safe_bool(profile.get("default_profile_image")),
                "has_url": len(_safe_str(profile.get("url"))) > 0,
                "has_location": len(_safe_str(profile.get("location"))) > 0,
                "has_description": len(description) > 0,
                # Graph / tweet counts
                "num_tweets_available": len(tweets) if tweets else 0,
                "num_following": len(neighbor.get("following", [])),
                "num_follower_neighbors": len(neighbor.get("follower", [])),
            })
    df = pd.DataFrame(records)
    df["label_str"] = df["label"].map({0: "Human", 1: "Bot"})
    return df


# ── Plotting ─────────────────────────────────────────────────────────────────

def set_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })


def plot_class_split_distribution(df):
    """Bar chart: class balance per split."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Class balance overall
    counts = df["label_str"].value_counts()
    colors = ["#4C9BE8", "#E8724C"]
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 50, f"{v:,}", ha="center", fontweight="bold")
    axes[0].set_title("Class Distribution (all splits)")
    axes[0].set_ylabel("Count")

    # (b) Per-split breakdown
    ct = df.groupby(["split", "label_str"]).size().unstack(fill_value=0)
    ct = ct.reindex(["train", "dev", "test"])
    ct.plot.bar(ax=axes[1], color=colors, edgecolor="white", linewidth=1.2)
    axes[1].set_title("Class Distribution per Split")
    axes[1].set_ylabel("Count")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(title="Class")

    fig.suptitle("1. Class & Split Balance", fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(OUT_DIR / "01_class_split_distribution.png")
    plt.close(fig)
    print("  -> 01_class_split_distribution.png")


def plot_numeric_distributions(df):
    """Histograms of key numeric features, split by label."""
    features = [
        ("followers_count", "Followers Count (log scale)", True),
        ("friends_count", "Friends Count (log scale)", True),
        ("statuses_count", "Statuses Count (log scale)", True),
        ("favourites_count", "Favourites Count (log scale)", True),
        ("listed_count", "Listed Count (log scale)", True),
        ("age_days", "Account Age (days)", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for ax, (col, title, use_log) in zip(axes, features):
        for label, color in [("Human", "#4C9BE8"), ("Bot", "#E8724C")]:
            vals = df.loc[df["label_str"] == label, col].dropna()
            if use_log:
                vals = vals.clip(lower=1)
                ax.hist(np.log10(vals), bins=50, alpha=0.55, label=label, color=color, edgecolor="white")
                ax.set_xlabel("log10(value)")
            else:
                ax.hist(vals, bins=50, alpha=0.55, label=label, color=color, edgecolor="white")
            ax.set_title(title)
        ax.legend()

    fig.suptitle("2. Numeric Feature Distributions (Human vs Bot)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_numeric_distributions.png")
    plt.close(fig)
    print("  -> 02_numeric_distributions.png")


def plot_ratio_distributions(df):
    """KDE plots for derived ratio features."""
    ratios = [
        ("follower_friend_ratio", "Follower / Friend Ratio"),
        ("statuses_per_follower", "Statuses / Follower"),
        ("tweets_per_day", "Tweets per Day"),
        ("listed_per_follower", "Listed / Follower"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()

    for ax, (col, title) in zip(axes, ratios):
        for label, color in [("Human", "#4C9BE8"), ("Bot", "#E8724C")]:
            vals = df.loc[df["label_str"] == label, col].dropna()
            # Clip extreme outliers for better visualization
            q99 = vals.quantile(0.99)
            vals_clipped = vals.clip(upper=q99)
            vals_clipped.plot.kde(ax=ax, label=label, color=color, linewidth=2)
        ax.set_title(title)
        ax.legend()
        ax.set_ylabel("Density")

    fig.suptitle("3. Ratio Feature Distributions (clipped at 99th pctl)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_ratio_distributions.png")
    plt.close(fig)
    print("  -> 03_ratio_distributions.png")


def plot_binary_features(df):
    """Grouped bar chart for binary features by class."""
    binary_cols = ["verified", "default_profile", "default_profile_image", "has_url", "has_location", "has_description"]
    rates = df.groupby("label_str")[binary_cols].mean().T
    rates.columns.name = None

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(binary_cols))
    width = 0.35
    bars1 = ax.bar(x - width / 2, rates["Human"], width, label="Human", color="#4C9BE8", edgecolor="white")
    bars2 = ax.bar(x + width / 2, rates["Bot"], width, label="Bot", color="#E8724C", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in binary_cols], rotation=25, ha="right")
    ax.set_ylabel("Proportion (True)")
    ax.set_title("4. Binary Feature Rates by Class", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Annotate
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_binary_features.png")
    plt.close(fig)
    print("  -> 04_binary_features.png")


def plot_correlation_heatmap(df):
    """Correlation heatmap of numeric features."""
    numeric_cols = [
        "followers_count", "friends_count", "statuses_count", "favourites_count",
        "listed_count", "follower_friend_ratio", "statuses_per_follower",
        "listed_per_follower", "tweets_per_day", "age_days",
        "name_length", "screen_name_length", "description_length",
        "digits_in_screen_name",
    ]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.8})
    ax.set_title("5. Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_correlation_heatmap.png")
    plt.close(fig)
    print("  -> 05_correlation_heatmap.png")


def plot_screen_name_analysis(df):
    """Screen name length & digit count distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Screen name length
    for label, color in [("Human", "#4C9BE8"), ("Bot", "#E8724C")]:
        vals = df.loc[df["label_str"] == label, "screen_name_length"]
        axes[0].hist(vals, bins=30, alpha=0.55, label=label, color=color, edgecolor="white")
    axes[0].set_title("Screen Name Length")
    axes[0].set_xlabel("Length")
    axes[0].legend()

    # (b) Digits in screen name
    for label, color in [("Human", "#4C9BE8"), ("Bot", "#E8724C")]:
        vals = df.loc[df["label_str"] == label, "digits_in_screen_name"]
        axes[1].hist(vals, bins=range(0, 16), alpha=0.55, label=label, color=color, edgecolor="white", density=True)
    axes[1].set_title("Digits in Screen Name (normalized)")
    axes[1].set_xlabel("Number of digits")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    fig.suptitle("6. Screen Name Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_screen_name_analysis.png")
    plt.close(fig)
    print("  -> 06_screen_name_analysis.png")


def plot_tweet_and_neighbor_stats(df):
    """Number of tweets available and neighbor counts."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Tweets available
    for label, color in [("Human", "#4C9BE8"), ("Bot", "#E8724C")]:
        vals = df.loc[df["label_str"] == label, "num_tweets_available"]
        axes[0].hist(vals, bins=40, alpha=0.55, label=label, color=color, edgecolor="white")
    axes[0].set_title("Tweets Available per User")
    axes[0].set_xlabel("# Tweets")
    axes[0].legend()

    # (b) Following neighbors
    for label, color in [("Human", "#4C9BE8"), ("Bot", "#E8724C")]:
        vals = df.loc[df["label_str"] == label, "num_following"]
        axes[1].hist(vals, bins=range(0, 12), alpha=0.55, label=label, color=color, edgecolor="white", density=True)
    axes[1].set_title("Following Neighbors (in graph)")
    axes[1].set_xlabel("# Following")
    axes[1].legend()

    # (c) Follower neighbors
    for label, color in [("Human", "#4C9BE8"), ("Bot", "#E8724C")]:
        vals = df.loc[df["label_str"] == label, "num_follower_neighbors"]
        axes[2].hist(vals, bins=range(0, 12), alpha=0.55, label=label, color=color, edgecolor="white", density=True)
    axes[2].set_title("Follower Neighbors (in graph)")
    axes[2].set_xlabel("# Followers")
    axes[2].legend()

    fig.suptitle("7. Tweet & Graph Neighbor Statistics", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_tweet_neighbor_stats.png")
    plt.close(fig)
    print("  -> 07_tweet_neighbor_stats.png")


def plot_boxplots_key_features(df):
    """Box plots for the most discriminative features."""
    features = [
        "followers_count", "friends_count", "statuses_count",
        "follower_friend_ratio", "tweets_per_day", "age_days",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for ax, col in zip(axes, features):
        data_h = df.loc[df["label_str"] == "Human", col].dropna()
        data_b = df.loc[df["label_str"] == "Bot", col].dropna()
        # Clip to 95th percentile for visibility
        q95 = df[col].dropna().quantile(0.95)
        bp = ax.boxplot(
            [data_h.clip(upper=q95), data_b.clip(upper=q95)],
            tick_labels=["Human", "Bot"],
            patch_artist=True,
            widths=0.5,
        )
        bp["boxes"][0].set_facecolor("#4C9BE8")
        bp["boxes"][1].set_facecolor("#E8724C")
        for box in bp["boxes"]:
            box.set_alpha(0.7)
        ax.set_title(col.replace("_", " ").title())

    fig.suptitle("8. Key Feature Box Plots (clipped at 95th pctl)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_boxplots_key_features.png")
    plt.close(fig)
    print("  -> 08_boxplots_key_features.png")


def print_summary_stats(df):
    """Print textual summary to console and save to file."""
    lines = []
    lines.append("=" * 70)
    lines.append("TWIBOT-20 DATASET — DESCRIPTIVE STATISTICS")
    lines.append("=" * 70)

    lines.append(f"\nTotal users: {len(df):,}")
    lines.append(f"  Train: {(df['split'] == 'train').sum():,}")
    lines.append(f"  Dev:   {(df['split'] == 'dev').sum():,}")
    lines.append(f"  Test:  {(df['split'] == 'test').sum():,}")

    lines.append(f"\nClass balance:")
    for label in ["Human", "Bot"]:
        n = (df["label_str"] == label).sum()
        lines.append(f"  {label}: {n:,}  ({100 * n / len(df):.1f}%)")

    lines.append(f"\n{'─' * 70}")
    lines.append("NUMERIC FEATURES — Summary Statistics")
    lines.append(f"{'─' * 70}")

    numeric_cols = [
        "followers_count", "friends_count", "statuses_count", "favourites_count",
        "listed_count", "follower_friend_ratio", "statuses_per_follower",
        "listed_per_follower", "tweets_per_day", "age_days",
        "name_length", "screen_name_length", "description_length",
        "digits_in_screen_name",
    ]

    desc = df[numeric_cols].describe().T
    desc["missing"] = df[numeric_cols].isna().sum()
    lines.append(desc.to_string())

    lines.append(f"\n{'─' * 70}")
    lines.append("NUMERIC FEATURES — By Class (median)")
    lines.append(f"{'─' * 70}")
    median_by_class = df.groupby("label_str")[numeric_cols].median().T
    lines.append(median_by_class.to_string())

    lines.append(f"\n{'─' * 70}")
    lines.append("BINARY FEATURES — Proportion by Class")
    lines.append(f"{'─' * 70}")
    binary_cols = ["verified", "default_profile", "default_profile_image", "has_url", "has_location", "has_description"]
    binary_rates = df.groupby("label_str")[binary_cols].mean().T
    lines.append(binary_rates.to_string())

    text = "\n".join(lines)
    print(text)

    summary_path = OUT_DIR / "summary_stats.txt"
    summary_path.write_text(text, encoding="utf-8")
    print(f"\n  -> summary_stats.txt saved")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    set_style()
    df = load_all()

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")

    print("Generating plots...")
    plot_class_split_distribution(df)
    plot_numeric_distributions(df)
    plot_ratio_distributions(df)
    plot_binary_features(df)
    plot_correlation_heatmap(df)
    plot_screen_name_analysis(df)
    plot_tweet_and_neighbor_stats(df)
    plot_boxplots_key_features(df)

    print("\nComputing summary statistics...")
    print_summary_stats(df)

    print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()