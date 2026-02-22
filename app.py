"""Streamlit frontend for TwiBot-20 bot detection."""

import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import matplotlib.patches as mpatches      # noqa: E402
import networkx as nx                      # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
import streamlit as st                     # noqa: E402
import torch                               # noqa: E402
import torch.nn.functional as F            # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

# ---------------------------------------------------------------------------
# Allow imports from graph/
# ---------------------------------------------------------------------------
GRAPH_DIR = str(Path(__file__).resolve().parent / "graph")
if GRAPH_DIR not in sys.path:
    sys.path.insert(0, GRAPH_DIR)

from config import Config            # noqa: E402
from data_loader import build_graph  # noqa: E402
from evaluate import evaluate_test   # noqa: E402
from models import build_model       # noqa: E402
from train import get_device         # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARCHIVE_DIR = Path(__file__).resolve().parent / "archive"
SPLITS = {
    "train": ARCHIVE_DIR / "train.json",
    "dev": ARCHIVE_DIR / "dev.json",
    "test": ARCHIVE_DIR / "test.json",
}

_HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)

FEATURE_NAMES = [
    "followers_count", "friends_count", "statuses_count", "favourites_count",
    "listed_count", "followers/friends ratio", "statuses/followers ratio",
    "listed/followers ratio", "age_days", "tweets_per_day",
    "name_length", "screen_name_length", "description_length",
    "digits_in_screen_name", "verified", "default_profile",
    "default_profile_image", "has_url", "has_location", "has_description",
]

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset…")
def load_all_users() -> tuple[list[dict], pd.DataFrame]:
    """Load every user from all splits and build a lightweight search index."""
    all_entries: list[dict] = []
    for split_name, path in SPLITS.items():
        with open(path) as f:
            entries = json.load(f)
        for entry in entries:
            entry["_split"] = split_name
        all_entries.extend(entries)

    rows = []
    for i, entry in enumerate(all_entries):
        profile = entry.get("profile", {})
        rows.append({
            "idx": i,
            "ID": str(entry["ID"]).strip(),
            "screen_name": str(profile.get("screen_name", "")).strip(),
            "name": str(profile.get("name", "")).strip(),
            "label": int(entry.get("label", -1)),
            "split": entry["_split"],
            "followers_count": int(str(profile.get("followers_count", 0)).strip() or 0),
            "verified": str(profile.get("verified", "")).strip().lower() == "true",
        })
    index_df = pd.DataFrame(rows)
    return all_entries, index_df


@st.cache_data(show_spinner="Building hashtag index…")
def build_hashtag_index(_all_entries: list[dict]) -> dict[str, list[int]]:
    hashtag_to_users: dict[str, list[int]] = {}
    for i, entry in enumerate(_all_entries):
        tweets = entry.get("tweet") or []
        seen_tags: set[str] = set()
        for t in tweets:
            if t is None:
                continue
            for tag in _HASHTAG_RE.findall(str(t)):
                seen_tags.add(tag.lower())
        for tag in seen_tags:
            hashtag_to_users.setdefault(tag, []).append(i)
    return hashtag_to_users


@st.cache_data(show_spinner="Collecting all hashtags…")
def get_all_hashtags(_all_entries: list[dict]) -> list[tuple[str, int]]:
    tag_counts: Counter = Counter()
    for entry in _all_entries:
        tweets = entry.get("tweet") or []
        seen_tags: set[str] = set()
        for t in tweets:
            if t is None:
                continue
            for tag in _HASHTAG_RE.findall(str(t)):
                seen_tags.add(tag.lower())
        for tag in seen_tags:
            tag_counts[tag] += 1
    return tag_counts.most_common()


@st.cache_data(show_spinner="Building domain index…")
def build_domain_index(_all_entries: list[dict]) -> dict[str, list[int]]:
    domain_to_users: dict[str, list[int]] = {}
    for i, entry in enumerate(_all_entries):
        for domain in entry.get("domain") or []:
            domain_to_users.setdefault(domain, []).append(i)
    return domain_to_users


# ---------------------------------------------------------------------------
# Model + graph caching
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading GNN model and graph (~30s)…")
def load_model_and_graph(model_type: str):
    """Load the full graph + trained model once. Cached across reruns."""
    cfg = Config()
    cfg.model_type = model_type
    checkpoint_path = os.path.join(cfg.checkpoint_dir, f"best_{model_type}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")
    data = build_graph(cfg)
    in_dim = data.x.shape[1]
    model = build_model(model_type, in_dim, cfg)
    device = get_device(cfg)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    data = data.to(device)
    model.eval()
    uid_to_idx = {uid: idx for idx, uid in enumerate(data.user_ids)}
    return model, data, cfg, uid_to_idx


@st.cache_data(show_spinner="Running inference on the full graph…")
def compute_all_probs(_model, _data, _cfg, model_type: str) -> np.ndarray:
    """Run inference once and cache the full probability matrix (N, 2)."""
    with torch.no_grad():
        edge_type = _data.edge_type if model_type == "rgcn" else None
        logits = _model(_data.x, _data.edge_index, edge_type)
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()


@st.cache_data(show_spinner="Evaluating model on test set…")
def cached_evaluate_test(_model, _data, _cfg, model_type: str) -> dict:
    return evaluate_test(_model, _data, _cfg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def search_users(index_df: pd.DataFrame, query: str, max_results: int = 50) -> pd.DataFrame:
    q = query.strip().lower()
    if not q:
        return index_df.head(0)
    mask = (
        index_df["screen_name"].str.lower().str.contains(q, na=False)
        | index_df["name"].str.lower().str.contains(q, na=False)
        | index_df["ID"].str.contains(q, na=False)
    )
    return index_df[mask].head(max_results)


def format_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _safe_int(val, default=0) -> int:
    if val is None or str(val).strip().lower() in ("none", ""):
        return default
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return default


def extract_raw_feature_values(entry: dict) -> list:
    """Extract the 20 raw (pre-normalization) feature values for display."""
    profile = entry.get("profile", {})
    followers = _safe_int(profile.get("followers_count"))
    friends = _safe_int(profile.get("friends_count"))
    statuses = _safe_int(profile.get("statuses_count"))
    favourites = _safe_int(profile.get("favourites_count"))
    listed = _safe_int(profile.get("listed_count"))
    name = str(profile.get("name") or "").strip()
    if name.lower() == "none":
        name = ""
    screen_name = str(profile.get("screen_name") or "").strip()
    if screen_name.lower() == "none":
        screen_name = ""
    description = str(profile.get("description") or "").strip()
    if description.lower() == "none":
        description = ""
    url = str(profile.get("url") or "").strip()
    if url.lower() == "none":
        url = ""
    location = str(profile.get("location") or "").strip()
    if location.lower() == "none":
        location = ""

    age_days = 0.0
    created_str = str(profile.get("created_at") or "").strip()
    if created_str and created_str.lower() != "none":
        try:
            dt = datetime.strptime(created_str, "%a %b %d %H:%M:%S %z %Y")
            collection = datetime(2022, 2, 1, tzinfo=dt.tzinfo)
            age_days = max((collection - dt).total_seconds() / 86400, 0.0)
        except ValueError:
            pass

    verified = str(profile.get("verified", "")).strip().lower() in ("true", "1")
    default_profile = str(profile.get("default_profile", "")).strip().lower() in ("true", "1")
    default_image = str(profile.get("default_profile_image", "")).strip().lower() in ("true", "1")

    return [
        followers,
        friends,
        statuses,
        favourites,
        listed,
        round(followers / (friends + 1), 2),
        round(statuses / (followers + 1), 2),
        round(listed / (followers + 1), 4),
        round(age_days),
        round(statuses / (age_days + 1), 2),
        len(name),
        len(screen_name),
        len(description),
        sum(c.isdigit() for c in screen_name),
        int(verified),
        int(default_profile),
        int(default_image),
        int(len(url) > 0),
        int(len(location) > 0),
        int(len(description) > 0),
    ]


def compute_feature_importance(model, data, cfg, node_idx: int):
    """Perturbation-based feature importance for a single node.

    Strategy: for each feature, negate its normalized value (flip sign).
    This turns a value 2σ above the mean into 2σ below the mean, which
    reveals the actual directional contribution of each feature — not just
    whether it differs from the mean.  Binary features (0/1) are flipped
    to their opposite value.
    """
    model.eval()
    x_backup = data.x[node_idx].clone()
    edge_type = data.edge_type if cfg.model_type == "rgcn" else None

    with torch.no_grad():
        logits_base = model(data.x, data.edge_index, edge_type)
        p_bot_base = F.softmax(logits_base, dim=1)[node_idx, 1].item()

    n_features = data.x.shape[1]
    importances = np.zeros(n_features)
    for j in range(n_features):
        original_val = data.x[node_idx, j].item()
        # Negate: flip the sign of the normalized value
        data.x[node_idx, j] = -original_val
        with torch.no_grad():
            logits_pert = model(data.x, data.edge_index, edge_type)
            p_bot_pert = F.softmax(logits_pert, dim=1)[node_idx, 1].item()
        importances[j] = p_bot_base - p_bot_pert
        data.x[node_idx, j] = original_val

    # Restore from backup to be safe
    data.x[node_idx] = x_backup
    return p_bot_base, importances


def generate_user_summary(entry: dict, p_bot: float, all_probs_arr, uid_to_idx_map) -> str:
    """Generate a short natural-language summary of a user profile."""
    profile = entry.get("profile", {})
    screen_name = str(profile.get("screen_name", "")).strip()
    name = str(profile.get("name", "")).strip()
    followers = _safe_int(profile.get("followers_count"))
    friends = _safe_int(profile.get("friends_count"))
    statuses = _safe_int(profile.get("statuses_count"))
    verified = str(profile.get("verified", "")).strip().lower() == "true"
    default_image = str(profile.get("default_profile_image", "")).strip().lower() == "true"
    desc = str(profile.get("description", "")).strip()
    if desc.lower() in ("none", ""):
        desc = ""
    domains = entry.get("domain") or []

    # Account age
    age_years = 0
    created_str = str(profile.get("created_at", "")).strip()
    if created_str and created_str.lower() != "none":
        try:
            dt = datetime.strptime(created_str, "%a %b %d %H:%M:%S %z %Y")
            collection = datetime(2022, 2, 1, tzinfo=dt.tzinfo)
            age_years = max((collection - dt).days / 365.25, 0)
        except ValueError:
            pass

    # Neighbor bot rate
    neighbor = entry.get("neighbor") or {}
    following_ids = [str(x) for x in (neighbor.get("following") or [])]
    follower_ids = [str(x) for x in (neighbor.get("follower") or [])]
    all_nids = list(set(following_ids + follower_ids))
    valid_nids = [nid for nid in all_nids if nid in uid_to_idx_map]
    n_neighbor_bots = sum(
        1 for nid in valid_nids
        if all_probs_arr[uid_to_idx_map[nid], 1] >= 0.5
    )

    # Build summary parts
    parts = []

    # Identity
    if name and name.lower() != "none":
        parts.append(f"**@{screen_name}** ({name})")
    else:
        parts.append(f"**@{screen_name}**")

    # Verdict
    if p_bot >= 0.8:
        parts.append(f"is predicted as a **bot** with high confidence ({p_bot:.0%}).")
    elif p_bot >= 0.5:
        parts.append(f"is predicted as a **probable bot** ({p_bot:.0%}).")
    elif p_bot >= 0.2:
        parts.append(f"is predicted as a **probable human** ({1 - p_bot:.0%} confidence).")
    else:
        parts.append(f"is predicted as **human** with high confidence ({1 - p_bot:.0%}).")

    summary = " ".join(parts)

    # Profile details
    details = []
    if verified:
        details.append("The account is **verified**.")
    if age_years >= 1:
        details.append(f"It was created **{age_years:.0f} years** before data collection.")
    details.append(
        f"It has **{format_number(followers)}** followers, "
        f"follows **{format_number(friends)}** accounts, "
        f"and has posted **{format_number(statuses)}** tweets."
    )
    if default_image:
        details.append("It still uses the **default profile image**, a common bot indicator.")
    if not desc:
        details.append("The account has **no bio**.")
    if domains:
        details.append(f"Active in: {', '.join(domains)}.")

    summary += " " + " ".join(details)

    # Neighbor context
    if valid_nids:
        neighbor_bot_pct = n_neighbor_bots / len(valid_nids) * 100
        summary += (
            f" In its social neighborhood, **{n_neighbor_bots}/{len(valid_nids)}** "
            f"connections ({neighbor_bot_pct:.0f}%) are also predicted as bots."
        )

    return summary


# ---------------------------------------------------------------------------
# UI setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TwiBot – Bot Detector",
    page_icon="🤖",
    layout="wide",
)

st.title("TwiBot – Bot Detector")
st.caption("Graph Neural Network bot detection on the TwiBot-20 dataset")

# Load data
all_entries, index_df = load_all_users()
hashtag_index = build_hashtag_index(all_entries)
domain_index = build_domain_index(all_entries)

# --- Sidebar (shared) ------------------------------------------------------
st.sidebar.header("Settings")
model_type = st.sidebar.radio(
    "Model",
    options=["rgcn", "gat"],
    horizontal=True,
    help="R-GCN uses relation-specific weights; GAT uses multi-head attention.",
)

# Load model + graph (cached)
model, data, cfg, uid_to_idx = load_model_and_graph(model_type)
all_probs = compute_all_probs(model, data, cfg, model_type)

st.sidebar.markdown("---")
st.sidebar.header("User Search")
query = st.sidebar.text_input(
    "Search by name, @handle or ID",
    placeholder="e.g. SHAQ, elonmusk, 17461978",
)

# Random user button
if st.sidebar.button("🎲 Random user"):
    st.session_state["random_idx"] = random.randint(0, len(all_entries) - 1)

results = search_users(index_df, query)

if query and results.empty:
    st.sidebar.warning("No users found.")

selected_idx = None
if not results.empty:
    options_list = []
    for _, row in results.iterrows():
        label_tag = "🤖 Bot" if row["label"] == 1 else "👤 Human"
        verified = " ✓" if row["verified"] else ""
        options_list.append(
            f"@{row['screen_name']}{verified} — {row['name']}  [{label_tag}]"
        )
    choice = st.sidebar.selectbox(
        f"{len(results)} result(s)",
        range(len(options_list)),
        format_func=lambda i: options_list[i],
    )
    selected_idx = int(results.iloc[choice]["idx"])

# Random user overrides search selection
if "random_idx" in st.session_state:
    selected_idx = st.session_state.pop("random_idx")
    random_entry = all_entries[selected_idx]
    random_sn = str(random_entry.get("profile", {}).get("screen_name", "")).strip()
    st.sidebar.success(f"Random: @{random_sn}")

# Tabs
tab_dash, tab_user, tab_hashtag, tab_domain, tab_perf, tab_graph, tab_explain = st.tabs([
    "🏠 Dashboard",
    "👤 User Search",
    "#️⃣ Hashtag Explorer",
    "🌐 Domain Explorer",
    "📊 Model Performance",
    "🕸️ Social Graph",
    "🔍 Explicability",
])

# ===================================================================
# TAB 0 — Dashboard
# ===================================================================
with tab_dash:
    # Key metrics row
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Users", f"{len(index_df):,}")
    n_total_bots = int((all_probs[
        [uid_to_idx[str(all_entries[i]['ID']).strip()] for i in range(len(all_entries))]
    , 1] >= 0.5).sum())
    n_total_humans = len(index_df) - n_total_bots
    d2.metric("Predicted Bots", f"{n_total_bots:,}")
    d3.metric("Predicted Humans", f"{n_total_humans:,}")
    d4.metric("Predicted Bot Rate", f"{n_total_bots / len(index_df):.1%}")

    st.markdown("---")

    # Model performance summary + domain breakdown side by side
    dash_left, dash_right = st.columns(2)

    with dash_left:
        st.subheader(f"Model: {model_type.upper()}")
        metrics_dash = cached_evaluate_test(model, data, cfg, model_type)
        perf_col1, perf_col2 = st.columns(2)
        perf_col1.metric("Accuracy", f"{metrics_dash['accuracy']:.1%}")
        perf_col1.metric("Precision", f"{metrics_dash['precision']:.1%}")
        perf_col2.metric("F1 Score", f"{metrics_dash['f1']:.1%}")
        perf_col2.metric("Recall", f"{metrics_dash['recall']:.1%}")

        test_labels_d = metrics_dash["test_labels"]
        n_test_d = len(test_labels_d)
        st.caption(f"Evaluated on {n_test_d:,} test users")

    with dash_right:
        st.subheader("Bot Rate by Domain")
        all_domains_dash = sorted(domain_index.keys())
        dom_rows_dash = []
        for dom in all_domains_dash:
            idxs = domain_index[dom]
            p_bots_d = np.array([
                all_probs[uid_to_idx[str(all_entries[i]["ID"]).strip()], 1]
                for i in idxs
            ])
            n_b = int((p_bots_d >= 0.5).sum())
            dom_rows_dash.append({
                "Domain": dom,
                "Users": len(idxs),
                "Bot Rate": n_b / len(idxs) * 100 if idxs else 0,
            })
        dom_df_dash = pd.DataFrame(dom_rows_dash)
        st.dataframe(
            dom_df_dash,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Bot Rate": st.column_config.ProgressColumn(
                    "Bot Rate", format="%.0f%%", min_value=0, max_value=100,
                ),
            },
        )

    st.markdown("---")

    # P(bot) distribution for all labeled users
    st.subheader("Overall P(Bot) Distribution")
    all_labeled_probs = np.array([
        all_probs[uid_to_idx[str(all_entries[i]["ID"]).strip()], 1]
        for i in range(len(all_entries))
    ])
    fig_dash, ax_dash = plt.subplots(figsize=(10, 4))
    ax_dash.hist(all_labeled_probs, bins=50, color="#6c5ce7", alpha=0.8, edgecolor="white")
    ax_dash.axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="Threshold (0.5)")
    ax_dash.set_xlabel("P(Bot)")
    ax_dash.set_ylabel("Number of users")
    ax_dash.set_title("Distribution of Bot Probability Across All Users")
    ax_dash.legend()
    st.pyplot(fig_dash)
    plt.close(fig_dash)

    st.markdown("---")

    # Top 10 most popular hashtags with bot rates
    st.subheader("Top 10 Hashtags")
    all_hashtags_dash = get_all_hashtags(all_entries)
    top_ht_rows = []
    for tag, user_count in all_hashtags_dash[:10]:
        user_idxs = hashtag_index[tag]
        p_bots_ht = np.array([
            all_probs[uid_to_idx[str(all_entries[i]["ID"]).strip()], 1]
            for i in user_idxs
        ])
        n_b_ht = int((p_bots_ht >= 0.5).sum())
        top_ht_rows.append({
            "Hashtag": f"#{tag}",
            "Users": user_count,
            "Bot Rate": n_b_ht / user_count * 100 if user_count > 0 else 0,
        })
    top_ht_df = pd.DataFrame(top_ht_rows)
    st.dataframe(
        top_ht_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Bot Rate": st.column_config.ProgressColumn(
                "Bot Rate", format="%.0f%%", min_value=0, max_value=100,
            ),
        },
    )

# ===================================================================
# TAB 1 — User Search
# ===================================================================
with tab_user:
    if selected_idx is None:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total users", f"{len(index_df):,}")
        col2.metric("Bots", f"{(index_df['label'] == 1).sum():,}")
        col3.metric("Humans", f"{(index_df['label'] == 0).sum():,}")
        st.info("Use the sidebar to search for a user, or click **🎲 Random user**.")
    else:
        entry = all_entries[selected_idx]
        profile = entry.get("profile", {})
        user_id = str(entry["ID"]).strip()

        # Get prediction from cached probs
        node_idx = uid_to_idx[user_id]
        p_human = float(all_probs[node_idx, 0])
        p_bot = float(all_probs[node_idx, 1])
        true_label = int(entry.get("label", -1))

        st.markdown("---")

        verdict_col, prob_col = st.columns([1, 2])

        with verdict_col:
            if p_bot >= 0.5:
                st.error("### 🤖 Likely Bot")
            else:
                st.success("### 👤 Likely Human")
            if true_label == 1:
                st.caption("Ground truth: **Bot**")
            elif true_label == 0:
                st.caption("Ground truth: **Human**")

        with prob_col:
            st.markdown("#### Bot Probability")
            st.progress(p_bot, text=f"{p_bot:.1%}")
            pcol1, pcol2 = st.columns(2)
            pcol1.metric("P(Bot)", f"{p_bot:.2%}")
            pcol2.metric("P(Human)", f"{p_human:.2%}")
            predicted_label = 1 if p_bot >= 0.5 else 0
            if true_label >= 0:
                if predicted_label == true_label:
                    st.success("Prediction matches ground truth ✓")
                else:
                    st.warning("Prediction differs from ground truth ✗")

        # Summary blurb
        summary_text = generate_user_summary(entry, p_bot, all_probs, uid_to_idx)
        st.info(summary_text)

        st.markdown("---")

        st.subheader(f"@{str(profile.get('screen_name', '')).strip()}")
        st.caption(f"ID: {user_id} · Split: {entry.get('_split', 'unknown')}")

        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        followers = _safe_int(profile.get("followers_count"))
        friends = _safe_int(profile.get("friends_count"))
        statuses = _safe_int(profile.get("statuses_count"))
        favourites = _safe_int(profile.get("favourites_count"))

        info_col1.metric("Followers", format_number(followers))
        info_col2.metric("Following", format_number(friends))
        info_col3.metric("Tweets", format_number(statuses))
        info_col4.metric("Favourites", format_number(favourites))

        desc = str(profile.get("description", "")).strip()
        if desc and desc.lower() not in ("none", ""):
            st.markdown(f"**Bio:** {desc}")

        loc = str(profile.get("location", "")).strip()
        if loc and loc.lower() not in ("none", ""):
            st.markdown(f"**Location:** {loc}")

        created = str(profile.get("created_at", "")).strip()
        if created and created.lower() not in ("none", ""):
            st.markdown(f"**Joined:** {created}")

        verified = str(profile.get("verified", "")).strip().lower() == "true"
        default_profile = str(profile.get("default_profile", "")).strip().lower() == "true"
        default_image = str(profile.get("default_profile_image", "")).strip().lower() == "true"

        tag_parts = []
        if verified:
            tag_parts.append("✅ Verified")
        if default_profile:
            tag_parts.append("⚠️ Default profile")
        if default_image:
            tag_parts.append("⚠️ Default profile image")
        if tag_parts:
            st.markdown(" · ".join(tag_parts))

        # Key indicators
        st.markdown("---")
        st.subheader("Key Indicators")

        ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
        ratio_ff = followers / (friends + 1)
        tweets_per_follower = statuses / (followers + 1)

        age_days = 0
        created_str = str(profile.get("created_at", "")).strip()
        if created_str and created_str.lower() not in ("none", ""):
            try:
                dt = datetime.strptime(created_str, "%a %b %d %H:%M:%S %z %Y")
                collection = datetime(2022, 2, 1, tzinfo=dt.tzinfo)
                age_days = max((collection - dt).days, 0)
            except ValueError:
                pass

        tweets_per_day = statuses / (age_days + 1) if age_days > 0 else 0
        screen_name = str(profile.get("screen_name", "")).strip()
        digits_in_name = sum(c.isdigit() for c in screen_name)

        ind_col1.metric("Followers / Following ratio", f"{ratio_ff:.2f}",
                        help="Low ratio is a bot signal.")
        ind_col2.metric("Tweets / Follower", f"{tweets_per_follower:.2f}",
                        help="High ratio is a bot signal.")
        ind_col3.metric("Tweets / Day", f"{tweets_per_day:.2f}",
                        help="Very high tweet rates can indicate automation.")
        ind_col4.metric("Digits in @handle", str(digits_in_name),
                        help="Auto-generated bot accounts often have many digits.")

        # Social graph summary
        st.markdown("---")
        st.subheader("Social Graph")
        neighbor = entry.get("neighbor") or {}
        following_list = neighbor.get("following") or []
        follower_list = neighbor.get("follower") or []
        g_col1, g_col2 = st.columns(2)
        g_col1.metric("Following (in graph)", len(following_list))
        g_col2.metric("Followers (in graph)", len(follower_list))

        # Tweets sample
        tweets = entry.get("tweet") or []
        if tweets:
            st.markdown("---")
            st.subheader("Recent Tweets")
            for t in tweets[:10]:
                t_clean = str(t).strip() if t else ""
                if t_clean:
                    st.markdown(f"> {t_clean}")
            if len(tweets) > 10:
                with st.expander(f"Show all {len(tweets)} tweets"):
                    for t in tweets[10:]:
                        t_clean = str(t).strip() if t else ""
                        if t_clean:
                            st.markdown(f"> {t_clean}")

# ===================================================================
# TAB 2 — Hashtag Explorer
# ===================================================================
with tab_hashtag:
    st.header("Hashtag Explorer")
    st.markdown(
        "Search for a hashtag to see which users tweeted it "
        "and the **predicted** bot rate from the GNN model."
    )

    ht_query = st.text_input(
        "Search hashtag",
        placeholder="e.g. MAGA, covid, ad, BlackLivesMatter",
        key="ht_search",
    )

    ht_query_clean = ht_query.strip().lstrip("#").lower()

    if ht_query_clean:
        matching_tags = [
            (tag, len(indices))
            for tag, indices in hashtag_index.items()
            if ht_query_clean in tag
        ]
        matching_tags.sort(key=lambda x: x[1], reverse=True)

        if not matching_tags:
            st.warning(f"No hashtag matching **#{ht_query_clean}** found.")
        else:
            if len(matching_tags) == 1:
                chosen_tag = matching_tags[0][0]
            else:
                tag_options = [
                    f"#{tag}  ({count} users)"
                    for tag, count in matching_tags[:100]
                ]
                tag_choice = st.selectbox(
                    f"{len(matching_tags)} hashtag(s) found",
                    range(len(tag_options)),
                    format_func=lambda i: tag_options[i],
                )
                chosen_tag = matching_tags[tag_choice][0]

            user_indices = hashtag_index[chosen_tag]
            n_users = len(user_indices)

            # Get predictions from cached probs
            p_bots = np.array([
                all_probs[uid_to_idx[str(all_entries[i]["ID"]).strip()], 1]
                for i in user_indices
            ])
            predicted_labels = (p_bots >= 0.5).astype(int)
            n_predicted_bots = int(predicted_labels.sum())
            n_predicted_humans = n_users - n_predicted_bots
            predicted_bot_rate = n_predicted_bots / n_users if n_users > 0 else 0

            st.markdown("---")
            st.subheader(f"#{chosen_tag}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Users", f"{n_users:,}")
            m2.metric("Predicted Bots", f"{n_predicted_bots:,}")
            m3.metric("Predicted Humans", f"{n_predicted_humans:,}")
            m4.metric("Predicted Bot Rate", f"{predicted_bot_rate:.1%}")

            st.markdown("#### Predicted Bot Rate")
            st.progress(
                predicted_bot_rate,
                text=(
                    f"{predicted_bot_rate:.1%} of users who tweeted "
                    f"#{chosen_tag} are predicted as bots"
                ),
            )

            dataset_bot_rate = (index_df["label"] == 1).sum() / len(index_df)
            delta = predicted_bot_rate - dataset_bot_rate
            if abs(delta) > 0.01:
                if delta > 0:
                    st.warning(
                        f"This hashtag has a **higher** predicted bot rate "
                        f"than the dataset average "
                        f"({predicted_bot_rate:.1%} vs {dataset_bot_rate:.1%})"
                    )
                else:
                    st.success(
                        f"This hashtag has a **lower** predicted bot rate "
                        f"than the dataset average "
                        f"({predicted_bot_rate:.1%} vs {dataset_bot_rate:.1%})"
                    )
            else:
                st.info(
                    f"Predicted bot rate is close to the dataset average "
                    f"({dataset_bot_rate:.1%})"
                )

            # Distribution
            st.markdown("---")
            st.subheader("P(Bot) Distribution")
            st.caption(
                "How many users fall into each probability range. "
                "Users on the right (> 50%) are predicted as bots."
            )
            bin_edges = np.arange(0, 1.1, 0.1)
            bin_labels = [
                f"{int(lo*100)}-{int(hi*100)}%"
                for lo, hi in zip(bin_edges[:-1], bin_edges[1:])
            ]
            bin_indices = np.digitize(p_bots, bin_edges[1:], right=True)
            bin_counts = [int((bin_indices == i).sum()) for i in range(len(bin_labels))]
            chart_df = pd.DataFrame({"range": bin_labels, "users": bin_counts})
            st.bar_chart(chart_df, x="range", y="users", x_label="P(Bot)", y_label="Users")

            # User table
            st.markdown("---")
            st.subheader("Users")
            table_rows = []
            for j, i in enumerate(user_indices):
                e = all_entries[i]
                p = e.get("profile", {})
                p_bot_val = float(p_bots[j])
                table_rows.append({
                    "screen_name": "@" + str(p.get("screen_name", "")).strip(),
                    "name": str(p.get("name", "")).strip(),
                    "P(Bot)": p_bot_val * 100,
                    "prediction": "🤖 Bot" if p_bot_val >= 0.5 else "👤 Human",
                    "followers": _safe_int(p.get("followers_count")),
                    "following": _safe_int(p.get("friends_count")),
                    "tweets": _safe_int(p.get("statuses_count")),
                    "verified": (
                        "✓"
                        if str(p.get("verified", "")).strip().lower() == "true"
                        else ""
                    ),
                })

            table_df = pd.DataFrame(table_rows)
            table_df = table_df.sort_values("P(Bot)", ascending=False)
            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "P(Bot)": st.column_config.ProgressColumn(
                        "P(Bot)", format="%.0f%%", min_value=0, max_value=100,
                    ),
                    "followers": st.column_config.NumberColumn(format="%d"),
                    "following": st.column_config.NumberColumn(format="%d"),
                    "tweets": st.column_config.NumberColumn(format="%d"),
                },
            )
    else:
        st.markdown("---")
        st.subheader("Top Hashtags by Usage")
        st.caption(
            "Showing ground-truth bot rates for the top hashtags. "
            "Select a hashtag above to see **predicted** bot rates."
        )
        all_hashtags = get_all_hashtags(all_entries)
        top_n = 30
        top_rows = []
        for tag, user_count in all_hashtags[:top_n]:
            user_idxs = hashtag_index[tag]
            labels = [int(all_entries[i].get("label", -1)) for i in user_idxs]
            n_bots = sum(1 for l in labels if l == 1)
            bot_rate = (n_bots / user_count * 100) if user_count > 0 else 0
            top_rows.append({
                "hashtag": f"#{tag}",
                "users": user_count,
                "bots": n_bots,
                "humans": user_count - n_bots,
                "bot_rate": bot_rate,
            })

        top_df = pd.DataFrame(top_rows)
        st.dataframe(
            top_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "bot_rate": st.column_config.ProgressColumn(
                    "Bot Rate", format="%.0f%%", min_value=0, max_value=100,
                ),
            },
        )
        st.caption(f"Total unique hashtags in the dataset: **{len(all_hashtags):,}**")

# ===================================================================
# TAB 3 — Domain Explorer
# ===================================================================
with tab_domain:
    st.header("Domain Explorer")
    st.markdown("Explore predicted bot rates across topical domains.")

    all_domains = sorted(domain_index.keys())

    # Overview: bot rate per domain
    st.markdown("---")
    st.subheader("Bot Rate by Domain")

    overview_rows = []
    for dom in all_domains:
        idxs = domain_index[dom]
        p_bots_dom = np.array([
            all_probs[uid_to_idx[str(all_entries[i]["ID"]).strip()], 1]
            for i in idxs
        ])
        n_bots_dom = int((p_bots_dom >= 0.5).sum())
        overview_rows.append({
            "domain": dom,
            "users": len(idxs),
            "predicted_bots": n_bots_dom,
            "predicted_humans": len(idxs) - n_bots_dom,
            "bot_rate": n_bots_dom / len(idxs) * 100 if idxs else 0,
        })

    overview_df = pd.DataFrame(overview_rows)
    st.dataframe(
        overview_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "bot_rate": st.column_config.ProgressColumn(
                "Predicted Bot Rate", format="%.0f%%", min_value=0, max_value=100,
            ),
        },
    )

    # Bar chart
    fig_dom, ax_dom = plt.subplots(figsize=(8, 4))
    colors_dom = ["coral" if r > 50 else "steelblue" for r in overview_df["bot_rate"]]
    bars = ax_dom.barh(overview_df["domain"], overview_df["bot_rate"], color=colors_dom)
    ax_dom.set_xlabel("Predicted Bot Rate (%)")
    ax_dom.set_xlim(0, 100)
    for bar, val in zip(bars, overview_df["bot_rate"]):
        ax_dom.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%", va="center", fontsize=10)
    ax_dom.set_title("Predicted Bot Rate by Domain")
    st.pyplot(fig_dom)
    plt.close(fig_dom)

    # Filter by domain
    st.markdown("---")
    st.subheader("Filter by Domain")
    selected_domains = st.multiselect(
        "Select domain(s)",
        all_domains,
        default=all_domains,
    )

    if selected_domains:
        # Union of users across selected domains
        filtered_set: set[int] = set()
        for dom in selected_domains:
            filtered_set.update(domain_index.get(dom, []))
        filtered_indices = sorted(filtered_set)

        p_bots_filtered = np.array([
            all_probs[uid_to_idx[str(all_entries[i]["ID"]).strip()], 1]
            for i in filtered_indices
        ])
        n_filtered = len(filtered_indices)
        n_bots_f = int((p_bots_filtered >= 0.5).sum())
        bot_rate_f = n_bots_f / n_filtered if n_filtered > 0 else 0

        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Users", f"{n_filtered:,}")
        fc2.metric("Predicted Bots", f"{n_bots_f:,}")
        fc3.metric("Predicted Humans", f"{n_filtered - n_bots_f:,}")
        fc4.metric("Predicted Bot Rate", f"{bot_rate_f:.1%}")

        # User table
        dom_table_rows = []
        for j, i in enumerate(filtered_indices):
            e = all_entries[i]
            p = e.get("profile", {})
            p_bot_val = float(p_bots_filtered[j])
            dom_table_rows.append({
                "screen_name": "@" + str(p.get("screen_name", "")).strip(),
                "name": str(p.get("name", "")).strip(),
                "domains": ", ".join(e.get("domain") or []),
                "P(Bot)": p_bot_val * 100,
                "prediction": "🤖 Bot" if p_bot_val >= 0.5 else "👤 Human",
                "followers": _safe_int(p.get("followers_count")),
            })

        dom_table_df = pd.DataFrame(dom_table_rows)
        dom_table_df = dom_table_df.sort_values("P(Bot)", ascending=False)
        st.dataframe(
            dom_table_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "P(Bot)": st.column_config.ProgressColumn(
                    "P(Bot)", format="%.0f%%", min_value=0, max_value=100,
                ),
                "followers": st.column_config.NumberColumn(format="%d"),
            },
        )

# ===================================================================
# TAB 4 — Model Performance
# ===================================================================
with tab_perf:
    st.header("Model Performance")
    st.markdown(f"Test set evaluation for the **{model_type.upper()}** model.")

    metrics = cached_evaluate_test(model, data, cfg, model_type)

    # Metrics row
    st.markdown("---")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    mc2.metric("F1 Score", f"{metrics['f1']:.2%}")
    mc3.metric("Precision", f"{metrics['precision']:.2%}")
    mc4.metric("Recall", f"{metrics['recall']:.2%}")

    test_labels = metrics["test_labels"]
    test_probs = metrics["test_probs"]
    n_test = len(test_labels)
    n_test_bots = int((test_labels == 1).sum())
    n_test_humans = n_test - n_test_bots
    st.caption(
        f"Test set: {n_test:,} users ({n_test_humans:,} humans, {n_test_bots:,} bots)"
    )

    # Confusion matrix + P(bot) distribution side by side
    st.markdown("---")
    cm_col, dist_col = st.columns(2)

    with cm_col:
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(
            metrics["confusion_matrix"], display_labels=["Human", "Bot"]
        )
        disp.plot(ax=ax_cm, cmap="Blues", values_format="d")
        ax_cm.set_title(f"{model_type.upper()} – Test Set")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with dist_col:
        st.subheader("P(Bot) Distribution")
        fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
        human_probs = test_probs[test_labels == 0]
        bot_probs = test_probs[test_labels == 1]
        ax_dist.hist(
            human_probs, bins=50, alpha=0.6,
            label=f"Human (n={len(human_probs)})", color="steelblue", density=True,
        )
        ax_dist.hist(
            bot_probs, bins=50, alpha=0.6,
            label=f"Bot (n={len(bot_probs)})", color="coral", density=True,
        )
        ax_dist.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Threshold")
        ax_dist.set_xlabel("P(Bot)")
        ax_dist.set_ylabel("Density")
        ax_dist.legend()
        ax_dist.set_title("P(Bot) for Humans vs Bots (Test Set)")
        st.pyplot(fig_dist)
        plt.close(fig_dist)

# ===================================================================
# TAB 5 — Social Graph
# ===================================================================
with tab_graph:
    st.header("Social Network Graph")
    st.markdown("Visualize the social neighborhood of a selected user.")

    if selected_idx is None:
        st.info("Select a user from the sidebar to see their social graph.")
    else:
        entry = all_entries[selected_idx]
        user_id = str(entry["ID"]).strip()
        profile = entry.get("profile", {})
        screen_name = str(profile.get("screen_name", "")).strip()

        neighbor = entry.get("neighbor") or {}
        following_ids = [str(x) for x in (neighbor.get("following") or [])]
        follower_ids = [str(x) for x in (neighbor.get("follower") or [])]

        if not following_ids and not follower_ids:
            st.warning(
                f"@{screen_name} has no connections in the TwiBot-20 graph. "
                f"About 9% of users in the dataset have no recorded neighbors."
            )
        else:
            # Build ego graph
            G = nx.DiGraph()
            G.add_node(user_id, label=f"@{screen_name}", node_type="center")

            all_neighbor_ids = list(set(following_ids + follower_ids))
            valid_neighbor_ids = [nid for nid in all_neighbor_ids if nid in uid_to_idx]
            missing_count = len(all_neighbor_ids) - len(valid_neighbor_ids)

            for nid in valid_neighbor_ids:
                idx = uid_to_idx[nid]
                p_bot_n = float(all_probs[idx, 1])
                # Try to find screen_name from index
                match = index_df[index_df["ID"] == nid]
                if not match.empty:
                    nlabel = "@" + match.iloc[0]["screen_name"]
                else:
                    nlabel = nid
                G.add_node(nid, label=nlabel, p_bot=p_bot_n)

            for nid in following_ids:
                if nid in uid_to_idx:
                    G.add_edge(user_id, nid, relation="following")
            for nid in follower_ids:
                if nid in uid_to_idx:
                    G.add_edge(nid, user_id, relation="follower")

            # Render
            fig_g, ax_g = plt.subplots(figsize=(10, 8))

            pos = nx.spring_layout(G, seed=42, k=2)

            node_colors = []
            node_sizes = []
            for node in G.nodes():
                if node == user_id:
                    node_colors.append("gold")
                    node_sizes.append(900)
                else:
                    p_b = G.nodes[node].get("p_bot", 0.5)
                    node_colors.append("coral" if p_b >= 0.5 else "steelblue")
                    node_sizes.append(400)

            edge_colors = [
                "#1f77b4" if G.edges[e].get("relation") == "following" else "#2ca02c"
                for e in G.edges()
            ]

            nx.draw_networkx_nodes(
                G, pos, node_color=node_colors, node_size=node_sizes,
                ax=ax_g, edgecolors="black", linewidths=0.5,
            )
            nx.draw_networkx_edges(
                G, pos, edge_color=edge_colors, arrows=True,
                arrowsize=15, ax=ax_g, alpha=0.7,
            )
            labels = {node: G.nodes[node].get("label", node)[:15] for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax_g)

            legend_elements = [
                mpatches.Patch(color="gold", label="Selected user"),
                mpatches.Patch(color="coral", label="Predicted bot"),
                mpatches.Patch(color="steelblue", label="Predicted human"),
                mpatches.Patch(color="#1f77b4", label="Following →"),
                mpatches.Patch(color="#2ca02c", label="← Follower"),
            ]
            ax_g.legend(handles=legend_elements, loc="upper left", fontsize=8)
            ax_g.set_title(f"Social Graph of @{screen_name}")
            ax_g.axis("off")

            st.pyplot(fig_g)
            plt.close(fig_g)

            # Stats below graph
            n_neighbor_bots = sum(
                1 for nid in valid_neighbor_ids
                if all_probs[uid_to_idx[nid], 1] >= 0.5
            )
            st.markdown(
                f"**{n_neighbor_bots}** of **{len(valid_neighbor_ids)}** "
                f"neighbors are predicted as bots."
            )
            if missing_count > 0:
                st.caption(
                    f"{missing_count} neighbor(s) are outside the labeled dataset "
                    f"and not shown."
                )

            # Neighbor table
            st.markdown("---")
            st.subheader("Neighbors")
            neighbor_rows = []
            for nid in valid_neighbor_ids:
                idx_n = uid_to_idx[nid]
                p_bot_n = float(all_probs[idx_n, 1])
                match = index_df[index_df["ID"] == nid]
                sn = "@" + match.iloc[0]["screen_name"] if not match.empty else nid
                rel = []
                if nid in following_ids:
                    rel.append("following")
                if nid in follower_ids:
                    rel.append("follower")
                neighbor_rows.append({
                    "screen_name": sn,
                    "relationship": ", ".join(rel),
                    "P(Bot)": p_bot_n * 100,
                    "prediction": "🤖 Bot" if p_bot_n >= 0.5 else "👤 Human",
                })

            neighbor_df = pd.DataFrame(neighbor_rows)
            neighbor_df = neighbor_df.sort_values("P(Bot)", ascending=False)
            st.dataframe(
                neighbor_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "P(Bot)": st.column_config.ProgressColumn(
                        "P(Bot)", format="%.0f%%", min_value=0, max_value=100,
                    ),
                },
            )

# ===================================================================
# TAB 6 — Explicability
# ===================================================================
with tab_explain:
    st.header("Explicability")
    st.markdown(
        "Understand which features drive the bot prediction for a selected user. "
        "We use **perturbation-based** feature importance: each feature is set to "
        "the population mean and the change in P(Bot) is measured."
    )

    if selected_idx is None:
        st.info("Select a user from the sidebar to see feature explanations.")
    else:
        entry = all_entries[selected_idx]
        user_id = str(entry["ID"]).strip()
        profile = entry.get("profile", {})
        screen_name = str(profile.get("screen_name", "")).strip()
        node_idx = uid_to_idx[user_id]

        with st.spinner("Computing feature importance (20 perturbations)…"):
            p_bot_base, importances = compute_feature_importance(
                model, data, cfg, node_idx
            )

        st.markdown("---")

        if p_bot_base >= 0.5:
            st.error(f"### 🤖 @{screen_name} — P(Bot) = {p_bot_base:.1%}")
        else:
            st.success(f"### 👤 @{screen_name} — P(Bot) = {p_bot_base:.1%}")

        # Top 3 explanations
        top_3 = np.argsort(np.abs(importances))[-3:][::-1]
        raw_vals = extract_raw_feature_values(entry)

        st.subheader("Top Explanations")
        for j in top_3:
            fname = FEATURE_NAMES[j]
            imp = importances[j]
            direction = "increases" if imp > 0 else "decreases"
            st.markdown(
                f"- **{fname}** = `{raw_vals[j]}` → "
                f"{direction} P(Bot) by {abs(imp):.3f}"
            )

        # Bar chart
        st.markdown("---")
        st.subheader("Feature Importance")
        st.caption(
            "Positive values (coral) push toward Bot; "
            "negative values (blue) push toward Human."
        )

        fig_imp, ax_imp = plt.subplots(figsize=(10, 7))
        sorted_idx = np.argsort(np.abs(importances))
        sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]
        sorted_vals = importances[sorted_idx]
        colors_imp = ["coral" if v > 0 else "steelblue" for v in sorted_vals]

        ax_imp.barh(range(len(sorted_names)), sorted_vals, color=colors_imp)
        ax_imp.set_yticks(range(len(sorted_names)))
        ax_imp.set_yticklabels(sorted_names)
        ax_imp.set_xlabel("Impact on P(Bot)")
        ax_imp.set_title(
            f"Feature Importance for @{screen_name} (P(Bot) = {p_bot_base:.2%})"
        )
        ax_imp.axvline(0, color="black", linewidth=0.5)

        legend_imp = [
            mpatches.Patch(color="coral", label="Pushes toward Bot"),
            mpatches.Patch(color="steelblue", label="Pushes toward Human"),
        ]
        ax_imp.legend(handles=legend_imp, loc="lower right")
        fig_imp.tight_layout()
        st.pyplot(fig_imp)
        plt.close(fig_imp)

        # Feature values table
        st.markdown("---")
        st.subheader("Feature Values")
        feat_table = []
        for j in range(len(FEATURE_NAMES)):
            feat_table.append({
                "Feature": FEATURE_NAMES[j],
                "Raw Value": raw_vals[j],
                "Importance": round(importances[j], 4),
                "Direction": "🤖 Bot signal" if importances[j] > 0 else "👤 Human signal",
            })

        feat_df = pd.DataFrame(feat_table)
        feat_df = feat_df.sort_values("Importance", key=abs, ascending=False)
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
