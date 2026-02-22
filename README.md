# HackEurope-2026-HOGA

An advanced bot detection system developed during **HackEurope 2026**. The project combines three complementary machine learning paradigms to detect automated or coordinated malicious behavior on the [TwiBot-20](https://twibot20.github.io/) dataset (~230k Twitter users).

- 🌲 **Random Forest Classifier** — structured feature learning on engineered tabular features
- 🤖 **Fine-Tuned BERT Encoder** — semantic understanding of tweet content
- 🕸 **Graph Neural Network (GNN)** — relational modeling of the social graph

These models are combined using an **ensemble strategy**, allowing the system to capture bot activity across structured, textual, and network dimensions.

## Screenshots

| Hashtag Explorer | User Search |
|:---:|:---:|
| ![Hashtag Explorer](screenshots/hashtag_explorer.png) | ![User Search](screenshots/user_search.png) |

## Why Ensemble?

Each model detects different signals:

| Model | Signal | Input |
|-------|--------|-------|
| Random Forest | Behavioral anomalies | 20 engineered profile features |
| BERT | Linguistic patterns | Tweet text |
| GNN (GAT / R-GCN) | Coordinated network activity | Social graph + profile features |

Individually, each model has strengths and limitations. When combined, they provide higher accuracy, better generalization, and increased robustness.

## Models

### 1. Graph Neural Network (GNN)

The GNN models relationships between users as a graph structure. Instead of analyzing accounts independently, it learns from interaction patterns, network connectivity, community structures, and coordinated behavior.

Two architectures are available, selectable from the app sidebar:

- **GAT** (Graph Attention Network) — 2-layer GAT with 4 attention heads, 128 hidden dim
- **R-GCN** (Relational GCN) — 2-layer R-GCN with relation-specific weights (following / follower)

**Input features (20):** followers count, friends count, statuses count, favourites count, listed count, followers/friends ratio, statuses/followers ratio, listed/followers ratio, account age, tweets per day, name length, screen name length, description length, digits in screen name, verified, default profile, default profile image, has URL, has location, has description.

### 2. Fine-Tuned BERT Encoder

A BERT encoder fine-tuned to extract deep semantic representations from user tweet text. The model weights are stored in SafeTensors format (`mon_modele_safetensors/`).

### 3. Random Forest Classifier

A classical ensemble-based model trained on engineered tabular features. The trained model is serialized as `bot_detector.pkl`.

## Interactive Dashboard

The Streamlit app provides 7 interactive tabs:

| Tab | Description |
|-----|-------------|
| 🏠 Dashboard | Global metrics, P(Bot) distribution, top hashtags, bot rate by domain |
| 👤 User Search | Search any user by name/@handle/ID, see prediction + profile details |
| #️⃣ Hashtag Explorer | Search hashtags, see predicted bot rate among users who tweeted them |
| 🌐 Domain Explorer | Compare predicted bot rates across topical domains |
| 📊 Model Performance | Accuracy, F1, precision, recall, confusion matrix on the test set |
| 🕸️ Social Graph | Visualize the ego-network of a selected user with bot/human coloring |
| 🔍 Explicability | Perturbation-based feature importance for individual predictions |

## Project Structure

```
├── app.py                          # Streamlit frontend (main entry point)
├── graph/                          # GNN module
│   ├── config.py                   #   Hyperparameters and paths
│   ├── data_loader.py              #   Graph construction from JSON splits
│   ├── feature_extractor.py        #   20-feature extraction + normalization
│   ├── models.py                   #   GAT and R-GCN architectures
│   ├── train.py                    #   Training loop
│   ├── evaluate.py                 #   Test set evaluation + probability export
│   ├── predict.py                  #   Inference on new users
│   ├── visualize.py                #   Training curves and embeddings
│   ├── download_data.py            #   Kaggle dataset downloader
│   └── checkpoints/                #   Trained model weights (.pt)
├── archive/                        # TwiBot-20 data splits
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── mon_modele_safetensors/         # Fine-tuned BERT weights
├── bot_detector.pkl                # Trained Random Forest model
├── Ensembling.ipynb                # Ensemble strategy notebook
├── Fine_Tuning_API.ipynb           # BERT fine-tuning notebook
├── HackEurope_Feature_engineering.ipynb  # Feature engineering notebook
├── descriptive_stats.py            # Descriptive statistics generator
└── stats_output/                   # Generated plots and summary stats
```

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Aziz-Chebil/HackEurope-2026-Hoga.git
cd HackEurope-2026-Hoga
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
```

- On **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

- On **Windows:**
  ```bash
  venv\Scripts\activate
  ```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

## Data

The project uses the [TwiBot-20](https://twibot20.github.io/) dataset. The data splits (`train.json`, `dev.json`, `test.json`) should be placed in the `archive/` directory.

To download the dataset automatically via Kaggle:

```bash
python graph/download_data.py
```

> Requires `kagglehub` and a valid [Kaggle API token](https://www.kaggle.com/docs/api).

## Running the App

Once the environment is set up and dependencies are installed:

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Team

**Hoga** — HackEurope 2026
