# HackEurope-2026-HOGA

HackEurope-2026-Hoga is an advanced bot detection system developed during HackEurope 2026.
The project combines three complementary machine learning paradigms to detect automated or coordinated malicious behavior:

🌲 Random Forest Classifier (structured feature learning)

🤖 Fine-Tuned BERT Encoder (semantic/textual understanding)

🕸 Graph Neural Network (GNN) (relational/structural modeling)

These models are combined using an ensemble strategy, allowing the system to capture bot activity across structured, textual, and network dimensions.

## Why Ensemble?

Each model detects different signals:

Random Forest → behavioral anomalies

BERT → linguistic patterns

GNN → coordinated network activity

Individually, each model has strengths and limitations. When combined, they provide:

Higher accuracy

Better generalization

Increased robustness

## 1️⃣ Graph Neural Network (GNN)

The GNN models relationships between users/entities as a graph structure. Instead of analyzing accounts independently, it learns from Interaction patterns, Network connectivity, Community structures, and Coordinated behavior.

## 2️⃣ Fine-Tuned BERT Encoder

We fine-tuned a BERT encoder to extract deep semantic representations from textual data.

## 3️⃣ Random Forest Classifier

A classical ensemble-based model trained on engineered tabular features.


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
pip install numpy pandas scikit-learn torch transformers torch-geometric matplotlib seaborn
```

---
## Running the Project

Once the environment is set up and dependencies are installed, run:

```bash
python app.py
```

