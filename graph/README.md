# Graph-Based Twitter Bot Detection

Module de détection de bots Twitter par **Graph Neural Networks** (GNN) sur le dataset **TwiBot-20**. Ce module produit une probabilité P(bot) par utilisateur, destinée à être combinée avec les modules feature-based et text-based via stacking.

---

## Table des matières

1. [Fondements théoriques](#1-fondements-théoriques)
   - [Pourquoi un graphe ?](#11-pourquoi-un-graphe-)
   - [Message Passing Neural Networks](#12-message-passing-neural-networks)
   - [GAT : Graph Attention Network](#13-gat--graph-attention-network)
   - [R-GCN : Relational Graph Convolutional Network](#14-r-gcn--relational-graph-convolutional-network)
   - [Transductive Learning](#15-transductive-learning)
2. [Dataset TwiBot-20](#2-dataset-twibot-20)
3. [Architecture du pipeline](#3-architecture-du-pipeline)
4. [Description des fichiers](#4-description-des-fichiers)
5. [Installation et exécution](#5-installation-et-exécution)
6. [Hyperparamètres](#6-hyperparamètres)
7. [Outputs](#7-outputs)
8. [Performances attendues](#8-performances-attendues)

---

## 1. Fondements théoriques

### 1.1 Pourquoi un graphe ?

Twitter est naturellement un **graphe social dirigé** : les utilisateurs se suivent mutuellement (ou non), formant un réseau de relations. L'hypothèse centrale est que les bots ont des **patterns structurels** distincts des humains :

- **Comportement de suivi anormal** : les bots suivent en masse d'autres comptes mais sont peu suivis en retour, ou inversement ils se suivent entre eux (clusters de bots).
- **Homophilie** : les bots tendent à interagir avec d'autres bots. Un utilisateur dont la majorité des voisins sont des bots a une forte probabilité d'être lui-même un bot.
- **Position topologique** : les bots occupent des positions spécifiques dans le graphe (périphérie, hubs artificiels, composantes denses).

Un classifieur classique (features-only) ignore ces signaux relationnels. Un GNN les exploite directement en propageant de l'information le long des arêtes du graphe.

### 1.2 Message Passing Neural Networks

Les GNN modernes reposent sur le paradigme de **message passing** (Gilmer et al., 2017). À chaque couche, chaque noeud :

1. **Collecte** les représentations de ses voisins (messages)
2. **Agrège** ces messages (somme, moyenne, attention...)
3. **Met à jour** sa propre représentation en combinant son état actuel et l'agrégation reçue

Formellement, pour un noeud `v` à la couche `l` :

```
m_v^(l) = AGGREGATE({ h_u^(l-1) : u in N(v) })
h_v^(l) = UPDATE(h_v^(l-1), m_v^(l))
```

Où `N(v)` désigne l'ensemble des voisins de `v`, `h_v^(l)` sa représentation à la couche `l`, et `AGGREGATE`/`UPDATE` sont des fonctions différentiables apprises.

Avec **2 couches** de message passing, chaque noeud intègre l'information de ses **voisins à 2 sauts** (2-hop neighborhood). Cela signifie que la représentation d'un utilisateur encode non seulement son propre profil, mais aussi les profils de ses amis et des amis de ses amis.

### 1.3 GAT : Graph Attention Network

Le **Graph Attention Network** (Veličković et al., 2018) introduit un mécanisme d'**attention** dans l'agrégation : tous les voisins ne contribuent pas de manière égale.

**Mécanisme d'attention :**

Pour chaque paire de noeuds connectés `(v, u)`, un coefficient d'attention `alpha_vu` est calculé :

```
e_vu = LeakyReLU(a^T [W*h_v || W*h_u])
alpha_vu = softmax_u(e_vu) = exp(e_vu) / sum_{k in N(v)} exp(e_vk)
```

Où `W` est une matrice de projection apprise, `||` dénote la concaténation, et `a` est un vecteur d'attention appris. Le softmax normalise les coefficients sur l'ensemble des voisins.

La nouvelle représentation est alors :

```
h_v^(l) = ELU( sum_{u in N(v)} alpha_vu * W * h_u^(l-1) )
```

**Multi-head attention :**

Pour stabiliser l'apprentissage, GAT utilise `K` têtes d'attention indépendantes dont les résultats sont concaténés (couche intermédiaire) ou moyennés (couche finale) :

```
h_v^(l) = || _{k=1}^{K} ELU( sum_{u in N(v)} alpha_vu^k * W^k * h_u^(l-1) )
```

**Intuition pour la détection de bots :** L'attention permet au modèle d'apprendre à pondérer différemment les voisins. Par exemple, un voisin vérifié et ancien devrait avoir plus d'influence qu'un compte récemment créé avec zéro tweet. Le modèle apprend ces pondérations automatiquement.

**Notre architecture GAT :**

```
Input [N, 20]
  -> GATConv(20, 128, heads=4, dropout=0.3)    # output: [N, 512]
  -> ELU
  -> Dropout(0.3)
  -> GATConv(512, 128, heads=1, dropout=0.3)    # output: [N, 128]
  -> ELU
  -> Linear(128, 2)                              # output: [N, 2] (logits)
```

- La 1ère couche utilise **4 têtes d'attention** concaténées : 4 x 128 = 512 dimensions
- La 2ème couche utilise **1 seule tête** pour réduire à 128 dimensions
- L'activation **ELU** (au lieu de ReLU) est le choix standard pour GAT car elle permet des valeurs négatives, évitant le problème des "neurones morts"
- GAT traite toutes les arêtes uniformément (pas de distinction following/follower au niveau du type d'arête)

### 1.4 R-GCN : Relational Graph Convolutional Network

Le **Relational GCN** (Schlichtkrull et al., 2018) étend le GCN classique aux **graphes multi-relationnels** : chaque type d'arête a ses propres poids.

**Formulation :**

```
h_v^(l) = ReLU( sum_{r in R} sum_{u in N_r(v)} (1/c_{v,r}) * W_r^(l) * h_u^(l-1) + W_0^(l) * h_v^(l-1) )
```

Où :
- `R` est l'ensemble des types de relations (ici : `{following, follower}`)
- `N_r(v)` sont les voisins de `v` via la relation `r`
- `W_r^(l)` est la matrice de poids **spécifique à la relation `r`** à la couche `l`
- `W_0^(l)` est une transformation "self-loop" (le noeud préserve sa propre information)
- `c_{v,r}` est un facteur de normalisation (typiquement `|N_r(v)|`)

**Intuition pour la détection de bots :** La distinction entre "following" et "follower" est cruciale. Un bot qui suit 10 000 comptes mais n'est suivi par personne a un pattern très différent d'un humain influent. R-GCN apprend des transformations séparées pour chaque direction, capturant cette asymétrie.

**Notre architecture R-GCN :**

```
Input [N, 20]
  -> RGCNConv(20, 128, num_relations=2)     # output: [N, 128]
  -> ReLU
  -> Dropout(0.3)
  -> RGCNConv(128, 128, num_relations=2)    # output: [N, 128]
  -> ReLU
  -> Linear(128, 2)                          # output: [N, 2] (logits)
```

- 2 relations : `0 = following` (A suit B), `1 = follower` (B suit A)
- Chaque couche apprend 2 matrices de poids distinctes + 1 self-loop
- Activation **ReLU** (standard pour R-GCN)

### 1.5 Transductive Learning

Notre approche est **transductive** (et non inductive) :

- Les **~180K utilisateurs non labellisés** participent au graphe et au message passing
- Mais la **loss n'est calculée que sur les noeuds labellisés** du train set
- À l'inférence, le modèle produit des prédictions pour tous les noeuds, y compris ceux vus pendant l'entraînement

Cela signifie que les noeuds non labellisés enrichissent les représentations via la propagation d'information, même s'ils ne contribuent pas directement à la loss. C'est un avantage majeur des GNN : exploiter la structure du graphe complet.

---

## 2. Dataset TwiBot-20

Le dataset provient du benchmark **TwiBot-20** (Feng et al., 2021), référence pour la détection de bots Twitter.

### Structure des données

```
archive/
  train.json    # 8,278 utilisateurs (3,632 humains, 4,646 bots)
  dev.json      # 2,365 utilisateurs (1,062 humains, 1,303 bots)
  test.json     # 1,183 utilisateurs (543 humains, 640 bots)
```

Chaque entrée JSON contient :

| Champ       | Description |
|-------------|-------------|
| `ID`        | Identifiant Twitter unique |
| `profile`   | Métadonnées du profil (followers_count, friends_count, verified, created_at, ...) |
| `tweet`     | Liste des 200 derniers tweets (texte brut) |
| `neighbor`  | Dict `{"following": [id1, ...], "follower": [id2, ...]}` — max 10 par type |
| `domain`    | Catégories thématiques du compte |
| `label`     | `"0"` = humain, `"1"` = bot |

### Graphe construit

À partir des champs `neighbor`, nous construisons un graphe dirigé :

- **191,582 noeuds** : 11,826 labellisés + 179,756 voisins non labellisés
- **208,716 arêtes** : 105,701 "following" + 103,015 "follower"
- Les voisins non labellisés n'ont pas de profil dans le dataset : leurs features sont initialisées à zéro

---

## 3. Architecture du pipeline

```
main.py                    # Orchestration
  |
  |-- data_loader.py       # 1. Charge les JSON, construit le graphe PyG
  |     |
  |     +-- feature_extractor.py   # Extrait 20 features par noeud
  |
  |-- models.py            # 2. Définit GAT et R-GCN
  |
  |-- train.py             # 3. Boucle d'entraînement full-batch + early stopping
  |
  |-- evaluate.py          # 4. Métriques test + export P(bot) en CSV
  |
  +-- visualize.py         # 5. Courbes de loss, confusion matrix, t-SNE
```

**Flux de données :**

```
JSON files
    |
    v
[data_loader] -> Data(x, edge_index, edge_type, y, masks)
    |
    v
[train] -> forward pass complet sur tout le graphe
    |        loss calculée uniquement sur train_mask
    |        early stopping sur val_mask
    v
[evaluate] -> métriques sur test_mask
    |          P(bot) pour tous les labellisés -> CSV
    v
[visualize] -> plots PNG
```

---

## 4. Description des fichiers

### `config.py`

Dataclass centralisant tous les hyperparamètres et chemins. Les valeurs par défaut sont ajustables via les arguments CLI de `main.py`.

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `model_type` | `"gat"` | Architecture : `"gat"` ou `"rgcn"` |
| `hidden_dim` | `128` | Dimension des couches cachées du GNN |
| `gat_heads` | `4` | Nombre de têtes d'attention (GAT uniquement) |
| `num_relations` | `2` | Nombre de types de relations (R-GCN) |
| `dropout` | `0.3` | Taux de dropout |
| `lr` | `1e-3` | Learning rate (Adam) |
| `weight_decay` | `5e-4` | Régularisation L2 |
| `epochs` | `100` | Nombre max d'époques |
| `patience` | `10` | Époques sans amélioration avant early stopping |
| `seed` | `42` | Graine aléatoire pour reproductibilité |
| `collection_date` | `"2022-02-01"` | Date de référence pour calculer l'âge des comptes |

### `data_loader.py`

Charge les 3 fichiers JSON et construit un objet `torch_geometric.data.Data`.

**Étapes :**
1. **Chargement** : `train.json`, `dev.json`, `test.json`
2. **Indexation** : mapping `user_id -> index` (0 à N-1). Les utilisateurs labellisés recevront les indices 0 à 11825, puis les voisins non labellisés prennent les indices suivants.
3. **Construction des arêtes** : pour chaque utilisateur labellisé, ses listes `following` et `follower` deviennent des arêtes dirigées dans `edge_index` (tensor `[2, E]`), avec un `edge_type` (0 ou 1).
4. **Extraction des features** : appelle `feature_extractor.py` pour construire la matrice `x` de shape `[N, 20]`.
5. **Labels et masques** : tensor `y` (`0`=humain, `1`=bot, `-1`=non labellisé) et masques booléens `train_mask`, `val_mask`, `test_mask`.

**Convention des arêtes :**
- `following` (type 0) : `user -> target` (l'utilisateur suit la cible)
- `follower` (type 1) : `follower -> user` (le suiveur suit l'utilisateur)

### `feature_extractor.py`

Extrait un vecteur de **20 features** à partir du profil de chaque utilisateur.

**14 features numériques** (normalisées par `StandardScaler`) :

| # | Feature | Intuition |
|---|---------|-----------|
| 1 | `followers_count` | Popularité du compte |
| 2 | `friends_count` | Nombre de comptes suivis |
| 3 | `statuses_count` | Volume total de tweets |
| 4 | `favourites_count` | Engagement passif |
| 5 | `listed_count` | Curatorship — indicateur de crédibilité |
| 6 | `followers / (friends + 1)` | **Ratio followers/friends** — les bots ont souvent un ratio très bas (suivent beaucoup, peu de retour) ou artificiellement élevé |
| 7 | `statuses / (followers + 1)` | **Ratio tweets/followers** — un compte qui tweete énormément pour peu de followers est suspect |
| 8 | `listed / (followers + 1)` | Ratio qualité d'audience — les humains influents sont plus souvent dans des listes |
| 9 | `age_days` | Ancienneté du compte (jours depuis `created_at` jusqu'au 01/02/2022) |
| 10 | `statuses / (age_days + 1)` | **Tweets par jour** — les bots ont souvent un rythme de publication anormalement régulier ou élevé |
| 11 | `len(name)` | Longueur du nom affiché |
| 12 | `len(screen_name)` | Longueur du nom d'utilisateur |
| 13 | `len(description)` | Longueur de la bio |
| 14 | `digits_in_screen_name` | **Nombre de chiffres dans le pseudo** — les bots générés automatiquement ont souvent des noms type `user389271` |

**6 features binaires** (0 ou 1, non normalisées) :

| # | Feature | Intuition |
|---|---------|-----------|
| 15 | `verified` | Les comptes vérifiés sont presque toujours humains |
| 16 | `default_profile` | Profil non personnalisé — signal de bot |
| 17 | `default_profile_image` | Pas de photo de profil — signal fort de bot |
| 18 | `has_url` | Présence d'un lien dans le profil |
| 19 | `has_location` | Localisation renseignée |
| 20 | `has_description` | Bio non vide |

**Normalisation :** le `StandardScaler` de scikit-learn est fit uniquement sur les noeuds labellisés (pour éviter que les ~180K noeuds à zéro ne biaisent la moyenne). Les noeuds non labellisés (voisins sans profil) conservent des features à zéro après normalisation, ce qui correspond approximativement à la moyenne.

### `models.py`

Définit les deux architectures GNN et une factory `build_model()`.

**`GATBotDetector`** : 2 couches GATConv + classifieur linéaire. La première couche avec multi-head attention (4 têtes), la seconde avec une seule tête. Total : ~78K paramètres.

**`RGCNBotDetector`** : 2 couches RGCNConv + classifieur linéaire. Chaque couche apprend des poids séparés pour les relations following et follower.

Les deux modèles sauvegardent les embeddings de l'avant-dernière couche dans `self.embedding` pour la visualisation t-SNE.

### `train.py`

Boucle d'entraînement **full-batch** : à chaque époque, un forward pass complet sur tout le graphe (191K noeuds, 208K arêtes).

**Pourquoi full-batch et non mini-batch ?**
Le mini-batch via `NeighborLoader` de PyG nécessite `pyg-lib` ou `torch-sparse`, qui ne sont pas disponibles pour toutes les combinaisons PyTorch/Python/OS. Notre graphe est suffisamment petit pour tenir en mémoire (contrairement au graphe complet TwiBot-20 de 33M arêtes mentionnés dans le papier — notre sous-ensemble n'en contient que 208K).

**Fonctionnement :**
- **Forward pass** : le modèle traite tous les noeuds en une seule passe
- **Loss** : `CrossEntropyLoss` calculée uniquement sur les noeuds `train_mask` (8,278 noeuds)
- **Validation** : évaluée sur `val_mask` après chaque époque
- **Early stopping** : si la val loss ne s'améliore pas pendant `patience` époques consécutives, l'entraînement s'arrête et le meilleur modèle est restauré
- **Checkpoint** : le meilleur modèle (selon la val loss) est sauvegardé dans `checkpoints/`

**Détection du device** : auto-détection CUDA > MPS > CPU.

### `evaluate.py`

Évaluation et export des résultats.

**`evaluate_test()`** : calcule les métriques sur le test set :
- Accuracy, F1-score, Precision, Recall
- Matrice de confusion
- Classification report détaillé

**`save_probabilities()`** : exporte un CSV avec les colonnes :
- `user_id` : identifiant Twitter
- `split` : train / val / test
- `true_label` : 0 (humain) ou 1 (bot)
- `p_bot_graph` : probabilité P(bot) prédite par le GNN

Ce CSV est le **livrable principal** du module : il sera fusionné avec les probabilités des modules feature-based et text-based pour le stacking final.

**`compute_probabilities()`** : fait un forward pass complet, applique un softmax sur les logits, et retourne P(bot) = softmax(logits)[:, 1] pour chaque noeud labellisé. Retourne également les embeddings de l'avant-dernière couche pour la visualisation.

### `visualize.py`

Génère 4 visualisations sauvegardées en PNG dans `output/` :

1. **Courbes de loss** : train loss et val loss au fil des époques, + courbes d'accuracy. Permet de vérifier la convergence et le surentraînement.

2. **Matrice de confusion** : visualisation des vrais positifs, faux positifs, vrais négatifs, faux négatifs sur le test set.

3. **Distribution de P(bot)** : histogrammes superposés des probabilités P(bot) pour les humains vs les bots. Un bon modèle montrera deux distributions bien séparées, avec les humains à gauche (proche de 0) et les bots à droite (proche de 1).

4. **t-SNE des embeddings** : projection 2D des représentations apprises par le GNN (avant-dernière couche). Permet de visualiser si le GNN a appris à séparer les deux classes dans l'espace latent. Sous-échantillonné à 3000 points pour la lisibilité.

### `main.py`

Point d'entrée qui orchestre le pipeline complet :

1. Parse les arguments CLI
2. Fixe les seeds (reproductibilité)
3. Charge les données et construit le graphe
4. Instancie le modèle
5. Entraîne avec early stopping
6. Évalue sur le test set
7. Exporte les probabilités en CSV
8. Génère les visualisations

---

## 5. Installation et exécution

### Dépendances

```bash
pip install torch torch-geometric scikit-learn pandas numpy matplotlib seaborn
```

Sur un serveur avec GPU CUDA (ex: Onyxia), installer aussi les extensions PyG pour le mini-batch :
```bash
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-<VERSION>+<CUDA>.html
```

### Exécution

```bash
cd graph/

# GAT (défaut)
python main.py --model gat --epochs 50

# R-GCN
python main.py --model rgcn --epochs 50

# Sur CPU (si MPS pose problème)
python main.py --model gat --device cpu

# Sans t-SNE (plus rapide)
python main.py --model gat --skip_tsne

# Tous les arguments
python main.py --model rgcn --epochs 100 --lr 0.001 --hidden_dim 128 --dropout 0.3 --patience 10 --seed 42 --device auto
```

---

## 6. Hyperparamètres

Les valeurs par défaut sont calibrées d'après la littérature (BotRGCN, TwiBot-22 benchmark) :

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Hidden dim | 128 | Bon compromis expressivité / surentraînement pour ~12K noeuds labellisés |
| GAT heads | 4 | Standard dans la littérature GAT. Plus de têtes = plus de sous-espaces d'attention |
| Dropout | 0.3 | Régularisation modérée — les GNN surentraînent facilement sur des petits graphes |
| LR | 1e-3 | Standard pour Adam sur des tâches de classification |
| Weight decay | 5e-4 | Régularisation L2 légère |
| Patience | 10 | Assez d'époques pour confirmer que le modèle stagne |

---

## 7. Outputs

Après exécution, le dossier `output/` contient :

```
output/
  graph_probabilities_gat.csv       # P(bot) pour chaque utilisateur labellisé
  loss_curves_gat.png               # Courbes train/val loss + accuracy
  confusion_matrix_gat.png          # Matrice de confusion test set
  prob_distribution_gat.png         # Distribution P(bot) humains vs bots
  tsne_embeddings_gat.png           # t-SNE des embeddings GNN
```

Et le dossier `checkpoints/` contient le meilleur modèle :

```
checkpoints/
  best_gat.pt                       # State dict du meilleur modèle
```

---

## 8. Performances attendues

D'après le benchmark TwiBot-22 (Feng et al., NeurIPS 2022 Datasets Track) :

| Méthode | Accuracy sur TwiBot-20 |
|---------|------------------------|
| GCN | 77.5% |
| GAT | 83.3% |
| BotRGCN | 85.8% |
| RGT | 86.6% |

Avec nos 20 features de profil (sans embeddings textuels), nous visons **80-85% accuracy**. L'écart avec BotRGCN/RGT s'explique par l'absence de features textuelles (embeddings de tweets, embeddings de description), qui seront apportées par le module text-based dans le stacking final.

---

## Références

- Veličković et al. (2018). *Graph Attention Networks.* ICLR 2018.
- Schlichtkrull et al. (2018). *Modeling Relational Data with Graph Convolutional Networks.* ESWC 2018.
- Gilmer et al. (2017). *Neural Message Passing for Quantum Chemistry.* ICML 2017.
- Feng et al. (2021). *TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark.* CIKM 2021.
- Feng et al. (2022). *TwiBot-22: Towards Graph-Based Twitter Bot Detection.* NeurIPS 2022 Datasets Track.
