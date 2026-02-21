# Graph-Based Twitter Bot Detection

Module de detection de bots Twitter par **Graph Neural Networks** (GNN) sur le dataset **TwiBot-20**. Ce module produit une probabilite P(bot) par utilisateur, destinee a etre combinee avec les modules feature-based et text-based via stacking.

---

## Table des matieres

1. [Fondements theoriques](#1-fondements-theoriques)
   - [Pourquoi un graphe ?](#11-pourquoi-un-graphe-)
   - [Message Passing Neural Networks](#12-message-passing-neural-networks)
   - [GAT : Graph Attention Network](#13-gat--graph-attention-network)
   - [R-GCN : Relational Graph Convolutional Network](#14-r-gcn--relational-graph-convolutional-network)
   - [Transductive Learning](#15-transductive-learning)
2. [Dataset TwiBot-20](#2-dataset-twibot-20)
3. [Architecture du pipeline](#3-architecture-du-pipeline)
4. [Description des fichiers](#4-description-des-fichiers)
5. [Installation et execution](#5-installation-et-execution)
6. [Hyperparametres](#6-hyperparametres)
7. [Outputs](#7-outputs)
8. [Performances attendues](#8-performances-attendues)

---

## 1. Fondements theoriques

### 1.1 Pourquoi un graphe ?

Twitter est naturellement un **graphe social dirige** : les utilisateurs se suivent mutuellement (ou non), formant un reseau de relations. L'hypothese centrale est que les bots ont des **patterns structurels** distincts des humains :

- **Comportement de suivi anormal** : les bots suivent en masse d'autres comptes mais sont peu suivis en retour, ou inversement ils se suivent entre eux (clusters de bots).
- **Homophilie** : les bots tendent a interagir avec d'autres bots. Un utilisateur dont la majorite des voisins sont des bots a une forte probabilite d'etre lui-meme un bot.
- **Position topologique** : les bots occupent des positions specifiques dans le graphe (peripherie, hubs artificiels, composantes denses).

Un classifieur classique (features-only) ignore ces signaux relationnels. Un GNN les exploite directement en propageant de l'information le long des aretes du graphe.

### 1.2 Message Passing Neural Networks

Les GNN modernes reposent sur le paradigme de **message passing** (Gilmer et al., 2017). A chaque couche, chaque noeud :

1. **Collecte** les representations de ses voisins (messages)
2. **Agrege** ces messages (somme, moyenne, attention...)
3. **Met a jour** sa propre representation en combinant son etat actuel et l'agregation recue

Formellement, pour un noeud `v` a la couche `l` :

```
m_v^(l) = AGGREGATE({ h_u^(l-1) : u in N(v) })
h_v^(l) = UPDATE(h_v^(l-1), m_v^(l))
```

Ou `N(v)` designe l'ensemble des voisins de `v`, `h_v^(l)` sa representation a la couche `l`, et `AGGREGATE`/`UPDATE` sont des fonctions differentiables apprises.

Avec **2 couches** de message passing, chaque noeud integre l'information de ses **voisins a 2 sauts** (2-hop neighborhood). Cela signifie que la representation d'un utilisateur encode non seulement son propre profil, mais aussi les profils de ses amis et des amis de ses amis.

### 1.3 GAT : Graph Attention Network

Le **Graph Attention Network** (Velickovic et al., 2018) introduit un mecanisme d'**attention** dans l'aggregation : tous les voisins ne contribuent pas de maniere egale.

**Mecanisme d'attention :**

Pour chaque paire de noeuds connectes `(v, u)`, un coefficient d'attention `alpha_vu` est calcule :

```
e_vu = LeakyReLU(a^T [W*h_v || W*h_u])
alpha_vu = softmax_u(e_vu) = exp(e_vu) / sum_{k in N(v)} exp(e_vk)
```

Ou `W` est une matrice de projection apprise, `||` denote la concatenation, et `a` est un vecteur d'attention appris. Le softmax normalise les coefficients sur l'ensemble des voisins.

La nouvelle representation est alors :

```
h_v^(l) = ELU( sum_{u in N(v)} alpha_vu * W * h_u^(l-1) )
```

**Multi-head attention :**

Pour stabiliser l'apprentissage, GAT utilise `K` tetes d'attention independantes dont les resultats sont concatenes (couche intermediaire) ou moyennes (couche finale) :

```
h_v^(l) = || _{k=1}^{K} ELU( sum_{u in N(v)} alpha_vu^k * W^k * h_u^(l-1) )
```

**Intuition pour la detection de bots :** L'attention permet au modele d'apprendre a ponderer differemment les voisins. Par exemple, un voisin verifie et ancien devrait avoir plus d'influence qu'un compte recemment cree avec zero tweet. Le modele apprend ces ponderations automatiquement.

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

- La 1ere couche utilise **4 tetes d'attention** concatenees : 4 x 128 = 512 dimensions
- La 2eme couche utilise **1 seule tete** pour reduire a 128 dimensions
- L'activation **ELU** (au lieu de ReLU) est le choix standard pour GAT car elle permet des valeurs negatives, evitant le probleme des "neurones morts"
- GAT traite toutes les aretes uniformement (pas de distinction following/follower au niveau du type d'arete)

### 1.4 R-GCN : Relational Graph Convolutional Network

Le **Relational GCN** (Schlichtkrull et al., 2018) etend le GCN classique aux **graphes multi-relationnels** : chaque type d'arete a ses propres poids.

**Formulation :**

```
h_v^(l) = ReLU( sum_{r in R} sum_{u in N_r(v)} (1/c_{v,r}) * W_r^(l) * h_u^(l-1) + W_0^(l) * h_v^(l-1) )
```

Ou :
- `R` est l'ensemble des types de relations (ici : `{following, follower}`)
- `N_r(v)` sont les voisins de `v` via la relation `r`
- `W_r^(l)` est la matrice de poids **specifique a la relation `r`** a la couche `l`
- `W_0^(l)` est une transformation "self-loop" (le noeud preserve sa propre information)
- `c_{v,r}` est un facteur de normalisation (typiquement `|N_r(v)|`)

**Intuition pour la detection de bots :** La distinction entre "following" et "follower" est cruciale. Un bot qui suit 10 000 comptes mais n'est suivi par personne a un pattern tres different d'un humain influent. R-GCN apprend des transformations separees pour chaque direction, capturant cette asymetrie.

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

- Les **~180K utilisateurs non labellises** participent au graphe et au message passing
- Mais la **loss n'est calculee que sur les noeuds labellises** du train set
- A l'inference, le modele produit des predictions pour tous les noeuds, y compris ceux vus pendant l'entrainement

Cela signifie que les noeuds non labellises enrichissent les representations via la propagation d'information, meme s'ils ne contribuent pas directement a la loss. C'est un avantage majeur des GNN : exploiter la structure du graphe complet.

---

## 2. Dataset TwiBot-20

Le dataset provient du benchmark **TwiBot-20** (Feng et al., 2021), reference pour la detection de bots Twitter.

### Structure des donnees

```
archive/
  train.json    # 8,278 utilisateurs (3,632 humains, 4,646 bots)
  dev.json      # 2,365 utilisateurs (1,062 humains, 1,303 bots)
  test.json     # 1,183 utilisateurs (543 humains, 640 bots)
```

Chaque entree JSON contient :

| Champ       | Description |
|-------------|-------------|
| `ID`        | Identifiant Twitter unique |
| `profile`   | Metadonnees du profil (followers_count, friends_count, verified, created_at, ...) |
| `tweet`     | Liste des 200 derniers tweets (texte brut) |
| `neighbor`  | Dict `{"following": [id1, ...], "follower": [id2, ...]}` — max 10 par type |
| `domain`    | Categories thematiques du compte |
| `label`     | `"0"` = humain, `"1"` = bot |

### Graphe construit

A partir des champs `neighbor`, nous construisons un graphe dirige :

- **191,582 noeuds** : 11,826 labellises + 179,756 voisins non labellises
- **208,716 aretes** : 105,701 "following" + 103,015 "follower"
- Les voisins non labellises n'ont pas de profil dans le dataset : leurs features sont initialisees a zero

---

## 3. Architecture du pipeline

```
main.py                    # Orchestration
  |
  |-- data_loader.py       # 1. Charge les JSON, construit le graphe PyG
  |     |
  |     +-- feature_extractor.py   # Extrait 20 features par noeud
  |
  |-- models.py            # 2. Definit GAT et R-GCN
  |
  |-- train.py             # 3. Boucle d'entrainement full-batch + early stopping
  |
  |-- evaluate.py          # 4. Metriques test + export P(bot) en CSV
  |
  +-- visualize.py         # 5. Courbes de loss, confusion matrix, t-SNE
```

**Flux de donnees :**

```
JSON files
    |
    v
[data_loader] -> Data(x, edge_index, edge_type, y, masks)
    |
    v
[train] -> forward pass complet sur tout le graphe
    |        loss calculee uniquement sur train_mask
    |        early stopping sur val_mask
    v
[evaluate] -> metriques sur test_mask
    |          P(bot) pour tous les labellises -> CSV
    v
[visualize] -> plots PNG
```

---

## 4. Description des fichiers

### `config.py`

Dataclass centralisant tous les hyperparametres et chemins. Les valeurs par defaut sont ajustables via les arguments CLI de `main.py`.

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| `model_type` | `"gat"` | Architecture : `"gat"` ou `"rgcn"` |
| `hidden_dim` | `128` | Dimension des couches cachees du GNN |
| `gat_heads` | `4` | Nombre de tetes d'attention (GAT uniquement) |
| `num_relations` | `2` | Nombre de types de relations (R-GCN) |
| `dropout` | `0.3` | Taux de dropout |
| `lr` | `1e-3` | Learning rate (Adam) |
| `weight_decay` | `5e-4` | Regularisation L2 |
| `epochs` | `100` | Nombre max d'epoques |
| `patience` | `10` | Epoques sans amelioration avant early stopping |
| `seed` | `42` | Graine aleatoire pour reproductibilite |
| `collection_date` | `"2022-02-01"` | Date de reference pour calculer l'age des comptes |

### `data_loader.py`

Charge les 3 fichiers JSON et construit un objet `torch_geometric.data.Data`.

**Etapes :**
1. **Chargement** : `train.json`, `dev.json`, `test.json`
2. **Indexation** : mapping `user_id -> index` (0 a N-1). Les utilisateurs labellises recevront les indices 0 a 11825, puis les voisins non labellises prennent les indices suivants.
3. **Construction des aretes** : pour chaque utilisateur labellise, ses listes `following` et `follower` deviennent des aretes dirigees dans `edge_index` (tensor `[2, E]`), avec un `edge_type` (0 ou 1).
4. **Extraction des features** : appelle `feature_extractor.py` pour construire la matrice `x` de shape `[N, 20]`.
5. **Labels et masques** : tensor `y` (`0`=humain, `1`=bot, `-1`=non labellise) et masques booleens `train_mask`, `val_mask`, `test_mask`.

**Convention des aretes :**
- `following` (type 0) : `user -> target` (l'utilisateur suit la cible)
- `follower` (type 1) : `follower -> user` (le suiveur suit l'utilisateur)

### `feature_extractor.py`

Extrait un vecteur de **20 features** a partir du profil de chaque utilisateur.

**14 features numeriques** (normalisees par `StandardScaler`) :

| # | Feature | Intuition |
|---|---------|-----------|
| 1 | `followers_count` | Popularite du compte |
| 2 | `friends_count` | Nombre de comptes suivis |
| 3 | `statuses_count` | Volume total de tweets |
| 4 | `favourites_count` | Engagement passif |
| 5 | `listed_count` | Curatorship — indicateur de credibilite |
| 6 | `followers / (friends + 1)` | **Ratio followers/friends** — les bots ont souvent un ratio tres bas (suivent beaucoup, peu de retour) ou artificiellement eleve |
| 7 | `statuses / (followers + 1)` | **Ratio tweets/followers** — un compte qui tweete enormement pour peu de followers est suspect |
| 8 | `listed / (followers + 1)` | Ratio qualite d'audience — les humains influents sont plus souvent dans des listes |
| 9 | `age_days` | Anciennete du compte (jours depuis `created_at` jusqu'au 01/02/2022) |
| 10 | `statuses / (age_days + 1)` | **Tweets par jour** — les bots ont souvent un rythme de publication anormalement regulier ou eleve |
| 11 | `len(name)` | Longueur du nom affiche |
| 12 | `len(screen_name)` | Longueur du nom d'utilisateur |
| 13 | `len(description)` | Longueur de la bio |
| 14 | `digits_in_screen_name` | **Nombre de chiffres dans le pseudo** — les bots generes automatiquement ont souvent des noms type `user389271` |

**6 features binaires** (0 ou 1, non normalisees) :

| # | Feature | Intuition |
|---|---------|-----------|
| 15 | `verified` | Les comptes verifies sont presque toujours humains |
| 16 | `default_profile` | Profil non personnalise — signal de bot |
| 17 | `default_profile_image` | Pas de photo de profil — signal fort de bot |
| 18 | `has_url` | Presence d'un lien dans le profil |
| 19 | `has_location` | Localisation renseignee |
| 20 | `has_description` | Bio non vide |

**Normalisation :** le `StandardScaler` de scikit-learn est fit uniquement sur les noeuds labellises (pour eviter que les ~180K noeuds a zero ne biaisent la moyenne). Les noeuds non labellises (voisins sans profil) conservent des features a zero apres normalisation, ce qui correspond approximativement a la moyenne.

### `models.py`

Definit les deux architectures GNN et une factory `build_model()`.

**`GATBotDetector`** : 2 couches GATConv + classifieur lineaire. La premiere couche avec multi-head attention (4 tetes), la seconde avec une seule tete. Total : ~78K parametres.

**`RGCNBotDetector`** : 2 couches RGCNConv + classifieur lineaire. Chaque couche apprend des poids separes pour les relations following et follower.

Les deux modeles sauvegardent les embeddings de l'avant-derniere couche dans `self.embedding` pour la visualisation t-SNE.

### `train.py`

Boucle d'entrainement **full-batch** : a chaque epoque, un forward pass complet sur tout le graphe (191K noeuds, 208K aretes).

**Pourquoi full-batch et non mini-batch ?**
Le mini-batch via `NeighborLoader` de PyG necessite `pyg-lib` ou `torch-sparse`, qui ne sont pas disponibles pour toutes les combinaisons PyTorch/Python/OS. Notre graphe est suffisamment petit pour tenir en memoire (contrairement au graphe complet TwiBot-20 de 33M aretes mentionnes dans le papier — notre sous-ensemble n'en contient que 208K).

**Fonctionnement :**
- **Forward pass** : le modele traite tous les noeuds en une seule passe
- **Loss** : `CrossEntropyLoss` calculee uniquement sur les noeuds `train_mask` (8,278 noeuds)
- **Validation** : evaluee sur `val_mask` apres chaque epoque
- **Early stopping** : si la val loss ne s'ameliore pas pendant `patience` epoques consecutives, l'entrainement s'arrete et le meilleur modele est restaure
- **Checkpoint** : le meilleur modele (selon la val loss) est sauvegarde dans `checkpoints/`

**Detection du device** : auto-detection CUDA > MPS > CPU.

### `evaluate.py`

Evaluation et export des resultats.

**`evaluate_test()`** : calcule les metriques sur le test set :
- Accuracy, F1-score, Precision, Recall
- Matrice de confusion
- Classification report detaille

**`save_probabilities()`** : exporte un CSV avec les colonnes :
- `user_id` : identifiant Twitter
- `split` : train / val / test
- `true_label` : 0 (humain) ou 1 (bot)
- `p_bot_graph` : probabilite P(bot) predite par le GNN

Ce CSV est le **livrable principal** du module : il sera fusionne avec les probabilites des modules feature-based et text-based pour le stacking final.

**`compute_probabilities()`** : fait un forward pass complet, applique un softmax sur les logits, et retourne P(bot) = softmax(logits)[:, 1] pour chaque noeud labellise. Retourne egalement les embeddings de l'avant-derniere couche pour la visualisation.

### `visualize.py`

Genere 4 visualisations sauvegardees en PNG dans `output/` :

1. **Courbes de loss** : train loss et val loss au fil des epoques, + courbes d'accuracy. Permet de verifier la convergence et le surentrainement.

2. **Matrice de confusion** : visualisation des vrais positifs, faux positifs, vrais negatifs, faux negatifs sur le test set.

3. **Distribution de P(bot)** : histogrammes superposes des probabilites P(bot) pour les humains vs les bots. Un bon modele montrera deux distributions bien separees, avec les humains a gauche (proche de 0) et les bots a droite (proche de 1).

4. **t-SNE des embeddings** : projection 2D des representations apprises par le GNN (avant-derniere couche). Permet de visualiser si le GNN a appris a separer les deux classes dans l'espace latent. Sous-echantillonne a 3000 points pour la lisibilite.

### `main.py`

Point d'entree qui orchestre le pipeline complet :

1. Parse les arguments CLI
2. Fixe les seeds (reproductibilite)
3. Charge les donnees et construit le graphe
4. Instancie le modele
5. Entraine avec early stopping
6. Evalue sur le test set
7. Exporte les probabilites en CSV
8. Genere les visualisations

---

## 5. Installation et execution

### Dependances

```bash
pip install torch torch-geometric scikit-learn pandas numpy matplotlib seaborn
```

Sur un serveur avec GPU CUDA (ex: Onyxia), installer aussi les extensions PyG pour le mini-batch :
```bash
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-<VERSION>+<CUDA>.html
```

### Execution

```bash
cd graph/

# GAT (defaut)
python main.py --model gat --epochs 50

# R-GCN
python main.py --model rgcn --epochs 50

# Sur CPU (si MPS pose probleme)
python main.py --model gat --device cpu

# Sans t-SNE (plus rapide)
python main.py --model gat --skip_tsne

# Tous les arguments
python main.py --model rgcn --epochs 100 --lr 0.001 --hidden_dim 128 --dropout 0.3 --patience 10 --seed 42 --device auto
```

---

## 6. Hyperparametres

Les valeurs par defaut sont calibrees d'apres la litterature (BotRGCN, TwiBot-22 benchmark) :

| Parametre | Valeur | Justification |
|-----------|--------|---------------|
| Hidden dim | 128 | Bon compromis expressivite / surentrainement pour ~12K noeuds labellises |
| GAT heads | 4 | Standard dans la litterature GAT. Plus de tetes = plus de sous-espaces d'attention |
| Dropout | 0.3 | Regularisation moderee — les GNN surentrainent facilement sur des petits graphes |
| LR | 1e-3 | Standard pour Adam sur des taches de classification |
| Weight decay | 5e-4 | Regularisation L2 legere |
| Patience | 10 | Assez d'epoques pour confirmer que le modele stagne |

---

## 7. Outputs

Apres execution, le dossier `output/` contient :

```
output/
  graph_probabilities_gat.csv       # P(bot) pour chaque utilisateur labellise
  loss_curves_gat.png               # Courbes train/val loss + accuracy
  confusion_matrix_gat.png          # Matrice de confusion test set
  prob_distribution_gat.png         # Distribution P(bot) humains vs bots
  tsne_embeddings_gat.png           # t-SNE des embeddings GNN
```

Et le dossier `checkpoints/` contient le meilleur modele :

```
checkpoints/
  best_gat.pt                       # State dict du meilleur modele
```

---

## 8. Performances attendues

D'apres le benchmark TwiBot-22 (Feng et al., NeurIPS 2022 Datasets Track) :

| Methode | Accuracy sur TwiBot-20 |
|---------|------------------------|
| GCN | 77.5% |
| GAT | 83.3% |
| BotRGCN | 85.8% |
| RGT | 86.6% |

Avec nos 20 features de profil (sans embeddings textuels), nous visons **80-85% accuracy**. L'ecart avec BotRGCN/RGT s'explique par l'absence de features textuelles (embeddings de tweets, embeddings de description), qui seront apportees par le module text-based dans le stacking final.

---

## References

- Velickovic et al. (2018). *Graph Attention Networks.* ICLR 2018.
- Schlichtkrull et al. (2018). *Modeling Relational Data with Graph Convolutional Networks.* ESWC 2018.
- Gilmer et al. (2017). *Neural Message Passing for Quantum Chemistry.* ICML 2017.
- Feng et al. (2021). *TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark.* CIKM 2021.
- Feng et al. (2022). *TwiBot-22: Towards Graph-Based Twitter Bot Detection.* NeurIPS 2022 Datasets Track.
