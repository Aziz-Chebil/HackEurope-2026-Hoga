def create_csv():
    import pandas as pd
    import json

    # ── Load all splits ───────────────────────────────────────────────────────────
    def load_json(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    train_data = load_json("train.json")
    dev_data   = load_json("dev.json")
    test_data  = load_json("test.json")
    def parse_entries(data, split_name):
        rows = []
        for entry in data:
            profile = entry.get('profile', {}) or {}
            tweets  = entry.get('tweet', [])  or []

            row = {
                # Identity
                'ID'              : entry.get('ID'),
                'split'           : split_name,
                'label'           : int(entry.get('label')),  # ← fixed


                # Profile features
                'screen_name'     : profile.get('screen_name', ''),
                'name'            : profile.get('name', ''),
                'description'     : profile.get('description', ''),
                'location'        : profile.get('location', ''),
                'followers_count' : profile.get('followers_count', 0),
                'friends_count'   : profile.get('friends_count', 0),
                'statuses_count'  : profile.get('statuses_count', 0),
                'favourites_count': profile.get('favourites_count', 0),
                'listed_count'    : profile.get('listed_count', 0),
                'verified'        : profile.get('verified', False),
                'created_at'      : profile.get('created_at', ''),
                'default_profile' : profile.get('default_profile', False),
                'default_profile_image': profile.get('default_profile_image', False),

                # Tweet features
                'tweet_count'     : len(tweets),
                'tweets': [t for t in tweets if isinstance(t, str)],

                # Graph features
                'neighbors'       : entry.get('neighbor', {}) or {},
                'domain'          : entry.get('domain', ''),
            }
            rows.append(row)
            return rows
    test_rows  = parse_entries(test_data,  'test')
    test_df  = pd.DataFrame(test_rows)
    return test_df


def predict_bot_probability(df, model, features):
    """
    Prend un dataset, retourne la probabilité que chaque utilisateur soit un bot ou un humain.
    
    Parameters:
        df      : DataFrame avec les features déjà créées
        model   : RandomForestClassifier déjà entraîné
        features: liste des features utilisées
    
    Returns:
        DataFrame avec les probabilités
    """
    df = df.copy()
    
    # Vérifier que toutes les features existent
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Features manquantes : {missing}")
    
    X = df[features]
    
    # predict_proba retourne [[prob_human, prob_bot], ...]
    probs = model.predict_proba(X)
    
    results = pd.DataFrame({
        'prob_human'   : probs[:, 0].round(3),
        'prob_bot'     : probs[:, 1].round(3),
        'prediction'   : ['Bot' if p > 0.5 else 'Human' for p in probs[:, 1]],
        'confidence'   : [max(p) for p in probs].copy()
    })
    
    
    return results
import pickle
import pandas as pd
# ── Charger le modèle ─────────────────────────────────────────────────────────
with open('bot_detector.pkl', 'rb') as f:
    saved = pickle.load(f)
import re
# ── Feature Engineering ───────────────────────────────────────────────────────
def create_features3(df):
    df = df.copy()

    # 1. Followers / Friends ratio
    df['followers_count'] = pd.to_numeric(df['followers_count'], errors='coerce').fillna(0)
    df['friends_count']   = pd.to_numeric(df['friends_count'],   errors='coerce').fillna(0)
    df['follow_ratio']    = df['followers_count'] / (df['friends_count'] + 1)  # +1 to avoid division by zero

    df['default_profile']       = df['default_profile']
    df['default_profile_image'] = df['default_profile_image']
    df['verified']              = df['verified']

    # 3. Description length
    df['description_length'] = df['description'].fillna('').apply(len)

    # 4. Verified (boolean → int)


    # 5. Numeric columns
    df['statuses_count'] = pd.to_numeric(df['statuses_count'], errors='coerce').fillna(0)
    df['listed_count']   = pd.to_numeric(df['listed_count'],   errors='coerce').fillna(0)

    from datetime import datetime, timezone

    df['account_age_days'] = df['created_at'].apply(
    lambda x: (datetime.now(timezone.utc) - pd.to_datetime(x, errors='coerce', utc=True)).days
    if pd.notnull(x) else 0
    )


    # Tweets par jour (activité)
    df['tweets_per_day'] = df['statuses_count'] / (df['account_age_days'] + 1)
        # Convertir la colonne tweets depuis string CSV vers liste
    df['tweets'] = df['tweets'].apply(
        lambda x: eval(x) if isinstance(x, str) else x
    )

        # Ratio de retweets
    df['retweet_ratio'] = df['tweets'].apply(
        lambda x: sum(1 for t in x if t.startswith('RT ')) / (len(x) + 1)
    )
     # Ratio de tweets avec URLs
    df['url_ratio'] = df['tweets'].apply(
        lambda x: sum(1 for t in x if 'https://' in t) / (len(x) + 1)
    )
    # Longueur moyenne des tweets
    df['avg_tweet_length'] = df['tweets'].apply(
        lambda x: sum(len(t) for t in x) / (len(x) + 1)
    )
        # Diversité des hashtags
    df['hashtag_diversity'] = df['tweets'].apply(
        lambda x: len(set(re.findall(r'#\w+', ' '.join(x)))) / (len(x) + 1)
    )
    bool_cols = ['default_profile', 'default_profile_image', 'verified']

    for col in bool_cols:
        df[col] = df[col].astype(str).str.strip().str.lower().eq('true')
        df[col]  = df[col].astype(str).str.strip().str.lower().eq('true')

    return df

model    = saved['model']
features = saved['features']

df = create_csv()
df = create_features3(df)

results = predict_bot_probability(df, model,features)