import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

def get_brand_text(description, industry, target_audience):
    """Combine brand inputs into a single text for relevance matching."""
    return f"{description} {industry} {target_audience}"

def compute_centroid_embeddings(posts_train, model, sample_size, seed):
    """
    Compute mean embedding for each category based on a seeded sample of training captions.
    Returns a dictionary of category -> embedding vector.
    """
    categories = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel', 'other']
    centroids = {}
    for cat in categories:
        cat_posts = posts_train[posts_train['category'] == cat]
        n_sample = min(len(cat_posts), sample_size)
        if n_sample == 0:
            continue
        sampled = cat_posts.sample(n=n_sample, random_state=seed)
        captions = sampled['caption'].fillna('').tolist()
        embs = model.encode(captions)
        centroids[cat] = np.mean(embs, axis=0)
    return centroids

def get_hashtag_text(hashtag, captions):
    """Concatenate hashtag and its sampled captions."""
    text = hashtag
    for cap in captions:
        if str(cap).strip():
            text += f" {cap}"
    return text

def compute_hashtag_embeddings(eligible_hashtags, hashtags_df, posts_df, model, sample_size, seed):
    """
    Compute embeddings for each eligible hashtag using concatenation of its captions.
    Optimized with groupby to avoid O(N^2) filtering.
    """
    valid_hashtags = hashtags_df[hashtags_df['hashtag'].isin(set(eligible_hashtags))]
    grouped = valid_hashtags.groupby('hashtag')
    post_captions = posts_df.set_index('post_id')['caption']
    
    texts = []
    ht_list = []
    
    for h in eligible_hashtags:
        try:
            ph = grouped.get_group(h)
        except KeyError:
            ph = pd.DataFrame(columns=hashtags_df.columns)
            
        n_sample = min(len(ph), sample_size)
        if n_sample > 0:
            sampled = ph.sample(n=n_sample, random_state=seed)
            caps = post_captions.reindex(sampled['post_id']).fillna('').tolist()
        else:
            caps = []
            
        text = get_hashtag_text(h, caps)
        texts.append(text)
        ht_list.append(h)
        
    logging.info(f"Encoding {len(texts)} hashtag texts with MiniLM...")
    embs = model.encode(texts)
    
    return pd.DataFrame({
        'hashtag': ht_list,
        'embedding': list(embs)
    })

def predict_category(brand_text, model, centroids, tau):
    """
    Predict category for a brand text.
    Returns (category, score, match_type).
    """
    b_emb = model.encode([brand_text])[0]
    
    best_cat = None
    best_score = -1.0
    
    for cat, c_emb in centroids.items():
        if cat == 'other': continue
        score = cosine_similarity([b_emb], [c_emb])[0][0]
        if score > best_score:
            best_score = score
            best_cat = cat
            
    if best_score < tau:
        return 'all', best_score, 'fallback_all'
    
    return best_cat, best_score, 'matched'

def calculate_brand_relevance(brand_text, model, ht_embeddings_df):
    """
    Calculate relevance between a brand text and a pool of hashtag embeddings.
    """
    b_emb = model.encode([brand_text])[0]
    embs = np.vstack(ht_embeddings_df['embedding'].values)
    sims = cosine_similarity([b_emb], embs)[0]
    
    res = ht_embeddings_df[['hashtag']].copy()
    res['relevance'] = sims
    return res
