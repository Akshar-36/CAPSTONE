import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from trend import calculate_trend_signals

def get_candidates_for_brand(brand_text, trends_df, h_emb_df, category_centroids, model, config):
    brand_emb = model.encode([brand_text])[0]
    
    # Map to category
    best_cat = None
    best_sim = -1
    for cat, cent in category_centroids.items():
        sim = cosine_similarity([brand_emb], [cent])[0][0]
        if sim > best_sim:
            best_sim = sim
            best_cat = cat
            
    if best_sim < config['tau']:
        best_cat = 'all'
        
    pool_trends = trends_df[trends_df['pool'] == best_cat]
    if pool_trends.empty:
        return pd.DataFrame(), best_cat
        
    eligible_hashtags = pool_trends['hashtag'].tolist()
    
    pool_emb_df = h_emb_df[h_emb_df['hashtag'].isin(eligible_hashtags)]
    
    if pool_emb_df.empty:
        return pd.DataFrame(), best_cat
        
    emb_matrix = np.stack(pool_emb_df['embedding'].values)
    sims = cosine_similarity([brand_emb], emb_matrix)[0]
    
    pool_emb_df = pool_emb_df.copy()
    pool_emb_df['relevance'] = sims
    pool_emb_df = pool_emb_df.reset_index()
    
    candidates = pool_emb_df.sort_values('relevance', ascending=False).head(config['candidate_pool_size'])
    candidates = candidates.merge(pool_trends[['hashtag', 'trend_score', 'trend_direction']], on='hashtag', how='inner')
    
    return candidates, best_cat

def calculate_historical_engagement(pool, candidate_hashtags, history_df):
    pool_history = history_df[history_df['pool'] == pool]
    hist_eng = pool_history[pool_history['hashtag'].isin(candidate_hashtags)][['hashtag', 'historical_engagement']]
    return hist_eng

def rank_hashtags(candidates, config):
    if candidates.empty:
        return candidates
        
    def min_max_norm(series):
        if series.max() == series.min():
            return np.zeros(len(series))
        return (series - series.min()) / (series.max() - series.min())
        
    c = candidates.copy()
    c['norm_relevance'] = min_max_norm(c['relevance'])
    c['norm_trend'] = min_max_norm(c['trend_score'])
    c['norm_history'] = min_max_norm(c['historical_engagement'])
    
    w_rel, w_trend, w_hist = config['weights_hashtag']
    c['score'] = (w_rel * c['norm_relevance']) + (w_trend * c['norm_trend']) + (w_hist * c['norm_history'])
    
    c = c.sort_values('score', ascending=False).reset_index(drop=True)
    return c
