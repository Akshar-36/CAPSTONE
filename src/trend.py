import pandas as pd
import numpy as np
import itertools
from scipy.stats import spearmanr
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def calculate_trend_signals(posts, hashtags, origin, config, pool='all'):
    period_days = config['period_days']
    origin_dt = pd.to_datetime(origin)
    origin = pd.to_datetime(origin, utc=True) if origin_dt.tzinfo is None else origin_dt
    prev_start = origin - pd.Timedelta(days=2 * period_days)
    curr_start = origin - pd.Timedelta(days=period_days)
    
    # filter time
    pool_posts = posts[(posts['posted_at'] > prev_start) & (posts['posted_at'] <= origin)].copy()
    if pool != 'all':
        pool_posts = pool_posts[pool_posts['category'] == pool]
        
    if pool_posts.empty:
        return pd.DataFrame()
        
    curr_pool_posts = pool_posts[pool_posts['posted_at'] > curr_start].copy()
    
    winsor_pct = config['er_winsor_pct']
    if not curr_pool_posts['engagement_rate'].isna().all():
        er_cap = curr_pool_posts['engagement_rate'].quantile(winsor_pct)
        curr_pool_posts['engagement_rate_win'] = curr_pool_posts['engagement_rate'].clip(upper=er_cap)
    else:
        curr_pool_posts['engagement_rate_win'] = np.nan
        
    pool_posts['is_current'] = pool_posts['posted_at'] > curr_start
    
    ph = pool_posts[['post_id', 'username', 'posted_at', 'is_current']].merge(hashtags, on='post_id')
    
    hashtag_stats = ph.groupby('hashtag').agg(
        posts_cur=('is_current', 'sum'),
        posts_prev=('is_current', lambda x: (~x).sum()),
        users_total=('username', 'nunique')
    ).reset_index()
    
    hashtag_stats['posts_total'] = hashtag_stats['posts_cur'] + hashtag_stats['posts_prev']
    
    eligible = hashtag_stats[
        (hashtag_stats['posts_total'] >= config['min_period_posts']) & 
        (hashtag_stats['users_total'] >= config['min_distinct_users'])
    ].copy()
    
    if eligible.empty:
        return pd.DataFrame()
        
    pool_cur_posts = pool_posts['is_current'].sum()
    pool_prev_posts = (~pool_posts['is_current']).sum()
    
    if pool_cur_posts == 0 or pool_prev_posts == 0:
        return pd.DataFrame()
        
    eligible['share_cur'] = eligible['posts_cur'] / pool_cur_posts
    eligible['share_prev'] = eligible['posts_prev'] / pool_prev_posts
    
    eligible['frequency_raw'] = eligible['share_cur'] * 10000
    
    a = config.get('growth_smoothing', 1e-4)
    eligible['growth_raw'] = np.log((eligible['share_cur'] + a) / (eligible['share_prev'] + a))
    
    curr_ph = curr_pool_posts[['post_id', 'posted_at', 'engagement_rate_win']].merge(hashtags, on='post_id')
    
    half_life = period_days / 4
    age_days = (origin - curr_ph['posted_at']).dt.total_seconds() / 86400.0
    curr_ph['recency_raw'] = np.exp(-np.log(2) * age_days / half_life)
    
    eng_rec = curr_ph.groupby('hashtag').agg(
        engagement_raw=('engagement_rate_win', 'mean'),
        recency_raw=('recency_raw', 'mean')
    ).reset_index()
    
    eligible = eligible.merge(eng_rec, on='hashtag', how='left')
    
    eligible['engagement_raw'] = eligible['engagement_raw'].fillna(0)
    eligible['recency_raw'] = eligible['recency_raw'].fillna(0)
    
    for col in ['frequency', 'growth', 'engagement', 'recency']:
        eligible[col] = eligible[f'{col}_raw'].rank(pct=True)
        
    eligible['pool'] = pool
    eligible['reference_date'] = origin.strftime('%Y-%m-%d')
    
    w = config.get('weights_trend')
    if w is not None:
        eligible['trend_score'] = (
            w[0] * eligible['frequency'] + 
            w[1] * eligible['growth'] + 
            w[2] * eligible['engagement'] + 
            w[3] * eligible['recency']
        )
        
        theta = config.get('theta')
        if theta is not None:
            eligible['trend_direction'] = 'stable'
            eligible.loc[eligible['growth_raw'] > theta, 'trend_direction'] = 'rising'
            eligible.loc[eligible['growth_raw'] < -theta, 'trend_direction'] = 'falling'
            
    return eligible

def prepare_backtest_data(posts, hashtags, origin, config, pool, a):
    cfg = config.copy()
    cfg['growth_smoothing'] = a
    # arbitrary weights for prep, overwritten during evaluation
    cfg['weights_trend'] = (0.25, 0.25, 0.25, 0.25)
    
    signals = calculate_trend_signals(posts, hashtags, origin, cfg, pool)
    if signals.empty:
        return None
        
    period_days = config['period_days']
    origin_dt = pd.to_datetime(origin, utc=True) if pd.to_datetime(origin).tzinfo is None else pd.to_datetime(origin)
    next_start = origin_dt
    next_end = origin_dt + pd.Timedelta(days=period_days)
    
    next_posts = posts[(posts['posted_at'] > next_start) & (posts['posted_at'] <= next_end)].copy()
    if pool != 'all':
        next_posts = next_posts[next_posts['category'] == pool]
        
    pool_next_posts = len(next_posts)
    if pool_next_posts == 0:
        return None
        
    next_ph = next_posts[['post_id']].merge(hashtags, on='post_id')
    next_counts = next_ph.groupby('hashtag').size().reset_index(name='posts_next')
    
    df = signals.merge(next_counts, on='hashtag', how='left')
    df['posts_next'] = df['posts_next'].fillna(0)
    df['share_next'] = df['posts_next'] / pool_next_posts
    
    df['target'] = np.log((df['share_next'] + 1e-5) / (df['share_cur'] + 1e-5))
    df = df.dropna(subset=['target'])
    
    if len(df) < 5:
        return None
        
    return df

def evaluate_backtest_weights(df, weights, origin, pool, a):
    df_eval = df.copy()
    df_eval['trend_score'] = (
        weights[0] * df_eval['frequency'] + 
        weights[1] * df_eval['growth'] + 
        weights[2] * df_eval['engagement'] + 
        weights[3] * df_eval['recency']
    )
    df_eval = df_eval.dropna(subset=['trend_score', 'target'])
    if len(df_eval) < 5:
        return None
        
    corr, _ = spearmanr(df_eval['trend_score'], df_eval['target'])
    corr_freq, _ = spearmanr(df_eval['frequency'], df_eval['target'])
    corr_growth, _ = spearmanr(df_eval['growth'], df_eval['target'])
    
    return {
        'pool': pool,
        'origin': origin,
        'weights': weights,
        'a': a,
        'corr': corr,
        'corr_freq': corr_freq,
        'corr_growth': corr_growth,
        'n_hashtags': len(df_eval)
    }

def run_backtest_for_origin(posts, hashtags, origin, config, pool, weights, a):
    df = prepare_backtest_data(posts, hashtags, origin, config, pool, a)
    if df is None:
        return None
    return evaluate_backtest_weights(df, weights, origin, pool, a)

def generate_weight_combinations():
    combos = []
    for w1 in np.arange(0, 1.1, 0.1):
        for w2 in np.arange(0, 1.1 - w1, 0.1):
            for w3 in np.arange(0, 1.1 - w1 - w2, 0.1):
                w4 = round(1.0 - w1 - w2 - w3, 1)
                if w4 >= -1e-9:
                    combos.append((round(w1,1), round(w2,1), round(w3,1), max(0.0, w4)))
    return combos
