import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
from dotenv import find_dotenv

PROJECT_ROOT = os.path.dirname(find_dotenv())

def load_config():
    with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
        return yaml.safe_load(f)

def build_features():
    print("Task: Phase 4 Feature Engineering started")
    config = load_config()
    print("CHECKPOINT 1 - Configuration loaded: DONE")
    
    # Load clean data
    df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'data/processed/posts_clean.parquet'))
    print(f"CHECKPOINT 2 - Data loaded ({len(df)} rows): DONE")
    
    # Filter out flagged rows (low_followers, er_gt_1) and null ER
    mask_valid = (~df['low_followers']) & (~df['er_gt_1']) & (df['engagement_rate'].notnull())
    df = df[mask_valid].copy()
    print(f"CHECKPOINT 3 - Filtered valid rows ({len(df)} rows): DONE")
    
    # Target
    df['y'] = np.log1p(100 * df['engagement_rate'])
    
    # Save original category before get_dummies
    df['original_category'] = df['category']
    
    # Cold-start features
    df['log_followers'] = np.log1p(df['followers'])
    df['hour_of_day'] = df['posted_at'].dt.hour
    df['day_of_week'] = df['posted_at'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Hashtag count
    ht_df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'data/processed/hashtags_clean.parquet'))
    ht_counts = ht_df.groupby('post_id').size().rename('hashtag_count')
    df = df.join(ht_counts, on='post_id')
    df['hashtag_count'] = df['hashtag_count'].fillna(0).astype(int)
    
    # Caption length
    df['caption_length'] = df['caption'].fillna('').str.len()
    
    # n_media
    df['n_media'] = np.where(df['media_type'] == 'GraphSidecar', df['sidecar_child_count'].fillna(1), 1).astype(int)
    
    # One-hot encoding for category and media_type
    df = pd.get_dummies(df, columns=['category', 'media_type'], drop_first=False)
    
    print("CHECKPOINT 4 - Cold-start features engineered: DONE")
    
    # Historical Features
    # Sort by account and time
    df = df.sort_values(['username', 'posted_at']).reset_index(drop=True)
    
    # historical_avg_engagement
    df['historical_avg_engagement'] = df.groupby('username')['engagement_rate'].transform(lambda x: x.expanding().mean().shift(1))
    
    # days_since_last_post
    df['days_since_last_post'] = df.groupby('username')['posted_at'].diff().dt.total_seconds() / (24*3600)
    
    # posts_last_30d
    df = df.set_index('posted_at')
    # using closed='left' so it doesn't count the current post
    posts_30d = df.groupby('username').rolling('30D', closed='left')['post_id'].count()
    posts_30d = posts_30d.reset_index(level=0, drop=True)
    df['posts_last_30d'] = posts_30d
    df = df.reset_index()
    
    print("CHECKPOINT 5 - Historical features engineered: DONE")
    
    # Primary split: account-holdout (80/10/10) stratified by category
    users = df[['username', 'original_category']].drop_duplicates('username')
    
    train_users, temp_users = train_test_split(
        users['username'], test_size=0.2, 
        stratify=users['original_category'], random_state=config['seed']
    )
    temp_categories = users[users['username'].isin(temp_users)]['original_category']
    val_users, test_users = train_test_split(
        temp_users, test_size=0.5, 
        stratify=temp_categories, random_state=config['seed']
    )
    
    df['split'] = 'train'
    df.loc[df['username'].isin(val_users), 'split'] = 'val'
    df.loc[df['username'].isin(test_users), 'split'] = 'test'
    
    # Robustness split: time split
    t_80, t_90 = pd.to_datetime(config['time_cutoffs'][0], utc=True), pd.to_datetime(config['time_cutoffs'][1], utc=True)
    df['time_split'] = 'train'
    df.loc[(df['posted_at'] > t_80) & (df['posted_at'] <= t_90), 'time_split'] = 'val'
    df.loc[df['posted_at'] > t_90, 'time_split'] = 'test'
    
    print("CHECKPOINT 6 - Data splits generated: DONE")
    
    # Save to parquet
    cat_cols = [c for c in df.columns if c.startswith('category_') or c.startswith('media_type_')]
    base_cols = ['post_id', 'username', 'posted_at', 'split', 'time_split', 'y', 'engagement_rate', 'original_category']
    cold_features = ['log_followers', 'hour_of_day', 'day_of_week', 'is_weekend', 'hashtag_count', 'caption_length', 'usertag_count', 'n_media'] + cat_cols
    
    cold_df = df[base_cols + cold_features]
    full_features = cold_features + ['historical_avg_engagement', 'days_since_last_post', 'posts_last_30d']
    full_df = df[base_cols + full_features]
    
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    cold_df.to_parquet(os.path.join(output_dir, 'features_cold.parquet'), engine='pyarrow')
    full_df.to_parquet(os.path.join(output_dir, 'features_full.parquet'), engine='pyarrow')
    
    print("CHECKPOINT 7 - Parquet files saved: DONE")
    print("DONE!! - Phase 4 Feature Engineering script completed successfully.")

if __name__ == '__main__':
    build_features()
