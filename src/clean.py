import os
import yaml
import psycopg2
import pandas as pd
import hashlib
from dotenv import load_dotenv, find_dotenv

# Get project root from find_dotenv()
PROJECT_ROOT = os.path.dirname(find_dotenv())

def load_config():
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def clean_data():
    print("Task: Phase 2 Data Cleaning started")
    print("CHECKPOINT 1 - Configuration loaded: DONE")
    
    load_dotenv(find_dotenv())
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    
    print("CHECKPOINT 2 - Database connected: DONE")
    
    # Read posts and influencers
    query_posts = """
    SELECT 
        rp.post_id, rp.username, rp.caption, rp.likes, rp.comments,
        rp.media_type, rp.sidecar_child_count, rp.usertag_count,
        rp.is_sponsored, rp.comments_disabled, rp.posted_at,
        i.category, i.followers
    FROM raw_posts rp
    LEFT JOIN influencers i ON rp.username = i.username
    """
    df_posts = pd.read_sql(query_posts, conn)
    print(f"CHECKPOINT 3 - Posts loaded ({len(df_posts)} rows): DONE")
    
    # Read hashtags
    df_hashtags = pd.read_sql("SELECT post_id, hashtag FROM post_hashtags", conn)
    print(f"CHECKPOINT 4 - Hashtags loaded ({len(df_hashtags)} rows): DONE")
    conn.close()
    
    # Normalize category
    df_posts['category'] = df_posts['category'].replace('fasion', 'fashion')
    
    # has_followers flag
    df_posts['has_followers'] = df_posts['followers'].notnull()
    
    # Engagement rate
    # Computed only where comments are non-null and followers > 0
    mask_er = df_posts['comments'].notnull() & (df_posts['followers'] > 0) & df_posts['has_followers']
    df_posts['engagement_rate'] = None
    df_posts.loc[mask_er, 'engagement_rate'] = (df_posts['likes'] + df_posts['comments']) / df_posts['followers']
    df_posts['engagement_rate'] = df_posts['engagement_rate'].astype(float)
    
    # Likes rate
    mask_lr = (df_posts['followers'] > 0) & df_posts['has_followers']
    df_posts['likes_rate'] = None
    df_posts.loc[mask_lr, 'likes_rate'] = df_posts['likes'] / df_posts['followers']
    df_posts['likes_rate'] = df_posts['likes_rate'].astype(float)
    
    # Flags
    config = load_config()
    min_followers = config.get('min_followers')
    if min_followers is not None:
        df_posts['low_followers'] = df_posts['followers'] < min_followers
    else:
        df_posts['low_followers'] = False
        
    df_posts['er_gt_1'] = df_posts['engagement_rate'] > 1.0
    
    print("CHECKPOINT 5 - Data cleaning applied: DONE")
    
    # Save to parquet
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    df_posts.to_parquet(os.path.join(output_dir, 'posts_clean.parquet'), engine='pyarrow')
    df_hashtags.to_parquet(os.path.join(output_dir, 'hashtags_clean.parquet'), engine='pyarrow')
    print("CHECKPOINT 6 - Parquet files saved: DONE")
    
    # Data Dictionary and Quality Report
    report = []
    report.append("# Phase 2 Quality Report and Data Dictionary\n")
    report.append("## Dataset Version Tag")
    report.append(f"- **Posts Row Count**: {len(df_posts)}")
    report.append(f"- **Hashtags Row Count**: {len(df_hashtags)}")
    
    # Hash of post_ids to uniquely identify dataset content
    post_ids_joined = "".join(sorted(df_posts['post_id'].astype(str).tolist()))
    dataset_hash = hashlib.sha256(post_ids_joined.encode('utf-8')).hexdigest()
    report.append(f"- **Dataset Hash (SHA256 of post_ids)**: `{dataset_hash}`\n")
    
    report.append("## Duplicate Report")
    duplicate_posts = df_posts['post_id'].duplicated().sum()
    report.append(f"- **Duplicate post_ids**: {duplicate_posts}")
    if duplicate_posts == 0:
        report.append("- *No duplicates found, data is clean.*")
    report.append("\n## Missing Values Report")
    
    missing = df_posts.isnull().sum()
    report.append("| Column | Missing Count | % Missing |")
    report.append("|---|---|---|")
    for col, count in missing.items():
        pct = (count / len(df_posts)) * 100
        report.append(f"| {col} | {count} | {pct:.2f}% |")
        
    report.append("\n## Data Dictionary")
    report.append("| Column | Type | Description |")
    report.append("|---|---|---|")
    report.append("| `post_id` | string | Unique identifier for the Instagram post |")
    report.append("| `username` | string | Author's username |")
    report.append("| `caption` | string | Full text caption of the post |")
    report.append("| `likes` | integer | Number of likes on the post |")
    report.append("| `comments` | integer | Number of comments (null if disabled or missing) |")
    report.append("| `media_type` | string | GraphImage, GraphSidecar, or GraphVideo |")
    report.append("| `sidecar_child_count` | integer | Number of images/videos in a carousel |")
    report.append("| `usertag_count` | integer | Number of tagged users in the post |")
    report.append("| `is_sponsored` | boolean | True if sponsored edge exists |")
    report.append("| `comments_disabled` | boolean | True if comments were disabled by the author |")
    report.append("| `posted_at` | datetime | UTC timestamp of when the post was created |")
    report.append("| `category` | string | Influencer category (e.g. fashion, beauty) |")
    report.append("| `followers` | integer | Number of followers of the author at scrape time |")
    report.append("| `has_followers` | boolean | True if a match in influencers.txt was found |")
    report.append("| `engagement_rate` | float | (likes + comments) / followers |")
    report.append("| `likes_rate` | float | likes / followers |")
    report.append("| `low_followers` | boolean | True if followers < min_followers |")
    report.append("| `er_gt_1` | boolean | True if engagement_rate > 1.0 |")
    
    report_path = os.path.join(output_dir, 'quality_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
        
    print("CHECKPOINT 7 - Quality report generated: DONE")
    print("DONE!! - Phase 2 data cleaning completed successfully.")

if __name__ == '__main__':
    clean_data()
