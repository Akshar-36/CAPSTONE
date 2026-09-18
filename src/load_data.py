import os
import glob
import csv
import psycopg2
from io import StringIO
from dotenv import load_dotenv
from parse_post import parse_post, load_influencers

load_dotenv()

def get_connection():
    return psycopg2.connect(os.environ["DATABASE_URL"])

def create_staging_tables(cursor):
    cursor.execute("""
    CREATE TEMP TABLE stg_influencers (LIKE influencers INCLUDING ALL);
    CREATE TEMP TABLE stg_raw_posts (LIKE raw_posts INCLUDING ALL);
    CREATE TEMP TABLE stg_post_hashtags (LIKE post_hashtags INCLUDING ALL);
    """)

def load_influencers_to_db(cursor, influencers_data):
    f = StringIO()
    writer = csv.writer(f)
    for username, data in influencers_data.items():
        writer.writerow([username, data['category'], data['followers']])
    
    f.seek(0)
    cursor.copy_expert("COPY stg_influencers(username, category, followers) FROM STDIN WITH CSV", f)
    
    cursor.execute("""
    INSERT INTO influencers (username, category, followers)
    SELECT username, category, followers FROM stg_influencers
    ON CONFLICT (username) DO NOTHING;
    """)

def load_posts_to_db(cursor, files):
    post_f = StringIO()
    hashtag_f = StringIO()
    
    post_writer = csv.writer(post_f)
    hashtag_writer = csv.writer(hashtag_f)
    
    empty_files = []
    error_files = []
    
    for filepath in files:
        if os.path.getsize(filepath) == 0:
            empty_files.append(filepath)
            continue
            
        try:
            row = parse_post(filepath)
        except Exception as e:
            error_files.append((filepath, str(e)))
            continue
            
        post_id = str(row['post_id'])
        username = row['username']
        caption = row['caption']
        likes = row['likes']
        comments = row['comments']
        media_type = row['media_type']
        sidecar_child_count = row['sidecar_child_count']
        usertag_count = row['usertag_count']
        is_sponsored = row['is_sponsored']
        comments_disabled = row['comments_disabled']
        posted_at = row['timestamp'].isoformat() if row['timestamp'] else None
        
        post_writer.writerow([
            post_id, username, caption, likes, comments, media_type,
            sidecar_child_count, usertag_count, is_sponsored, comments_disabled, posted_at
        ])
        
        unique_hashtags = set(h.lower() for h in row['hashtags'])
        for h in unique_hashtags:
            hashtag_writer.writerow([post_id, h])
            
    print(f"Empty files skipped: {len(empty_files)}")
    print(f"Error files skipped: {len(error_files)}")
    
    post_f.seek(0)
    hashtag_f.seek(0)
    
    cursor.copy_expert("COPY stg_raw_posts(post_id, username, caption, likes, comments, media_type, sidecar_child_count, usertag_count, is_sponsored, comments_disabled, posted_at) FROM STDIN WITH CSV", post_f)
    cursor.copy_expert("COPY stg_post_hashtags(post_id, hashtag) FROM STDIN WITH CSV", hashtag_f)
    
    cursor.execute("""
    INSERT INTO raw_posts (post_id, username, caption, likes, comments, media_type, sidecar_child_count, usertag_count, is_sponsored, comments_disabled, posted_at)
    SELECT post_id, username, caption, likes, comments, media_type, sidecar_child_count, usertag_count, is_sponsored, comments_disabled, posted_at 
    FROM stg_raw_posts
    ON CONFLICT (post_id) DO NOTHING;
    """)
    
    cursor.execute("""
    INSERT INTO post_hashtags (post_id, hashtag)
    SELECT post_id, hashtag FROM stg_post_hashtags
    ON CONFLICT (post_id, hashtag) DO NOTHING;
    """)

def main():
    conn = get_connection()
    conn.autocommit = False
    cursor = conn.cursor()
    
    try:
        create_staging_tables(cursor)
        
        print("Loading influencers...")
        influencers = load_influencers('data/raw/influencers.txt')
        load_influencers_to_db(cursor, influencers)
        
        print("Loading posts...")
        files = glob.glob('data/raw/info/*.info')
        load_posts_to_db(cursor, files)
        
        conn.commit()
        print("Load completed successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error loading data: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()
