import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(os.environ["DATABASE_URL"])

def setup_db():
    conn = get_connection()
    conn.autocommit = True
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS influencers (
        username TEXT PRIMARY KEY,
        category TEXT,
        followers INTEGER
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS raw_posts (
        post_id TEXT PRIMARY KEY,
        username TEXT,
        caption TEXT,
        likes INTEGER,
        comments INTEGER,
        media_type TEXT,
        sidecar_child_count INTEGER,
        usertag_count INTEGER,
        is_sponsored BOOLEAN,
        comments_disabled BOOLEAN,
        posted_at TIMESTAMPTZ,
        loaded_at TIMESTAMPTZ DEFAULT now()
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS post_hashtags (
        post_id TEXT REFERENCES raw_posts(post_id),
        hashtag TEXT,
        PRIMARY KEY (post_id, hashtag)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trends (
        run_id TEXT,
        pool TEXT,
        hashtag TEXT,
        reference_date DATE,
        frequency NUMERIC,
        growth NUMERIC,
        engagement NUMERIC,
        recency NUMERIC,
        trend_score NUMERIC,
        trend_direction TEXT,
        PRIMARY KEY (run_id, pool, hashtag)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id SERIAL PRIMARY KEY,
        request_id TEXT,
        input_json JSONB,
        output_json JSONB,
        model_version TEXT,
        generated_at TIMESTAMPTZ DEFAULT now()
    );
    """)

    cursor.close()
    conn.close()

if __name__ == "__main__":
    setup_db()
    print("Database setup complete.")
