import os
import json
import logging
from datetime import datetime, timezone
import pandas as pd
import yaml
import jsonschema
import joblib
from sentence_transformers import SentenceTransformer

from relevance import get_brand_text, predict_category
from hashtags import get_candidates_for_brand, calculate_historical_engagement, rank_hashtags
from content_type import recommend_content_type
from posting_time import get_recommended_posting_time
from engagement_model import EngagementModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO)

class RecommenderPipeline:
    def __init__(self):
        logging.info("Initializing RecommenderPipeline and loading artifacts...")
        with open(os.path.join(PROJECT_ROOT, 'config.yaml')) as f:
            self.config = yaml.safe_load(f)
            
        with open(os.path.join(PROJECT_ROOT, 'contract', 'input_schema.json')) as f:
            self.input_schema = json.load(f)
            
        with open(os.path.join(PROJECT_ROOT, 'contract', 'output_schema.json')) as f:
            self.output_schema = json.load(f)
            
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.trends_df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'artifacts/v1.0/trends.parquet'))
        
        with open(os.path.join(PROJECT_ROOT, 'artifacts/v1.0/category_centroids.pkl'), 'rb') as f:
            self.centroids = joblib.load(f)
            
        self.h_emb_df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'artifacts/v1.0/hashtag_embeddings.parquet'))
        
        self.history_df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'artifacts/v1.0/historical_engagement.parquet'))
        
        self.content_type_df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'artifacts/v1.0/content_type.parquet'))
        self.posting_time_df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'artifacts/v1.0/posting_time.parquet'))
        
        self.engagement_model = EngagementModel()
        logging.info("All artifacts loaded.")

    def _validate(self, instance, schema):
        try:
            jsonschema.validate(instance=instance, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            return False, str(e.message), e.json_path
        return True, "", ""

    def recommend(self, brand_input: dict) -> dict:
        is_valid, msg, path = self._validate(brand_input, self.input_schema)
        if not is_valid:
            return {
                "error": {
                    "code": "INVALID_INPUT",
                    "message": msg,
                    "field": path
                }
            }

        try:
            brand_text = get_brand_text(
                brand_input.get('business_description', ''),
                brand_input.get('industry', ''),
                brand_input.get('target_audience', '')
            )
            
            # 1. Mapped Category
            # We must use predict_category to also get category_match
            best_cat, best_score, match_type = predict_category(brand_text, self.model, self.centroids, self.config['tau'])
            
            # 2. Hashtags Candidates
            # get_candidates_for_brand actually predicts category internally again, but it uses the exact same logic.
            # We'll pass the text and it will map. 
            candidates, _ = get_candidates_for_brand(
                brand_text, self.trends_df, self.h_emb_df, self.centroids, self.model, self.config
            )
            
            # 3. Hashtags History & Ranking
            if not candidates.empty:
                candidate_hashtags = candidates['hashtag'].tolist()
                hist_eng = calculate_historical_engagement(best_cat, candidate_hashtags, self.history_df)
                candidates = candidates.merge(hist_eng, on='hashtag', how='left')
                candidates['historical_engagement'] = candidates['historical_engagement'].fillna(0.0)
                ranked_hashtags = rank_hashtags(candidates, self.config)
            else:
                ranked_hashtags = pd.DataFrame()

            if not ranked_hashtags.empty:
                top_n = ranked_hashtags.head(self.config['n_hashtags'])
                
                # Trending topics: top n_topics by trend_score
                trending_df = top_n.sort_values('trend_score', ascending=False).head(self.config['n_topics'])
                trending_topics = [h.replace('#', '') for h in trending_df['hashtag'].tolist()]
                
                recommended_topic = top_n.iloc[0]['hashtag'].replace('#', '')
                recommended_hashtags = top_n['hashtag'].tolist()
                
                hashtag_details = []
                for _, row in top_n.iterrows():
                    hashtag_details.append({
                        "hashtag": row['hashtag'],
                        "relevance": float(row['relevance']),
                        "trend_score": float(row['trend_score']),
                        "trend_direction": row['trend_direction'],
                        "historical_engagement": float(row['historical_engagement']),
                        "score": float(row['score'])
                    })
            else:
                trending_topics = []
                recommended_topic = ""
                recommended_hashtags = []
                hashtag_details = []

            # 4. Content Type
            try:
                ct_res = recommend_content_type(best_cat, self.content_type_df)
                content_type_val = ct_res['value']
                median_rel_eng = ct_res['median_relative_engagement_by_type']
            except ValueError:
                content_type_val = 'image'
                median_rel_eng = {"image": 1.0, "carousel": 1.0}

            # 5. Posting Time
            tz_str = brand_input.get('timezone', 'UTC')
            try:
                pt_res = get_recommended_posting_time(best_cat, self.posting_time_df, tz_str)
            except ValueError:
                pt_res = {"day": "Monday", "start": "12:00", "end": "13:00", "timezone": tz_str}
                
            # 6. Prediction
            hour = int(pt_res['start'].split(':')[0])
            days_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            day_of_week = days_map.get(pt_res['day'], 0)
            
            pred_res = self.engagement_model.predict(
                category=best_cat,
                media_type=content_type_val,
                hour_of_day=hour,
                day_of_week=day_of_week,
                hashtag_count=len(recommended_hashtags),
                follower_count=brand_input.get('follower_count')
            )
            
            # Assemble output
            output = {
                "recommended_topic": recommended_topic,
                "trending_topics": trending_topics,
                "recommended_hashtags": recommended_hashtags,
                "hashtag_details": hashtag_details,
                "recommended_content_type": {
                    "value": content_type_val,
                    "median_relative_engagement_by_type": {
                        "image": float(median_rel_eng.get("image", 1.0)),
                        "carousel": float(median_rel_eng.get("carousel", 1.0))
                    }
                },
                "predicted_engagement": {
                    "value": float(pred_res['value']),
                    "unit": pred_res['unit'],
                    "label": pred_res['label']
                },
                "recommended_posting_time": pt_res,
                "data_basis": {
                    "dataset": "Kim et al., WWW '20",
                    "data_period": "2012-02-12 to 2019-05-15",
                    "reference_date": "2019-05-15",
                    "data_type": "historical",
                    "category": best_cat,
                    "category_match": match_type
                },
                "model_version": "v1.0",
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Validate output
            is_valid, msg, path = self._validate(output, self.output_schema)
            if not is_valid:
                return {
                    "error": {
                        "code": "INTERNAL",
                        "message": f"Output validation failed: {msg}",
                        "field": path
                    }
                }
                
            return output
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "error": {
                    "code": "INTERNAL",
                    "message": str(e)
                }
            }
