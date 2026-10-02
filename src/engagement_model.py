import pandas as pd
import numpy as np
import joblib
import json
import os
import yaml
from dotenv import find_dotenv

PROJECT_ROOT = os.path.dirname(find_dotenv())

def load_config():
    with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
        return yaml.safe_load(f)

class EngagementModel:
    def __init__(self):
        self.config = load_config()
        self.allowlist = self.config['feature_allowlist']['cold']
        
        model_path = os.path.join(PROJECT_ROOT, 'artifacts/v1.0/model.joblib')
        thresholds_path = os.path.join(PROJECT_ROOT, 'artifacts/v1.0/label_thresholds.json')
        fallbacks_path = os.path.join(PROJECT_ROOT, 'artifacts/v1.0/fallback_medians.json')
        
        if not os.path.exists(model_path) or not os.path.exists(thresholds_path) or not os.path.exists(fallbacks_path):
            raise FileNotFoundError("Artifacts not found. Run Phase 6 and remediation first.")
            
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        
        with open(thresholds_path, 'r') as f:
            self.thresholds = json.load(f)
            
        with open(fallbacks_path, 'r') as f:
            self.fallbacks = json.load(f)
            
    def get_label(self, er, pool):
        pool_key = pool if pool in self.thresholds else 'all'
        t = self.thresholds.get(pool_key, {'low': 0.0, 'high': 1.0})
        if er < t['low']:
            return 'Low'
        elif er < t['high']:
            return 'Medium'
        else:
            return 'High'

    def predict(self, category, media_type, hour_of_day, day_of_week, hashtag_count, follower_count=None, caption_length=None, usertag_count=None, n_media=None):
        inf_category = 'other' if category == 'all' else category
        
        pool_key = category if category in self.fallbacks else 'all'
        fallbacks = self.fallbacks.get(pool_key, self.fallbacks['all'])
        
        follower_count = follower_count if follower_count is not None else fallbacks['follower_count']
        caption_length = caption_length if caption_length is not None else fallbacks['caption_length']
        usertag_count = usertag_count if usertag_count is not None else fallbacks['usertag_count']
        
        if n_media is None:
            n_media = 1 if media_type == 'image' else fallbacks['n_media_carousel']
            
        data = {
            'log_followers': np.log1p(follower_count),
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'is_weekend': 1 if day_of_week in [5, 6] else 0,
            'hashtag_count': hashtag_count,
            'caption_length': caption_length,
            'usertag_count': usertag_count,
            'n_media': n_media
        }
        
        categories = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'other', 'pet', 'travel']
        for cat in categories:
            data[f'category_{cat}'] = 1 if inf_category == cat else 0
            
        media_types = ['carousel', 'image']
        for mt in media_types:
            data[f'media_type_{mt}'] = 1 if media_type == mt else 0
            
        df = pd.DataFrame([data], columns=self.allowlist)
        
        y_pred = self.model.predict(df)[0]
        er_pred = (np.expm1(y_pred)) / 100.0
        er_pred = max(0.0, er_pred)
        
        label = self.get_label(er_pred, category)
        
        return {
            'value': float(er_pred),
            'unit': 'engagement_rate',
            'label': label
        }
