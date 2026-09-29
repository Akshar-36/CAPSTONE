import pandas as pd
import pytz
from datetime import datetime, timedelta

def get_recommended_posting_time(pool: str, artifact_df: pd.DataFrame, tz_str: str = "UTC") -> dict:
    """
    Returns the recommended 1-hour posting window for a given category pool.
    Matches the Phase 5e contract schema.
    """
        
    pool_name = pool if pool in artifact_df['pool'].values else 'all'
    pool_data = artifact_df[artifact_df['pool'] == pool_name]
    
    if pool_data.empty:
        raise ValueError(f"No bucket data found for pool '{pool_name}'.")
        
    best_b = pool_data.loc[pool_data['shrunk_mean'].idxmax()]
    
    day_name = best_b['day_name']
    hour = int(best_b['hour'])
    
    # DST simplification: use offset on the dataset reference date (2019-05-15 UTC)
    ref_dt = datetime(2019, 5, 15, hour, 0, tzinfo=pytz.UTC)
    
    days_map_rev = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    ref_day_of_week = ref_dt.weekday()
    target_day_of_week = days_map_rev[day_name]
    
    day_diff = (target_day_of_week - ref_day_of_week) % 7
    bucket_utc = ref_dt + timedelta(days=day_diff)
    
    try:
        target_tz = pytz.timezone(tz_str)
    except pytz.UnknownTimeZoneError:
        target_tz = pytz.UTC
        tz_str = "UTC"
        
    bucket_local = bucket_utc.astimezone(target_tz)
    
    days_map_fwd = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    
    start_str = bucket_local.strftime('%H:%M')
    end_local = bucket_local + timedelta(hours=1)
    end_str = end_local.strftime('%H:%M')
    
    return {
        "day": days_map_fwd[bucket_local.weekday()],
        "start": start_str,
        "end": end_str,
        "timezone": tz_str
    }
