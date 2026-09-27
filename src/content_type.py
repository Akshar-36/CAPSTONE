import pandas as pd

def recommend_content_type(pool: str, artifact_df: pd.DataFrame) -> dict:
    """
    Given a mapped category (pool), return the recommended content type.
    If the pool is not found, fallback to 'all'.
    """
    pool_name = pool if pool in artifact_df['pool'].values else 'all'
    
    # Filter for the specific pool
    pool_data = artifact_df[artifact_df['pool'] == pool_name]
    
    # Identify the winner (highest relative_engagement_median)
    winner = pool_data.loc[pool_data['relative_engagement_median'].idxmax()]['media_type']
    
    # Create the dictionary mapping media type to its median relative engagement
    medians = dict(zip(pool_data['media_type'], pool_data['relative_engagement_median']))
    
    return {
        "value": str(winner),
        "median_relative_engagement_by_type": {str(k): float(v) for k, v in medians.items()}
    }
