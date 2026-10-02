import json
import sys
sys.path.append('./src') 

from pipeline import RecommenderPipeline

# 1. Initialize the pipeline
pipeline = RecommenderPipeline()

# 2.  input
test_input = {
  "brand_name": "burgerkIg",
  "business_description": "sell burger and fries",
  "industry": "Food",
  "target_audience": "everyone",
  "brand_tone": "Friendly",
  "follower_count": 5000,
  "timezone": "Asia/Kolkata"
}

# 3.CHALJA PLS !!!!!!!
try:
    print("Running recommendation pipeline...\n")
    output = pipeline.recommend(test_input)
    
    # 4. Print the output as a nicely formatted JSON string
    print(json.dumps(output, indent=2))
    
except Exception as e:
    print(f"An error occurred: {e}")
