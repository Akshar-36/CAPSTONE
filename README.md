# Social Media Data/ML Recommendation Engine

## Overview

This is the Data/ML engine we built for our Instagram marketing SaaS. When you hand it a brand's description, it figures out what they should post about, which hashtags to use, whether an image or carousel will work better, when to post, and what kind of engagement to expect. 

We didn't want an LLM just guessing at these things, so all of these recommendations are grounded in historical Instagram data. The LLM still writes the actual caption, but this engine makes sure it's working from a real, data-backed foundation.

## How It Works

We split the system into two pieces so the online pipeline doesn't have to repeat the expensive data processing:

**1. Offline Analytical Pipeline (The heavy lifting)**
```text
Historical Data → Cleaning → Feature Engineering → Trend Detection → Brand Relevance
→ Hashtag Ranking → Content Type → Posting Time → Engagement Prediction → Serialized Artifacts
```

**2. Online Inference Pipeline (What runs per request)**
```text
Input JSON → RecommenderPipeline.recommend() → Output JSON
```

The reason for this split is simple: a user request shouldn't have to scan 1.3 million historical posts. All the expensive data processing happens offline. Once the artifacts (like trend scores, category embeddings, and the trained Random Forest) are serialized into `artifacts/v1.0/`, the online pipeline just loads those lookups and runs a quick model inference.

## What's Under the Hood?

We built six main ML components into the pipeline:

- **Trend Detection**: We score every hashtag based on recency, growth, frequency, and historical engagement, combining them into one weighted trend score.
- **Brand Relevance**: We map a brand's free-text description to one of 9 predefined semantic categories (or a fallback pool if nothing fits) using `all-MiniLM-L6-v2` sentence embeddings.
- **Hashtag Ranking**: We rank candidate hashtags using a min-max normalized blend of semantic relevance, trend momentum, and historical engagement.
- **Content Type Recommendation**: We use median relative engagement with bootstrap validation and confidence intervals to figure out if images or carousels historically perform better for the matched category.
- **Posting Time Recommendation**: We apply exponential recency weighting and Bayesian shrinkage toward the global pool mean to find the strongest-performing one-hour posting window.
- **Engagement Prediction**: We trained a Random Forest regressor to predict expected engagement rate from the engineered Phase 6 feature set (which includes `log_followers`, `hour_of_day`, `day_of_week`, `is_weekend`, `hashtag_count`, `caption_length`, `usertag_count`, `n_media`, plus category and media-type one-hot features). It then buckets the final result into Low, Medium, or High.

## Model Results

### Brand Relevance (Category Mapping)

We evaluated category mapping on `all-MiniLM-L6-v2` sentence embeddings. In our Phase 5b evaluation, we got these results:

| Model | Category Mapping Accuracy | P@10 |
|---|---|---|
| MiniLM (Production) | 43.87% | 40% |
| TF-IDF (Evaluated Baseline) | 48.31% | 50% |

*(P@10 = precision at 10 — the fraction of relevant/correct items among the model's top 10 retrieved matches.)*

We evaluated both MiniLM and TF-IDF. Even though TF-IDF performed better in this specific Phase 5b evaluation, we kept MiniLM as the embedding model for the production MVP. One issue we found is that once we apply our strict τ cutoff end-to-end, a lot of queries get pushed into the fallback pool instead of finding a real match. See **Limitations** for why fixing that threshold is a priority.

### Engagement Prediction

When training the model, our target was a log-transformed engagement rate: `y = ln(1 + 100 × ER)`. This kept a handful of viral outliers from wrecking the loss function. The predictions are inverse-transformed back to a raw engagement rate before they hit the output contract, so a result of `0.0266` is a real engagement rate, not a log-space value.

Here's what we saw during validation:

| Model | Val RMSE | Val R² |
|---|---|---|
| Global median | 0.6700 | -0.002 |
| Category + media median | 0.6558 | 0.040 |
| Linear Regression | 0.6233 | 0.133 |
| XGBoost | 0.5993 | 0.198 |
| **Random Forest (selected)** | **0.5620** | **0.295** |

**Final Random Forest, test set:** RMSE 0.606, R² 0.193.

The R² dropping from 0.295 (validation) to 0.193 (test) isn't a red flag. We specifically held out the test split at the account level to stop the model from memorizing account-specific quirks. An R² around 0.2 here means the model captures some of the available signal, but organic engagement remains highly difficult to predict. 

## End-to-End Input

We use strict JSON validated against `contract/input_schema.json`:
```json
{
  "brand_name": "Lumiere Cosmetics",
  "business_description": "Cruelty-free, vegan makeup brand focusing on bold eyeshadow palettes and long-lasting lipsticks.",
  "industry": "Beauty & Cosmetics",
  "target_audience": "Young adults, makeup enthusiasts, cruelty-free advocates",
  "brand_tone": "Bold, empowering, vibrant",
  "follower_count": 15000,
  "timezone": "America/New_York"
}
```

## End-to-End Output

And here's the structured output, validated against `contract/output_schema.json`:
```json
{
  "recommended_topic": "makeupartistsworldwide",
  "trending_topics": ["toofaced", "sephora", "instabeauty"],
  "recommended_hashtags": [
    "#makeupartistsworldwide",
    "#undiscoveredmuas",
    "#colourpopme",
    "#abhprsearch",
    "#instabeauty"
  ],
  "hashtag_details": [
    {
      "hashtag": "#makeupartistsworldwide",
      "relevance": 0.5147,
      "trend_score": 0.5862,
      "trend_direction": "falling",
      "historical_engagement": 0.0701,
      "score": 0.6932
    }
  ],
  "recommended_content_type": {
    "value": "carousel",
    "median_relative_engagement_by_type": {
      "image": 1.0,
      "carousel": 1.0356
    }
  },
  "predicted_engagement": {
    "value": 0.0266,
    "unit": "engagement_rate",
    "label": "Medium"
  },
  "recommended_posting_time": {
    "day": "Saturday",
    "start": "10:00",
    "end": "11:00",
    "timezone": "America/New_York"
  },
  "data_basis": {
    "dataset": "Kim et al., WWW '20",
    "data_period": "2012-02-12 to 2019-05-15",
    "reference_date": "2019-05-15",
    "data_type": "historical",
    "category": "beauty",
    "category_match": "matched"
  },
  "model_version": "v1.0",
  "generated_at": "2026-09-27T12:11:09Z"
}
```

Notice the `data_basis` field. We added this so the backend (or anyone debugging) knows exactly which category a request matched to, and what historical data is driving the recommendation. We didn't want the engine to be a total black box.

## Project Structure

```text
artifacts/     precomputed models, parquet lookups, and JSON metadata the pipeline reads at runtime
contract/      strict JSON input/output schemas, plus sample integration outputs
data/          raw and processed .parquet files (not pushed to remote)
notebooks/     00-11, sequenced: cleaning → EDA → feature extraction → model training
src/           core logic — pipeline.py, engagement_model.py, trend.py, etc.
tests/         pytest suite covering schema validation, determinism, and leakage checks
```

## Running the Project

### Setup
```bash
pip install -r requirements.txt
```

### Getting a recommendation
```python
from src.pipeline import RecommenderPipeline

pipeline = RecommenderPipeline(artifact_dir="artifacts/v1.0")

result = pipeline.recommend({
    "brand_name": "...",
    "business_description": "...",
    "industry": "...",
    "target_audience": "..."
})
```

### Running the tests
```bash
python -m pytest tests/test_pipeline.py -v
```

## Backend Integration

This piece of the project is totally headless—no UI, no routes. The backend validates an incoming request, passes the dict to `RecommenderPipeline.recommend()`, and gets back a data-grounded recommendation object shaped exactly like `output_schema.json`. From there, the backend folds that object into an LLM prompt to generate the actual Instagram post.

## How We Evaluated This

We made sure to evaluate the major ML components using held-out or temporal validation procedures that made sense for each task:

- **Category mapping** — evaluated on held-out brand/account profiles using category mapping accuracy and P@10.
- **Hashtag ranking** — backtested against realized historical engagement across temporal splits, not just eyeballed.
- **Posting time** — validated through the same time-based shrinkage and decay mechanisms used at inference, so training and serving behavior match.
- **Content-type superiority** — confirmed with bootstrapped 95% confidence intervals, not a bare point estimate.
- **Engagement model leakage** — guarded against with account-level holdout splits (an account's posts never span both train and test) and permutation importance checks on the Random Forest's features.
- **Temporal robustness** — evaluated using a timestamp-based train/validation/test split to assess performance under temporal distribution changes.
- **Contracts** — enforced with `jsonschema` in the pytest suite, so a malformed input/output fails loudly in CI rather than quietly in production.

## The Dataset

- **Source:** Kim et al., *WWW '20* — Instagram profiles and posts.
- **Period:** 2012-02-12 to 2019-05-15.
- **Size:** 1,313,584 valid posts across 32,593 eligible accounts after cleaning.
- **Target metric:** engagement rate, `(likes + comments) / followers`.

## Honest Limitations

The biggest limitation right now is the age of the dataset. Because it ends in mid-2019, the model has no concept of Reels or modern short-form video behavior, and its idea of "trending" is stuck in a pre-TikTok Instagram era. It's also skewed toward certain influencer subcultures (like mommy bloggers and fashion accounts), which bleeds into our fallback candidate pool and makes recommendations less useful for brands outside those niches.

Another sharp edge is our fallback threshold (`τ = 0.343`). In our 10-brand recommendation-quality diagnostic, 9 of the 10 test brands fell back to the global pool. (Note: this was a small diagnostic sample and should not be interpreted as a population-wide rate.) This fallback behavior drags down relevance quality for anything that isn't a clean fit for one of the 9 predefined categories, making it the first thing worth revisiting.

Finally, on the prediction side, the current test R² of approximately 0.19 reflects the difficulty of predicting organic engagement from the available historical features. Organic engagement is driven by a lot of randomness the model simply can't see, which makes it a challenging prediction problem.

## What's Next?

- Move to a modern (2024+) dataset that actually captures Reels and short-form video engagement.
- Explore per-category or dynamic `τ` thresholds instead of one fixed cutoff that pushes too many queries into the fallback pool.
- Look into replacing the Random Forest with a small fine-tuned language model (e.g., RoBERTa) that regresses directly on text, rather than hand-engineered features.
- Move serialized artifacts off Pickle/Joblib and onto a safer, faster production format like ONNX or Safetensors.

## Technologies Used

- Python 3
- pandas, NumPy — data manipulation
- scikit-learn — Random Forest, preprocessing
- XGBoost — evaluated as a comparison baseline
- Sentence-Transformers (`all-MiniLM-L6-v2`) — embeddings
- jsonschema — contract validation
- pytest — testing

## More Docs

If you want to dig into our methodology, known technical debt, or implementation rules, check these out:
- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
- [DATA_ML_MASTER_TECHNICAL_AUDIT.md](DATA_ML_MASTER_TECHNICAL_AUDIT.md)
- [RECOMMENDATION_QUALITY_AUDIT.md](RECOMMENDATION_QUALITY_AUDIT.md)
- [HANDOFF.md](HANDOFF.md)
- [CONTEXT.md](CONTEXT.md)
- [contract/input_schema.json](contract/input_schema.json)
- [contract/output_schema.json](contract/output_schema.json)
