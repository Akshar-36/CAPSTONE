# INDEPENDENT ENGINEERING / CODEX HANDOFF DOCUMENT
**CAPSTONE Project — Data/ML Component**

## 1. DOCUMENT PURPOSE
This document represents the current technical state of the Data/ML system and serves as an independent engineering handoff. It documents the implementation as it actually exists today in the repository, detailing the methodologies, code architecture, data flow, models, artifacts, evaluation results, structural assumptions, limitations, and known technical debt. 

While the original MVP was developed under a strict "frozen specification" protocol to ensure stable delivery, future engineers or autonomous agents (such as Codex) may independently evaluate improvements and departures outside that frozen specification. This document gives you the exact context required to inspect the implementation, identify weaknesses, propose improvements, and safely modify the system without unknowingly using stale artifacts or violating expected backend contracts.

## 2. GOVERNING DOCUMENTS
The system was originally built adhering strictly to the following foundational documents. 
* **AGENTS.md:** Contains operational rules, behavioral guardrails, and non-negotiable development constraints (R1-R17). It was the highest authority preventing scope creep and data hallucination.
* **CONTEXT.md:** Provides the product background (AI SaaS for Instagram post generation), dataset boundaries (Kim et al. 2012-2019), and business constraints.
* **HANDOFF.md:** The active state tracker for pipeline phase progression.
* **IMPLEMENTATION_GUIDE.md:** The frozen technical authority for all Data/ML methodologies, architectures, and metrics executed during the MVP phase.

**Important:** These documents describe the controlled MVP that was implemented. Future optimization may intentionally depart from the frozen MVP methodology if a new engineering decision is made and evaluated separately. You do not need to treat the guide as an unbreakable law if you are explicitly tasked with upgrading the system, provided you evaluate the changes properly.

## 3. CURRENT SYSTEM — EXECUTIVE TECHNICAL SUMMARY
The Data/ML component is an offline-analytical and online-inference backbone for an Instagram recommendation SaaS. It does NOT generate final LLM post text. It takes a brand's description, maps it to a category, and recommends data-driven hashtags, optimal posting times, content types (image vs carousel), and predicts the post's expected engagement label (Low/Medium/High). 

The system relies on a heavy offline data-engineering and training pipeline that materializes 9 specialized artifacts. The online `RecommenderPipeline` performs precomputed-artifact inference with no raw-dataset scan. It does not scan the raw historical dataset during recommendation generation because required information has been materialized into runtime artifacts, ensuring decoupled, low-latency execution.

**Data Flow Diagram:**
```text
Raw Historical Data (Kim et al. 2012-2019)
       ↓
Load / Parse (GraphQL JSON)
       ↓
Cleaning (Exclude 0-followers, clean text)
       ↓
Feature Engineering (Cold-start vs Full metrics)
       ↓
┌──────────────────────────────────┐
│ Offline analytical/model stages  │
│                                  │
│ Trend Detection (Recency/Growth) │
│ Brand Relevance (MiniLM cosine)  │
│ Hashtag Ranking (Normalized)     │
│ Content Type (Bayesian Lift)     │
│ Posting Time (Shrinkage)         │
│ Engagement Prediction (RF)       │
└──────────────────────────────────┘
       ↓
Serialized Artifacts (artifacts/v1.0/)
       ↓
RecommenderPipeline (src/pipeline.py)
       ↓
Validated JSON (contract/output_schema.json)
       ↓
Backend Integration
```

## 4. REPOSITORY MAP

| Path | Purpose | Used By | Runtime/Offline | Important Dependencies |
|---|---|---|---|---|
| `data/` | Stores raw and processed parquet data files. | Offline notebooks, `src/*.py` | Offline Only | Kim et al. dataset |
| `src/clean.py`, `features.py` | Data parsing, cleaning, feature extraction. | Offline notebooks | Offline Only | pandas |
| `src/trend.py`, `relevance.py` | Phase 5a/5b core logic. | `04`, `05` notebooks | Offline Only | `SentenceTransformers` |
| `src/hashtags.py`, `content_type.py`, `posting_time.py`, `engagement_model.py` | Phase 5c, 5d, 5e, 6 core logic. | Offline notebooks | Offline Only | `xgboost`, `scikit-learn` |
| `notebooks/` | Numbered sequential execution flow for offline model building. | Developers | Offline Only | All `src/` modules |
| `artifacts/v1.0/` | Serialized model, parquets, json dictionaries. | `src/pipeline.py` | Runtime | Produced by notebooks |
| `src/pipeline.py` | The main `RecommenderPipeline` API class. | Backend, Demos | Runtime | `artifacts/v1.0/` |
| `contract/` | JSON schema files defining input/output interfaces. | `src/pipeline.py` | Runtime | `jsonschema` |
| `config.yaml` | Centralized hyperparameter and threshold repository. | All | Offline / Runtime | `yaml` |
| `tests/` | Pytest suite enforcing schema and logic determinism. | CI/CD | Evaluation Only | `pytest` |

## 5. DATASET TECHNICAL SPECIFICATION
* **Source:** Kim et al., WWW '20 Dataset (Instagram profiles and posts).
* **Temporal Range:** 2012-02-12 to 2019-05-15.
* **Raw Files:** Originally GraphQL JSON metadata parsed into `raw_posts` (1,484,362 rows) and `influencers` (33,934 rows).
* **Cleaning & Filtering (`src/clean.py`):**
  * Posts without followers (`followers <= 0`) or with disabled comments are flagged and filtered during feature usage to prevent division by zero or skewed metrics.
  * *Null behavior:* Missing comments are kept null, never filled with zero, preventing artificial engagement deflation.
* **Engagement Rate Formula:** `(likes + comments) / followers`.
* **Cleaned Dataset Sizes:** `posts_clean.parquet` contains 1,313,584 valid rows for modeling.

## 6. FEATURE ENGINEERING SPECIFICATION

| Feature | Dataset | Type | Formula/Construction | Source Columns | Leakage Risk | Used by Model? |
|---|---|---|---|---|---|---|
| `log_followers` | Cold | Float | `np.log1p(followers)` | followers | Low (captured at scrape time) | Yes |
| `hour_of_day` | Cold | Int | `posted_at.dt.hour` | posted_at | None | Yes |
| `day_of_week` | Cold | Int | `posted_at.dt.dayofweek` | posted_at | None | Yes |
| `is_weekend` | Cold | Int | `day_of_week >= 5` | posted_at | None | Yes |
| `hashtag_count` | Cold | Int | Number of `#` extracted | caption | None | Yes |
| `usertag_count` | Cold | Int | Number of `@` extracted | caption | None | Yes |
| `caption_length` | Cold | Int | `len(caption)` | caption | None | Yes |
| `n_media` | Cold | Int | Media items count | media_type, edge_sidecar | None | Yes |
| `media_type_*` | Cold | OHE | Image vs Carousel dummy | media_type | None | Yes |
| `category_*` | Cold | OHE | Category dummy | original_category | None | Yes |
| `y` (Target) | Cold/Full | Float | `np.log1p(100 * ER)` | likes, comments, followers | HIGH | Target |
| `historical_avg_eng` | Full | Float | Mean ER of past posts | engagement_rate | HIGH | No |
| `days_since_last` | Full | Float | Diff to prev post | posted_at | Medium | No |

**Why Full Features are excluded:** Full historical features require scanning a user's exact past posting history. In a cold-start SaaS system where the brand is new to the platform, we lack historical posts. Therefore, `features_full.parquet` is generated purely for theoretical reference/comparison, but the MVP inference pipeline strictly relies on `features_cold.parquet`.

## 7. PHASE 5a — TREND ENGINE
* **Intended Methodology:** Calculate recency, growth, frequency, and engagement metrics for hashtags up to a reference date.
* **Actual Implementation (`src/trend.py`):**
  * **Eligibility:** `min_period_posts=25`, `min_distinct_users=5`.
  * **Frequency:** `share_cur * 10000`
  * **Growth:** `log((share_cur + a) / (share_prev + a))`
  * **Engagement:** Winsorized at 99th percentile (`er_winsor_pct = 0.99`).
  * **Recency:** `exp(-ln2 * age_days / (period_days/4))`
  * **Score:** Weighted sum of the 4 normalized signals.
* **Measured Tuning Result:** Backtest tuning achieved a Spearman correlation of `0.0765` against future share changes.
* **Parameters Selected:** The model heavily favored recency. `weights_trend`: `[0.0, 0.0, 0.0, 1.0]`. `growth_smoothing` (`a`) = `0.001`. `theta` = `0.10311`.
* **Artifact:** `trends.parquet`. (Note: The exact final trend signal row count was truncated in the notebook logs and is not verified here, but it exists and is loaded successfully).

## 8. PHASE 5b — BRAND RELEVANCE ENGINE
* **Methodology:** Pretrained `all-MiniLM-L6-v2` (SentenceTransformers) generates 384-dimensional dense vectors. A brand text is mapped via cosine similarity to the closest precomputed category centroid (`centroid_sample=500`).
* **Experimental vs. Final Implementation:**
  * *Experiment:* The team attempted to represent hashtags by embedding individual captions and averaging them. This yielded 81% P@10 accuracy. However, during end-to-end integration, this representation proved poorly calibrated, resulting in a 68.95% mapping accuracy and broken fallback behaviors (39.0% P@10).
  * *Final Implementation:* Reverted to concatenation. Up to 20 captions (`hashtag_caption_sample=20`) are concatenated into a single string per hashtag, then embedded.
* **Measured Results (Concatenation):** MiniLM P@10: `0.4000`, TF-IDF Baseline P@10: `0.5000`. (Note: TF-IDF performed slightly better on the test set, but MiniLM was selected per the frozen guide architecture).
* **Fallback Behavior:** Governed by `tau = 0.34308`. If the brand's cosine similarity to all centroids falls below this 5th percentile calibration bound, the pipeline maps the brand to the `all` fallback pool to prevent nonsensical mappings.

## 9. PHASE 5c — HASHTAG ENGINE
* **Implementation:** During inference, candidate hashtags are filtered from `trends.parquet` for the chosen pool (size=`50`). 
* **Scoring:** The pipeline calculates a combined score using Min-Max normalized metrics: `score = (w_rel * norm_rel) + (w_trend * norm_trend) + (w_hist * norm_hist)`.
* **Parameters:** `weights_hashtag` = `[0.333, 0.333, 0.334]`, `n_hashtags` = `8`. (Weights were explicitly zero-shot, not tuned to overfit).
* **Historical Engagement Handling:** `src/hashtags.py` was remediated during Phase 7. The intended guide required calculating historical engagement on-the-fly. The actual implementation precomputes this via `historical_engagement.parquet` to avoid loading raw posts at runtime.
* **Validation (Mean Realized ER on temporal cutoff):**
  * Overall Proposed: `0.0567`
  * Overall Frequent: `0.0525`
  * Overall Random: `0.0421`

## 10. PHASE 5d — CONTENT TYPE ENGINE
* **Implementation:** Analyzes relative engagement (ER / account median ER) to determine if images or carousels perform better.
* **Thresholds:** `min_account_posts=10`, `min_type_posts=100`.
* **Data Validated:** `1,312,297` valid posts from `32,593` eligible accounts.
* **Results:** Bootstrap 95% CIs of the median difference (Carousel - Image) proved Carousel was universally superior in relative engagement for almost all categories (e.g. `all`: [0.0225, 0.0280]). `fitness` had conflicting train/test splits, but `carousel` won globally.
* **Artifact:** `content_type.parquet` stores the pool, median relative engagement for images/carousels, and the absolute winner.

## 11. PHASE 5e — POSTING TIME ENGINE
* **Implementation Amendment:** The original guide proposed a simple frequency lookup. The implementation was amended to include exponential recency weighting and Bayesian-style shrinkage towards the global pool mean to prevent low-sample hour buckets from spiking erroneously.
* **Parameter Grid Search:**
  * Tested candidate half-lives (14, 28, 56, 90 days) and shrinkage constants $k$ (10, 50, 100, 500).
  * Selection metric: Validation macro-average relative engagement across all pools.
  * Selected parameters: `time_half_life_days=14`, `shrinkage_k=10`.
* **Holdout Lift Results:**
  * For the `all` pool (Monday 02:00), the selected bucket provided a +0.1661 lift over the pool mean, and +0.1563 over the most-posted bucket.
  * Bootstrapped CI for lift vs pool mean: [-0.0305, 0.4232].
* **Timezone Conversion:** The system uses standard python `pytz` offsets. Because the dataset's audience geography is unknown (mixture), timestamps were maintained in UTC. The pipeline handles target timezones by calculating the delta between the requested timezone and UTC on the specific reference date (`2019-05-15`) to properly preserve historical DST states.

## 12. PHASE 6 — ENGAGEMENT MODEL
* **Target:** $y = \ln(1 + 100 \times ER)$.
* **Splits:** Account-holdout split (80 train / 10 val / 10 test) capped at `train_rows=400000` to prevent data leakage across users. Time splits were additionally generated to test robustness.
* **Models Evaluated & Metrics:**

| Model | Val RMSE | Val MAE | Val R² | Purpose |
|---|---|---|---|---|
| Global Median | 0.6700 | 0.5398 | -0.0020 | Baseline |
| Category+Media Median | 0.6558 | 0.5263 | 0.0401 | Baseline |
| Linear Regression | 0.6233 | 0.4993 | 0.1329 | ML Baseline |
| Random Forest | 0.5620 | 0.4472 | 0.2949 | Final Candidate |
| XGBoost | 0.5993 | 0.4777 | 0.1983 | Candidate |

* **Selection Rule:** The guide required selecting the model with the lowest Validation RMSE. Random Forest (0.5620) decisively beat XGBoost (0.5993).
* **Final Parameters:** Random Forest (`n_estimators=100`, `min_samples_leaf=50`, `max_features=0.3`).
* **Final Test Metrics (Random Forest):**
  * Test RMSE: `0.606374`
  * Test MAE: `0.482893`
  * Test R²: `0.193189` (Modest, due to the extreme noise of organic virality)
* **Classification Results:** Low/Medium/High thresholds generated per category via 33rd/67th percentiles.
  * Accuracy: `0.4484` (Majority baseline: `0.3372`).
* **Feature Importance (Permutation):** `log_followers` (0.1199) completely dominates predictive power, followed distantly by `usertag_count` (0.0442) and `hashtag_count` (0.0309).

## 13. PHASE 7 PIPELINE
* **Implementation (`src/pipeline.py`):** The `RecommenderPipeline` class orchestrates inference.
* **Initialization:** It loads 9 specific artifacts during `__init__`. It caches them in memory.
* **Inference (`recommend(input_dict)`):**
  1. Validates `input_dict` against `input_schema.json`.
  2. Embeds `business_description` + `industry` + `target_audience`.
  3. Uses `cosine_similarity` against `category_centroids.pkl`. If score < `tau` (0.343), falls back to `all`.
  4. Retrieves missing features (e.g., median caption lengths) from `fallback_medians.json`.
  5. Computes normalized ranking using `historical_engagement.parquet` and `trends.parquet`.
  6. Looks up content type from `content_type.parquet`.
  7. Looks up optimal UTC hour from `posting_time.parquet` and shifts it to the requested `timezone`.
  8. Passes the constructed feature vector to `model.joblib`.
  9. Compares predicted ER against `label_thresholds.json`.
  10. Assembles and validates output against `output_schema.json`.

## 14. ARTIFACT SPECIFICATION

| Artifact | Format | Producer | Runtime Consumer | Purpose |
|---|---|---|---|---|
| `trends.parquet` | Parquet | `trend.py` | `pipeline.py` | Trending hashtag pool |
| `content_type.parquet` | Parquet | `content_type.py` | `pipeline.py` | Media type recommendations |
| `historical_engagement.parquet` | Parquet | `hashtags.py` (Phase 7 fix) | `pipeline.py` | Precomputed-artifact hashtag history lookup |
| `posting_time.parquet` | Parquet | `posting_time.py` | `pipeline.py` | Temporal recommendations |
| `category_centroids.pkl` | Pickle | `relevance.py` | `pipeline.py` | Brand-to-category mapping |
| `hashtag_embeddings.parquet` | Parquet | `relevance.py` | `pipeline.py` | Semantic hashtag matching |
| `fallback_medians.json` | JSON | `pipeline.py` (Phase 7 fix) | `pipeline.py` | Fill missing RF features |
| `model.joblib` | Joblib | `engagement_model.py` | `pipeline.py` | Random Forest prediction |
| `label_thresholds.json` | JSON | `engagement_model.py` | `pipeline.py` | Low/Med/High classification |

*(Note: `metrics.json` is generated for documentation, but not consumed at runtime).*

## 15. API / CONTRACT SPECIFICATION
The API requires exact adherence to JSON schemas located in `contract/`.

**Input Schema Constraints:**
* Requires: `brand_name`, `business_description`, `industry`, `target_audience`.
* Optional: `brand_tone`, `follower_count` (int >= 0), `timezone` (string).
* Extra properties are forbidden.

**Output Schema Constraints:**
* Provides a strictly typed response containing `recommended_topic`, `trending_topics`, `recommended_hashtags` (must start with `#`), detailed relevance/trend scores bounded `[0, 1]` or `[-1, 1]`, `recommended_content_type` (`image` or `carousel`), `predicted_engagement` (value and `Low/Medium/High` enum), and `recommended_posting_time` (start/end `HH:MM`).

## 16. DATA/ML → BACKEND INTERFACE
The Data/ML component is designed as a headless analytical engine. It does not handle HTTP requests, database transactions, or final LLM text generation. 

**What Data/ML currently exposes:**
The core integration point is the `RecommenderPipeline` class in `src/pipeline.py`, specifically the `recommend(input_dict: dict) -> dict` method.

**What input it expects:**
It expects a Python dictionary that strictly validates against `contract/input_schema.json`. This dictionary must contain the brand's name, business description, industry, and target audience, along with optional parameters like follower count and target timezone.

**What happens internally after receiving that input:**
1. The pipeline maps the descriptive text to one of the 9 dataset categories (or `all` as a fallback) using semantic embedding similarity (MiniLM).
2. It relies on precomputed-artifact inference with no raw-dataset scan. The pipeline does not scan the raw historical dataset during recommendation generation because required information has been materialized into runtime artifacts. It uses these artifacts to fetch trending and historically relevant hashtags for that specific category.
3. It determines the historically optimal content type (image vs carousel) and the optimal posting time window, converted to the user's timezone.
4. It extracts these features and feeds them into the Random Forest `model.joblib` to predict the engagement rate and classify it as Low, Medium, or High.

**What final object it returns:**
It returns a nested Python dictionary that strictly validates against `contract/output_schema.json`. This object contains all the recommendations (topic, hashtags, time, content type) and predicted engagement alongside a `data_basis` tracking object for explainability.

**How that returned object is intended to be consumed by the backend:**
Data/ML produces the analytical recommendation parameters. It does not generate the final captions/images and does not currently implement the LLM/backend transport. The correct architecture is:

```text
Backend/application provides brand_input
             ↓
RecommenderPipeline.recommend()
             ↓
Data/ML inference
             ↓
Schema-validated recommendation dict
             ↓
Downstream backend/LLM layer
             ↓
Final application content
```

The downstream backend/LLM layer is expected to take the strict, schema-validated recommendation dictionary and use its values to construct a precise, constrained prompt for a Generative LLM before returning the final content to the frontend.

## 17. TESTING SPECIFICATION
The test suite (`tests/test_pipeline.py`) aggressively enforces determinism and schema compliance.

| Test | What It Protects Against | Result |
|---|---|---|
| `test_valid_input` | Output schema violations | Passes |
| `test_invalid_input` | Missing required fields, invalid types | Passes |
| `test_unknown_industry` | Ensures fallback to `all` pool triggers correctly | Passes |
| `test_empty_and_non_english` | Ensures pipeline doesn't crash on junk text | Passes |
| `test_deterministic` | Verifies identical inputs yield perfectly identical outputs | Passes |
| `test_value_ranges` | Prevents probabilities/scores exceeding bounds `[0, 1]` | Passes |
| `test_timezone_conversion` | Prevents DST or offset drift during conversion | Passes |
| `test_leakage_allowlist` | Ensures target variables are blocked from RF feature sets | Passes |
| `test_loader_idempotency` | Verifies single-load singleton artifact logic | Passes |

## 18. KNOWN LIMITATIONS
* **Data Limits:** The dataset spans up to 2019-05-15. It knows nothing about Instagram Reels, TikTok-style engagement shifts, or the COVID-19 era of social media behavior. 
* **Follower Point-in-Time:** Follower counts in the dataset were captured at the *time of scraping*, rather than exactly when the post was made. This limits temporal follower-growth modeling.
* **Timezone Mixture:** Because actual audience geographies are unknown, the data treats UTC as a unified mixture. Specific local nuances in target timezones are extrapolated entirely via offset logic.
* **Model Limits:** The R² is 0.193. The model is statistically better than a baseline, but organic engagement contains overwhelming random variance that cold-start features cannot explain.

## 19. TECHNICAL DEBT & IMPROVEMENT OPPORTUNITIES (For Future Engineers)
* **Data Quality:** Future iterations MUST replace the Kim et al. dataset with modern (2024+) scraping data including Reels to provide relevant real-world business value.
* **Evaluation Methodology:** The brand relevance fallback (`tau`) was calibrated globally. It could be calibrated per-category to prevent overly aggressive fallback triggers.
* **Model Architecture:** The Random Forest cannot ingest raw text. Experimenting with fine-tuning a small LLM (e.g. RoBERTa) directly on the engagement regression task might yield significantly higher R².
* **Runtime Performance:** `all-MiniLM-L6-v2` is run on CPU natively. Batching or moving inference to an ONNX runtime could optimize throughput for high-scale backend deployments.
* **Artifact Management:** Joblib and Pickle pose security risks if models are injected. Consider migrating artifacts to Safetensors or ONNX.

## 20. SAFE MODIFICATION GUIDE FOR FUTURE ENGINEERS
**Safe to change independently (Requires metric re-evaluation):**
* Random Forest hyperparameters.
* Hashtag ranking weights (`weights_hashtag`).
* Temporal decay functions (`time_half_life_days`).

**Changes requiring artifact regeneration:**
* Altering the trend signals invalidates `trends.parquet`.
* Altering the categorical definitions invalidates `category_centroids.pkl`.

**Changes requiring model retraining:**
* Adding/removing features in `config.yaml`'s `feature_allowlist` requires retraining `model.joblib` and updating `fallback_medians.json`.

**Changes requiring contract changes:**
* Altering the output format (e.g. replacing `Low/Medium/High` with 1-5 percentiles) requires explicit backend team coordination and an update to `contract/output_schema.json`.

## 21. DEPENDENCY / INVALIDATION MAP
```text
Raw Data (posts_clean)
  ├──→ Trend Engine ──────→ trends.parquet
  ├──→ Relevance Engine ──→ category_centroids.pkl, hashtag_embeddings.parquet
  ├──→ Content Engine ────→ content_type.parquet
  ├──→ Time Engine ───────→ posting_time.parquet
  ├──→ Hashtag Engine ────→ historical_engagement.parquet
  └──→ Engagement Model ──→ model.joblib, label_thresholds.json, fallback_medians.json
```
*If Raw Data changes, EVERY artifact must be regenerated.*
*If Trend methodology changes, only `trends.parquet` must be regenerated.*

## 22. CURRENT STATE VS FUTURE STATE
This document reflects the **Current MVP Implementation**, completely frozen and validated against the initial specification. Future states—such as incorporating deep learning for engagement prediction, upgrading to modern datasets, or rewriting the pipeline in Rust—are highly encouraged but must be formally evaluated against the baselines explicitly recorded in this document.

## 23. FINAL INDEPENDENT-ENGINEER SUMMARY
* **What does the system currently do?** Maps a brand's text to a category and provides statistically grounded offline-computed recommendations for hashtags, post timing, content type, and expected engagement.
* **What data does it depend on?** 1.3M cleaned posts from Kim et al. (2012-2019).
* **What models are used?** Pretrained `all-MiniLM-L6-v2` for embeddings; `Random Forest` for engagement regression.
* **What are the strongest measured results?** Content-type analysis generally selected carousel based on higher median relative engagement, although some pool-level estimates had uncertainty and fitness showed train/test disagreement. RF achieves 0.193 R², outperforming global medians drastically.
* **What can an engineer safely experiment with?** ML architecture, ranking weights, and hyperparameters, provided the backend contracts remain intact.
* **What must be inspected before modifying?** The `contract/` JSON schemas and the `RecommenderPipeline` initialization sequence to understand the exact structure the backend requires.

DONE!! INDEPENDENT ENGINEERING HANDOFF COMPLETE — IMPLEMENTATION, ARTIFACTS, RESULTS, DEPENDENCIES, AND IMPROVEMENT SURFACES DOCUMENTED
