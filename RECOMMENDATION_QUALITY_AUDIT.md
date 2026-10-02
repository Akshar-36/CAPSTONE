# Recommendation Quality Audit

## 1. Executive Summary

An end-to-end recommendation quality diagnostic was performed across 10 synthetic brand profiles spanning various industries. While the Data/ML pipeline's implementation is mathematically and structurally correct according to the build specification, the semantic quality of the recommendations fails for 90% of the test cases. The underlying issues do not stem from code defects, but rather from heuristic threshold tuning (`tau = 0.343`), embedding similarity mismatches, and severe historical dataset biases that heavily pollute the fallback candidate pool.

## 2. Diagnostic Results (10 Brands)

| Brand | Expected Category | Predicted Category | Match Type | Hashtag Relevance | Notes / Extracted Hashtags |
| :--- | :--- | :--- | :--- | :--- | :--- |
| B1 (Fashion) | `fashion` | `all` | `fallback_all` | Questionable | Contains fashion tags (`#ootdfashion`, `#styleiswhat`), but also irrelevant demographic tags (`#mommylifestyle`). |
| B2 (Beauty) | `beauty` | `all` | `fallback_all` | Questionable | Contains beauty tags (`#makeupartistsworldwide`), but includes irrelevant geographic/demographic tags (`#nashvilleblogger`, `#mommylifestyle`). |
| B3 (Fitness) | `fitness` | `all` | `fallback_all` | Irrelevant | Fitness app for athletes yielded mommy-blogger tags (`#postpartum`, `#motherhoodthroughinstagram`). |
| B4 (Food) | `food` | `all` | `fallback_all` | Irrelevant | Vegan meal kit yielded irrelevant fashion and geographic tags (`#streetstyle`, `#indianfashionblogger`, `#india`). |
| B5 (Travel) | `travel` | `all` | `fallback_all` | Irrelevant | Solo young-adult backpacking brand yielded family tags (`#travelingwithkids`, `#familytravelblogger`). |
| B6 (Interior) | `interior` | `interior` | `matched` | Relevant | Successfully matched; yielded highly relevant hashtags (`#colourmyhome`, `#interiormilk`, `#urbanjunglebloggers`). |
| B7 (Pet) | `pet` | `all` | `fallback_all` | Questionable | Contains pet tags (`#dogsthathike`), but includes irrelevant tags (`#hairgoals`). |
| B8 (Other) | `other` | `all` | `fallback_all` | Irrelevant | B2B SaaS platform yielded mommy/influencer tags (`#sahm`, `#brandambassador`). |
| B9 (Other) | `other` | `all` | `fallback_all` | Irrelevant | Financial advisory firm yielded lifestyle/food tags (`#momswhoblog`, `#vscofood`, `#bakestagram`). |
| B10 (Family) | `family` | `all` | `fallback_all` | Questionable | Yielded general mom-blogger tags (`#candidchildhood`, `#sahm`, `#boymom`). |

## 3. Identified Quality Failures

### A. The Mapping Failure (Tau Threshold)
9 out of 10 brands failed to match their intended category and defaulted to `fallback_all`. The pipeline uses a cosine similarity threshold (`tau = 0.343`) to map a brand's text embedding to category centroid embeddings. Because category centroids are often derived from short, single-word labels (e.g., "fashion", "food") or narrow clusters, complex brand descriptions (e.g., "Vegan meal kit delivery service providing quick, plant-based dinners for busy professionals") fail to reach the `0.343` similarity threshold. As a result, the mapper rejects the true category and triggers the fallback mechanism.

### B. The Candidate Pool Failure (Dataset Bias)
Because 90% of brands hit the `fallback_all` route, their hashtag candidate pool becomes the global `all` category. The historical dataset (Kim et al., WWW '20) is not uniformly distributed across industries; it is heavily biased towards specific Instagram influencer demographics—primarily mommy bloggers, fashion influencers, travel bloggers, and regional (e.g., Indian) fashion segments. Consequently, the global candidate pool is severely polluted with niche demographic tags (`#mommylifestyle`, `#sahm`, `#indianfashionblogger`). When the pipeline attempts to recommend hashtags from this pool, it is forced to select from a heavily skewed distribution.

### C. The Ranking Failure (Embedding Mismatch)
The pipeline ranks candidate hashtags by computing the cosine similarity between the brand description embedding and the raw hashtag text embedding (e.g., matching "Vegan meal kit..." against "#indianfashionblogger"). SentenceTransformer models struggle to find meaningful semantic alignment between dense, descriptive paragraphs and concatenated, unspaced hashtag strings. This weak semantic linkage causes the ranking algorithm to latch onto spurious correlations in the biased global pool, surfacing entirely irrelevant hashtags (such as suggesting `#travelingwithkids` for a solo backpacking agency, or `#postpartum` for a general fitness app).

## 4. Conclusion

The Data/ML pipeline functions correctly according to the strict implementation guidelines, contracts, and schema requirements. However, the semantic quality of the recommendations is currently unusable for general brands. This is a direct consequence of combining a high rejection threshold (`tau=0.343`), which forces most brands into a fallback state, with a historical dataset that suffers from extreme demographic and topical bias. While the pipeline is technically sound, the underlying data and mapping heuristics require fundamental revision before the recommendations can be considered semantically viable.
