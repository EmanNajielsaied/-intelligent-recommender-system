# FocusMate Pro – Recommender System update

## Included
- `recommender_engine.py`: Content-Based Filtering using TF-IDF and cosine similarity, plus feedback-aware re-ranking.
- `integration_patch.py`: Code snippets to connect the engine to the existing Streamlit app.

## Install dependency
```bash
pip install scikit-learn pandas
```
(Your current app already uses pandas.)

## Apply
1. Put `recommender_engine.py` in the same folder as `focus.py` and `load_model.py`.
2. Add the imports and `RECOMMENDER_CATALOG` initialization shown at the top of `integration_patch.py`.
3. In the analysis results section, replace the old recommendation selection/display loop with the supplied block. Keep the existing BERT inference and `user_state` calculation.
4. Run the app and give ratings to several recommendations. Ratings are stored in `feedback.csv`.

## Important implementation note
This is a ready-to-integrate recommendation module, not a verified replacement of every line in the original large `focus.py`. The original app has multiple nested display loops, so the integration block must replace the old recommendation block rather than be pasted after it; otherwise the app will display duplicate recommendations.

## Algorithm
- TF-IDF represents the resource metadata as vectors.
- Cosine similarity ranks resources against the current user profile.
- Exact subject/state matches and previous ratings adjust the score slightly.
- The interface shows an explanation for each recommendation.
