"""
FocusMate Pro - Content-Based Recommendation Engine
---------------------------------------------------
Uses TF-IDF + cosine similarity to rank study resources.
Uses explicit user ratings from feedback.csv to re-rank resources.

This is a content-based recommender with feedback-aware ranking,
not collaborative filtering and not an LLM.

Enhancements in this version:
1. resource_id includes user_state so each (resource, state) tuple is
   a distinct rankable item. This makes state_boost actually work.
2. feedback_adjustment weight is 0.4 (was 0.12), so explicit ratings can
   meaningfully change ranking order.
3. query_text (the user's free-text input) is folded into the profile,
   so resources whose description matches the user's words rank higher.
4. load_feedback is hardened against missing header, bad rows, and
   out-of-range ratings.
5. validate_resources() removes duplicate URLs from the catalog.
6. Hard state filter: when user_state is 'excellent' or 'need_improve',
   resources tagged exclusively for the opposite state are excluded
   (unless the user has previously rated them 4+).
7. Guaranteed diversity: the top-N list always includes at least one
   'general' resource (subject='عام'), unless the catalog has none.
8. Each returned item carries a detailed `match_details` dict for the
   UI to show 'why this was recommended'.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FEEDBACK_COLUMNS = ["resource_id", "rating", "subject", "user_state", "timestamp"]


def _slug(value: str) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^a-z0-9\u0600-\u06FF]+", "-", value)
    return value.strip("-")[:100] or "resource"


def flatten_recommendation_catalog(
    personalized_recommendations: dict,
    study_materials: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Convert the existing PERSONALIZED_VIDEO_RECOMMENDATIONS nested dict
    into a flat list the recommender can rank.

    Each (resource, state) tuple is emitted as a separate item, so the
    same URL appearing under two states is treated as two distinct
    rankable items. This is what allows state_boost to have a real effect.
    """
    resources = []
    study_materials = study_materials or {}

    for state, state_data in (personalized_recommendations or {}).items():
        for title, url, description in state_data.get("general_focus", []):
            resources.append({
                "resource_id": f"{_slug(url or title)}::{state}",
                "title": title,
                "url": url,
                "description": description,
                "subject": "عام",
                "user_state": state,
                "difficulty": "متنوع",
                "tags": "focus productivity study skills",
                "kind": "general_focus",
            })

        for subject, items in state_data.get("subject_specific", {}).items():
            for title, url, description in items:
                material = study_materials.get(subject, {})
                resources.append({
                    "resource_id": f"{_slug(url or title)}::{state}",
                    "title": title,
                    "url": url,
                    "description": description,
                    "subject": subject,
                    "user_state": state,
                    "difficulty": material.get("difficulty", "متنوع"),
                    "tags": f"{subject} {material.get('description', '')}",
                    "kind": "subject_specific",
                })

    return resources


def validate_resources(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove resources with empty or duplicate URLs.
    When duplicates exist, keep the first occurrence (caller can pre-sort
    by rating if desired).
    """
    seen_urls: set[str] = set()
    valid: list[dict[str, Any]] = []
    for item in resources:
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        valid.append(item)
    return valid


def load_feedback(feedback_path: str | Path = "feedback.csv") -> pd.DataFrame:
    """
    Load explicit ratings from feedback.csv.

    Robustness rules:
    - If the file does not exist, return an empty DataFrame.
    - If the first line does not start with 'resource_id', assume there is
      no header and assign FEEDBACK_COLUMNS manually.
    - Malformed rows are skipped rather than crashing the whole read.
    - Only rows with a numeric rating in [1, 5] are kept.
    """
    path = Path(feedback_path)
    if not path.exists():
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)

    try:
        first_line = ""
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.strip():
                    first_line = line.strip()
                    break

        has_header = first_line.startswith("resource_id")

        if has_header:
            data = pd.read_csv(path, on_bad_lines="skip")
        else:
            data = pd.read_csv(path, header=None, on_bad_lines="skip")
            if len(data.columns) == len(FEEDBACK_COLUMNS):
                data.columns = FEEDBACK_COLUMNS
            else:
                return pd.DataFrame(columns=FEEDBACK_COLUMNS)

        for column in FEEDBACK_COLUMNS:
            if column not in data.columns:
                data[column] = None

        data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
        data = data.dropna(subset=["resource_id", "rating"])
        data = data[(data["rating"] >= 1) & (data["rating"] <= 5)]
        return data
    except (OSError, pd.errors.ParserError, ValueError, IndexError):
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)


def save_feedback(
    resource_id: str,
    rating: int,
    subject: str = "",
    user_state: str = "",
    feedback_path: str | Path = "feedback.csv",
) -> bool:
    """Save an explicit 1-5 star rating; returns False if writing fails."""
    if not 1 <= int(rating) <= 5:
        raise ValueError("rating must be between 1 and 5")

    path = Path(feedback_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "resource_id": resource_id,
        "rating": int(rating),
        "subject": subject,
        "user_state": user_state,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    try:
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FEEDBACK_COLUMNS)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
        return True
    except OSError:
        return False


def _state_filter(
    resources: list[dict[str, Any]],
    user_state: str,
    average_ratings: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Hard state filter:
    - If user is 'excellent', drop resources tagged ONLY for 'need_improve'
      unless user rated them 4+.
    - If user is 'need_improve', drop resources tagged ONLY for 'excellent'
      unless user rated them 4+.
    """
    if user_state not in ("excellent", "need_improve"):
        return resources

    opposite = "need_improve" if user_state == "excellent" else "excellent"
    filtered: list[dict[str, Any]] = []
    for item in resources:
        states = str(item.get("user_state", "")).split("|")
        rating = average_ratings.get(item.get("resource_id"))
        if opposite in states and user_state not in states:
            # Tagged only for opposite state; keep only if user liked it.
            if rating is not None and rating >= 4:
                filtered.append(item)
            # else: drop
        else:
            filtered.append(item)
    return filtered


def recommend_resources(
    resources: list[dict[str, Any]],
    subject: str,
    user_state: str,
    mood: str = "",
    difficulty: str = "",
    query_text: str = "",
    feedback_path: str | Path = "feedback.csv",
    top_n: int = 6,
) -> list[dict[str, Any]]:
    """
    Rank resources with a content similarity score and feedback adjustment.

    Similarity uses TF-IDF over resource metadata and a user profile query.
    The user profile includes the free-text `query_text`, so resources whose
    description matches the user's own words rank higher.

    Past average ratings adjust ranking meaningfully:
      rating 1 -> -0.4
      rating 3 ->  0.0
      rating 5 -> +0.4

    Additional post-processing:
    - Hard state filter (excellent vs need_improve).
    - Guaranteed diversity: at least one 'general' resource in top-N
      (unless the catalog has none).
    """
    if not resources:
        return []

    subject = str(subject or "")
    user_state = str(user_state or "")
    mood = str(mood or "")
    difficulty = str(difficulty or "")
    query_text = str(query_text or "")

    feedback = load_feedback(feedback_path)
    average_ratings: dict[str, float] = {}
    if not feedback.empty:
        average_ratings = feedback.groupby("resource_id")["rating"].mean().to_dict()

    # Apply hard state filter BEFORE TF-IDF so 'general' resources stay.
    filtered_resources = _state_filter(resources, user_state, average_ratings)
    if not filtered_resources:
        filtered_resources = resources

    documents = []
    for item in filtered_resources:
        documents.append(" ".join([
            str(item.get("title", "")),
            str(item.get("description", "")),
            str(item.get("subject", "")),
            str(item.get("difficulty", "")),
            str(item.get("tags", "")),
            str(item.get("user_state", "")),
            str(item.get("kind", "")),
        ]))

    profile = " ".join([
        subject, subject, difficulty, difficulty,
        user_state, user_state, mood,
        query_text, query_text,
        "study learning focus educational resource",
    ])

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        matrix = vectorizer.fit_transform(documents + [profile])
        similarities = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    except ValueError:
        similarities = [0.0] * len(filtered_resources)

    ranked: list[dict[str, Any]] = []
    for item, similarity in zip(filtered_resources, similarities):
        item_subject = str(item.get("subject", ""))
        item_state = str(item.get("user_state", ""))
        rating = average_ratings.get(item.get("resource_id"))

        subject_boost = 0.20 if subject and item_subject == subject else (
            0.03 if item_subject == "عام" else 0.0
        )
        state_boost = 0.08 if user_state and user_state in item_state.split("|") else 0.0
        feedback_adjustment = (
            0.0 if rating is None
            else ((float(rating) - 3.0) / 2.0) * 0.4
        )

        final_score = float(similarity) + subject_boost + state_boost + feedback_adjustment

        explanation: list[str] = []
        if item_subject == subject and subject:
            explanation.append(f"متوافق مع مادة {subject}")
        elif item_subject == "عام":
            explanation.append("مورد عام لتحسين التركيز")
        if user_state and user_state in item_state.split("|"):
            explanation.append("مناسب لحالة التركيز الحالية")
        if difficulty and str(item.get("difficulty", "")) == difficulty:
            explanation.append("متوافق مع مستوى الصعوبة")
        if rating is not None:
            explanation.append(f"متوسط تقييماتك السابقة: {float(rating):.1f}/5")
        if not explanation:
            explanation.append("اختير بناءً على تشابه وصف المورد مع احتياجك")

        ranked_item = dict(item)
        ranked_item["similarity_score"] = round(float(similarity), 4)
        ranked_item["final_score"] = round(final_score, 4)
        ranked_item["average_user_rating"] = (
            round(float(rating), 2) if rating is not None else None
        )
        ranked_item["why_recommended"] = "؛ ".join(explanation)
        ranked_item["match_details"] = {
            "similarity": round(float(similarity), 4),
            "subject_boost": round(subject_boost, 4),
            "state_boost": round(state_boost, 4),
            "feedback_adjustment": round(feedback_adjustment, 4),
            "final_score": round(final_score, 4),
            "matched_subject": item_subject == subject and bool(subject),
            "matched_state": user_state in item_state.split("|") if user_state else False,
        }
        ranked.append(ranked_item)

    ranked.sort(key=lambda row: row["final_score"], reverse=True)
    top_items = ranked[:max(1, int(top_n))]

    # Diversity guarantee: ensure at least one 'general' item in top-N.
    has_general = any(item.get("subject") == "عام" for item in top_items)
    if not has_general and len(ranked) > len(top_items):
        general_candidates = [r for r in ranked if r.get("subject") == "عام"]
        if general_candidates:
            top_items[-1] = general_candidates[0]

    return top_items