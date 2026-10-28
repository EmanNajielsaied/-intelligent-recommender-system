"""
FocusMate Pro - Content-Based Recommendation Engine
---------------------------------------------------
Uses TF-IDF + cosine similarity to rank study resources.
Uses explicit user ratings from feedback.csv to re-rank resources.

This is a content-based recommender with feedback-aware ranking,
not collaborative filtering and not an LLM.

Changes in this version:
1. resource_id now includes the user_state, so each (resource, state)
   tuple is treated as a distinct rankable item. This makes state_boost
   actually work (previously, merged states defeated the boost).
2. feedback_adjustment weight raised from 0.12 to 0.4, so explicit
   ratings can meaningfully change ranking order.
3. Duplicate URLs across states are preserved as separate items;
   deduplication (if desired) should happen at the UI layer.
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

    Existing tuple format: (title, embed_url, description)

    Each (resource, state) tuple is emitted as a separate item, so that
    a resource tagged for "excellent" and the same URL tagged for
    "need_improve" are treated as two distinct rankable items. This is
    what allows state_boost to have a real effect on ranking.
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


def load_feedback(feedback_path: str | Path = "feedback.csv") -> pd.DataFrame:
    path = Path(feedback_path)
    if not path.exists():
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)

    try:
        data = pd.read_csv(path)
        for column in FEEDBACK_COLUMNS:
            if column not in data.columns:
                data[column] = None
        data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
        return data.dropna(subset=["resource_id", "rating"])
    except (OSError, pd.errors.ParserError, ValueError):
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)


def save_feedback(
    resource_id: str,
    rating: int,
    subject: str = "",
    user_state: str = "",
    feedback_path: str | Path = "feedback.csv",
) -> bool:
    """Save an explicit 1–5 star rating; returns False if writing fails."""
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


def recommend_resources(
    resources: list[dict[str, Any]],
    subject: str,
    user_state: str,
    mood: str = "",
    difficulty: str = "",
    feedback_path: str | Path = "feedback.csv",
    top_n: int = 6,
) -> list[dict[str, Any]]:
    """
    Rank resources with a content similarity score and feedback adjustment.

    Similarity uses TF-IDF over resource metadata and a user profile query.
    Past average ratings adjust ranking meaningfully:
      rating 1 -> -0.4
      rating 3 ->  0.0
      rating 5 -> +0.4
    This weight (0.4) is large enough to change ordering relative to
    subject_boost (0.2) and state_boost (0.08), as intended.
    """
    if not resources:
        return []

    subject = str(subject or "")
    user_state = str(user_state or "")
    mood = str(mood or "")
    difficulty = str(difficulty or "")

    documents = []
    for item in resources:
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
        "study learning focus educational resource",
    ])

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        matrix = vectorizer.fit_transform(documents + [profile])
        similarities = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    except ValueError:
        similarities = [0.0] * len(resources)

    feedback = load_feedback(feedback_path)
    average_ratings = {}
    if not feedback.empty:
        average_ratings = feedback.groupby("resource_id")["rating"].mean().to_dict()

    ranked = []
    for item, similarity in zip(resources, similarities):
        item_subject = str(item.get("subject", ""))
        item_state = str(item.get("user_state", ""))
        rating = average_ratings.get(item.get("resource_id"))

        # Explicit, transparent relevance adjustments.
        subject_boost = 0.20 if subject and item_subject == subject else (
            0.03 if item_subject == "عام" else 0.0
        )
        state_boost = 0.08 if user_state and user_state in item_state.split("|") else 0.0
        # Feedback weight 0.4 (was 0.12): strong enough to reorder.
        feedback_adjustment = (
            0.0 if rating is None
            else ((float(rating) - 3.0) / 2.0) * 0.4
        )

        final_score = float(similarity) + subject_boost + state_boost + feedback_adjustment
        explanation = []
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
        ranked.append(ranked_item)

    ranked.sort(key=lambda row: row["final_score"], reverse=True)
    return ranked[:max(1, int(top_n))]