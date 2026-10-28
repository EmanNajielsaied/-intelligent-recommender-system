# FocusMate Pro integration patch
# 1) Add this import beside your other imports:
from recommender_engine import (
    flatten_recommendation_catalog,
    recommend_resources,
    save_feedback,
)

# 2) After PERSONALIZED_VIDEO_RECOMMENDATIONS and STUDY_MATERIALS are defined:
RECOMMENDER_CATALOG = flatten_recommendation_catalog(
    PERSONALIZED_VIDEO_RECOMMENDATIONS,
    STUDY_MATERIALS,
)

# 3) In the analysis-results section, after user_state is determined,
# replace the existing hard-coded recommendation-selection/display loops
# with the block below. Keep your existing BERT prediction and mood logic.

recommended_items = recommend_resources(
    resources=RECOMMENDER_CATALOG,
    subject=selected_subject,
    user_state=user_state,
    mood=selected_mood,
    feedback_path="feedback.csv",
    top_n=6,
)

st.markdown("### 🎯 توصيات مخصصة حسب احتياجك")
if not recommended_items:
    st.info("لا توجد موارد مناسبة حاليًا.")
else:
    for item in recommended_items:
        with st.container(border=True):
            st.markdown(f"#### {item['title']}")
            st.write(item.get("description", ""))
            st.caption(
                f"درجة الملاءمة: {item['final_score']:.3f} "
                f"| {item['why_recommended']}"
            )

            if item.get("url"):
                st.video(item["url"])

            rating_key = f"rating_{item['resource_id']}"
            rating = st.select_slider(
                "قيّمي التوصية",
                options=[1, 2, 3, 4, 5],
                value=3,
                key=rating_key,
                help="1 = غير مناسبة، 5 = مناسبة جدًا",
            )
            if st.button("حفظ التقييم", key=f"save_{item['resource_id']}"):
                saved = save_feedback(
                    resource_id=item["resource_id"],
                    rating=rating,
                    subject=selected_subject,
                    user_state=user_state,
                    feedback_path="feedback.csv",
                )
                if saved:
                    st.success("تم حفظ تقييمك، وسيؤثر في ترتيب التوصيات القادمة.")
                else:
                    st.error("تعذر حفظ التقييم. تحققي من صلاحيات الكتابة للملف.")

# Notes:
# - Existing BERT model remains responsible only for text/state classification.
# - The recommender is content-based (TF-IDF + cosine similarity), with ratings
#   used as a small ranking adjustment.
# - This is not collaborative filtering because it does not learn from other users.
