import streamlit as st
import random
import torch
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from load_model import load_sentiment_model, predict_text
from recommender_engine import (
    flatten_recommendation_catalog,
    recommend_resources,
    save_feedback,
)
from pathlib import Path
import time
import json

# ----------------------------
# Page Config & Custom CSS
# ----------------------------
st.set_page_config(
    page_title="FocusMate Pro - مدرب التركيز الذكي",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    .gradient-text {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .elegant-card {
        background: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.8);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    .elegant-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }
    .elegant-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px 20px 0 0;
    }
    .stats-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.95) 100%);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(226, 232, 240, 0.6);
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    .video-card-professional {
        background: white;
        border-radius: 16px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(226, 232, 240, 0.8);
        box-shadow: 0 6px 25px rgba(0,0,0,0.07);
        transition: all 0.3s ease;
    }
    .video-card-professional:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.15);
    }
    .custom-progress {
        height: 12px;
        border-radius: 6px;
        background: #e2e8f0;
        overflow: hidden;
    }
    .custom-progress-fill {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #4fd1c5 0%, #38b2ac 100%);
        transition: width 0.6s ease;
    }
    .status-badge-pro {
        display: inline-flex;
        align-items: center;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 5px;
        backdrop-filter: blur(10px);
    }
    .status-excellent-pro {
        background: linear-gradient(135deg, rgba(212, 252, 121, 0.15) 0%, rgba(150, 230, 161, 0.15) 100%);
        color: #22543d;
        border: 1px solid rgba(56, 161, 105, 0.2);
    }
    .status-good-pro {
        background: linear-gradient(135deg, rgba(163, 189, 237, 0.15) 0%, rgba(105, 145, 199, 0.15) 100%);
        color: #1a365d;
        border: 1px solid rgba(49, 130, 206, 0.2);
    }
    .status-need-improve-pro {
        background: linear-gradient(135deg, rgba(240, 147, 251, 0.15) 0%, rgba(245, 87, 108, 0.15) 100%);
        color: #742a2a;
        border: 1px solid rgba(229, 62, 62, 0.2);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #f8fafc;
        padding: 4px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 14px 32px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    .stRadio > div { flex-direction: row; gap: 10px; }
    .stRadio > div > label {
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 15px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .stRadio > div > label:hover {
        border-color: #667eea;
        background: #f8fafc;
    }
    .stRadio > div > label[data-baseweb="radio"]:has(input:checked) {
        border-color: #667eea;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    .floating-icon { animation: float 6s ease-in-out infinite; }
    .subject-card { transition: all 0.3s ease; cursor: pointer; }
    .subject-card:hover { transform: scale(1.02); }
    .material-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.95) 100%);
        border-radius: 16px;
        padding: 25px;
        border: 1px solid rgba(226, 232, 240, 0.6);
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        height: 100%;
    }
    .material-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.1);
        border-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Initialize Session State
# ----------------------------
if 'user_text' not in st.session_state:
    st.session_state.user_text = ""
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.datetime.now()
if 'draft' not in st.session_state:
    st.session_state.draft = ""
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "🎯"
if 'selected_library_subject' not in st.session_state:
    st.session_state.selected_library_subject = "شبكات الحاسوب"
if "analysis_active" not in st.session_state:
    st.session_state.analysis_active = False
if "history_saved" not in st.session_state:
    st.session_state.history_saved = False
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


def activate_analysis():
    st.session_state.analysis_active = True
    st.session_state.history_saved = False
    st.session_state.analysis_result = None


def start_new_session():
    st.session_state.user_text = ""
    st.session_state.analysis_active = False
    st.session_state.history_saved = False
    st.session_state.analysis_result = None
    st.session_state.start_time = datetime.datetime.now()


# ----------------------------
# Helper Functions
# ----------------------------
def load_history_data():
    """Load history data with proper error handling"""
    if not Path("history.csv").exists():
        return pd.DataFrame(columns=["time", "text", "status", "confidence", "mood", "focus_score", "subject"])

    try:
        data = pd.read_csv("history.csv")

        if len(data.columns) == 1 or 'time' not in data.columns:
            data = pd.read_csv("history.csv", header=None, on_bad_lines='skip')
            if len(data.columns) >= 7:
                data.columns = ["time", "text", "status", "confidence", "mood", "focus_score", "subject"][:len(data.columns)]

        required_columns = ["time", "text", "status", "confidence", "mood", "focus_score", "subject"]
        for col in required_columns:
            if col not in data.columns:
                data[col] = None

        data["time"] = pd.to_datetime(data["time"], errors='coerce')
        data["confidence"] = pd.to_numeric(data["confidence"], errors='coerce')
        data["focus_score"] = pd.to_numeric(data["focus_score"], errors='coerce')

        data = data.dropna(subset=["time"])

        return data

    except Exception:
        return pd.DataFrame(columns=["time", "text", "status", "confidence", "mood", "focus_score", "subject"])


def save_history_entry(entry):
    """Save a new entry to history.csv"""
    try:
        headers = ["time", "text", "status", "confidence", "mood", "focus_score", "subject"]
        entry_df = pd.DataFrame([entry], columns=headers)

        if Path("history.csv").exists():
            try:
                existing_data = pd.read_csv("history.csv")
                if len(existing_data.columns) == len(headers):
                    entry_df.to_csv("history.csv", mode='a', header=False, index=False)
                else:
                    combined = pd.concat([existing_data, entry_df], ignore_index=True)
                    combined.to_csv("history.csv", index=False)
            except Exception:
                entry_df.to_csv("history.csv", index=False)
        else:
            entry_df.to_csv("history.csv", index=False)

        return True
    except Exception:
        return False


def calculate_streak(data):
    """Calculate current streak from history data"""
    if data.empty:
        return 0
    try:
        data = data.copy()
        data['date'] = pd.to_datetime(data['time']).dt.date
        unique_dates = sorted(set(data['date']), reverse=True)

        streak = 0
        current_date = datetime.date.today()

        for date in unique_dates:
            if date == current_date or date == current_date - datetime.timedelta(days=1):
                streak += 1
                current_date = date
            else:
                break

        return streak
    except Exception:
        return 0


# ----------------------------
# Personalized Video Recommendations Database
# ----------------------------
PERSONALIZED_VIDEO_RECOMMENDATIONS = {
    "excellent": {
        "general_focus": [
            ("🚀 تقنيات التعلم المتقدم", "https://www.youtube.com/embed/IlU-zDU6aQ0", "تعلم كيف تتعمق في المواضيع الصعبة"),
            ("🎯 التخطيط الاستراتيجي للدراسة", "https://www.youtube.com/embed/3s4cL6I2k4M", "كيف تخطط لأهدافك الأكاديمية"),
            ("💪 تطوير العقلية الناجحة", "https://www.youtube.com/embed/F6CQh6rjV4o", "بناء عقلية المتفوقين")
        ],
        "subject_specific": {
            "شبكات الحاسوب": [
                ("🌐 الشبكات المتقدمة", "https://www.youtube.com/embed/qiQR5rTSshw", "مفاهيم متقدمة في الشبكات"),
                ("🔒 أمن الشبكات", "https://www.youtube.com/embed/3bSIHivxGR4", "حماية الشبكات من الاختراقات")
            ],
            "البرمجة": [
                ("💻 هياكل البيانات المعقدة", "https://www.youtube.com/embed/RBSGKlAvoiM", "هياكل بيانات متقدمة"),
                ("⚡ تحسين الخوارزميات", "https://www.youtube.com/embed/09_LlHjoEiY", "كيف تحسن من أداء كودك")
            ],
            "NLP": [
                ("🤖 الذكاء الاصطناعي المتقدم", "https://www.youtube.com/embed/8S3qyygmX_c", "تقنيات NLP حديثة"),
                ("🧠 فهم Transformers", "https://www.youtube.com/embed/TQQlZhbC5ps", "شبكات Transformer كاملة")
            ]
        }
    },
    "good": {
        "general_focus": [
            ("📚 تقنيات المراجعة الفعالة", "https://www.youtube.com/embed/Z-zNHHpXoMM", "Active Recall و Spaced Repetition"),
            ("⏱️ إدارة الوقت للطلاب", "https://www.youtube.com/embed/iONDebHX9qk", "كيف تنظم وقتك للدراسة"),
            ("🤝 التعلم التعاوني", "https://www.youtube.com/embed/rWEwv_qobpU", "كيف تدرس مع الآخرين")
        ],
        "subject_specific": {
            "شبكات الحاسوب": [
                ("🔗 أساسيات الشبكات", "https://www.youtube.com/embed/0y6FtKsg6J4", "طبقات OSI السبعة"),
                ("📡 بروتوكول TCP/IP", "https://www.youtube.com/embed/PpsEaqJV_A0", "فهم TCP/IP عملياً")
            ],
            "البرمجة": [
                ("🐍 بايثون للمتوسطين", "https://www.youtube.com/embed/_uQrJ0TkZlc", "مستوى متوسط في بايثون"),
                ("🔧 حل المشكلات البرمجية", "https://www.youtube.com/embed/uQdy914JRKQ", "تقنيات حل المشكلات")
            ],
            "NLP": [
                ("📝 معالجة اللغة الطبيعية", "https://www.youtube.com/embed/MN__ZByYJa0", "مقدمة عملية للـNLP"),
                ("🎯 تحليل المشاعر", "https://www.youtube.com/embed/MN__ZByYJa0", "كيف تحلل مشاعر النصوص")
            ]
        }
    },
    "need_improve": {
        "general_focus": [
            ("🧘‍♀️ تمارين الاسترخاء السريعة", "https://www.youtube.com/embed/tEmt1Znux58", "5 دقائق فقط للتركيز"),
            ("🍅 تقنية Pomodoro", "https://www.youtube.com/embed/mNBmG24djoY", "25 دقيقة دراسة، 5 دقائق راحة"),
            ("🎧 إدارة التشتت", "https://www.youtube.com/embed/9GorqroigqM", "كيف تتخلص من الملهيات")
        ],
        "subject_specific": {
            "شبكات الحاسوب": [
                ("🔌 أساسيات الشبكات للمبتدئين", "https://www.youtube.com/embed/qiQR5rTSshw", "مقدمة بسيطة للشبكات"),
                ("📊 فهم الشبكات البسيطة", "https://www.youtube.com/embed/PpsEaqJV_A0", "مفاهيم أساسية فقط")
            ],
            "البرمجة": [
                ("🐍 بايثون للمبتدئين", "https://www.youtube.com/embed/_uQrJ0TkZlc", "أول خطواتك في البرمجة"),
                ("🎯 أساسيات الخوارزميات", "https://www.youtube.com/embed/RBSGKlAvoiM", "مقدمة للخوارزميات")
            ],
            "NLP": [
                ("🤖 مقدمة للذكاء الاصطناعي", "https://www.youtube.com/embed/8S3qyygmX_c", "ما هو الذكاء الاصطناعي؟"),
                ("📝 معالجة النصوص البسيطة", "https://www.youtube.com/embed/MN__ZByYJa0", "أساسيات معالجة اللغة")
            ]
        }
    }
}

# ----------------------------
# Study Materials Database
# ----------------------------
STUDY_MATERIALS = {
    "شبكات الحاسوب": {
        "description": "دراسة شبكات الحاسوب، البروتوكولات، والأمان",
        "icon": "🌐",
        "difficulty": "متوسط",
        "color": "#4299e1",
        "modules": [
            {"name": "مقدمة في الشبكات", "duration": "2 ساعة", "completed": False},
            {"name": "بروتوكول TCP/IP", "duration": "3 ساعات", "completed": False},
            {"name": "أمن الشبكات", "duration": "4 ساعات", "completed": False},
            {"name": "شبكات الوايرلس", "duration": "2.5 ساعة", "completed": False}
        ],
        "resources": [
            {"type": "📘 كتاب", "name": "شبكات الحاسوب للمبتدئين", "link": "#"},
            {"type": "🎥 فيديو", "name": "دورة كاملة في الشبكات", "link": "#"},
            {"type": "📄 مقال", "name": "دليل بروتوكولات الشبكات", "link": "#"}
        ]
    },
    "البرمجة": {
        "description": "تعلم البرمجة وهياكل البيانات والخوارزميات",
        "icon": "💻",
        "difficulty": "متنوع",
        "color": "#48bb78",
        "modules": [
            {"name": "أساسيات البرمجة", "duration": "3 ساعات", "completed": False},
            {"name": "هياكل البيانات", "duration": "5 ساعات", "completed": False},
            {"name": "الخوارزميات", "duration": "6 ساعات", "completed": False},
            {"name": "مشاريع عملية", "duration": "8 ساعات", "completed": False}
        ],
        "resources": [
            {"type": "📘 كتاب", "name": "تعلم البرمجة من الصفر", "link": "#"},
            {"type": "🎥 فيديو", "name": "دورة برمجة متكاملة", "link": "#"},
            {"type": "💻 مشروع", "name": "تطبيقات عملية", "link": "#"}
        ]
    },
    "NLP": {
        "description": "معالجة اللغة الطبيعية والذكاء الاصطناعي",
        "icon": "🤖",
        "difficulty": "متقدم",
        "color": "#ed8936",
        "modules": [
            {"name": "مقدمة في NLP", "duration": "2 ساعة", "completed": False},
            {"name": "معالجة النصوص", "duration": "3 ساعات", "completed": False},
            {"name": "نماذج اللغة", "duration": "4 ساعات", "completed": False},
            {"name": "التطبيقات العملية", "duration": "5 ساعات", "completed": False}
        ],
        "resources": [
            {"type": "📘 كتاب", "name": "مدخل إلى معالجة اللغة", "link": "#"},
            {"type": "🎥 فيديو", "name": "دورة NLP متقدمة", "link": "#"},
            {"type": "🔬 بحث", "name": "أوراق بحثية مهمة", "link": "#"}
        ]
    },
    "قواعد البيانات": {
        "description": "تصميم وإدارة قواعد البيانات",
        "icon": "🗃️",
        "difficulty": "متوسط",
        "color": "#9f7aea",
        "modules": [
            {"name": "مقدمة في قواعد البيانات", "duration": "2 ساعة", "completed": False},
            {"name": "لغة SQL", "duration": "4 ساعات", "completed": False},
            {"name": "التصميم المعماري", "duration": "3 ساعات", "completed": False}
        ],
        "resources": [
            {"type": "📘 كتاب", "name": "قواعد البيانات الشامل", "link": "#"},
            {"type": "🎥 فيديو", "name": "تعلم SQL من الصفر", "link": "#"},
            {"type": "💾 تمرين", "name": "تمارين عملية", "link": "#"}
        ]
    },
    "تطوير الويب": {
        "description": "تطوير مواقع وتطبيقات الويب",
        "icon": "🌐",
        "difficulty": "مبتدئ إلى متقدم",
        "color": "#f56565",
        "modules": [
            {"name": "HTML & CSS", "duration": "3 ساعات", "completed": False},
            {"name": "JavaScript", "duration": "5 ساعات", "completed": False},
            {"name": "أطر العمل", "duration": "6 ساعات", "completed": False}
        ],
        "resources": [
            {"type": "📘 كتاب", "name": "تطوير الويب الحديث", "link": "#"},
            {"type": "🎥 فيديو", "name": "دورة تطوير ويب كاملة", "link": "#"},
            {"type": "🌐 مشروع", "name": "بناء موقع متكامل", "link": "#"}
        ]
    },
    "Machine Learning": {
        "description": "تعلم الآلة والتنبؤ بالبيانات",
        "icon": "🧠",
        "difficulty": "متقدم",
        "color": "#ed64a6",
        "modules": [
            {"name": "أساسيات ML", "duration": "3 ساعات", "completed": False},
            {"name": "الخوارزميات", "duration": "6 ساعات", "completed": False},
            {"name": "التطبيقات العملية", "duration": "8 ساعات", "completed": False}
        ],
        "resources": [
            {"type": "📘 كتاب", "name": "تعلم الآلة خطوة بخطوة", "link": "#"},
            {"type": "🎥 فيديو", "name": "دورة ML متقدمة", "link": "#"},
            {"type": "📊 بيانات", "name": "مجموعات بيانات للتدريب", "link": "#"}
        ]
    }
}

# قاعدة بيانات فيديوهات يوتيوب للمواد الدراسية
SUBJECT_YOUTUBE_VIDEOS = {
    "شبكات الحاسوب": {
        "مقدمة عامة": [
            ("مقدمة في شبكات الحاسوب - أساسيات الشبكات", "https://www.youtube.com/embed/qiQR5rTSshw", "تعلم أساسيات الشبكات والمفاهيم الأساسية", "🟢 للمبتدئين"),
            ("ما هي شبكات الحاسوب ولماذا نتعلمها؟", "https://www.youtube.com/embed/0y6FtKsg6J4", "فهم أهمية الشبكات في عالم التكنولوجيا", "🎯 تمهيدي"),
            ("تاريخ وتطور شبكات الحاسوب", "https://www.youtube.com/embed/3bSIHivxGR4", "رحلة تطور الشبكات من البداية حتى الآن", "📜 تاريخي")
        ],
        "البروتوكولات الأساسية": [
            ("شرح بروتوكول TCP/IP بالتفصيل", "https://www.youtube.com/embed/PpsEaqJV_A0", "فهم بروتوكول TCP/IP عملياً", "🔧 عملي"),
            ("مقارنة بين TCP و UDP", "https://www.youtube.com/embed/0y6FtKsg6J4", "الفرق بين بروتوكولات الاتصال", "⚖️ مقارنة"),
            ("بروتوكول HTTP و HTTPS", "https://www.youtube.com/embed/3bSIHivxGR4", "بروتوكولات الويب والأمان", "🌐 ويب")
        ],
        "أمن الشبكات": [
            ("أساسيات أمن الشبكات للمبتدئين", "https://www.youtube.com/embed/qiQR5rTSshw", "حماية الشبكات من الهجمات", "🛡️ أمن"),
            ("الفرق بين Firewall و VPN", "https://www.youtube.com/embed/0y6FtKsg6J4", "مقارنة أدوات حماية الشبكات", "🔒 مقارنة"),
            ("كيف تحمي شبكتك المنزلية", "https://www.youtube.com/embed/3bSIHivxGR4", "نصائح عملية لحماية الشبكات", "🏠 منزلي")
        ],
        "شبكات الوايرلس": [
            ("شبكات Wi-Fi الحديثة", "https://www.youtube.com/embed/PpsEaqJV_A0", "تقنيات الوايرلس المتطورة", "📶 لاسلكي"),
            ("الفرق بين 2.4GHz و 5GHz", "https://www.youtube.com/embed/0y6FtKsg6J4", "مقارنة ترددات الشبكات اللاسلكية", "📊 تقني"),
            ("تطبيقات الشبكات اللاسلكية", "https://www.youtube.com/embed/3bSIHivxGR4", "استخدامات الشبكات في الحياة العملية", "💼 تطبيقي")
        ]
    },
    "البرمجة": {
        "مقدمة للبرمجة": [
            ("ما هي البرمجة ولماذا تتعلمها؟", "https://www.youtube.com/embed/_uQrJ0TkZlc", "بداية رحلتك في عالم البرمجة", "🚀 انطلاق"),
            ("الفرق بين لغات البرمجة", "https://www.youtube.com/embed/RBSGKlAvoiM", "مقارنة بين Python, Java, C++", "⚖️ مقارنة"),
            ("كيف تختار لغة البرمجة المناسبة", "https://www.youtube.com/embed/09_LlHjoEiY", "دليل اختيار لغة البرمجة الأولى", "🎯 توجيه")
        ],
        "بايثون للمبتدئين": [
            ("بايثون من الصفر للإحتراف", "https://www.youtube.com/embed/_uQrJ0TkZlc", "دورة كاملة في لغة بايثون", "🐍 شامل"),
            ("المتغيرات والعمليات في بايثون", "https://www.youtube.com/embed/RBSGKlAvoiM", "أساسيات البرمجة في بايثون", "🔢 أساسيات"),
            ("الدوال والوحدات في بايثون", "https://www.youtube.com/embed/09_LlHjoEiY", "البرمجة الوظيفية في بايثون", "⚙️ متقدم")
        ],
        "تطوير الويب": [
            ("HTML و CSS للمبتدئين", "https://www.youtube.com/embed/_uQrJ0TkZlc", "أساسيات تطوير الواجهات", "🌐 ويب"),
            ("JavaScript من الصفر", "https://www.youtube.com/embed/RBSGKlAvoiM", "برمجة الويب التفاعلية", "🎨 تفاعلي"),
            ("الفرق بين Frontend و Backend", "https://www.youtube.com/embed/09_LlHjoEiY", "فهم جوانب تطوير الويب", "⚖️ مقارنة")
        ],
        "هياكل البيانات": [
            ("مقدمة لهياكل البيانات", "https://www.youtube.com/embed/_uQrJ0TkZlc", "أهمية هياكل البيانات في البرمجة", "📊 أساسي"),
            ("المصفوفات والقوائم في البرمجة", "https://www.youtube.com/embed/RBSGKlAvoiM", "هياكل البيانات الأساسية", "🔢 أساسيات"),
            ("الأشجار والخرائط في البرمجة", "https://www.youtube.com/embed/09_LlHjoEiY", "هياكل البيانات المتقدمة", "🌳 متقدم")
        ]
    },
    "NLP": {
        "مقدمة في NLP": [
            ("ما هو معالجة اللغة الطبيعية؟", "https://www.youtube.com/embed/8S3qyygmX_c", "مقدمة شاملة لمجال NLP", "🤖 تمهيدي"),
            ("تطبيقات NLP في الحياة العملية", "https://www.youtube.com/embed/MN__ZByYJa0", "استخدامات الذكاء الاصطناعي في اللغة", "💼 تطبيقي"),
            ("الفرق بين NLP و ML", "https://www.youtube.com/embed/TQQlZhbC5ps", "مقارنة بين المجالات المختلفة", "⚖️ مقارنة")
        ],
        "نماذج اللغة": [
            ("شرح نماذج Transformers", "https://www.youtube.com/embed/8S3qyygmX_c", "فهم أحدث نماذج معالجة اللغة", "🧠 متقدم"),
            ("BERT و GPT مقارنة", "https://www.youtube.com/embed/MN__ZByYJa0", "مقارنة بين أشهر نماذج اللغة", "⚖️ مقارنة"),
            ("كيف تعمل نماذج الذكاء الاصطناعي", "https://www.youtube.com/embed/TQQlZhbC5ps", "آلية عمل النماذج اللغوية", "🔧 تقني")
        ],
        "التطبيقات العملية": [
            ("تحليل المشاعر باستخدام NLP", "https://www.youtube.com/embed/8S3qyygmX_c", "كيفية تحليل مشاعر النصوص", "😊 تطبيقي"),
            ("الترجمة الآلية بالذكاء الاصطناعي", "https://www.youtube.com/embed/MN__ZByYJa0", "تطبيقات الترجمة في NLP", "🌍 ترجمة"),
            ("التلخيص الآلي للنصوص", "https://www.youtube.com/embed/TQQlZhbC5ps", "تقنيات تلخيص النصوص تلقائياً", "📝 تلخيص")
        ]
    },
    "قواعد البيانات": {
        "مقدمة لقواعد البيانات": [
            ("ما هي قواعد البيانات ولماذا مهمة؟", "https://www.youtube.com/embed/qiQR5rTSshw", "أساسيات قواعد البيانات", "🗃️ أساسي"),
            ("الفرق بين SQL و NoSQL", "https://www.youtube.com/embed/0y6FtKsg6J4", "مقارنة أنواع قواعد البيانات", "⚖️ مقارنة"),
            ("تصميم قواعد البيانات", "https://www.youtube.com/embed/3bSIHivxGR4", "كيف تصمم قاعدة بيانات فعالة", "🎨 تصميم")
        ],
        "لغة SQL": [
            ("SQL من الصفر للإحتراف", "https://www.youtube.com/embed/PpsEaqJV_A0", "دورة كاملة في لغة SQL", "📊 شامل"),
            ("الفرق بين JOINs المختلفة", "https://www.youtube.com/embed/0y6FtKsg6J4", "مقارنة عمليات الربط في SQL", "🔗 مقارنة"),
            ("الاستعلامات المتقدمة في SQL", "https://www.youtube.com/embed/3bSIHivxGR4", "تقنيات متقدمة في SQL", "🚀 متقدم")
        ],
        "قواعد البيانات العلائقية": [
            ("MySQL للمبتدئين", "https://www.youtube.com/embed/qiQR5rTSshw", "تعلم MySQL من البداية", "🐬 عملي"),
            ("PostgreSQL vs MySQL", "https://www.youtube.com/embed/0y6FtKsg6J4", "مقارنة بين أنظمة قواعد البيانات", "⚖️ مقارنة"),
            ("تحسين أداء قواعد البيانات", "https://www.youtube.com/embed/3bSIHivxGR4", "نصائح لتحسين سرعة قواعد البيانات", "⚡ أداء")
        ]
    },
    "تطوير الويب": {
        "الأساسيات": [
            ("تطوير الويب من الصفر", "https://www.youtube.com/embed/_uQrJ0TkZlc", "رحلة كاملة في تطوير الويب", "🌐 شامل"),
            ("الفرق بين Frontend و Backend", "https://www.youtube.com/embed/RBSGKlAvoiM", "فهم جوانب تطوير الويب", "⚖️ مقارنة"),
            ("مسار تعلم تطوير الويب", "https://www.youtube.com/embed/09_LlHjoEiY", "خطة لتعلم تطوير الويب", "🗺️ خطة")
        ],
        "Frontend Development": [
            ("HTML5 و CSS3 كامل", "https://www.youtube.com/embed/_uQrJ0TkZlc", "دورة شاملة في تطوير الواجهات", "🎨 واجهات"),
            ("JavaScript الحديث ES6+", "https://www.youtube.com/embed/RBSGKlAvoiM", "أحدث ميزات JavaScript", "⚡ حديث"),
            ("React.js للمبتدئين", "https://www.youtube.com/embed/09_LlHjoEiY", "تعلم React من البداية", "⚛️ إطار عمل")
        ],
        "Backend Development": [
            ("Node.js و Express", "https://www.youtube.com/embed/_uQrJ0TkZlc", "تطوير الخلفية باستخدام Node.js", "🖥️ خلفية"),
            ("الفرق بين Django و Flask", "https://www.youtube.com/embed/RBSGKlAvoiM", "مقارنة أطر عمل بايثون", "🐍 مقارنة"),
            ("RESTful APIs كامل", "https://www.youtube.com/embed/09_LlHjoEiY", "بناء واجهات برمجة التطبيقات", "🔌 API")
        ]
    },
    "Machine Learning": {
        "المفاهيم الأساسية": [
            ("ما هو تعلم الآلة؟", "https://www.youtube.com/embed/qiQR5rTSshw", "مقدمة شاملة لتعلم الآلة", "🧠 تمهيدي"),
            ("الفرق بين ML و AI و DL", "https://www.youtube.com/embed/0y6FtKsg6J4", "مقارنة مصطلحات الذكاء الاصطناعي", "⚖️ مقارنة"),
            ("تطبيقات تعلم الآلة", "https://www.youtube.com/embed/3bSIHivxGR4", "استخدامات تعلم الآلة في الحياة", "💼 تطبيقي")
        ],
        "الخوارزميات": [
            ("خوارزميات التعلم الآلي", "https://www.youtube.com/embed/PpsEaqJV_A0", "شرح الخوارزميات الأساسية", "📊 أساسيات"),
            ("Linear Regression كامل", "https://www.youtube.com/embed/0y6FtKsg6J4", "شرح الانحدار الخطي بالتفصيل", "📈 انحدار"),
            ("خوارزميات التصنيف", "https://www.youtube.com/embed/3bSIHivxGR4", "خوارزميات مثل SVM و Decision Trees", "🌲 تصنيف")
        ],
        "التطبيق العملي": [
            ("مشروع ML كامل من الصفر", "https://www.youtube.com/embed/qiQR5rTSshw", "بناء مشروع تعلم آلي كامل", "🚀 مشروع"),
            ("استخدام TensorFlow و PyTorch", "https://www.youtube.com/embed/0y6FtKsg6J4", "مقارنة بين المكتبات الشهيرة", "⚖️ مقارنة"),
            ("تنقية البيانات للتعلم الآلي", "https://www.youtube.com/embed/3bSIHivxGR4", "تحضير البيانات للنماذج", "🧹 بيانات")
        ]
    }
}

# ----------------------------
# Content-Based Recommender Catalog
# ----------------------------
RECOMMENDER_CATALOG = flatten_recommendation_catalog(
    PERSONALIZED_VIDEO_RECOMMENDATIONS,
    STUDY_MATERIALS,
)

# ----------------------------
# Professional Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 25px 0;">
        <div style="position: relative; display: inline-block;">
            <div style="width: 100px; height: 100px; border-radius: 50%; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        display: flex; align-items: center; justify-content: center;
                        margin: 0 auto 15px auto;">
                <span style="font-size: 2.5rem; color: white;">👩‍🎓</span>
            </div>
        </div>
        <h2 style="margin: 10px 0 5px 0; color: #2d3748;">إيمان ناجي</h2>
        <p style="color: #718096; margin: 0 0 15px 0; font-size: 1rem;">طالبة هندسة حاسوب</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 نظرة عامة")

    data_side = load_history_data()

    col1, col2 = st.columns(2)
    with col1:
        total_sessions = len(data_side) if not data_side.empty else 0
        st.markdown(f"""
        <div class="stats-card">
            <div style="color: #718096; font-size: 0.9rem;">الجلسات</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #2d3748;">{total_sessions}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if not data_side.empty and 'focus_score' in data_side.columns:
            avg_focus = data_side["focus_score"].mean() * 100
            st.markdown(f"""
        <div class="stats-card">
            <div style="color: #718096; font-size: 0.9rem;">متوسط التركيز</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #2d3748;">{avg_focus:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
        else:
            st.markdown("""
        <div class="stats-card">
            <div style="color: #718096; font-size: 0.9rem;">متوسط التركيز</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #2d3748;">0%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧭 التنقل السريع")

    nav_options = [
        ("🎯", "التحليل الذكي", "تحليل حالة التركيز"),
        ("📚", "المكتبة التعليمية", "موارد مخصصة للدراسة"),
        ("📊", "لوحة التحكم", "إحصائيات وتقارير"),
        ("⚙️", "الإعدادات", "تخصيص الحساب")
    ]

    for icon, label, _desc in nav_options:
        if st.button(f"{icon} {label}", use_container_width=True, key=f"nav_{label}"):
            st.session_state.active_tab = icon
            st.rerun()

    st.markdown("---")
    with st.expander("💡 نصيحة اليوم", expanded=True):
        tips = [
            "التركيز لمدة 25 دقيقة ثم استراحة 5 دقائق يحسن الإنتاجية بنسبة 40%",
            "الموسيقى الكلاسيكية تزيد التركيز والإبداع",
            "تخطيط اليوم قبل البدء يقلل التوتر ويحسن الأداء",
            "التمارين الرياضية الخفيفة تزيد تدفق الدم للدماغ",
            "النوم الكافي أهم عامل لتحسين الذاكرة والتركيز",
            "شرب الماء بانتظام يحسن الوظائف المعرفية"
        ]
        st.info(random.choice(tips))

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    try:
        tokenizer, model = load_sentiment_model(".")
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ فشل تحميل النموذج: {str(e)}")
        return None, None


tokenizer, model = load_model()

if model:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0fff4 0%, #e6fffa 100%); 
                padding: 15px 25px; 
                border-radius: 12px; 
                border: 1px solid #c6f6d5;
                margin: 20px 0; display: flex; align-items: center;">
        <span style="font-size: 1.5rem; margin-right: 15px;">✅</span>
        <div>
            <strong style="color: #22543d;">النموذج الذكي جاهز للعمل</strong>
            <p style="color: #38a169; margin: 5px 0 0 0; font-size: 0.95rem;">
                نظام التحليل الآلي يعمل بكفاءة لتحليل حالتك وتقديم توصيات مخصصة
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.stop()

# ----------------------------
# Professional Header
# ----------------------------
col_header1, col_header2, col_header3 = st.columns([1, 2, 1])

with col_header2:
    st.markdown("""
    <div style="text-align: center; padding: 30px 0;">
        <div class="floating-icon" style="font-size: 4rem; margin-bottom: 15px;">🧠</div>
        <h1 style="font-size: 3.2rem; font-weight: 800; margin-bottom: 10px;" class="gradient-text">
            FocusMate Pro
        </h1>
        <p style="font-size: 1.2rem; color: #4a5568; max-width: 600px; margin: 0 auto;">
            نظام ذكي متكامل لتحسين التركيز والإنتاجية الأكاديمية
        </p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Motivational Section
# ----------------------------
col_mot1, col_mot2, col_mot3 = st.columns([2, 1, 1])

with col_mot1:
    quotes = [
        ("🚀 النجاح يبدأ عندما تتوقف عن الأعذار وتبدأ بالأفعال", "إيمان ناجي"),
        ("💡 العلم نور.. والعقل السليم في الجسم السليم", "مثل عربي"),
        ("🎯 ليس المهم سرعتك في التعلم، بل استمراريتك في التعلم", "إيمان ناجي"),
        ("📚 الاستثمار في التعليم يدفع أفضل الفوائد", "بنجامين فرانكلين"),
        ("🧠 العقل كالبارود يشتعل بالشرر وينطفئ بالغبار", "ابن خلدون")
    ]
    selected_quote, author = random.choice(quotes)

    st.markdown(f"""
    <div class="elegant-card">
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        width: 50px; height: 50px; border-radius: 12px; 
                        display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                <span style="color: white; font-size: 1.5rem;">💭</span>
            </div>
            <div>
                <h3 style="margin: 0; color: #2d3748;">اقتباس اليوم</h3>
                <p style="margin: 5px 0 0 0; color: #718096;">لإلهامك اليومي</p>
            </div>
        </div>
        <p style="font-size: 1.2rem; font-style: italic; color: #4a5568; margin: 15px 0; line-height: 1.6;">
            "{selected_quote}"
        </p>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">
            <div style="color: #667eea; font-weight: 600;">— {author}</div>
            <div style="color: #718096; font-size: 0.9rem;">{datetime.datetime.now().strftime("%d %B %Y")}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_mot2:
    st.markdown(f"""
    <div class="elegant-card" style="text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 15px;">⏱️</div>
        <h4 style="margin: 0 0 10px 0; color: #2d3748;">الوقت الحالي</h4>
        <div style="font-size: 1.8rem; color: #667eea; font-weight: 700; margin-bottom: 10px;">
            {datetime.datetime.now().strftime("%I:%M %p")}
        </div>
        <p style="color: #718096; font-size: 0.9rem; margin: 0;">{datetime.datetime.now().strftime("%A, %d %B")}</p>
    </div>
    """, unsafe_allow_html=True)

with col_mot3:
    if not data_side.empty:
        today = datetime.datetime.now().date()
        today_data = data_side[data_side["time"].dt.date == today]
        today_sessions = len(today_data)
    else:
        today_sessions = 0

    st.markdown(f"""
    <div class="elegant-card" style="text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 15px;">📊</div>
        <h4 style="margin: 0 0 10px 0; color: #2d3748;">جلسات اليوم</h4>
        <div style="font-size: 1.8rem; color: #48bb78; font-weight: 700; margin-bottom: 10px;">
            {today_sessions}
        </div>
        <div class="custom-progress">
            <div class="custom-progress-fill" style="width: {min(today_sessions * 20, 100)}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Main Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["🎯 التحليل الذكي", "📚 المكتبة المخصصة", "📊 لوحة التحكم"])

# ==========================================================
# TAB 1 – التحليل الذكي
# ==========================================================
with tab1:
    st.markdown("""
    <div class="elegant-card" style="margin-bottom: 30px;">
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        width: 60px; height: 60px; border-radius: 15px; 
                        display: flex; align-items: center; justify-content: center; margin-right: 20px;">
                <span style="color: white; font-size: 2rem;">🎯</span>
            </div>
            <div>
                <h2 style="margin: 0; color: #2d3748;">محلل التركيز الذكي</h2>
                <p style="margin: 5px 0 0 0; color: #718096;">حلل حالتك واحصل على توصيات مخصصة بناءً على حالتك المزاجية والمادة الدراسية</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_middle, col_right = st.columns([2, 1, 1])

    with col_left:
        st.markdown("### 1️⃣ اختر حالتك الحالية")
        mood_options = {
            "😊 ممتاز - طاقة عالية وتركيز قوي": {"emoji": "😊", "color": "#38a169", "score": 0.9},
            "🙂 جيد - منتج لكن يمكنني الأفضل": {"emoji": "🙂", "color": "#3182ce", "score": 0.7},
            "😐 متوسط - أحتاج بعض التحفيز": {"emoji": "😐", "color": "#d69e2e", "score": 0.5},
            "😟 مشتت - صعوبة في التركيز": {"emoji": "😟", "color": "#ed8936", "score": 0.3},
            "😫 متعب - بحاجة للراحة": {"emoji": "😫", "color": "#e53e3e", "score": 0.2}
        }
        selected_mood = st.radio(
            "ما هي حالتك الآن؟",
            options=list(mood_options.keys()),
            key="mood_selector"
        )
        mood_info = mood_options[selected_mood]
        st.markdown(f"""
        <div style="background: {mood_info['color']}10; 
                    border: 2px solid {mood_info['color']}20;
                    border-radius: 16px;
                    padding: 20px;
                    margin: 15px 0;">
            <div style="display: flex; align-items: center;">
                <div style="background: {mood_info['color']}; 
                            width: 60px; height: 60px; 
                            border-radius: 15px; 
                            display: flex; align-items: center; justify-content: center; 
                            margin-right: 20px;">
                    <span style="font-size: 2rem; color: white;">{mood_info['emoji']}</span>
                </div>
                <div>
                    <h4 style="margin: 0; color: #2d3748;">{selected_mood.split(' - ')[0]}</h4>
                    <p style="margin: 5px 0 0 0; color: #4a5568; font-size: 0.95rem;">
                        {selected_mood.split(' - ')[1]}
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 2️⃣ اكتب عن حالتك")

        with st.expander("💡 أمثلة للكتابة", expanded=False):
            examples = [
                "أشعر أنني أتمكن من المادة جيداً وأريد استغلال هذه الطاقة للتعمق أكثر",
                "أعاني من التشتت بسبب الضوضاء في المكان وأحتاج لتحسين بيئة الدراسة",
                "متحمس للتعلم ولكن أجد صعوبة في البداية وأحتاج لبعض التحفيز",
                "أشعر بالإرهاق بعد يوم طويل وأحتاج لاستراتيجيات لاستعادة التركيز"
            ]
            for i, ex in enumerate(examples):
                if st.button(f"✏️ {ex[:50]}...", key=f"ex_{i}", use_container_width=True):
                    st.session_state.user_text = ex

        user_input = st.text_area(
            "صف مشاعرك والتحديات التي تواجهها:",
            value=st.session_state.get("user_text", ""),
            height=120,
            placeholder="مثال: اليوم أشعر أنني أريد دراسة الشبكات ولكن أشعر ببعض التشتت بسبب المشتتات حولي..."
        )

        if user_input:
            st.caption(f"📝 {len(user_input)} حرف مكتوب")

    with col_middle:
        st.markdown("### 3️⃣ اختر المادة الدراسية")
        selected_subject = st.selectbox(
            "ما هي المادة التي تريد دراستها؟",
            options=list(STUDY_MATERIALS.keys()),
            key="subject_selection"
        )
        if selected_subject:
            subject_info = STUDY_MATERIALS[selected_subject]
            st.markdown(f"""
            <div class="elegant-card" style="text-align: center; margin-top: 10px;">
                <div style="font-size: 3rem; margin-bottom: 15px; color: {subject_info['color']}">
                    {subject_info['icon']}
                </div>
                <h3 style="margin: 0 0 10px 0; color: #2d3748;">
                    {selected_subject}
                </h3>
                <p style="color: #4a5568; font-size: 0.9rem; margin-bottom: 15px;">
                    {subject_info['description']}
                </p>
                <div style="background: {subject_info['color']}15; 
                            color: {subject_info['color']}; 
                            padding: 6px 12px; 
                            border-radius: 20px; 
                            display: inline-block; 
                            font-size: 0.85rem; 
                            font-weight: 600;">
                    مستوى: {subject_info['difficulty']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### ⏱️ معلومات الجلسة")
        current_time = datetime.datetime.now()
        elapsed = current_time - st.session_state.start_time
        elapsed_minutes = elapsed.seconds // 60

        st.markdown(f"""
        <div class="elegant-card" style="text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 15px; color: #667eea;">⏱️</div>
            <h3 style="margin: 0 0 10px 0; color: #2d3748;">وقت الجلسة</h3>
            <div style="font-size: 2rem; color: #2d3748; font-weight: 700; margin-bottom: 15px;">
                {elapsed_minutes:02d}:{elapsed.seconds % 60:02d}
            </div>
            <div class="custom-progress">
                <div class="custom-progress-fill" style="width: {min(elapsed_minutes, 100)}%;"></div>
            </div>
            <p style="color: #718096; font-size: 0.9rem; margin-top: 15px;">
                بدأت الساعة {st.session_state.start_time.strftime('%I:%M %p')}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚡ إجراءات سريعة")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("💾 حفظ", use_container_width=True, help="حفظ المسودة الحالية"):
                if user_input:
                    st.session_state.draft = user_input
                    st.success("تم الحفظ!")
        with col_btn2:
            if st.button("🔄 جديد", use_container_width=True, help="بدء جلسة جديدة"):
                start_new_session()
                st.rerun()

        if user_input.strip():
            try:
                label, confidence = predict_text(user_input, tokenizer, model)
                score = confidence if label == 1 else (1 - confidence)
                if score >= 0.8:
                    status, status_class = "ممتاز", "status-excellent-pro"
                elif score >= 0.6:
                    status, status_class = "جيد", "status-good-pro"
                else:
                    status, status_class = "تحسين", "status-need-improve-pro"

                st.markdown(f"""
                <div class="elegant-card" style="margin-top: 20px; text-align: center;">
                    <h4 style="margin: 0 0 15px 0; color: #4a5568;">تحليل أولي</h4>
                    <span class="status-badge-pro {status_class}" style="margin-bottom: 15px;">
                        {status}
                    </span>
                    <div style="font-size: 1.8rem; color: #2d3748; font-weight: 700; margin-bottom: 10px;">
                        {score*100:.0f}%
                    </div>
                    <p style="color: #718096; font-size: 0.9rem; margin: 0;">
                        ثقة التحليل: {confidence:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                pass

    st.markdown("---")
    col_btn_left, col_btn_center, col_btn_right = st.columns([1, 2, 1])

    with col_btn_center:
        analyze_btn = st.button(
            "🚀 بدء التحليل الذكي المتقدم",
            type="primary",
            use_container_width=True,
            disabled=not (user_input.strip() and selected_subject),
            key="start_focus_analysis_button",
        )

    # -------
    # Run analysis (once) and persist results in session_state
    # -------
    if analyze_btn and user_input.strip() and selected_subject:
        with st.spinner("🔍 جارِ تحليل حالتك وتقديم توصيات مخصصة..."):
            time.sleep(1.2)
            try:
                label, confidence = predict_text(user_input, tokenizer, model)
                score = confidence if label == 1 else (1 - confidence)
                mood_score = mood_options[selected_mood]["score"]
                combined_score = (score + mood_score) / 2

                if combined_score >= 0.8:
                    user_state = "excellent"
                    status_text = "ممتاز! 🤩"
                    status_class = "status-excellent-pro"
                elif combined_score >= 0.6:
                    user_state = "good"
                    status_text = "جيد! 😊"
                    status_class = "status-good-pro"
                else:
                    user_state = "need_improve"
                    status_text = "تحسين! 🧘‍♀️"
                    status_class = "status-need-improve-pro"

                st.session_state.analysis_result = {
                    "label": label,
                    "confidence": confidence,
                    "score": score,
                    "combined_score": combined_score,
                    "user_state": user_state,
                    "status_text": status_text,
                    "status_class": status_class,
                    "selected_subject": selected_subject,
                    "selected_mood": selected_mood,
                    "user_input": user_input,
                }
                st.session_state.history_saved = False
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء التحليل: {str(e)}")

    # -------
    # Display results (persist across reruns)
    # -------
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        combined_score = result["combined_score"]
        status_text = result["status_text"]
        status_class = result["status_class"]
        user_state = result["user_state"]
        selected_subject_r = result["selected_subject"]
        selected_mood_r = result["selected_mood"]
        confidence_r = result["confidence"]

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; margin: 40px 0 30px 0;">
            <h2 style="color: #2d3748; margin-bottom: 10px;">📊 نتائج التحليل الذكي</h2>
            <p style="color: #718096;">توصيات مخصصة بناءً على حالتك ومادتك الدراسية</p>
        </div>
        """, unsafe_allow_html=True)

        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            st.markdown(f"""
            <div class="elegant-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 15px; color: #667eea;">🎯</div>
                <h4 style="margin: 0 0 10px 0; color: #4a5568;">التركيز الكلي</h4>
                <div style="font-size: 2.5rem; color: #2d3748; font-weight: 800; margin-bottom: 15px;">
                    {combined_score*100:.0f}%
                </div>
                <div class="custom-progress">
                    <div class="custom-progress-fill" style="width: {combined_score*100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_res2:
            st.markdown(f"""
            <div class="elegant-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 15px; color: #ed8936;">📈</div>
                <h4 style="margin: 0 0 10px 0; color: #4a5568;">حالتك العامة</h4>
                <div class="status-badge-pro {status_class}" style="font-size: 1.2rem; margin-bottom: 15px;">
                    {status_text}
                </div>
                <p style="color: #718096; font-size: 0.9rem; margin: 0;">
                    بناءً على تحليل مشاعرك
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_res3:
            st.markdown(f"""
            <div class="elegant-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 15px; color: #48bb78;">📚</div>
                <h4 style="margin: 0 0 10px 0; color: #4a5568;">المادة الدراسية</h4>
                <div style="font-size: 1.5rem; color: #2d3748; font-weight: 700; margin-bottom: 15px;">
                    {selected_subject_r}
                </div>
                <p style="color: #718096; font-size: 0.9rem; margin: 0;">
                    {STUDY_MATERIALS[selected_subject_r]['description']}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Detailed Analysis
        st.markdown("---")
        st.markdown("## 🎯 تحليل مفصل لمستويات التركيز")

        col_det1, col_det2 = st.columns(2)

        with col_det1:
            st.markdown("#### 📊 مقاييس التركيز")
            aspects = {
                "التركيز العقلي": random.randint(60, 100) if combined_score >= 0.7 else random.randint(30, 65),
                "الاستقرار العاطفي": random.randint(65, 95) if combined_score >= 0.7 else random.randint(40, 70),
                "الدافع الذاتي": random.randint(70, 100) if combined_score >= 0.7 else random.randint(35, 75),
                "جودة الفهم": random.randint(75, 100) if combined_score >= 0.7 else random.randint(45, 80)
            }
            for aspect, value in aspects.items():
                st.markdown(f"""
                <div style="margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: #4a5568; font-weight: 600;">{aspect}</span>
                        <span style="color: #2d3748; font-weight: 700;">{value}%</span>
                    </div>
                    <div class="custom-progress">
                        <div class="custom-progress-fill" style="width: {value}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_det2:
            st.markdown("#### 💡 توصيات فورية")
            if user_state == "excellent":
                tips = [
                    "✅ استغل طاقتك العالية للتعمق في المواضيع الصعبة",
                    "✅ ابدأ بالمهام التي تتطلب تفكيراً إبداعياً",
                    "✅ خطط لمشاريع طويلة المدى تستفيد من تركيزك الحالي",
                    "✅ استخدم تقنيات التعلم المتقدم لتحقيق أقصى استفادة"
                ]
            elif user_state == "good":
                tips = [
                    "📝 وزع وقتك بين المهام المختلفة لتحقيق التوازن",
                    "⏱️ خذ فترات راحة قصيرة كل 45 دقيقة للحفاظ على التركيز",
                    "🔄 راجع ما تعلمته بانتظام لتعزيز الذاكرة طويلة المدى",
                    "🤝 فكر في الدراسة الجماعية لتحفيز التفاعل والتعلم"
                ]
            else:
                tips = [
                    "🧘‍♀️ ابدأ بجلسات قصيرة (25 دقيقة) ثم زد المدة تدريجياً",
                    "🎧 اختر بيئة هادئة خالية من الملهيات والضوضاء",
                    "💧 خذ فترات راحة لشرب الماء والقيام بتمارين التمدد",
                    "🍅 استخدم تقنية بومودورو لتنظيم وقت الدراسة والراحة"
                ]
            for tip in tips:
                st.markdown(f"""
                <div style="background: #f8fafc; border-radius: 12px; padding: 15px; margin-bottom: 10px;">
                    <div style="display: flex; align-items: flex-start;">
                        <span style="margin-right: 10px; font-size: 1.2rem;">{tip.split(' ')[0]}</span>
                        <span style="color: #4a5568;">{' '.join(tip.split(' ')[1:])}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # -------
        # CONTENT-BASED RECOMMENDATIONS
        # -------
        st.markdown("---")
        st.markdown("## 🎥 فيديوهات مخصصة لك")
        st.caption(
            "يستخدم النظام TF-IDF وCosine Similarity، مع مراعاة المادة الدراسية "
            "وحالة التركيز والتقييمات السابقة."
        )

        recommended_items = recommend_resources(
            resources=RECOMMENDER_CATALOG,
            subject=selected_subject_r,
            user_state=user_state,
            mood=selected_mood_r,
            feedback_path="feedback.csv",
            top_n=6,
        )

        if not recommended_items:
            st.info("لا توجد موارد مناسبة حاليًا.")
        else:
            for item in recommended_items:
                with st.container(border=True):
                    st.markdown(f"#### {item['title']}")
                    st.write(item.get("description", ""))
                    st.caption(
                        f"درجة التشابه: {item['similarity_score']:.3f} | "
                        f"درجة الترتيب النهائية: {item['final_score']:.3f}"
                    )
                    st.info(f"💡 سبب الترشيح: {item['why_recommended']}")

                    if item.get("url"):
                        st.video(item["url"])

                    rating = st.select_slider(
                        "قيّمي التوصية من 1 إلى 5",
                        options=[1, 2, 3, 4, 5],
                        value=3,
                        key=f"rating_{item['resource_id']}",
                        help="1 = غير مناسبة، 5 = مناسبة جدًا",
                    )
                    if st.button("💾 حفظ التقييم", key=f"save_{item['resource_id']}"):
                        saved = save_feedback(
                            resource_id=item["resource_id"],
                            rating=rating,
                            subject=selected_subject_r,
                            user_state=user_state,
                            feedback_path="feedback.csv",
                        )
                        if saved:
                            st.success("تم حفظ تقييمك. سيؤثر في ترتيب التوصيات القادمة.")
                        else:
                            st.error(
                                "تعذر حفظ التقييم في feedback.csv. تحققي من صلاحيات الكتابة."
                            )

        # Additional Resources
        st.markdown("---")
        st.markdown("## 📚 موارد إضافية")
        tab_res1, tab_res2, tab_res3 = st.tabs(["📄 مقالات", "💪 تمارين", "🛠️ أدوات"])

        with tab_res1:
            articles = [
                ("كيفية تحسين التركيز أثناء الدراسة", "https://www.mawdoo3.com/", "مقال شامل عن تقنيات التركيز"),
                ("تقنيات إدارة الوقت للطلاب", "https://academy.hsoub.com/", "دليل عملي للإدارة الفعالة للوقت"),
                ("الغذاء والتركيز: ما العلاقة؟", "https://www.webteb.com/", "كيف يؤثر الغذاء على قدراتك العقلية")
            ]
            for title, url, desc in articles:
                st.markdown(f"""
                <div style="padding: 15px; background: #f8fafc; border-radius: 10px; margin-bottom: 10px;">
                    <h5 style="margin: 0 0 5px 0; color: #2d3748;">{title}</h5>
                    <p style="color: #718096; margin: 0 0 10px 0; font-size: 0.9rem;">{desc}</p>
                    <a href="{url}" target="_blank" style="color: #667eea; text-decoration: none; font-weight: 600;">
                        اقرأ المقال →
                    </a>
                </div>
                """, unsafe_allow_html=True)

        with tab_res2:
            exercises = [
                ("تنفس 4-7-8", "خذ شهيقاً لمدة 4 ثوان، احبس النفس 7 ثوان، زفير 8 ثوان"),
                ("تمارين شد العضلات", "شد وارتخاء مجموعات العضلات المختلفة لمدة 3 دقائق"),
                ("تمرين التركيز البصري", "ركز على نقطة ثابتة لمدة دقيقة دون تحريك العينين")
            ]
            for title, desc in exercises:
                st.markdown(f"""
                <div style="padding: 15px; background: #f0fff4; border-radius: 10px; margin-bottom: 10px;">
                    <h5 style="margin: 0 0 5px 0; color: #22543d;">{title}</h5>
                    <p style="color: #38a169; margin: 0; font-size: 0.9rem;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

        with tab_res3:
            tools = [
                ("Pomofocus", "https://pomofocus.io/", "تطبيق مجاني لتقنية بومودورو"),
                ("Forest App", "https://www.forestapp.cc/", "تطبيق للتركيز عن طريق زراعة الأشجار"),
                ("Noisli", "https://www.noisli.com/", "أصوات بيضاء لتحسين التركيز")
            ]
            for title, url, desc in tools:
                st.markdown(f"""
                <div style="padding: 15px; background: #ebf8ff; border-radius: 10px; margin-bottom: 10px;">
                    <h5 style="margin: 0 0 5px 0; color: #2c5282;">{title}</h5>
                    <p style="color: #4299e1; margin: 0 0 10px 0; font-size: 0.9rem;">{desc}</p>
                    <a href="{url}" target="_blank" style="color: #2c5282; text-decoration: none; font-weight: 600;">
                        تجربة الأداة →
                    </a>
                </div>
                """, unsafe_allow_html=True)

        # Save to history (once)
        st.markdown("---")
        if not st.session_state.history_saved:
            entry = {
                "time": datetime.datetime.now(),
                "text": result["user_input"],
                "status": status_text,
                "confidence": confidence_r,
                "mood": selected_mood_r,
                "focus_score": combined_score,
                "subject": selected_subject_r
            }
            if save_history_entry(entry):
                st.session_state.history_saved = True
                st.success("✅ تم حفظ تحليلك في السجل الشخصي!")
            else:
                st.warning("⚠️ تعذر حفظ الجلسة في history.csv.")

        # Next Steps
        st.markdown("### 🚀 الخطوات التالية")
        col_next1, col_next2, col_next3 = st.columns(3)

        with col_next1:
            if st.button("🔄 جلسة جديدة", use_container_width=True):
                start_new_session()
                st.rerun()

        with col_next2:
            report = {
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "focus_score": f"{combined_score*100:.0f}%",
                "status": status_text,
                "subject": selected_subject_r,
                "mood": selected_mood_r,
                "recommendations": tips
            }
            st.download_button(
                label="📥 تصدير التقرير",
                data=json.dumps(report, ensure_ascii=False, indent=2),
                file_name=f"focus_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )

        with col_next3:
            if st.button("📊 عرض التقدم", use_container_width=True):
                st.session_state.active_tab = "📊"
                st.rerun()

# ==========================================================
# TAB 2 – المكتبة التعليمية
# ==========================================================
with tab2:
    st.markdown("""
    <div class="elegant-card" style="margin-bottom: 30px;">
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <div style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); 
                        width: 60px; height: 60px; border-radius: 15px; 
                        display: flex; align-items: center; justify-content: center; margin-right: 20px;">
                <span style="color: white; font-size: 2rem;">📚</span>
            </div>
            <div>
                <h2 style="margin: 0; color: #2d3748;">المكتبة التعليمية المخصصة</h2>
                <p style="margin: 5px 0 0 0; color: #718096;">موارد تعليمية ذكية تتكيف مع حالتك وأهدافك الدراسية</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 كيف تعمل المكتبة التعليمية؟")
    steps = [
        ("1️⃣ اختر حالتك", "اختر حالتك المزاجية الحالية من الخيارات المتاحة"),
        ("2️⃣ اختر المادة", "حدد المادة الدراسية التي تريد التركيز عليها"),
        ("3️⃣ اكتب مشاعرك", "صف مشاعرك والتحديات التي تواجهها"),
        ("4️⃣ احصل على توصيات", "ستحصل على فيديوهات وموارد مخصصة لك")
    ]
    cols_steps = st.columns(4)
    for idx, (icon, text) in enumerate(steps):
        with cols_steps[idx]:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #f8fafc; border-radius: 12px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">{icon}</div>
                <p style="color: #4a5568; margin: 0; font-size: 0.9rem;">{text}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📚 اختر المادة الدراسية")
    st.markdown("اضغط على أي مادة لعرض فيديوهات يوتيوب تعليمية شاملة:")

    materials_list = list(STUDY_MATERIALS.items())

    cols1 = st.columns(3)
    for idx in range(3):
        with cols1[idx]:
            subject, info = materials_list[idx]
            is_selected = (subject == st.session_state.selected_library_subject)
            st.markdown(f"""
            <div class="material-card subject-card" 
                 style="text-align: center; 
                        border: 2px solid {info['color'] if is_selected else '#e2e8f0'}; 
                        background: {'#f0f9ff' if is_selected else 'white'};
                        transform: {'scale(1.02)' if is_selected else 'scale(1)'};">
                <div style="font-size: 3rem; margin-bottom: 15px; color: {info['color']}">
                    {info['icon']}
                </div>
                <h3 style="margin: 0 0 10px 0; color: #2d3748;">{subject}</h3>
                <p style="color: #4a5568; font-size: 0.9rem; margin-bottom: 15px;">{info['description']}</p>
                <div style="background: {info['color']}20; color: {info['color']}; 
                            padding: 6px 12px; border-radius: 20px; display: inline-block; 
                            font-size: 0.85rem; font-weight: 600;">
                    {info['difficulty']}
                </div>
                <div style="margin-top: 10px; color: #718096; font-size: 0.8rem;">
                    {len(SUBJECT_YOUTUBE_VIDEOS.get(subject, {}))} قسم • 
                    {sum(len(v) for v in SUBJECT_YOUTUBE_VIDEOS.get(subject, {}).values())} فيديو
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"👈 اختر {subject}", key=f"select_{subject}",
                        type="primary" if is_selected else "secondary",
                        use_container_width=True):
                st.session_state.selected_library_subject = subject
                st.rerun()

    cols2 = st.columns(3)
    for idx in range(3):
        with cols2[idx]:
            subject, info = materials_list[idx + 3]
            is_selected = (subject == st.session_state.selected_library_subject)
            st.markdown(f"""
            <div class="material-card subject-card" 
                 style="text-align: center; 
                        border: 2px solid {info['color'] if is_selected else '#e2e8f0'}; 
                        background: {'#f0f9ff' if is_selected else 'white'};
                        transform: {'scale(1.02)' if is_selected else 'scale(1)'};">
                <div style="font-size: 3rem; margin-bottom: 15px; color: {info['color']}">
                    {info['icon']}
                </div>
                <h3 style="margin: 0 0 10px 0; color: #2d3748;">{subject}</h3>
                <p style="color: #4a5568; font-size: 0.9rem; margin-bottom: 15px;">{info['description']}</p>
                <div style="background: {info['color']}20; color: {info['color']}; 
                            padding: 6px 12px; border-radius: 20px; display: inline-block; 
                            font-size: 0.85rem; font-weight: 600;">
                    {info['difficulty']}
                </div>
                <div style="margin-top: 10px; color: #718096; font-size: 0.8rem;">
                    {len(SUBJECT_YOUTUBE_VIDEOS.get(subject, {}))} قسم • 
                    {sum(len(v) for v in SUBJECT_YOUTUBE_VIDEOS.get(subject, {}).values())} فيديو
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"👈 اختر {subject}", key=f"select2_{subject}",
                        type="primary" if is_selected else "secondary",
                        use_container_width=True):
                st.session_state.selected_library_subject = subject
                st.rerun()

    st.markdown("---")

    selected_subject_lib = st.session_state.selected_library_subject
    subject_info_lib = STUDY_MATERIALS[selected_subject_lib]
    youtube_videos = SUBJECT_YOUTUBE_VIDEOS.get(selected_subject_lib, {})

    st.markdown(f"""
    <div style="text-align: center; margin: 30px 0; padding: 25px; 
                background: linear-gradient(135deg, {subject_info_lib['color']}15 0%, {subject_info_lib['color']}05 100%);
                border-radius: 20px; border: 2px solid {subject_info_lib['color']}30;">
        <div style="display: inline-flex; align-items: center; justify-content: center; 
                    background: {subject_info_lib['color']}; width: 80px; height: 80px; 
                    border-radius: 20px; margin-bottom: 20px;">
            <span style="font-size: 3rem; color: white;">{subject_info_lib['icon']}</span>
        </div>
        <h1 style="color: #2d3748; margin-bottom: 10px;">{selected_subject_lib}</h1>
        <p style="color: #4a5568; font-size: 1.2rem; max-width: 800px; margin: 0 auto;">
            {subject_info_lib['description']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    if youtube_videos:
        st.markdown("### 🎬 أقسام الفيديوهات التعليمية")
        tab_names = list(youtube_videos.keys())
        tabs = st.tabs([f"📖 {name}" for name in tab_names])

        for idx, (section_name, section_videos) in enumerate(youtube_videos.items()):
            with tabs[idx]:
                st.markdown(f"""
                <div style="padding: 20px; background: #f8fafc; border-radius: 15px; margin-bottom: 25px;">
                    <h3 style="color: #2d3748; margin-bottom: 10px;">{section_name}</h3>
                    <p style="color: #718096; margin-bottom: 15px;">
                        فيديوهات تعليمية متخصصة في {section_name.lower()} لـ {selected_subject_lib}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                for video_idx, (title, embed_url, description, level) in enumerate(section_videos):
                    col_vid1, col_vid2 = st.columns([2, 3])
                    with col_vid1:
                        st.markdown(f"""
                        <div style="margin-bottom: 20px; border-radius: 12px; overflow: hidden; 
                                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                            <iframe width="100%" height="220" src="{embed_url}" 
                            frameborder="0" allow="accelerometer; autoplay; clipboard-write; 
                            encrypted-media; gyroscope; picture-in-picture" 
                            allowfullscreen></iframe>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_vid2:
                        watch_url = embed_url.replace('embed/', 'watch?v=')
                        st.markdown(f"""
                        <div style="padding: 20px;">
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <div style="background: {subject_info_lib['color']}; color: white; 
                                            padding: 4px 12px; border-radius: 15px; 
                                            font-size: 0.85rem; font-weight: 600;">
                                    {level}
                                </div>
                                <div style="margin-right: 10px; color: #718096; font-size: 0.9rem;">
                                    الفيديو {video_idx + 1}
                                </div>
                            </div>
                            <h4 style="color: #2d3748; margin-bottom: 10px; font-size: 1.2rem;">{title}</h4>
                            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 15px;">{description}</p>
                            <a href="{watch_url}" target="_blank" 
                               style="background: {subject_info_lib['color']}; color: white; 
                                      text-decoration: none; padding: 8px 16px; border-radius: 8px; 
                                      font-weight: 600; display: inline-block;">
                                ▶️ شاهد على يوتيوب
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("---")
    else:
        st.warning(f"⚠️ جاري إضافة فيديوهات لـ {selected_subject_lib} في التحديث القادم")

    st.markdown("---")
    col_start1, col_start2, col_start3 = st.columns([1, 2, 1])
    with col_start2:
        if st.button(f"🚀 ابدأ رحلتك في تعلم {selected_subject_lib}",
                    type="primary", use_container_width=True):
            st.balloons()
            st.success(f"مبروك! بدأت رحلة تعلم {selected_subject_lib}. ننصح بمشاهدة الفيديوهات بالترتيب الموصى به.")

# ==========================================================
# TAB 3 – لوحة التحكم
# ==========================================================
with tab3:
    st.markdown("""
    <div class="elegant-card" style="margin-bottom: 30px;">
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <div style="background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%); 
                        width: 60px; height: 60px; border-radius: 15px; 
                        display: flex; align-items: center; justify-content: center; margin-right: 20px;">
                <span style="color: white; font-size: 2rem;">📊</span>
            </div>
            <div>
                <h2 style="margin: 0; color: #2d3748;">لوحة التحكم والإحصائيات</h2>
                <p style="margin: 5px 0 0 0; color: #718096;">تابع تقدمك وإحصائيات أدائك الدراسي</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    data = load_history_data()

    if not data.empty:
        streak = calculate_streak(data)

        st.markdown("### 📈 نظرة عامة على أدائك")
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

        with col_kpi1:
            st.markdown(f"""
            <div class="stats-card" style="text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 10px;">🔄</div>
                <div style="font-size: 0.9rem; color: #718096; margin-bottom: 5px;">إجمالي الجلسات</div>
                <div style="font-size: 2rem; color: #2d3748; font-weight: 700;">{len(data)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_kpi2:
            avg_focus = data["focus_score"].mean() * 100 if 'focus_score' in data.columns else 0
            st.markdown(f"""
            <div class="stats-card" style="text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 10px;">🎯</div>
                <div style="font-size: 0.9rem; color: #718096; margin-bottom: 5px;">متوسط التركيز</div>
                <div style="font-size: 2rem; color: #2d3748; font-weight: 700;">{avg_focus:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col_kpi3:
            best_focus = data["focus_score"].max() * 100 if 'focus_score' in data.columns else 0
            st.markdown(f"""
            <div class="stats-card" style="text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 10px;">⭐</div>
                <div style="font-size: 0.9rem; color: #718096; margin-bottom: 5px;">أفضل تركيز</div>
                <div style="font-size: 2rem; color: #2d3748; font-weight: 700;">{best_focus:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col_kpi4:
            st.markdown(f"""
            <div class="stats-card" style="text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 10px;">🔥</div>
                <div style="font-size: 0.9rem; color: #718096; margin-bottom: 5px;">متتالية النجاح</div>
                <div style="font-size: 2rem; color: #2d3748; font-weight: 700;">{streak} يوم</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 تحليلات مرئية")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### 📈 تطور التركيز عبر الوقت")
            if len(data) > 1 and "focus_score" in data.columns:
                chart_data = data.copy().sort_values("time")
                fig = px.line(chart_data, x="time", y="focus_score",
                             labels={"time": "التاريخ", "focus_score": "درجة التركيز"},
                             line_shape="spline")
                fig.update_traces(line=dict(color="#667eea", width=3),
                                 mode='lines+markers',
                                 marker=dict(size=6, color='#764ba2'))
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                  hovermode="x unified", height=350, margin=dict(t=0, b=0))
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            st.markdown("#### 📚 توزيع المواد الدراسية")
            if "subject" in data.columns and not data["subject"].isna().all():
                subject_counts = data["subject"].value_counts().reset_index()
                subject_counts.columns = ["subject", "count"]
                fig = px.pie(subject_counts, values="count", names="subject",
                            color_discrete_sequence=px.colors.sequential.RdBu, hole=0.4)
                fig.update_layout(showlegend=True, height=350, margin=dict(t=0, b=0))
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### ⏳ النشاط الحديث")
        recent_data = data.sort_values("time", ascending=False).head(5)
        for idx, row in recent_data.iterrows():
            try:
                time_str = row["time"].strftime("%I:%M %p")
                date_str = row["time"].strftime("%Y-%m-%d")
                score_display = f"{row['focus_score']*100:.0f}%" if pd.notna(row.get('focus_score')) else "N/A"
                text_preview = str(row['text'])[:80] + "..." if len(str(row['text'])) > 80 else str(row['text'])
                st.markdown(f"""
                <div style="padding: 20px; background: #f8fafc; border-radius: 12px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                        <div>
                            <div style="color: #4a5568; font-weight: 600; margin-bottom: 5px;">{date_str} - {time_str}</div>
                            <p style="color: #718096; margin: 0; font-size: 0.9rem;">{text_preview}</p>
                        </div>
                        <div style="background: #667eea; color: white; padding: 5px 12px; border-radius: 20px; font-weight: 600;">
                            {score_display}
                        </div>
                    </div>
                    <div style="display: flex; gap: 10px; margin-top: 10px;">
                        <span style="background: #e2e8f0; color: #4a5568; padding: 4px 10px; border-radius: 15px; font-size: 0.85rem;">
                            {row['subject'] if pd.notna(row.get('subject')) else 'غير محدد'}
                        </span>
                        <span style="background: #c6f6d5; color: #22543d; padding: 4px 10px; border-radius: 15px; font-size: 0.85rem;">
                            {row['status'] if pd.notna(row.get('status')) else 'غير محدد'}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                pass

        st.markdown("---")
        col_export1, col_export2, col_export3 = st.columns(3)
        with col_export1:
            csv_data = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 تصدير البيانات",
                data=csv_data,
                file_name=f"focusmate_data_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_export2:
            if st.button("🔄 تحديث البيانات", use_container_width=True):
                st.rerun()
        with col_export3:
            if st.button("🎯 جلسة جديدة", use_container_width=True):
                st.session_state.active_tab = "🎯"
                st.rerun()
    else:
        st.info("""
        <div style="text-align: center; padding: 50px 20px;">
            <div style="font-size: 4rem; margin-bottom: 20px;">📭</div>
            <h3 style="color: #4a5568; margin-bottom: 15px;">لا توجد بيانات بعد!</h3>
            <p style="color: #718096; margin-bottom: 30px; max-width: 500px; margin-left: auto; margin-right: auto;">
            ابدأ أول جلسة تحليل لترى إحصائياتك وتقدمك هنا. النظام سيتتبع أداءك ويقدم لك تقارير مفصلة.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 30px 0; color: #718096;">
    <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 20px;">
        <div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #4a5568; margin-bottom: 5px;">FocusMate Pro</div>
            <div style="font-size: 0.9rem;">نظام ذكي للتركيز والإنتاجية</div>
        </div>
        <div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #4a5568; margin-bottom: 5px;">الطالبة</div>
            <div style="font-size: 0.9rem;">إيمان ناجي</div>
        </div>
        <div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #4a5568; margin-bottom: 5px;">الإصدار</div>
            <div style="font-size: 0.9rem;">v2.4 Professional</div>
        </div>
    </div>
    <div style="border-top: 1px solid #e2e8f0; padding-top: 20px; margin-top: 20px;">
        <p style="margin: 0; font-size: 0.85rem;">
            © 2024 FocusMate Pro - جميع الحقوق محفوظة | مشروع تخرج - قسم هندسة الحاسوب - جامعة المنصورة
        </p>
        <p style="margin: 5px 0 0 0; font-size: 0.85rem;">
            تواصل معنا: <a href="mailto:iman.naji@example.com" style="color: #667eea; text-decoration: none;">iman.naji@example.com</a>
        </p>
    </div>
</div>
""", unsafe_allow_html=True)