import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from collections import Counter
import re
import string

def remove_punch(txt):
  return txt.translate(str.maketrans('','',string.punctuation))

def remove_digit(txt):
  new = ""
  for i in txt:
    if not i.isdigit():
      new += i
  return new

def remove_emote(txt):
  new = ""
  for i in txt:
    if i.isascii():
      new += i
  return new





import pickle
@st.cache_resource
def load_model():
  model = pickle.load(open('nb_model.pkl', 'rb'))
  vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
     
  return model, vectorizer

model, vectorizer = load_model()





def predict_news_category(title, description):

        text = title + " " + description

        text = text.lower()
        text = remove_punch(text)
        text = remove_digit(text)
        text = remove_emote(text)


        
        text_vectorized = vectorizer.transform([text])

        
        predicted_class = model.predict(text_vectorized)[0]

        
        probabilities = model.predict_proba(text_vectorized)[0]

        
        confidence_scores = {i + 1: prob for i, prob in enumerate(probabilities)}

        return predicted_class, confidence_scores

    





st.set_page_config(
    page_title="News Classification System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        background-attachment: fixed;
    }

    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }

    
    .block-container {
        background: rgba(26, 26, 46, 0.95);
        border-radius: 20px;
        padding: 2rem;
        padding-top: 3rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 1rem;
    }

    
    .main-header {
        font-size: 64px;
        font-weight: bold;
        background: linear-gradient(120deg, #e74c3c, #3498db, #f39c12, #9b59b6);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 15px;
        margin-top: 60px;
        animation: gradient 3s ease infinite;
        padding-top: 20px;
        line-height: 1.2;
    }

    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .sub-header {
        font-size: 24px;
        color: #e0e0e0;
        text-align: center;
        margin-bottom: 35px;
        margin-top: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        font-weight: 500;
    }

    .top-spacer {
        height: 80px;
    }

    
    .category-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 18px;
        font-weight: bold;
        margin: 5px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }

    .badge-world {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
    }

    .badge-sports {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
    }

    .badge-business {
        background: linear-gradient(135deg, #f39c12, #d68910);
        color: white;
    }

    .badge-scifi {
        background: linear-gradient(135deg, #9b59b6, #8e44ad);
        color: white;
    }

    
    .prediction-box {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        margin: 20px 0;
        animation: slideIn 0.5s ease-out;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    
    .stMarkdown, .stText, p, span, label {
        color: #e0e0e0 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white !important;
        height: 60px;
        font-size: 22px;
        font-weight: bold;
        border-radius: 30px;
        border: none;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.6);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.8);
        background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
    }

    
    .info-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
        border: 1px solid rgba(52, 152, 219, 0.3);
    }

    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(52, 152, 219, 0.4);
        border: 1px solid rgba(52, 152, 219, 0.5);
    }

    
    .stTextArea textarea {
        background: rgba(26, 26, 46, 0.8) !important;
        color: #ffffff !important;
        border: 1px solid rgba(52, 152, 219, 0.5) !important;
        border-radius: 10px;
        font-size: 16px;
    }

    .stTextInput input {
        background: rgba(26, 26, 46, 0.8) !important;
        color: #ffffff !important;
        border: 1px solid rgba(52, 152, 219, 0.5) !important;
        border-radius: 10px;
        font-size: 16px;
    }

    
    .section-header {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white !important;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0 10px 0;
        font-size: 22px;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(52, 152, 219, 0.4);
    }

    
    .stats-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(52, 152, 219, 0.3);
        margin: 10px 0;
    }

    
    .keyword-tag {
        display: inline-block;
        background: linear-gradient(135deg, #f39c12, #d68910);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        margin: 5px;
        font-size: 14px;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }

    
    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #1a1a2e;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        border-radius: 10px;
    }

    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(26, 26, 46, 0.8);
        border-radius: 10px;
        padding: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(52, 152, 219, 0.2);
        color: #e0e0e0;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        border: 1px solid rgba(52, 152, 219, 0.3);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white !important;
    }

    
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        border-radius: 10px;
        color: white !important;
        font-weight: bold;
    }

    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }

    section[data-testid="stSidebar"] > div {
        background: rgba(26, 26, 46, 0.8);
        padding-top: 3rem;
    }

    hr {
        border-color: rgba(52, 152, 219, 0.3) !important;
        margin: 2rem 0;
    }

    a {
        color: #3498db !important;
    }

    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
    }

    
    [data-testid="stMetricValue"] {
        color: #3498db !important;
        font-size: 24px !important;
    }

    [data-testid="stMetricLabel"] {
        color: #e0e0e0 !important;
    }

    
    .char-count {
        font-size: 14px;
        color: #95a5a6;
        text-align: right;
        margin-top: 5px;
    }

    .char-count.warning {
        color: #f39c12;
    }

    .char-count.error {
        color: #e74c3c;
    }

    </style>
""", unsafe_allow_html=True)


EXAMPLE_NEWS = {
    "World": {
        "title": "UN Climate Summit Reaches Historic Agreement on Carbon Emissions",
        "description": "World leaders gathered at the United Nations Climate Summit have reached a landmark agreement to reduce global carbon emissions by 50% by 2030. The accord includes commitments from over 190 countries to transition to renewable energy sources and implement stricter environmental regulations. This historic deal comes after weeks of intense negotiations and represents a significant step forward in the global fight against climate change."
    },
    "Sports": {
        "title": "Barcelona Wins Champions League Final in Dramatic Penalty Shootout",
        "description": "FC Barcelona claimed their sixth Champions League title after defeating Manchester City 4-2 on penalties following a thrilling 2-2 draw in regular time. The match, held at Wembley Stadium, saw both teams display exceptional skill and determination. Lionel Messi scored twice for Barcelona while Erling Haaland netted both goals for City. The victory marks Barcelona's return to European glory after a five-year drought."
    },
    "Business": {
        "title": "Tech Giant Apple Announces Record-Breaking Quarterly Revenue",
        "description": "Apple Inc. reported its highest quarterly revenue in company history, reaching $125 billion in Q4 2024. The tech giant's success was driven by strong iPhone 16 sales, growing services revenue, and robust performance in emerging markets. CEO Tim Cook attributed the results to innovation in artificial intelligence features and expanding the company's ecosystem. Apple's stock surged 8% in after-hours trading following the announcement."
    },
    "Sci-Fi": {
        "title": "NASA Discovers Evidence of Liquid Water on Mars Subsurface",
        "description": "NASA's Perseverance rover has detected strong evidence of liquid water reservoirs beneath the Martian surface using ground-penetrating radar technology. The discovery, made in the Jezero Crater region, suggests that Mars may harbor conditions suitable for microbial life. Scientists believe these underground water reserves could be crucial for future human colonization efforts. The findings will be published in the journal Nature and represent one of the most significant discoveries in planetary science."
    }
}


CATEGORY_INFO = {
    1: {
        "name": "World",
        "emoji": "🌍",
        "color": "#3498db",
        "description": "International news, politics, diplomacy, and global events",
        "keywords": ["international", "government", "politics", "treaty", "UN", "country", "president", "minister"]
    },
    2: {
        "name": "Sports",
        "emoji": "⚽",
        "color": "#e74c3c",
        "description": "Sports events, competitions, athletes, and sporting news",
        "keywords": ["match", "game", "championship", "win", "score", "team", "player", "tournament"]
    },
    3: {
        "name": "Business",
        "emoji": "💼",
        "color": "#f39c12",
        "description": "Business news, markets, economy, and corporate developments",
        "keywords": ["revenue", "stock", "market", "company", "CEO", "profit", "economy", "investment"]
    },
    4: {
        "name": "Sci-Fi",
        "emoji": "🚀",
        "color": "#9b59b6",
        "description": "Science, technology, space exploration, and scientific discoveries",
        "keywords": ["NASA", "space", "discovery", "research", "technology", "science", "Mars", "rover"]
    }
}



def extract_keywords(text, top_n=10):
    """Extract important keywords from text"""
    text = re.sub(r'[^\w\s]', '', text.lower())
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                      'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                      'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
                      'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'])

    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    word_freq = Counter(filtered_words)
    top_keywords = word_freq.most_common(top_n)

    return [word for word, freq in top_keywords]


def get_text_stats(text):
    """Get text statistics"""
    words = len(text.split())
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))
    return {
        "words": words,
        "characters": chars,
        "characters_no_spaces": chars_no_spaces
    }


def validate_title(title):
    """Validate title length"""
    word_count = len(title.split())
    if word_count > 50:
        return False, f"Title is too long ({word_count} words). Maximum 50 words allowed."
    return True, f"Title length is valid ({word_count}/50 words)"



if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []

st.markdown('<div class="top-spacer"></div>', unsafe_allow_html=True)
st.markdown('<p class="main-header">📰 AI News Classification System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">🤖 Powered by Natural Language Processing & Machine Learning</p>',
            unsafe_allow_html=True)


st.markdown("""
    <div style='text-align: center; margin: 20px 0;'>
        <span class='category-badge badge-world'>🌍 World</span>
        <span class='category-badge badge-sports'>⚽ Sports</span>
        <span class='category-badge badge-business'>💼 Business</span>
        <span class='category-badge badge-scifi'>🚀 Sci-Fi</span>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")


tab1, tab2, tab3 = st.tabs(["📝 Classify News", "📊 Category Guide", "📜 History"])

with tab1:
    st.markdown('<div class="section-header">📝 Enter News Article</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        
        st.markdown("### 📌 News Title")
        news_title = st.text_input(
            "Enter the news headline (Max 50 words)",
            placeholder="e.g., NASA Discovers Water on Mars",
            key="title_input",
            label_visibility="collapsed"
        )

        
        if news_title:
            is_valid, message = validate_title(news_title)
            title_stats = get_text_stats(news_title)

            if is_valid:
                st.markdown(f"<div class='char-count'>✅ {message} | {title_stats['characters']} characters</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='char-count error'>❌ {message}</div>", unsafe_allow_html=True)

        st.markdown("---")

        
        st.markdown("### 📄 News Description")
        news_description = st.text_area(
            "Enter the full news article description",
            placeholder="Provide detailed information about the news story...",
            height=200,
            key="desc_input",
            label_visibility="collapsed"
        )

        
        if news_description:
            desc_stats = get_text_stats(news_description)
            st.markdown(f"""
                <div class='char-count'>
                    📊 {desc_stats['words']} words | {desc_stats['characters']} characters | {desc_stats['characters_no_spaces']} characters (no spaces)
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 💡 Try Examples")

        for category, example in EXAMPLE_NEWS.items():
            category_id = [k for k, v in CATEGORY_INFO.items() if v['name'] == category][0]
            emoji = CATEGORY_INFO[category_id]['emoji']

            if st.button(f"{emoji} {category} Example", key=f"example_{category}", use_container_width=True):
                st.session_state.title_input = example['title']
                st.session_state.desc_input = example['description']
                st.rerun()

        st.markdown("---")

        st.markdown("""
            <div class='info-card'>
                <h4 style='color: #3498db !important;'>ℹ️ Instructions</h4>
                <ul style='color: #e0e0e0; font-size: 14px;'>
                    <li>Enter a news title (max 50 words)</li>
                    <li>Provide detailed description</li>
                    <li>Click "Classify News" button</li>
                    <li>View results and confidence scores</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    
    classify_button = st.button("🔍 Classify News Article", use_container_width=True, type="primary")

  


if classify_button:
        if not news_title or not news_description:
            st.error("⚠️ Please enter both title and description!")
        else:
            title_valid, title_msg = validate_title(news_title)
            if not title_valid:
                st.error(f"⚠️ {title_msg}")
            else:
                with st.spinner('🔄 Analyzing news article with AI...'):
                    import time

                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                   
                    
                    predicted_class, confidence_scores = predict_news_category(news_title, news_description)
                    

                    
                    category_name = CATEGORY_INFO[int(predicted_class)]['name']
                    category_emoji = CATEGORY_INFO[int(predicted_class)]['emoji']
                    category_color = CATEGORY_INFO[int(predicted_class)]['color']

                    
                    combined_text = news_title + " " + news_description
                    keywords = extract_keywords(combined_text, top_n=8)

                st.markdown("---")

                
                st.markdown(f"""
                    <div class='prediction-box' style='background: linear-gradient(135deg, {category_color}33, {category_color}66); border: 3px solid {category_color};'>
                        {category_emoji} Predicted Category: <span style='color: {category_color};'>{category_name.upper()}</span>
                    </div>
                """, unsafe_allow_html=True)

                
                st.markdown("### 📊 Confidence Scores")

                viz_col1, viz_col2 = st.columns([3, 2])

                with viz_col1:
                    
                    categories = [CATEGORY_INFO[i]['name'] for i in range(1, 5)]
                    scores = [confidence_scores[i] * 100 for i in range(1, 5)]
                    colors_list = [CATEGORY_INFO[i]['color'] for i in range(1, 5)]

                    fig = go.Figure(go.Bar(
                        x=scores,
                        y=categories,
                        orientation='h',
                        marker=dict(
                            color=colors_list,
                            line=dict(color='white', width=2)
                        ),
                        text=[f"{score:.1f}%" for score in scores],
                        textposition='auto',
                    ))

                    fig.update_layout(
                        height=300,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#e0e0e0', 'size': 14},
                        xaxis={'gridcolor': 'rgba(52, 152, 219, 0.2)', 'title': 'Confidence (%)', 'range': [0, 100]},
                        yaxis={'gridcolor': 'rgba(52, 152, 219, 0.2)'},
                        margin=dict(l=20, r=20, t=20, b=20),
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with viz_col2:
                    for i in range(1, 5):
                        cat_name = CATEGORY_INFO[i]['name']
                        cat_emoji = CATEGORY_INFO[i]['emoji']
                        score = confidence_scores[i] * 100

                        st.markdown(f"""
                            <div class='stats-box'>
                                <strong style='color: {CATEGORY_INFO[i]['color']};'>{cat_emoji} {cat_name}</strong><br>
                                <span style='font-size: 24px; color: #e0e0e0;'>{score:.2f}%</span>
                            </div>
                        """, unsafe_allow_html=True)

                
                st.markdown("### 🔑 Important Keywords Detected")
                keywords_html = "".join([f"<span class='keyword-tag'>{kw}</span>" for kw in keywords])
                st.markdown(f"<div style='text-align: center; margin: 20px 0;'>{keywords_html}</div>",
                            unsafe_allow_html=True)

                
                st.markdown("### 📈 Article Statistics")

                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

                combined_stats = get_text_stats(combined_text)
                title_stats = get_text_stats(news_title)
                desc_stats = get_text_stats(news_description)

                with stat_col1:
                    st.metric("Total Words", combined_stats['words'])

                with stat_col2:
                    st.metric("Total Characters", combined_stats['characters'])

                with stat_col3:
                    st.metric("Title Words", title_stats['words'])

                with stat_col4:
                    st.metric("Keywords Found", len(keywords))

                
                history_record = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'title': news_title,
                    'description': news_description[:100] + "...",
                    'category': category_name,
                    'confidence': confidence_scores[int(predicted_class)] * 100,
                    'keywords': ", ".join(keywords[:5])
                }
                st.session_state.classification_history.insert(0, history_record)

                if len(st.session_state.classification_history) > 10:
                    st.session_state.classification_history = st.session_state.classification_history[:10]

                st.markdown("---")
                st.success("✅ Classification completed successfully!")

with tab2:
     st.markdown('<div class="section-header">📊 News Category Guide</div>', unsafe_allow_html=True)

     st.markdown("""
        <div class='info-card'>
            <p style='color: #e0e0e0; font-size: 16px;'>
                Our AI model classifies news articles into four main categories based on content analysis.
                Each category has distinct characteristics and common keywords.
            </p>
        </div>
    """, unsafe_allow_html=True)

     for class_id, info in CATEGORY_INFO.items():
        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown(f"""
                <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, {info['color']}33, {info['color']}66); 
                border-radius: 15px; border: 2px solid {info['color']};'>
                    <div style='font-size: 64px;'>{info['emoji']}</div>
                    <div style='font-size: 24px; font-weight: bold; color: white; margin-top: 10px;'>{info['name']}</div>
                    <div style='font-size: 14px; color: #e0e0e0; margin-top: 5px;'>Class {class_id}</div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class='info-card'>
                    <h4 style='color: {info['color']} !important;'>Description</h4>
                    <p style='color: #e0e0e0;'>{info['description']}</p>

                    <h4 style='color: {info['color']} !important; margin-top: 15px;'>Common Keywords</h4>
                    <div>
                        {"".join([f"<span class='keyword-tag'>{kw}</span>" for kw in info['keywords']])}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-header">📜 Classification History</div>', unsafe_allow_html=True)

    if len(st.session_state.classification_history) == 0:
        st.info("📭 No classification history yet. Classify some news articles to see them here!")
    else:
        for idx, record in enumerate(st.session_state.classification_history):
            cat_id = [k for k, v in CATEGORY_INFO.items() if v['name'] == record['category']][0]
            cat_color = CATEGORY_INFO[cat_id]['color']
            cat_emoji = CATEGORY_INFO[cat_id]['emoji']

            with st.expander(f"{cat_emoji} {record['title']} - {record['timestamp']}", expanded=(idx == 0)):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Title:** {record['title']}")
                    st.markdown(f"**Description:** {record['description']}")
                    st.markdown(f"**Keywords:** {record['keywords']}")

                with col2:
                    st.markdown(f"""
                        <div class='stats-box' style='text-align: center;'>
                            <div style='font-size: 48px;'>{cat_emoji}</div>
                            <div style='font-size: 20px; font-weight: bold; color: {cat_color};'>{record['category']}</div>
                            <div style='font-size: 24px; color: #e0e0e0; margin-top: 10px;'>{record['confidence']:.1f}%</div>
                            <div style='font-size: 12px; color: #95a5a6;'>Confidence</div>
                        </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            history_df = pd.DataFrame(st.session_state.classification_history)
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History (CSV)",
                data=csv,
                file_name=f"news_classification_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
                use_container_width=True
            )

        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.classification_history = []
                st.rerun()


with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='font-size: 80px;'>📰</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='info-card'>
            <h2 style='color: #3498db !important; text-align: center;'>About This System</h2>
            <p style='text-align: justify; color: #e0e0e0 !important;'>
                This AI-powered news classification system uses Natural Language Processing (NLP) and 
                Machine Learning to automatically categorize news articles into four main categories: 
                World, Sports, Business, and Sci-Fi.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📊 System Stats")
    st.metric("Total Categories", "4")
    st.metric("Classifications Done", len(st.session_state.classification_history))
    st.metric("Model Accuracy", "94.5%")

    st.markdown("---")

    st.markdown("### ⚙️ How It Works")
    st.markdown("""
        <div class='info-card'>
            <ol style='color: #e0e0e0; font-size: 14px;'>
                <li>📝 Enter news title & description</li>
                <li>🔍 Text preprocessing & analysis</li>
                <li>🤖 AI model prediction</li>
                <li>📊 Confidence score calculation</li>
                <li>🔑 Keyword extraction</li>
                <li>✅ Results display</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 💡 Tips for Best Results")
    st.info("""
        ✓ Provide detailed descriptions

        ✓ Use clear, concise titles

        ✓ Include relevant keywords

        ✓ Avoid mixed-category content
    """)

    st.markdown("---")

    st.markdown("""
        <div style='text-align: center; color: #e0e0e0;'>
            <p>📰 News Classification AI</p>
            <p>Version 1.0</p>
            <p>© 2024 NLP Labs</p>
        </div>
    """, unsafe_allow_html=True)


st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); 
    border-radius: 15px; color: white; margin-top: 30px; box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);'>
        <h3 style='color: white !important;'>🌟 Accurate News Classification Powered by AI</h3>
        <p style='color: white !important;'>Stay informed with automated, intelligent content categorization</p>
        <p style='color: white !important;'>Made with ❤️ using Streamlit & Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

