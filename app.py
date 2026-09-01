import os
import nltk

nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=True)
import nltk
import streamlit as st
import pickle
import re
import numpy as np
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
import scipy.sparse as sp
from pathlib import Path

@st.cache_resource
def load_models():

    BASE_DIR = Path(__file__).resolve().parent

    with open(BASE_DIR / 'tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    with open(BASE_DIR / 'best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open(BASE_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open(BASE_DIR / 'selected_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)

    return tfidf, model, scaler, selected_features

tfidf, model, scaler, selected_features = load_models()

tfidf, model, scaler, selected_features = load_models()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w)
             for w in words
             if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def get_reasons(text, prediction):
    
    reasons_against = []   # why fake
    reasons_for     = []   # why real
    txt = text.lower()
  

def extract_custom_features(text):
    sensational = [
        'breaking', 'urgent', 'shocking', 'exposed',
        'exclusive', 'viral', 'share', 'forward',
        'unbelievable', 'secret', 'banned', 'hidden',
        'must read', 'alert', 'warning', 'bombshell'
    ]
    credibility = [
        'according to', 'reported', 'confirmed',
        'official', 'government', 'research', 'study',
        'data', 'percent', 'source', 'statement',
        'announced', 'verified', 'fact check'
    ]
    urgency = [
        'share this', 'forward to', 'tell everyone',
        'share karo', 'please share', 'must share',
        'spread the word', 'kindly forward'
    ]
    words = text.split()
    word_count     = len(words)
    char_count     = len(text)
    avg_word_len   = np.mean([len(w) for w in words]) if words else 0
    exclamation    = text.count('!')
    caps_count     = sum(1 for w in words if w.isupper() and len(w) > 2)
    sensational_sc = sum(1 for w in sensational if w in text.lower())
    credibility_sc = sum(1 for w in credibility if w in text.lower())
    urgency_sc     = sum(1 for p in urgency if p in text.lower())
    title_caps_ratio = 0.0
    return np.array([[
        word_count, char_count, avg_word_len,
        exclamation, caps_count,
        sensational_sc, credibility_sc,
        urgency_sc, title_caps_ratio
    ]])

def generate_explanation(text, prediction):
    warnings  = []
    positives = []
    text_lower = text.lower()
    words = text.split()

    urgency_phrases = [
        'kindly forward', 'share this', 'forward to all',
        'please share', 'must share', 'share karo',
        'tell everyone', 'spread the word'
    ]
    found_urgency = [p for p in urgency_phrases if p in text_lower]
    if found_urgency:
        warnings.append(
            f"Urgency language detected — phrases like "
            f"'{found_urgency[0]}' are commonly used in fake forwards"
        )

    sensational = [
        'breaking', 'urgent', 'shocking', 'exposed', 'exclusive',
        'viral', 'unbelievable', 'secret', 'banned', 'bombshell',
        'alert', 'hidden truth', 'must read'
    ]
    found_sensational = [w for w in sensational if w in text_lower]
    if found_sensational:
        warnings.append(
            f"Sensational words found — "
            f"'{', '.join(found_sensational[:3])}' — "
            f"real news rarely uses such language"
        )

    caps_words = [w for w in words if w.isupper() and len(w) > 2]
    if len(caps_words) > 2:
        warnings.append(
            f"Excessive CAPS usage — "
            f"'{', '.join(caps_words[:3])}' — "
            f"used to create panic or urgency"
        )

    excl_count = text.count('!')
    if excl_count > 1:
        warnings.append(
            f"Multiple exclamation marks ({excl_count}) — "
            f"credible sources rarely use excessive punctuation"
        )

    source_words = [
        'according to', 'reported by', 'source', 'confirmed',
        'official', 'government', 'ministry', 'aajtak',
        'ndtv', 'bbc', 'reuters', 'ani', 'pti'
    ]
    has_source = any(s in text_lower for s in source_words)
    if not has_source:
        warnings.append(
            "No credible source mentioned — "
            "authentic news always cites official sources"
        )

    has_url = bool(re.search(r'http\S+|www\S+', text))
    if has_url:
        warnings.append(
            "URL detected — always verify links "
            "before clicking or sharing"
        )

    credibility = [
        'according to', 'reported', 'confirmed', 'official',
        'government', 'research', 'study', 'data', 'percent',
        'source', 'statement', 'announced', 'verified'
    ]
    found_cred = [w for w in credibility if w in text_lower]
    if found_cred:
        positives.append(
            f"Credibility indicators present — "
            f"'{', '.join(found_cred[:3])}'"
        )

    if len(words) < 15:
        warnings.append(
            "Very short content — "
            "too little context to verify authenticity"
        )
    elif len(words) > 100:
        positives.append(
            "Detailed content — "
            "longer articles tend to be more credible"
        )

    q_count = text.count('?')
    if q_count > 2:
        warnings.append(
            f"Multiple questions ({q_count}) — "
            f"clickbait style often uses rhetorical questions"
        )
    return warnings, positives

def get_smart_advice(text, prediction):
    text_lower = text.lower()

    # ── Topic word lists ────────────────────────────────────
    education_words = [
        'school', 'college', 'university', 'student', 'exam',
        'result', 'admission', 'scholarship', 'fee', 'course',
        'degree', 'board', 'cbse', 'ugc', 'neet', 'jee',
        'upsc', 'teacher', 'education', 'study', 'campus',
        'internship', 'placement', 'marks', 'syllabus', 'class',
        'academy', 'institute', 'coaching', 'tuition', 'training'
    ]
    medical_words = [
        'health', 'medicine', 'doctor', 'hospital', 'disease',
        'vaccine', 'covid', 'cancer', 'treatment', 'drug',
        'virus', 'infection', 'symptoms', 'cure', 'medical',
        'patient', 'clinic', 'pharmacy', 'surgery', 'tablet'
    ]
    finance_words = [
        'bank', 'money', 'rupee', 'loan', 'share', 'stock',
        'market', 'rbi', 'sebi', 'investment', 'fraud',
        'account', 'transaction', 'tax', 'income', 'finance',
        'upi', 'payment', 'scam', 'wallet', 'atm', 'kyc'
    ]
    govt_words = [
        'government', 'modi', 'minister', 'parliament', 'scheme',
        'yojana', 'pib', 'official', 'policy', 'law', 'bill',
        'rajya sabha', 'lok sabha', 'cm', 'pm', 'central',
        'state govt', 'municipality', 'district', 'collector'
    ]
    election_words = [
        'election', 'vote', 'candidate', 'party', 'bjp',
        'congress', 'campaign', 'polling', 'ballot', 'ec',
        'constituency', 'seat', 'result', 'win', 'lose',
        'chunav', 'matdan', 'evm', 'manifesto', 'rally'
    ]
    legal_words = [
        'court', 'judge', 'case', 'arrest', 'police', 'fir',
        'supreme court', 'high court', 'verdict', 'bail',
        'crime', 'accused', 'law', 'legal', 'justice',
        'lawyer', 'advocate', 'hearing', 'petition', 'order'
    ]

    # ── Topic detection — education PEHLE check ho ──────────
    is_education = any(w in text_lower for w in education_words)
    is_medical   = any(w in text_lower for w in medical_words)
    is_finance   = any(w in text_lower for w in finance_words)
    is_govt      = any(w in text_lower for w in govt_words)
    is_election  = any(w in text_lower for w in election_words)
    is_legal     = any(w in text_lower for w in legal_words)

    # ── FAKE advice ─────────────────────────────────────────
    if prediction == 0:
        if is_education:
            return {
                'icon': '🎓',
                'topic': 'Education / Scholarship',
                'action': 'Fake scholarship and admission news is very '
                          'common in India. Never pay any fees based on '
                          'unverified messages. Always check official portals.',
                'sources': [
                    ('UGC India',               'ugc.ac.in'),
                    ('National Scholarship Portal', 'scholarships.gov.in'),
                    ('CBSE Official',            'cbse.gov.in'),
                    ('PIB Fact Check',           'pib.gov.in/factcheck'),
                ]
            }
        elif is_medical:
            return {
                'icon': '🏥',
                'topic': 'Medical / Health',
                'action': 'Medical misinformation can cause serious harm. '
                          'Never follow health advice from unverified forwards. '
                          'Always consult a doctor.',
                'sources': [
                    ('WHO',          'who.int'),
                    ('MoHFW India',  'mohfw.gov.in'),
                    ('PIB Fact Check', 'pib.gov.in/factcheck'),
                ]
            }
        elif is_finance:
            return {
                'icon': '🏦',
                'topic': 'Finance / Banking',
                'action': 'Financial fake news can cause panic and monetary '
                          'loss. Never share banking details based on forwards. '
                          'Verify before acting.',
                'sources': [
                    ('RBI Official', 'rbi.org.in'),
                    ('SEBI',         'sebi.gov.in'),
                    ('PIB Fact Check', 'pib.gov.in/factcheck'),
                ]
            }
        elif is_election:
            return {
                'icon': '🗳️',
                'topic': 'Election / Politics',
                'action': 'Election misinformation is dangerous for democracy. '
                          'Verify all political claims before sharing.',
                'sources': [
                    ('Election Commission', 'eci.gov.in'),
                    ('PIB Fact Check',      'pib.gov.in/factcheck'),
                    ('Voter Helpline',      '1950'),
                ]
            }
        elif is_legal:
            return {
                'icon': '⚖️',
                'topic': 'Legal / Crime',
                'action': 'Legal misinformation can harm reputations and '
                          'cause panic. Verify from official court records.',
                'sources': [
                    ('Supreme Court of India', 'sci.gov.in'),
                    ('Alt News',               'altnews.in'),
                    ('BOOM Live',              'boomlive.in'),
                ]
            }
        elif is_govt:
            return {
                'icon': '🏛️',
                'topic': 'Government / Policy',
                'action': 'Fake government scheme news spreads very fast. '
                          'Always verify from PIB before sharing.',
                'sources': [
                    ('PIB Fact Check', 'pib.gov.in/factcheck'),
                    ('MyGov India',    'mygov.in'),
                    ('Alt News',       'altnews.in'),
                ]
            }
        else:
            return {
                'icon': '📰',
                'topic': 'General News',
                'action': 'Do not share this content without verifying '
                          'from trusted fact-checkers.',
                'sources': [
                    ('Alt News',      'altnews.in'),
                    ('BOOM Live',     'boomlive.in'),
                    ('Vishvas News',  'vishvasnews.com'),
                ]
            }

    # ── REAL advice ─────────────────────────────────────────
    else:
        if is_education:
            return {
                'icon': '🎓',
                'topic': 'Education / Scholarship',
                'action': 'Content appears credible. Still verify admission '
                          'and scholarship details directly from official '
                          'portals before applying.',
                'sources': [
                    ('UGC India',               'ugc.ac.in'),
                    ('National Scholarship Portal', 'scholarships.gov.in'),
                    ('CBSE Official',            'cbse.gov.in'),
                ]
            }
        elif is_medical:
            return {
                'icon': '🏥',
                'topic': 'Medical / Health',
                'action': 'Content appears credible. Always consult a '
                          'qualified doctor before following any '
                          'medical advice.',
                'sources': [
                    ('WHO',         'who.int'),
                    ('MoHFW India', 'mohfw.gov.in'),
                ]
            }
        elif is_finance:
            return {
                'icon': '🏦',
                'topic': 'Finance / Banking',
                'action': 'Content appears credible. Verify financial '
                          'decisions from official RBI or SEBI portals '
                          'before acting.',
                'sources': [
                    ('RBI Official', 'rbi.org.in'),
                    ('SEBI',         'sebi.gov.in'),
                ]
            }
        elif is_election:
            return {
                'icon': '🗳️',
                'topic': 'Election / Politics',
                'action': 'Content appears credible. Verify election '
                          'results and candidate info from ECI official '
                          'website only.',
                'sources': [
                    ('Election Commission', 'eci.gov.in'),
                    ('PIB India',          'pib.gov.in'),
                ]
            }
        elif is_legal:
            return {
                'icon': '⚖️',
                'topic': 'Legal / Crime',
                'action': 'Content appears credible. Verify legal '
                          'judgments from official court websites.',
                'sources': [
                    ('Supreme Court of India', 'sci.gov.in'),
                    ('India Kanoon',           'indiankanoon.org'),
                ]
            }
        elif is_govt:
            return {
                'icon': '🏛️',
                'topic': 'Government / Policy',
                'action': 'Content appears credible. Confirm scheme '
                          'details from official government portals.',
                'sources': [
                    ('PIB India',   'pib.gov.in'),
                    ('MyGov India', 'mygov.in'),
                ]
            }
        else:
            return {
                'icon': '✅',
                'topic': 'General News',
                'action': 'Content appears credible. Always verify '
                          'specific statistics or claims from the '
                          'original source.',
                'sources': [
                    ('NDTV',      'ndtv.com'),
                    ('The Hindu', 'thehindu.com'),
                    ('BBC Hindi', 'bbc.com/hindi'),
                ]
            }
    

# ==================== UI ====================

st.set_page_config(
    page_title="VeriScan AI",
    page_icon="🔍",
    layout="centered"
)

# Clean professional light theme CSS
st.markdown("""
<style>
    /* Force light background
            
             everywhere */
    .stApp {
        background-color: #F7F9FC;
    }
    
    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }
    
    /* All text inputs light */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #1a1a2e !important;
        border: 2px solid #E0E6F0 !important;
        border-radius: 8px !important;
        font-size: 15px !important;
        font-family: 'Georgia', serif !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #2C5EE8 !important;
        box-shadow: 0 0 0 2px rgba(44,94,232,0.15) !important;
    }
    
    /* Selectbox light */
    .stSelectbox > div > div {
        background-color: #FFFFFF !important;
        border: 2px solid #E0E6F0 !important;
        border-radius: 8px !important;
        color: #1a1a2e !important;
    }
    
    /* Labels */
    .stSelectbox label, .stTextArea label {
        color: #2C5EE8 !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }
    
    /* Analyze button */
    .stButton > button {
        background: linear-gradient(135deg, #2C5EE8, #1a3fa8) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(44,94,232,0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(44,94,232,0.4) !important;
    }
    
    /* Result cards */
   .result-real {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border-left: 5px solid #2E7D32;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
    }
    
    .result-fake {
        background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
        border-left: 5px solid #C62828;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
    }
    
    .result-title-real {
        color: #1B5E20;
        font-size: 28px;
        font-weight: 800;
        margin: 0;
    }
    
    .result-title-fake {
        color: #B71C1C;
        font-size: 28px;
        font-weight: 800;
        margin: 0;
    }

.verdict-fake {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #ef4444;
}

.verdict-real {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #22c55e;
}

.score-bar-wrap {
    background: #E0E6F0;
    border-radius: 999px;
    height: 12px;
    margin: 0.5rem 0 1rem 0;
    overflow: hidden;
}

.score-bar-fill-fake {
    height: 12px;
    border-radius: 999px;
    background: linear-gradient(90deg, #ef4444, #f97316);
}

.score-bar-fill-real {
    height: 12px;
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e, #00d4ff);
}

    
    /* Section headers */
    .section-header {
        color: #2C5EE8;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 24px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #E0E6F0;
    }
    
    /* Warning cards */
    .flag-card {
        background: #FFF8E1;
        border: 1px solid #FFD54F;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #5D4037;
        font-size: 14px;
    }
    
    /* Positive cards */
    .positive-card {
        background: #E8F5E9;
        border: 1px solid #A5D6A7;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #2E7D32;
        font-size: 14px;
    }
    
    /* Stats row */
    .stat-box {
        background: #FFFFFF;
        border: 1px solid #E0E6F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    
    .stat-number {
        font-size: 28px;
        font-weight: 800;
        color: #2C5EE8;
    }
    
    .stat-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Platform badge */
    .platform-badge {
        display: inline-block;
        background: #EEF2FF;
        color: #2C5EE8;
        border: 1px solid #C7D7FF;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 13px;
        font-weight: 600;
    }
            
    .reason-box {
    background: #bfe3d0;
    border-left: 3px solid #00d4ff;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.92rem;
    color: #555;
}

.reason-box-warn {
    background: #bfe3d0;
    border-left: 3px solid #f97316;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.92rem;
    color:#555;
}

.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color:  #14e077;
    margin-bottom: 0.5rem;
}        
    
    /* Disclaimer */
    .disclaimer {
        background: #F0F4FF;
        border: 1px solid #C7D7FF;
        border-radius: 8px;
        padding: 12px 16px;
        color:#555 ;
        font-size: 13px;
        margin-top: 16px;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #E0E6F0;
        margin: 20px 0;
    }
    
    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div style="text-align:center; padding: 10px 0 24px 0;">
    <div style="font-size:42px; font-weight:900; color:#2C5EE8; 
                letter-spacing:-1px;">
        🔍 VeriScan AI
    </div>
    <div style="font-size:15px; color:#888; margin-top:4px;">
        Universal Fake News & Misinformation Detector
    </div>
    
</div>
""", unsafe_allow_html=True)

st.markdown('<hr>', unsafe_allow_html=True)

# ---- Platform Selector ----
platform_options = {
    "📱 WhatsApp Forward":     "WhatsApp",
    "📸 Instagram":            "Instagram",
    "👥 Facebook":             "Facebook",
    "🐦 Twitter / X":          "Twitter/X",
    "📺 Aaj Tak":              "Aaj Tak",
    "📡 NDTV":                 "NDTV",
    "📰 Amar Ujala":           "Amar Ujala",
    "📰 Dainik Bhaskar":       "Dainik Bhaskar",
    "📺 Zee News":             "Zee News",
    "📰 India Today":          "India Today",
    "📺 Republic Bharat":      "Republic Bharat",
    "▶️ YouTube":              "YouTube",
    "✈️ Telegram":             "Telegram",
    "🌐 Other News Website":   "Other News Website",
    "📲 Other Social Media":   "Other Social Media",
}

platform_key = st.selectbox(
    "SELECT CONTENT SOURCE",
    list(platform_options.keys())
)
platform = platform_options[platform_key]

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ---- Text Input ----
user_input = st.text_area(
    "PASTE CONTENT TO ANALYZE",
    height=160,
    placeholder="Paste any news headline, WhatsApp forward, "
                "social media post or news article here...",
)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1.5, 1])
with col2:
    analyze_btn = st.button(
        "🔍  Analyze Content",
        use_container_width=True
    )

# ---- Analysis ----

if analyze_btn:
    if user_input.strip() == "":
        st.warning("⚠️ Please paste some content to analyze.")

    elif len(user_input.split()) < 10:
        st.markdown("""
    <div style="background:#FFF3CD; border:1px solid #FFC107; 
                border-radius:10px; padding:16px; margin:10px 0;
                color:#856404;">
        <strong>⚠️ Content Too Short to Analyze</strong><br><br>
        Please enter at least <strong>10 words</strong> for accurate analysis.<br>
        VeriScan AI needs enough context to detect misinformation patterns.
        <br><br>
        <em>Example: Paste a full WhatsApp forward, news headline, 
        or article paragraph.</em>
    </div>
    """, unsafe_allow_html=True)
    else:
        with st.spinner("🔄 Scanning content..."):
            cleaned         = clean_text(user_input)
            tfidf_feat      = tfidf.transform([cleaned])
            custom_feat     = sp.csr_matrix(extract_custom_features(user_input))
            final_feat      = hstack([tfidf_feat, custom_feat])
            prediction      = model.predict(final_feat)[0]
            
            warnings, positives = generate_explanation(user_input, prediction)

        # credibility score (heuristic 0-100)
        base   = 85 if prediction == 1 else 25
        adjust = len(positives) * 5 - len(warnings) * 8
        score  = max(5, min(95, base + adjust))

        # ── VERDICT ─────────────────────────────────────────
        if prediction == 0:
           st.markdown("""
           <div class="result-fake">
               <div class="verdict-fake">
                    ❌ LIKELY FAKE / MISLEADING
               </div>
           </div>
           """, unsafe_allow_html=True)
        else:
           st.markdown("""
           <div class="result-real">
               <div class="verdict-real">
                   ✅ LIKELY CREDIBLE / REAL
               </div>
            </div>
            """, unsafe_allow_html=True)

        # credibility meter
        bar_class = "score-bar-fill-real" if prediction == 1 else "score-bar-fill-fake"
        st.markdown(
            f'<div class="section-label">Credibility Score: {score}/100</div>'
            f'<div class="score-bar-wrap">'
            f'<div class="{bar_class}" style="width:{score}%"></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Platform + stats badges
        word_count  = len(user_input.split())
        excl_count  = user_input.count('!')
        caps_count  = sum(1 for w in user_input.split()
                         if w.isupper() and len(w) > 2)

        st.markdown(f"""
        <div style="margin: 12px 0;">
            <span class="platform-badge">📱 {platform}</span>
            &nbsp;
            <span class="platform-badge">📝 {word_count} words</span>
            &nbsp;
            <span class="platform-badge">
                {'❌ Fake' if prediction == 0 else '✅ Real'}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Explanation section
        st.markdown(
            '<div class="section-header">'
            '🔎 Why VeriScan Flagged This Content'
            '</div>',
            unsafe_allow_html=True
        )

        if warnings:
            st.markdown("**🚩 Red Flags Detected:**")
            for w in warnings:
                st.markdown(
                    f'<div class="flag-card">⚠️ &nbsp;{w}</div>',
                    unsafe_allow_html=True
                )

        if positives:
            st.markdown(
                "<br>**✅ Credibility Signals:**",
                unsafe_allow_html=True
            )
            for p in positives:
                st.markdown(
                    f'<div class="positive-card">✅ &nbsp;{p}</div>',
                    unsafe_allow_html=True
                )

        if not warnings and not positives:
            st.info(
                "No specific patterns detected. "
                "Result is based on AI model training."
            )

         
        # Smart advice
        advice = get_smart_advice(user_input, prediction)

        st.markdown(
            f'<div class="section-header">'
            f'{advice["icon"]} What Should You Do?'
            f'</div>',
            unsafe_allow_html=True
        )
 
        advice_color = "#FFF3CD" if prediction == 0 else "#E8F5E9"
        border_color = "#FFC107" if prediction == 0 else "#4CAF50"
        text_color   = "#856404" if prediction == 0 else "#2E7D32"

        st.markdown(f"""
        <div style="background:{advice_color}; border:1px solid {border_color};
            border-radius:10px; padding:16px; margin:8px 0;
            color:{text_color};">
        <strong>Topic Detected: {advice['icon']} {advice['topic']}</strong>
        <br><br>
        {advice['action']}
        <br><br>
        <strong>Verify from these trusted sources:</strong><br>
        {''.join([f'&nbsp;&nbsp;• <a href="https://{src[1]}" target="_blank" style="color:{text_color}">{src[0]}</a><br>' for src in advice['sources']])}
        </div>
        """, unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Disclaimer:</strong> VeriScan AI is an 
            AI-based tool with ~85% accuracy. 
            Always cross-check important news from multiple 
            credible sources before sharing.
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="
text-align:center;
padding:25px;
color:#7a7a7a;
font-size:15px;
line-height:1.8;
">

<b>🔍 VeriScan AI | AI Powered Fake News Detector</b><br>

Helping users verify information before sharing
through <b>Machine Learning</b> &
<b>Explainable AI</b>.
<br><br>
Made with ❤️ by <b>Khushboo Khatun</b>

</div>
""", unsafe_allow_html=True)