import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Email Spam Detection",
    page_icon="🔐",
    layout="centered"
)

# -------------------------------
# Premium Cyber Email Theme
# -------------------------------
def add_premium_email_theme():
    st.markdown("""
    <style>

    /* Full Background - Email Security Theme */
    .stApp {
        background:
            linear-gradient(rgba(5,10,25,0.85), rgba(5,10,25,0.85)),
            url('https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&w=1470&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
        color: #e0f7ff;
    }

    /* Glass Effect Container */
    .main {
        background: rgba(15, 25, 45, 0.75);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 0 25px rgba(0,255,255,0.4);
    }

    /* Animated Title */
    h1 {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #00f7ff, #00c3ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 2s infinite alternate;
    }

    @keyframes glow {
        from { text-shadow: 0 0 10px #00f7ff; }
        to { text-shadow: 0 0 25px #00c3ff; }
    }

    p {
        text-align: center;
        font-size: 18px;
        color: #c8eaff;
    }

    textarea {
        background-color: rgba(0,0,0,0.6) !important;
        border: 2px solid #00c3ff !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        font-size: 16px !important;
    }

    .stButton>button {
        background: linear-gradient(90deg, #00c3ff, #0072ff);
        color: white;
        font-size: 18px;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        border: none;
        transition: 0.3s ease;
        box-shadow: 0 0 15px rgba(0,195,255,0.6);
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(0,195,255,0.9);
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00f7ff, #0072ff);
    }

    hr {
        border: 1px solid #00c3ff;
        opacity: 0.4;
        margin: 2rem 0;
    }

    footer {
        text-align: center;
        font-size: 14px;
        color: #a8eaff;
        margin-top: 2rem;
    }

    </style>
    """, unsafe_allow_html=True)

# Apply Theme
add_premium_email_theme()

# -------------------------------
# Title Section
# -------------------------------
st.markdown("<h1>🔐 AI Email Spam Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p>Protect your inbox using Machine Learning & NLP Technology</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------------
# Security Banner Image
# -------------------------------
st.image(
    "https://cdn-icons-png.flaticon.com/512/9068/9068670.png",
    width=140
)

# -------------------------------
# Dataset
# -------------------------------
data = {
    "text": [
        "You won a lottery!",
        "Claim prize by clicking!",
        "Click on amount to redeem reward!",
        "Limited offer for your account!",
        "Aadhar OTP",
        "Module Assessment",
        "Your ticket is confirmed with IRCTC",
        "Registration number for SSC Exams.",
        "Credit card offer for low interest.",
        "Win cash prize!",
        "ES Class is scheduled on Monday"
    ],
    "label": ["spam","spam","spam","spam","ham","ham","ham","ham","spam","spam","ham"]
}

df = pd.DataFrame(data)

# -------------------------------
# Train Model
# -------------------------------
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('nb', MultinomialNB(alpha=0.1))
])

model.fit(X_train, y_train)

# -------------------------------
# User Input
# -------------------------------
email_text = st.text_area(
    "✉️ Enter Email Text Below:",
    height=150,
    placeholder="Type or paste email content here..."
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Check Spam"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some email text.")
    else:
        prediction = model.predict([email_text])[0]
        probability = model.predict_proba([email_text])[0]

        spam_index = list(model.classes_).index("spam")
        ham_index = list(model.classes_).index("ham")

        if prediction == "spam":
            st.error("🚨 This is a SPAM Email")
            st.progress(float(probability[spam_index]))
            st.write("Spam Probability:", round(probability[spam_index]*100, 2), "%")
        else:
            st.success("✅ This is NOT a Spam Email")
            st.progress(float(probability[ham_index]))
            st.write("Ham Probability:", round(probability[ham_index]*100, 2), "%")

        st.info("Model Used: TF-IDF Vectorizer + Multinomial Naive Bayes")

# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<footer>Built with ❤️ using NLP | TF-IDF | Naive Bayes | Streamlit</footer>",
    unsafe_allow_html=True
)
