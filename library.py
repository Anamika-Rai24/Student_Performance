import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Library AI System",
    page_icon="📚",
    layout="centered"
)

# -------------------------------
# ULTRA Premium Library AI Theme
# -------------------------------
st.markdown("""
<style>

/* ===== Animated Gradient Overlay Background ===== */
.stApp {
    background:
        linear-gradient(rgba(15, 10, 5, 0.85), rgba(15, 10, 5, 0.85)),
        url("https://images.unsplash.com/photo-1507842217343-583bb7270b66?auto=format&fit=crop&w=1470&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Segoe UI', sans-serif;
}

/* Subtle animated background effect */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ===== Floating Glass Cards ===== */
.glass {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 25px;
    padding: 35px;
    margin-bottom: 30px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.7);
    border: 1px solid rgba(255, 215, 0, 0.4);
    color: #ffffff;
    transition: 0.4s ease;
}

.glass:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 45px rgba(255, 193, 7, 0.5);
}

/* ===== Title Styling ===== */
.title {
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    background: linear-gradient(90deg, #ffd700, #ffb300, #fff3b0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
    letter-spacing: 1px;
}

/* ===== Subtitle ===== */
.subtitle {
    text-align: center;
    font-size: 19px;
    color: #f5e6c8;
    margin-bottom: 35px;
    font-weight: 300;
}

/* ===== Sidebar Styling ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #140c04, #000000);
    border-right: 1px solid rgba(255,215,0,0.3);
}

/* ===== Sidebar Header ===== */
section[data-testid="stSidebar"] h2 {
    color: #ffd700;
}

/* ===== Buttons ===== */
.stButton>button {
    background: linear-gradient(135deg, #ffd700, #ff9800);
    color: black;
    border-radius: 40px;
    padding: 12px 28px;
    font-size: 17px;
    font-weight: 600;
    border: none;
    transition: 0.4s ease;
    box-shadow: 0 5px 20px rgba(255, 193, 7, 0.7);
}

.stButton>button:hover {
    transform: scale(1.08);
    box-shadow: 0 8px 25px rgba(255, 193, 7, 1);
}

/* ===== Metric Box Styling ===== */
[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.07);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255, 215, 0, 0.4);
    box-shadow: 0 6px 25px rgba(0,0,0,0.6);
}

/* ===== Dataframe Styling ===== */
[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.6);
}

/* ===== Scrollbar ===== */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(#ffd700, #ff9800);
    border-radius: 10px;
}

/* ===== Text Color ===== */
h1, h2, h3, h4, h5, h6, p, label {
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title Section
# -------------------------------
st.markdown('<div class="title">📚 Library AI Management System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict Frequent Library Users using Machine Learning</div>', unsafe_allow_html=True)

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("library_data_100.csv")

data = load_data()

# -------------------------------
# Dataset Preview
# -------------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.write("### 📄 Dataset Preview")
st.dataframe(data.head(), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Encoding
# -------------------------------
le = LabelEncoder()
data["FrequentUser"] = le.fit_transform(data["FrequentUser"])

X = data[["StudentAge", "BooksIssued", "LateReturns", "MembershipYears"]]
y = data["FrequentUser"]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model Training
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🧑‍🎓 Student Information")

age = st.sidebar.slider("Student Age", 10, 60, 20)
books = st.sidebar.slider("Books Issued", 0, 50, 5)
late = st.sidebar.slider("Late Returns", 0, 20, 1)
membership = st.sidebar.slider("Membership Years", 0, 10, 1)

# -------------------------------
# Prediction
# -------------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.write("### 🔮 Prediction Result")

if st.sidebar.button("✨ Predict Now"):
    input_data = [[age, books, late, membership]]
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(
            f"✅ **Frequent Library User**\n\n"
            f"Confidence: **{probability * 100:.1f}%**"
        )
    else:
        st.error(
            f"❌ **Not a Frequent Library User**\n\n"
            f"Confidence: **{probability * 100:.1f}%**"
        )

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Accuracy
# -------------------------------
accuracy = model.score(X_test, y_test)

st.markdown('<div class="glass">', unsafe_allow_html=True)
st.metric("📊 Model Accuracy", f"{accuracy * 100:.2f}%")
st.markdown('</div>', unsafe_allow_html=True)
