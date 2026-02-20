import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ---------- Page Config ----------
st.set_page_config(
    page_title="Disease Predictor",
    page_icon="🩺",
    layout="wide"
)

# ---------- Background Image ----------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: 
        linear-gradient(rgba(0, 40, 85, 0.55), rgba(0, 40, 85, 0.55)),
        url("https://images.unsplash.com/photo-1587351021759-3e566b6af7cc");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
    right: 2rem;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------- Title ----------
st.markdown(
    "<h1 style='text-align:center; color:white;'>🩺 Disease Prediction App</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center; color:white;'>Predict Disease Based on Symptoms using Machine Learning</h4>",
    unsafe_allow_html=True
)

# ---------- Load Data ----------
df = pd.read_csv("disease_prediction.csv")

# Feature selection
X = df.drop("Disease", axis=1)
y = df["Disease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# ---------- Sidebar Inputs ----------
st.sidebar.header("Select Symptoms (0 = No, 1 = Yes)")

fever = st.sidebar.selectbox("Fever 🤒", [0, 1])
cough = st.sidebar.selectbox("Cough 😷", [0, 1])
fatigue = st.sidebar.selectbox("Fatigue 😴", [0, 1])
body_ache = st.sidebar.selectbox("Body Ache 🤕", [0, 1])
headache = st.sidebar.selectbox("Headache 🤯", [0, 1])

# ---------- Prediction ----------
if st.sidebar.button("Predict Disease 🧪"):

    new_data = [[fever, cough, fatigue, body_ache, headache]]
    prediction = model.predict(new_data)[0]

    st.markdown(
        f"""
        <div style="
            background-color: rgba(0,0,0,0.7);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            color: white;
            font-size: 28px;">
            🩺 Predicted Disease: <br>
            <b>{prediction}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Data Preview ----------
with st.expander("📊 View Dataset"):
    st.dataframe(df)

# ---------- Footer ----------
st.markdown(
    "<p style='text-align:center; color:white;'>Made by Anamika using Streamlit</p>",
    unsafe_allow_html=True
)