import streamlit as st
import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack, csr_matrix
from nltk.stem import WordNetLemmatizer

# Download NLTK stop words and wordnet if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

# Define the advanced text cleaning function (from incident-classification-xgboost.py)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def clean_text_advanced(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(tokens)

# Custom transformer for preprocessing and feature combination (from incident-classification-xgboost.py)
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, X, y=None):
        # The vectorizer is already fitted
        return self

    def transform(self, X):
        text_features = self.vectorizer.transform(X['Narrative_Cleaned'])
        return hstack([text_features])

# Set the browser tab title and other configuration
st.set_page_config(
    page_title="Incident Classification Tool",
    page_icon="🛠️",
    layout="wide",
)

# Streamlit App Title
st.title("Incident Classification Tool (XGBoost)")
st.write("Classify incident descriptions using a pre-trained XGBoost model.")

# Initialize session state variables
if "model" not in st.session_state:
    st.session_state["model"] = None
if "label_encoders" not in st.session_state:
    st.session_state["label_encoders"] = None
if "vectorizer" not in st.session_state:
    st.session_state["vectorizer"] = None

# Load pre-trained XGBoost model, label encoders, and vectorizer
try:
    st.session_state["model"] = joblib.load("xgboost_model.pkl")
    st.session_state["label_encoders"] = joblib.load("label_encoders.pkl")
    st.session_state["vectorizer"] = joblib.load("vectorizer.pkl")
    st.success("XGBoost model, label encoders, and vectorizer loaded successfully!")
except FileNotFoundError as e:
    st.error(f"Error: Model file not found: {e.filename}")
    st.info("Please run 'incident-classification-xgboost.py' to train and save the model first.")
    st.stop()

# Define target columns (must match the order in which they were trained)
target_columns = list(st.session_state["label_encoders"].keys())

# Classification Section
st.subheader("Incident Classification")
new_description = st.text_area("Enter a new incident description:")

if st.button("Classify Incident"):
    if new_description.strip():
        # Prepare the input for the model
        processed_description = clean_text_advanced(new_description)
        input_df = pd.DataFrame({
            'Narrative_Cleaned': [processed_description]
        })

        # Make predictions
        predictions = st.session_state["model"].predict(input_df)

        st.write("### Classification Results:")
        for i, target_col in enumerate(target_columns):
            predicted_label_encoded = predictions[0, i] # Get the prediction for the current target
            # Inverse transform to get the original label
            predicted_label = st.session_state["label_encoders"][target_col].inverse_transform([predicted_label_encoded])[0]
            st.write(f"- **{target_col.replace('_', ' ')}:** {predicted_label}")
    else:
        st.error("Please enter a description to classify.")