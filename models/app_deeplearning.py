import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack

# Download NLTK stop words if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Define a text cleaning function
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Custom transformer for preprocessing and feature combination
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def fit(self, X, y=None):
        self.vectorizer.fit(X['Narrative_Cleaned'])
        return self

    def transform(self, X):
        text_features = self.vectorizer.transform(X['Narrative_Cleaned'])
        return hstack([text_features])

# Set the browser tab title
st.set_page_config(
    page_title="Incident Classification Tool (Deep Learning)",
    page_icon="🧠",
    layout="wide",
)

# Streamlit App Title
st.title("Incident Classification Tool (Deep Learning)")
st.write("Classify incident descriptions using a pre-trained Deep Learning model.")

# Load pre-trained Deep Learning model and other necessary objects
@st.cache_resource
def load_model_artifacts():
    try:
        model = tf.keras.models.load_model('deeplearning_model')
        feature_transformer = joblib.load('deeplearning_feature_transformer.pkl')
        label_encoders = joblib.load('deeplearning_label_encoders.pkl')
        return model, feature_transformer, label_encoders
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

model, feature_transformer, label_encoders = load_model_artifacts()

if model and feature_transformer and label_encoders:
    st.success("Deep Learning model and artifacts loaded successfully!")
else:
    st.error("Could not load model artifacts. Please ensure the files 'deeplearning_model', 'deeplearning_feature_transformer.pkl', and 'deeplearning_label_encoders.pkl' are present.")
    st.stop()

# Define target columns
target_columns = list(label_encoders.keys())

# Classification Section
st.subheader("Incident Classification")
new_description = st.text_area("Enter a new incident description:")

if st.button("Classify Incident"):
    if new_description.strip():
        # Prepare the input for the model
        processed_description = clean_text(new_description)
        input_df = pd.DataFrame({'Narrative_Cleaned': [processed_description]})

        # Transform the input using the loaded feature_transformer
        input_transformed = feature_transformer.transform(input_df)
        
        # Convert to dense array for the model
        input_dense = input_transformed.toarray()

        # Make predictions
        predictions_proba = model.predict(input_dense)
        
        # Get the predicted classes by finding the index of the max probability
        predictions = [np.argmax(proba, axis=1) for proba in predictions_proba]

        st.write("### Classification Results:")
        for i, target_col in enumerate(target_columns):
            predicted_label_encoded = predictions[i][0]
            predicted_label = label_encoders[target_col].inverse_transform([predicted_label_encoded])[0]
            st.write(f"- **{target_col.replace('_', ' ')}:** {predicted_label}")
    else:
        st.error("Please enter a description to classify.")