import streamlit as st
import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# --- Constants ---
MODEL_PATH = "logistic_regression.pkl"
LABEL_ENCODERS_PATH = "label_encoders_logistic_regression.pkl"
TRAINING_SCRIPT_NAME = "incident-classification-logisticregression.py"
CLEANED_NARRATIVE_COLUMN = 'Narrative_Cleaned'

# --- NLTK Downloads ---
def download_nltk_data():
    """Download necessary NLTK data if not present."""
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

# --- Text Processing ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text_advanced(text):
    """Advanced text cleaning function including lemmatization."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(tokens)

# --- Custom Transformer ---
# This class definition is required for joblib to load the pickled pipeline.
# It must match the definition used during model training.
class TextFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=5000):
        # The TfidfVectorizer instance will be replaced by the one loaded from the pickle file.
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit(self, X, y=None):
        # This method is not called during prediction, only for training.
        self.vectorizer.fit(X[CLEANED_NARRATIVE_COLUMN])
        return self

    def transform(self, X):
        return hstack([self.vectorizer.transform(X[CLEANED_NARRATIVE_COLUMN])])

# --- Model Loading ---
@st.cache_resource
def load_artifacts():
    """Load the trained model and label encoders."""
    try:
        model = joblib.load(MODEL_PATH)
        label_encoders = joblib.load(LABEL_ENCODERS_PATH)
        return model, label_encoders
    except FileNotFoundError as e:
        st.error(f"Error: Model or encoder file not found: {e.filename}")
        st.info(f"Please run '{TRAINING_SCRIPT_NAME}' to train and save the artifacts first.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading artifacts: {e}")
        st.stop()

# --- Streamlit App ---
st.set_page_config(
    page_title="Incident Classification Tool",
    page_icon="🛠️",
    layout="wide",
)

# Download NLTK data
download_nltk_data()

# Load model and encoders
model, label_encoders = load_artifacts()
if model and label_encoders:
    # The success message can be removed for a cleaner public-facing UI
    pass

# App Title
st.title("OSHA Severe Injury Classification Tool")
st.subheader("Powered by a Multi-Output Logistic Regression Model")

with st.expander("ℹ️ About this Tool"):
    st.write("""
    This tool uses a multi-output Logistic Regression model to automatically classify workplace incident reports.

    **Data Source:**
    The model was trained on a public dataset of severe injury reports from the U.S. Occupational Safety and Health Administration (OSHA), covering the period from January 1, 2015, to February 28, 2025. 
    Under federal regulation [29 CFR 1904.39](https://www.osha.gov/laws-regs/regulations/standardnumber/1904/1904.39), employers are required to report all severe work-related injuries, defined as an amputation, in-patient hospitalization, or loss of an eye. 
    The original data can be found on the [OSHA Severe Injury Reports page](https://www.osha.gov/severe-injury-reports).

    **Classification System:**
    The incident narratives are classified into four categories based on the **Occupational Injury and Illness Classification System (OIICS) Manual, Version 2.01**:
    - **Nature of Injury**: The primary physical characteristic of the injury (e.g., Fracture, Laceration).
    - **Part of Body**: The part of the body directly affected.
    - **Source of Injury**: The object, substance, or exposure that directly produced the injury.
    - **Event Type**: The manner in which the injury was produced (e.g., Fall, Slip, Trip).

    The goal of this application is to provide a quick, automated way to categorize incident narratives for analysis and reporting, demonstrating the capabilities of a machine learning model for this task.
    """)

with st.expander("🔬 Model Performance Analysis"):
    st.markdown("""
    ### Analysis of the OSHA Severe Injury Classification Model
    Welcome! This document provides a transparent look into the performance of the logistic regression model powering our new incident classification tool. This model was trained on over a decade of data (Jan 2015 - Feb 2025) from the U.S. Occupational Safety and Health Administration's (OSHA) Severe Injury Reports database.

    Understanding the source of this data is key. OSHA requires employers to report only specific, severe work-related injuries: amputations, in-patient hospitalizations, and loss of an eye. This scope directly influences the model's behavior, making it highly specialized. This report will walk you through its strengths and weaknesses within that context.

    Our goal is to automatically classify these severe incidents across four key dimensions based on the **Occupational Injury and Illness Classification System (OIICS) Manual, Version 2.01**:

    *   Nature of Injury
    *   Body Part Affected
    *   Type of Event
    *   Source of Injury

    ### Overall Performance
    At a high level, the model provides a solid baseline for classifying severe injuries. The overall accuracy—the percentage of incidents the model classifies correctly—shows its general effectiveness on this specific type of data.

    *   **Nature of Injury:** 65% Accuracy
    *   **Part of Body:** 64% Accuracy
    *   **Event Type:** 51% Accuracy
    *   **Source of Injury:** 47% Accuracy

    These metrics show that the model is most confident when determining the nature of an injury and the body part affected. It finds it more challenging to pinpoint the specific event and source, which are often more complex and nuanced. Given its specialized training data, the model's predictions should be seen as a helpful suggestion for categorizing severe incidents.

    ### Direct Result of the Data's Focus
    The model's performance is a direct reflection of the OSHA reporting requirements. It excels at identifying the very injuries it has seen most often—the ones that legally must be reported—while struggling with incidents that are less common in this severe-injury dataset.

    #### What the Model Does Well 👍
    The model is highly reliable when classifying the common, severe incidents that form the core of the OSHA database. Because the dataset is rich with examples of injuries like amputations and major fractures, the model has learned to identify them with high confidence. Here are examples of categories where the model excels:

    *   **Nature of Injury:** It is highly proficient with *Amputations, avulsions, enucleations* (F1-score of 0.75 on 995 cases) and *Fractures* (F1-score of 0.78 on 3060 cases). These are pillar categories for severe injury reporting.
    *   **Part of Body:** The model is exceptionally accurate with injuries to *Finger(s), fingernail(s)* (97% F1-score), a common site for amputations and severe crushing injuries.
    *   **Event Type:** It reliably identifies common industrial accidents like *Caught in running equipment or machinery* (74% F1-score) and *Fall on same level due to slipping* (82% F1-score), both of which frequently lead to hospitalization.
    *   **Source of Injury:** It's very good at recognizing incidents involving *Industrial vehicles* (79% F1-score) and *Roofs* (78% F1-score), which are common sources of severe accidents.

    #### What the Model Struggles With 👎
    The model's primary weakness is classifying injuries that, while potentially serious, are less frequently reported to OSHA or are secondary to a primary severe injury. An incident like a minor scratch would not be in the dataset unless it was part of a larger event that led to hospitalization. For many of these less-represented categories, the model scored a 0.00 F1-score, meaning it failed to correctly identify any of them. Examples include:

    *   **Nature of Injury:** *Anxiety, stress* (1 case) and *Burns and corrosions* (2 cases). These may not always result in immediate hospitalization and are thus rare in the data.
    *   **Part of Body:** Vague categories like *Arm(s), n.e.c.* (1 case) are difficult for the model to learn.
    *   **Event Type:** *Bites and stings* (1 case) are rarely severe enough to meet OSHA reporting criteria.
    *   **Source of Injury:** Specific machinery like *Agricultural and garden machinery* (1 case) is underrepresented.

    **Key Takeaway:** You can trust the model's predictions for incidents that clearly fall under the OSHA "severe" definition. For injuries that are less common in this dataset, its output should be reviewed carefully.

    ### A Note for Data Enthusiasts: Technical Breakdown 🤓
    For those interested in the technical details, a look at the precision, recall, and different averaging methods in the report reveals a classic case of an imbalanced dataset.

    **Macro Avg vs. Weighted Avg:** The most telling sign is the large gap between the "macro average" and "weighted average" for the F1-score. For the **Nature of Injury** model, the macro average F1-score is 0.33, while the weighted average is 0.65.
    *   The **macro average** treats every class equally. Because so many rare classes have a score of 0, this average is low.
    *   The **weighted average** gives more importance to classes with more samples. Since the model performs well on high-sample classes (like "Fractures"), this average is much higher. This gap confirms that the model's performance is driven by a few dominant classes.

    **Precision vs. Recall Trade-offs:** The report shows interesting trade-offs. For example, in the **Nature of Injury** report, look at *Blisters*:
    *   **Recall is 1.00:** The model correctly identified 100% of all true "Blister" incidents in the test set. It misses none.
    *   **Precision is 0.40:** However, when the model predicted "Blister," it was only correct 40% of the time. This means it incorrectly labeled other injuries as blisters quite often. This is a "high recall, low precision" scenario.
    """)

# Define target columns (must match the order in which they were trained)
if label_encoders:
    target_columns = list(label_encoders.keys())

# Classification Section
st.subheader("Incident Classification")
new_description = st.text_area("Enter a new incident description:", height=150)

if st.button("Classify Incident"):
    if new_description.strip():
        # Prepare the input for the model
        processed_description = clean_text_advanced(new_description)
        input_df = pd.DataFrame({CLEANED_NARRATIVE_COLUMN: [processed_description]})

        # Make predictions using the pipeline
        try:
            predictions = model.predict(input_df)
            probabilities = model.predict_proba(input_df)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.stop()

        st.write("### Classification Results:")
        results_data = []
        for i, target_col in enumerate(target_columns):
            predicted_label_encoded = predictions[0, i]
            confidence = probabilities[i][0, predicted_label_encoded]

            le = label_encoders[target_col]
            predicted_label = le.inverse_transform([predicted_label_encoded])[0]
            category = target_col.replace('_Generalized', '').replace('_', ' ')
            
            results_data.append({
                "Category": category,
                "Prediction": predicted_label,
                "Confidence": confidence * 100
            })

        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df,
                     use_container_width=True,
                     column_config={
                         "Category": st.column_config.TextColumn("Category", help="The classification category.", width="medium"),
                         "Prediction": st.column_config.TextColumn("Prediction", help="The model's predicted label for the category.", width="large"),
                         "Confidence": st.column_config.ProgressColumn(
                             "Confidence",
                             help="The model's confidence in the prediction (0-100%).",
                             format="%.1f%%",
                             min_value=0,
                             max_value=100,
                         ),
                     },
                     hide_index=True)
    else:
        st.error("Please enter a description to classify.")

st.markdown("---")
st.warning(
    "**Disclaimer:** This is a proof-of-concept tool. The predictions are generated by a machine learning model "
    "and may not be 100% accurate. It should not be used as a substitute for professional safety analysis or "
    "official reporting."
)