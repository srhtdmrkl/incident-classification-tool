import streamlit as st
import pandas as pd
import joblib
import re
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import warnings

# --- Main Streamlit App ---
st.set_page_config(
    page_title="Incident Classification Tool",
    page_icon="🛠️",
    layout="wide",
)

# Suppress all warnings
warnings.filterwarnings('ignore')

# --- Constants ---
LOGISTIC_REGRESSION_MODEL_PATH = "logistic_regression.pkl"
LOGISTIC_REGRESSION_LABEL_ENCODERS_PATH = "label_encoders_logistic_regression.pkl"
XGBOOST_MODELS_PATH = "xgboost_models_dict.pkl"
XGBOOST_LABEL_ENCODERS_PATH = "label_encoders_xgboost.pkl"
XGBOOST_LABEL_MAPPINGS_PATH = "label_mappings_xgboost.pkl"
BERT_MODEL_PATH = "bert_multi_label_model.pth"
BERT_TOKENIZER_PATH = "bert_tokenizer"
BERT_LABEL_ENCODERS_PATH = "bert_label_encoders.pkl"
CLEANED_NARRATIVE_COLUMN = 'Narrative_Cleaned'
TRAINING_SCRIPT_NAME = "incident-classification-all-models.py"

# --- NLTK Downloads ---
@st.cache_resource
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

# --- Custom Transformers and Functions ---
# This class is required for joblib to load the pickled LR and XGBoost pipelines.
class TextFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
    def fit(self, X, y=None):
        self.vectorizer.fit(X[CLEANED_NARRATIVE_COLUMN])
        return self
    def transform(self, X):
        return hstack([self.vectorizer.transform(X[CLEANED_NARRATIVE_COLUMN])])

# Text cleaning for Logistic Regression and XGBoost
download_nltk_data()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def clean_text_advanced(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(tokens)

# BERT Model Architecture (required to load the model)
class BERTMultiLabelClassifier(torch.nn.Module):
    def __init__(self, n_classes_dict):
        super(BERTMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(p=0.3)
        self.classifiers = torch.nn.ModuleDict({
            col: torch.nn.Linear(self.bert.config.hidden_size, n_classes)
            for col, n_classes in n_classes_dict.items()
        })
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(output.pooler_output)
        logits = {col: classifier(pooled_output) for col, classifier in self.classifiers.items()}
        return logits

# Text cleaning for BERT
def clean_text_for_bert(text):
    return str(text)

# --- Model Loading ---
@st.cache_resource
def load_logistic_regression_artifacts():
    try:
        model = joblib.load(LOGISTIC_REGRESSION_MODEL_PATH)
        label_encoders = joblib.load(LOGISTIC_REGRESSION_LABEL_ENCODERS_PATH)
        return model, label_encoders
    except FileNotFoundError as e:
        st.error(f"Error: Logistic Regression model file not found: {e.filename}")
        st.info(f"Please run the training script to train and save the artifacts first.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading Logistic Regression artifacts: {e}")
        return None, None

@st.cache_resource
def load_xgboost_artifacts():
    try:
        models = joblib.load(XGBOOST_MODELS_PATH)
        label_encoders = joblib.load(XGBOOST_LABEL_ENCODERS_PATH)
        label_mappings = joblib.load(XGBOOST_LABEL_MAPPINGS_PATH)
        return models, label_encoders, label_mappings
    except FileNotFoundError as e:
        st.error(f"Error: XGBoost model files not found: {e.filename}")
        st.info(f"Please run the training script to train and save the artifacts first.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading XGBoost artifacts: {e}")
        return None, None, None

@st.cache_resource
def load_bert_artifacts():
    try:
        label_encoder_classes = joblib.load(BERT_LABEL_ENCODERS_PATH)
        
        label_encoders = {}
        for col, classes in label_encoder_classes.items():
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            label_encoders[col] = le
            
        n_classes_dict = {col: len(le.classes_) for col, le in label_encoders.items()}

        model = BERTMultiLabelClassifier(n_classes_dict)
        model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()

        tokenizer = BertTokenizer.from_pretrained(BERT_TOKENIZER_PATH)
        return model, tokenizer, label_encoders
    except FileNotFoundError as e:
        st.error(f"Error: BERT model files not found: {e.filename}")
        st.info(f"Please run the training script to train and save the artifacts first.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading BERT artifacts: {e}")
        return None, None, None

st.title("OSHA Injury Classification Tool")
st.subheader("Compare Classification Models")

# --- About this Tool ---
with st.expander("ℹ️ About this Tool"):
    st.write("""
    This tool uses different machine learning models to automatically classify workplace injuries. You can choose between **Logistic Regression**, 
    **XGBoost**, and **BERT** models to see how they perform on the same incident description. 

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

# --- Model Performance Analysis ---
with st.expander("🔬 Model Performance Analysis"):
    st.markdown("""
### Analysis of the OSHA Injury Classification Models
Welcome! This document provides a transparent, comparative look into the performance of the three models powering our incident classification tool: Logistic Regression, XGBoost, and a BERT-based classifier. These models were trained on over a decade of data (Jan 2015 - Feb 2025) from the U.S. Occupational Safety and Health Administration's (OSHA) Severe Injury Reports database.
Understanding the data's origin is key. OSHA requires employers to report only specific, severe work-related injuries: amputations, in-patient hospitalizations, and loss of an eye. This scope makes our dataset highly specialized. To better handle the numerous rare incident types, we performed feature engineering by consolidating classifications into the 3rd-level OIICS hierarchy, grouping more specific sub-categories into their broader parent classes. This report will walk you through the strengths and weaknesses of each model within this context.

Our goal is to automatically classify severe incidents across four key dimensions based on the **Occupational Injury and Illness Classification System (OIICS) Manual, Version 2.01**:

* Nature of Injury
* Body Part Affected
* Type of Event
* Source of Injury

---
### Model Showdown: A Comparative Overview
While our initial Logistic Regression model provided a solid baseline, we introduced XGBoost and BERT to explore more advanced architectures. The **weighted F1-score**, which balances precision and recall while accounting for class imbalance, is our primary metric for comparison.

| Model               | Nature of Injury (F1) | Part of Body (F1) | Event Type (F1) | Source of Injury (F1) |
| ------------------- | :-------------------: | :---------------: | :-------------: | :-------------------: |
| 🥇 **BERT** |      **0.75**      |       0.72      |   **0.59**  |         0.35     |
| 🥈 **XGBoost** |      0.67         |    **0.70**    |   0.54    |      **0.50**     |
| 🥉 **Logistic Reg.** |      0.65       |       0.62    |   0.52     |         0.48       |

The results show a clear performance hierarchy. The **BERT** model, with its deep understanding of language context, excels at deciphering the nuanced descriptions in the `Nature of Injury` and `Event Type` fields. However, **XGBoost** proves to be highly effective for the more categorical `Part of Body` and `Source of Injury` classifications. Logistic Regression remains a consistent, albeit less powerful, baseline.

---
### 🔬 Case Studies
Aggregate metrics tell a story about overall performance, but individual cases can reveal fascinating nuances about how each model interprets information.

#### Case Study 1: The Sprained Wrist (Simple Model Wins on Specificity)
> "A prep cook, slipped and fell in the kitchen area of a coffee shop, resulting in a sprained wrist. The incident occurred when cook stepped on a wet floor near the dishwashing station, causing him to lose balance and fall, landing on his right wrist."

| Category         | Logistic Regression                       | XGBoost                               | BERT                                       |
| ---------------- | ----------------------------------------- | ------------------------------------- | ------------------------------------------ |
| **Nature** | ✅ **Sprains, strains, tears (21%)** | ❌ Fractures (24%)                    | ❌ Nonspecified injuries... (44%)          |
| **Part of Body** | ✅ Wrist(s) (57%)                         | ✅ Wrist(s) (89%)                     | ✅ Wrist(s) (68%)                          |
| **Event Type** | ✅ Fall on same level due to slipping (12%) | ✅ Fall on same level due to slipping (71%) | ✅ Fall on same level due to slipping (88%)  |
| **Source** | ✅ Floors (4%)                            | ✅ Floors (71%)                       | ✅ Floors (70%)                            |

In this first case, the **Logistic Regression** model was the only one to accurately identify the `Nature of Injury`. Its simpler approach likely created a strong, direct association with the explicit word "sprained," a detail the more complex models overlooked by generalizing from similar incidents.

#### Case Study 2: The Carpenter's Laceration (Complex Models Show Their Power)
> "A carpenter, sustained a minor laceration on his left index finger while using a circular saw at construction site. The incident occurred when the saw blade caught on a piece of wood, causing a momentary loss of control and a superficial cut to carpenter's finger."

| Category         | Logistic Regression                       | XGBoost                               | BERT                                       |
| ---------------- | ----------------------------------------- | ------------------------------------- | ------------------------------------------ |
| **Nature** | ✅ Cuts, lacerations (49%)                  | ✅ Cuts, lacerations (79%)            | ✅ Cuts, lacerations (92%)                 |
| **Part of Body** | ✅ Finger(s), fingernail(s) (62%)         | ✅ Finger(s), fingernail(s) (94%)     | ✅ Finger(s), fingernail(s) (90%)         |
| **Event Type** | ❌ Struck, caught... (0.2%)               | ✅ Injured by handheld object... (25%) | ❌ Struck, caught... (16%)               |
| **Source** | ✅ Cutting handtools—**powered** (17%)    | ✅ Cutting handtools—**powered** (51%) | ❌ Cutting handtools—**nonpowered** (15%) |

This example tells a different story. For the straightforward `Nature` and `Part of Body`, all models were correct, with BERT and XGBoost showing much higher confidence. More revealingly, XGBoost selected the most logical `Event Type`, and it correctly identified the `Source` as a **powered** tool—a critical detail that BERT got wrong.

---
### A Note for Data Enthusiasts: Technical Breakdown
A look at the precision, recall, and averaging methods reveals a classic case of an imbalanced dataset, even after our feature engineering efforts.

**Macro Avg vs. Weighted Avg:** The most telling sign is the large gap between the "macro" and "weighted" averages. Using the top-performing **BERT model for Nature of Injury** as an example:
* The **macro average F1-score is 0.26**. This average treats every class equally. Because so many rare classes have an F1-score of 0, this average is pulled down significantly.
* The **weighted average F1-score is 0.75**. This average is weighted by the number of samples in each class. Since the model performs exceptionally well on high-sample classes (like "Fractures"), this average is much higher. This gap is a clear indicator that the model's high-level performance is driven by a few dominant classes.
""")
            
# --- Model Selection ---
st.markdown("---")
st.subheader("Model Selection")
selected_model = st.radio("Choose a model:", ('Logistic Regression', 'XGBoost', 'BERT'), horizontal=True)

# Load artifacts based on selection
if selected_model == 'Logistic Regression':
    model, label_encoders = load_logistic_regression_artifacts()
    xgboost_models, xgboost_encoders, xgboost_mappings = None, None, None
    bert_model, bert_tokenizer, bert_encoders = None, None, None
elif selected_model == 'XGBoost':
    xgboost_models, xgboost_encoders, xgboost_mappings = load_xgboost_artifacts()
    model, label_encoders = None, None
    bert_model, bert_tokenizer, bert_encoders = None, None, None
else: # BERT
    bert_model, bert_tokenizer, bert_encoders = load_bert_artifacts()
    model, label_encoders = None, None
    xgboost_models, xgboost_encoders, xgboost_mappings = None, None, None

# --- Classification Section ---
st.markdown("---")
st.subheader("Incident Classification")
new_description = st.text_area("Enter a new incident description:", height=150)

if st.button("Classify Incident"):
    if not new_description.strip():
        st.error("Please enter a description to classify.")
    else:
        results_data = []

        if selected_model == 'Logistic Regression' and model and label_encoders:
            try:
                processed_description = clean_text_advanced(new_description)
                input_df = pd.DataFrame({CLEANED_NARRATIVE_COLUMN: [processed_description]})
                
                predictions = model.predict(input_df)
                probabilities = model.predict_proba(input_df)

                target_columns = list(label_encoders.keys())
                for i, target_col in enumerate(target_columns):
                    predicted_label_encoded = predictions[0, i]
                    confidence = probabilities[i][0, predicted_label_encoded]
                    le = label_encoders[target_col]
                    predicted_label = le.inverse_transform([predicted_label_encoded])[0]
                    category = target_col.replace('_Generalized', '').replace('_', ' ')
                    results_data.append({"Category": category, "Prediction": predicted_label, "Confidence": confidence * 100})
            except Exception as e:
                st.error(f"An error occurred with Logistic Regression model: {e}")

        elif selected_model == 'XGBoost' and xgboost_models and xgboost_encoders:
            try:
                processed_description = clean_text_advanced(new_description)
                input_df = pd.DataFrame({'Narrative_Cleaned': [processed_description]})
                
                target_columns = list(xgboost_encoders.keys())
                for col in target_columns:
                    current_model = xgboost_models[col]
                    label_encoder = xgboost_encoders[col]
                    mapping = xgboost_mappings[col]
                    inverse_mapping = {v: k for k, v in mapping.items()}

                    pred_sequential = current_model.predict(input_df)[0]
                    probabilities = current_model.predict_proba(input_df)
                    confidence = probabilities[0, pred_sequential]
                    pred_original_encoding = inverse_mapping[pred_sequential]
                    final_label = label_encoder.inverse_transform([pred_original_encoding])[0]
                    
                    category = col.replace('_Generalized', '').replace('_', ' ')
                    results_data.append({"Category": category, "Prediction": final_label, "Confidence": confidence * 100})
            except Exception as e:
                st.error(f"An error occurred with XGBoost model: {e}")

        elif selected_model == 'BERT' and bert_model and bert_tokenizer and bert_encoders:
            try:
                cleaned_description = clean_text_for_bert(new_description)
                encoding = bert_tokenizer.encode_plus(
                    cleaned_description, add_special_tokens=True, max_length=256,
                    return_token_type_ids=False, padding='max_length', truncation=True,
                    return_attention_mask=True, return_tensors='pt',
                )
                input_ids = encoding['input_ids']
                attention_mask = encoding['attention_mask']

                with torch.no_grad():
                    outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)

                target_columns = list(bert_encoders.keys())
                for col in target_columns:
                    logits = outputs[col]
                    probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
                    prediction_index = torch.argmax(logits, dim=1).cpu().numpy()[0]
                    
                    confidence = probabilities[prediction_index]
                    predicted_label = bert_encoders[col].inverse_transform([prediction_index])[0]
                    category = col.replace('_Generalized', '').replace('_', ' ')
                    results_data.append({"Category": category, "Prediction": predicted_label, "Confidence": confidence * 100})
            except Exception as e:
                st.error(f"An error occurred with BERT model: {e}")
        
        if results_data:
            st.write("### Classification Results:")
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df,
                         use_container_width=True,
                         column_config={
                             "Category": st.column_config.TextColumn("Category", width="medium"),
                             "Prediction": st.column_config.TextColumn("Prediction", width="large"),
                             "Confidence": st.column_config.ProgressColumn(
                                 "Confidence", format="%.1f%%", min_value=0, max_value=100,
                             ),
                         },
                         hide_index=True)
        else:
            st.warning("No results to display. Please check for errors above.")
    
st.markdown("---")
st.warning(
    "**Disclaimer:** This is a proof-of-concept tool. The predictions are generated by a machine learning model "
    "and may not be 100% accurate. It should not be used as a substitute for professional safety analysis or "
    "official reporting."
)
