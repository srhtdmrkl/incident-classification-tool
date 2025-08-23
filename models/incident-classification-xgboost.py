import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re
import nltk
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import numpy as np # Import numpy if not already imported
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import sys
import joblib # Import joblib
from nltk.stem import WordNetLemmatizer


# Download NLTK stop words if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    
nltk.download('wordnet')

# Load all mapping sheets from the Excel file at once
excel_path = 'data/oiics_201_code_list.xlsx'
sheet_mappings = pd.read_excel(excel_path, sheet_name=['Nature', 'Part', 'Event', 'Source'])

# Create mappings from the loaded sheets
nature_code_mapping = sheet_mappings['Nature'].set_index('CASE_CODE')['CASE_CODE_TITLE'].to_dict()
part_of_body_mapping = sheet_mappings['Part'].set_index('CASE_CODE')['CASE_CODE_TITLE'].to_dict()
event_code_mapping = sheet_mappings['Event'].set_index('CASE_CODE')['CASE_CODE_TITLE'].to_dict()
source_code_mapping = sheet_mappings['Source'].set_index('CASE_CODE')['CASE_CODE_TITLE'].to_dict()

# Load the data
try:
    df = pd.read_excel('data/January2015toApril2024.xlsx')
except FileNotFoundError:
    print("Error: The file 'data/January2015toApril2024.xlsx' was not found.")
    print("Please make sure the file path is correct.")
    exit()

# Drop rows with missing values
df.dropna(subset=['Final Narrative', 'NatureTitle', 'Part of Body Title', 'EventTitle', 'SourceTitle', 'Secondary Source Title', 'Amputation', 'Hospitalized'], inplace=True)

# --- Start of refined text cleaning ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def clean_text_advanced(text):
    text = str(text).lower()  # Convert to lowercase
    # Use more specific regex to preserve parts of codes
    text = re.sub(r'[^a-z0-9\s-]', '', text) 
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]  # Remove stop words and single characters
    return ' '.join(tokens)

# Apply advanced text cleaning to the narrative
df['Narrative_Cleaned'] = df['Final Narrative'].apply(clean_text_advanced)
# --- End of refined text cleaning ---

# --- Start of refined feature engineering for NatureTitle ---
def map_nature_code_refined(code):
    code_str = str(int(code))
    
    # Trim trailing zeros
    while code_str.endswith('0') and len(code_str) > 1:
        code_str = code_str[:-1]

    if len(code_str) >= 4:
        general_code = code_str[:3]
    else:
        general_code = code_str
    
    return nature_code_mapping.get(int(general_code), 'Other')

# Create the new, generalized NatureTitle feature
df['NatureTitle_Generalized'] = df['Nature'].apply(map_nature_code_refined)
# --- End of refined feature engineering for NatureTitle ---

# --- Start of refined feature engineering for Part of Body ---
def map_part_of_body_code_refined(code):
    code_str = str(int(code))
    
    # Trim trailing zeros
    while code_str.endswith('0') and len(code_str) > 1:
        code_str = code_str[:-1]

    if len(code_str) >= 4:
        general_code = code_str[:3]
    else:
        general_code = code_str
    
    return part_of_body_mapping.get(int(general_code), 'Other')

# Create the new, generalized Part of Body feature
df['PartofBody_Generalized'] = df['Part of Body'].apply(map_part_of_body_code_refined)
# --- End of refined feature engineering for Part of Body ---


# --- Start of new, refined feature engineering for EventTitle ---
def map_event_code_refined(code):
    code_str = str(int(code)) # Ensure the code is a string representation of an integer
    
    # Trim trailing zeros
    while code_str.endswith('0') and len(code_str) > 1:
        code_str = code_str[:-1]

    if len(code_str) >= 4:
        # For 4+ digit codes, take the first 3 digits
        general_code = code_str[:3]
    else:
        # For 1, 2, or 3-digit codes, use the code as is
        general_code = code_str
    
    # Map the general code to its title, defaulting to 'Other' if not found
    return event_code_mapping.get(int(general_code), 'Other')

# Create the new, generalized EventTitle feature using the refined function
df['EventTitle_Generalized'] = df['Event'].apply(map_event_code_refined)
# --- End of new, refined feature engineering for EventTitle ---

# --- Start of refined feature engineering for SourceTitle ---
def map_source_code_refined(code):
    code_str = str(int(code))
    
    # Trim trailing zeros
    while code_str.endswith('0') and len(code_str) > 1:
        code_str = code_str[:-1]

    if len(code_str) >= 4:
        general_code = code_str[:3]
    else:
        general_code = code_str
    
    return source_code_mapping.get(int(general_code), 'Other')

# Create the new, generalized SourceTitle feature
df['SourceTitle_Generalized'] = df['Source'].apply(map_source_code_refined)
# --- End of refined feature engineering for SourceTitle ---


# Select features and targets. The old 'SourceTitle' is now replaced with the new one.
features = ['Narrative_Cleaned','NatureTitle_Generalized', 'PartofBody_Generalized', 'EventTitle_Generalized', 'SourceTitle_Generalized']
targets = ['NatureTitle_Generalized', 'PartofBody_Generalized', 'EventTitle_Generalized', 'SourceTitle_Generalized']

# Separate features (X) and targets (y)
X = df[features]
y_str = df[targets].copy() # Keep string labels for now

# Create a mask for rows to keep (where all target classes appear at least 3 times)
# This is done on the string labels before any encoding to ensure that after
# filtering, the LabelEncoder will create a contiguous set of integers.
mask = pd.Series(True, index=y_str.index)
for col in targets:
    value_counts = y_str[col].value_counts()
    rare_classes = value_counts[value_counts < 5].index
    if len(rare_classes) > 0:
        # Update the mask to exclude rows with rare classes in the current column
        mask &= ~y_str[col].isin(rare_classes)

# Apply the mask to both features and string-based targets
X_filtered = X[mask]
y_filtered_str = y_str[mask]

# Now, encode the filtered target labels. This ensures contiguous labels.
y = pd.DataFrame(index=y_filtered_str.index)
label_encoders = {}
for col in targets:
    le = LabelEncoder()
    # Use fit_transform on the filtered data
    y[col] = le.fit_transform(y_filtered_str[col])
    label_encoders[col] = le

# Split the data into training and testing sets.
# The redundant post-split filtering and second split are removed.
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)

# Create a custom transformer for preprocessing and feature combination
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def fit(self, X, y=None):
        self.vectorizer.fit(X['Narrative_Cleaned'])
        return self

    def transform(self, X):
        text_features = self.vectorizer.transform(X['Narrative_Cleaned'])
        return hstack([text_features])

# Create the final pipeline with XGBoost
pipeline = Pipeline([
    ('features', CombinedAttributesAdder()), # Use the custom wrapper
    ('clf', MultiOutputClassifier(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')))
])

# Train the model with the corrected y_train format
print("Training the model...")
pipeline.fit(X_train, y_train.values.astype(int))
print("Model training complete.")
print("-" * 30)

# Save the trained pipeline and label encoders
joblib.dump(pipeline, 'xgboost_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Save the vectorizer from the pipeline
joblib.dump(pipeline.named_steps['features'].vectorizer, 'vectorizer.pkl')
print("Trained model and label encoders saved.")

# Make predictions
print("Making predictions...")
y_pred = pipeline.predict(X_test)
print("Predictions complete.")
print("-" * 30)

# Evaluate the model for each target and output to a file
print("Evaluating model performance...")
with open("classification_reports_xgboost.txt", "w") as f:
    for i, target_col in enumerate(targets):
        f.write(f"\n--- Classification Report for {target_col} ---\\n")
        
        y_test_labels = label_encoders[target_col].inverse_transform(y_test.iloc[:, i].values.astype(int))
        y_pred_labels = label_encoders[target_col].inverse_transform(y_pred[:, i].astype(int))

        report = classification_report(y_test_labels, y_pred_labels, zero_division=0)
        f.write(report + "\\n")
print("Classification reports have been written to classification_reports_xgboost.txt")
