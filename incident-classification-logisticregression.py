import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re
import nltk
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from nltk.stem import WordNetLemmatizer

# --- Constants ---
EXCEL_PATH = 'data/oiics_201_code_list.xlsx'
DATA_PATH = 'data/January2015toFebruary2025.xlsx'
MODEL_SAVE_PATH = 'logistic_regression.pkl'
LABEL_ENCODERS_SAVE_PATH = 'label_encoders_logistic_regression.pkl'
REPORT_SAVE_PATH = 'classification_reports_logistic_regression.txt'
TARGET_COLUMNS = ['Nature of Injury_Generalized', 'Part of Body_Generalized', 'Event Type_Generalized', 'Source of Injury_Generalized']
NARRATIVE_COLUMN = 'Final Narrative'
CLEANED_NARRATIVE_COLUMN = 'Narrative_Cleaned'
REQUIRED_COLUMNS = [NARRATIVE_COLUMN, 'NatureTitle', 'Part of Body Title', 'EventTitle', 'SourceTitle', 'Secondary Source Title']

# --- NLTK Downloads ---
def download_nltk_data():
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

# --- Data Loading Functions ---
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please make sure the file path is correct.")
        exit()

def load_mapping_sheets(excel_path):
    sheet_mappings = pd.read_excel(excel_path, sheet_name=['Nature', 'Part', 'Event', 'Source'])
    nature_code_mapping = sheet_mappings['Nature'].set_index('CASE_CODE')['CASE_CODE_TITLE'].to_dict()
    part_of_body_mapping = sheet_mappings['Part'].set_index('CASE_CODE')['CASE_CODE_TITLE'].to_dict()
    event_code_mapping = sheet_mappings['Event'].set_index('CASE_CODE')['CASE_CODE_TITLE'].to_dict()
    source_code_mapping = sheet_mappings['Source'].set_index('CASE_CODE')['CASE_CODE_TITLE'].to_dict()
    return nature_code_mapping, part_of_body_mapping, event_code_mapping, source_code_mapping

# --- Text Cleaning and Feature Engineering Functions ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text_advanced(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(tokens)

def map_generalized_code(code, mapping):
    code_str = str(int(code))
    while code_str.endswith('0') and len(code_str) > 1:
        code_str = code_str[:-1]
    
    if len(code_str) >= 4:
        general_code = code_str[:3]
    else:
        general_code = code_str
    
    return mapping.get(int(general_code), 'Other')

# --- Custom Transformer ---
class TextFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit(self, X, y=None):
        self.vectorizer.fit(X[CLEANED_NARRATIVE_COLUMN])
        return self

    def transform(self, X):
        text_features = self.vectorizer.transform(X[CLEANED_NARRATIVE_COLUMN])
        return hstack([text_features])

# --- Main Execution ---
if __name__ == "__main__":
    download_nltk_data()

    # Load data and mappings
    df = load_data(DATA_PATH)
    nature_map, part_map, event_map, source_map = load_mapping_sheets(EXCEL_PATH)

    # Drop rows with missing values
    df.dropna(subset=REQUIRED_COLUMNS, inplace=True)

    # Apply text cleaning
    df[CLEANED_NARRATIVE_COLUMN] = df[NARRATIVE_COLUMN].apply(clean_text_advanced)

    # Apply generalized code mapping
    df['Nature of Injury_Generalized'] = df['Nature'].apply(lambda x: map_generalized_code(x, nature_map))
    df['Part of Body_Generalized'] = df['Part of Body'].apply(lambda x: map_generalized_code(x, part_map))
    df['Event Type_Generalized'] = df['Event'].apply(lambda x: map_generalized_code(x, event_map))
    df['Source of Injury_Generalized'] = df['Source'].apply(lambda x: map_generalized_code(x, source_map))

    # Select features and targets
    features = [CLEANED_NARRATIVE_COLUMN, 'Nature of Injury_Generalized', 'Part of Body_Generalized', 'Event Type_Generalized', 'Source of Injury_Generalized']
    
    X = df[features]
    y_str = df[TARGET_COLUMNS].copy()

    # Filter out rare classes
    mask = pd.Series(True, index=y_str.index)
    for col in TARGET_COLUMNS:
        value_counts = y_str[col].value_counts()
        rare_classes = value_counts[value_counts < 2].index
        if len(rare_classes) > 0:
            mask &= ~y_str[col].isin(rare_classes)
    
    X_filtered = X[mask]
    y_filtered_str = y_str[mask]

    # Encode labels
    y = pd.DataFrame(index=y_filtered_str.index)
    label_encoders = {}
    for col in TARGET_COLUMNS:
        le = LabelEncoder()
        y[col] = le.fit_transform(y_filtered_str[col])
        label_encoders[col] = le

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)

    # Create and train pipeline
    pipeline = Pipeline([
        ('features', TextFeatureTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)))
    ])

    print("Training the model...")
    pipeline.fit(X_train, y_train.values.astype(int))
    print("Model training complete.")
    print("-" * 30)

    # Save model and label encoders
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    joblib.dump(label_encoders, LABEL_ENCODERS_SAVE_PATH)
    print("Trained model and label encoders saved.")

    # Make predictions
    print("Making predictions...")
    y_pred = pipeline.predict(X_test)
    print("Predictions complete.")
    print("-" * 30)

    # Evaluate and save reports
    print("Evaluating model performance...")
    with open(REPORT_SAVE_PATH, "w") as f:
        for i, target_col in enumerate(TARGET_COLUMNS):
            f.write(f"\n--- Classification Report for {target_col} ---\n")
            
            y_test_labels = label_encoders[target_col].inverse_transform(y_test.iloc[:, i].values.astype(int))
            y_pred_labels = label_encoders[target_col].inverse_transform(y_pred[:, i].astype(int))

            report = classification_report(y_test_labels, y_pred_labels, zero_division=0)
            f.write(report + "\n")
    print(f"Classification reports have been written to {REPORT_SAVE_PATH}")
