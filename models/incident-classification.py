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
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import sys

# Download NLTK stop words if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Load the data
try:
    df = pd.read_excel('data/January2015toApril2024.xlsx')
except FileNotFoundError:
    print("Error: The file 'data/January2015toApril2024.xlsx' was not found.")
    print("Please make sure the file path is correct.")
    exit()

# Drop rows with missing values
df.dropna(subset=['Final Narrative', 'NatureTitle', 'Part of Body Title', 'EventTitle', 'SourceTitle', 'Secondary Source Title', 'Amputation', 'Hospitalized'], inplace=True)

# Define a text cleaning function
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    return ' '.join(tokens)

# Apply text cleaning
df['Narrative_Cleaned'] = df['Final Narrative'].apply(clean_text)

# Select features and targets
features = ['Narrative_Cleaned', 'Amputation', 'Hospitalized']
targets = ['NatureTitle', 'Part of Body Title', 'EventTitle', 'SourceTitle']

# Separate features (X) and targets (y)
X = df[features]
y = df[targets].copy()

# Encode the target labels
label_encoders = {}
for col in targets:
    le = LabelEncoder()
    y.loc[:, col] = le.fit_transform(y[col]).astype(int)
    label_encoders[col] = le

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a custom transformer for preprocessing and feature combination
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def fit(self, X, y=None):
        self.vectorizer.fit(X['Narrative_Cleaned'])
        return self

    def transform(self, X):
        text_features = self.vectorizer.transform(X['Narrative_Cleaned'])
        # Convert additional features to a sparse matrix
        additional_features = csr_matrix(X[['Amputation', 'Hospitalized']].values)
        return hstack([text_features, additional_features])

# Create the final pipeline with Logistic Regression
pipeline = Pipeline([
    ('features', CombinedAttributesAdder()),
    ('clf', MultiOutputClassifier(LogisticRegression(solver='liblinear', random_state=42)))
])

# Train the model with the corrected y_train format
print("Training the model...")
pipeline.fit(X_train, y_train.values.astype(int))
print("Model training complete.")
print("-" * 30)

# Make predictions
print("Making predictions...")
y_pred = pipeline.predict(X_test)
print("Predictions complete.")
print("-" * 30)

# Evaluate the model for each target and output to a file
print("Evaluating model performance...")
with open("classification_reports.txt", "w") as f:
    for i, target_col in enumerate(targets):
        f.write(f"\n--- Classification Report for {target_col} ---\n")
        
        # Explicitly cast to integer type to avoid the TypeError
        y_test_labels = label_encoders[target_col].inverse_transform(y_test.iloc[:, i].values.astype(int))
        y_pred_labels = label_encoders[target_col].inverse_transform(y_pred[:, i].astype(int))

        report = classification_report(y_test_labels, y_pred_labels, zero_division=0)
        f.write(report + "\n")
print("Classification reports have been written to classification_reports.txt")