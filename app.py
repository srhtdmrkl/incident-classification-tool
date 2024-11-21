import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Set the browser tab title and other configuration
st.set_page_config(
    page_title="Incident Classification Tool",
    page_icon="🛠️",
    layout="wide",
)

# Streamlit App Title
st.title("Incident Classification Tool")
st.write("Use pre-trained models or train a model with your own dataset.")

# Initialize session state variables
if "models" not in st.session_state:
    st.session_state["models"] = None
if "vectorizer" not in st.session_state:
    st.session_state["vectorizer"] = None
if "training_logs" not in st.session_state:
    st.session_state["training_logs"] = []
if "description_mapping" not in st.session_state:
    st.session_state["description_mapping"] = None

# File Upload
st.subheader("Dataset Upload")
st.info(
    """
    Use pre-trained models for the default dataset or upload your own dataset in **CSV** or **Excel** format.
    """
)

uploaded_file = st.file_uploader("Upload your dataset file", type=["csv", "xlsx"])

# Load Data
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.success(f"File '{uploaded_file.name}' successfully uploaded!")
    else:
        df = pd.read_excel(uploaded_file)
        st.success(f"File '{uploaded_file.name}' successfully uploaded!")
else:
    # Load pre-trained models and default dataset
    df = pd.read_excel("osha-data.xlsx")  # Default dataset
    st.session_state["vectorizer"] = pickle.load(open("vectorizer.pkl", "rb"))
    st.session_state["models"] = {
        "Logistic Regression": pickle.load(open("logistic_regression.pkl", "rb")),
        "Linear SVC": pickle.load(open("linear_svc.pkl", "rb")),
    }
    st.warning("Using the default dataset with pre-trained models.")

# Show Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Show Column Selection Only for Uploaded Data
if uploaded_file:
    # Column Selection
    st.subheader("Select Columns")
    text_column = st.selectbox("Select the column containing incident descriptions:", df.columns)

    # Default Category Columns
    default_columns = ["nature_of_inj", "part_of_body", "event_type", "evn_factor"]

    # Filter only integer columns for selection
    integer_columns = [col for col in df.columns if pd.api.types.is_integer_dtype(df[col])]

    # Multiselect for category columns with defaults pre-selected
    category_columns = st.multiselect(
        "Select the category code columns (must be integer columns):",
        integer_columns,
        default=[col for col in default_columns if col in df.columns],
    )

    # Map code columns to description columns
    description_mapping = {}
    for category in category_columns:
        description_column = st.selectbox(
            f"Select the description column for '{category}':",
            df.columns,
            index=list(df.columns).index(category) + 1  # Assumes description column is next to the code column
        )
        description_mapping[category] = description_column

    # Save description_mapping to session state
    st.session_state["description_mapping"] = description_mapping

# Allow users to select the model
st.subheader("Select Model")
model_choice = st.selectbox(
    "Choose the model to use:",
    ["Logistic Regression", "Linear SVC"]
)

# Classification Section
if st.session_state["models"]:
    st.subheader("Incident Classification")
    new_description = st.text_area("Enter a new incident description:")
    if st.button("Analyze"):
        if new_description.strip():
            # Use pre-trained models
            vectorizer = st.session_state["vectorizer"]
            processed_description = vectorizer.transform([new_description])
            classifications = {
                category: st.session_state["models"][model_choice][category].predict(processed_description)[0]
                for category in category_columns
            }
            st.write("### Results:")
            for category, classification in classifications.items():
                # Retrieve description_mapping from session state
                code_to_description = dict(
                    zip(df[category], df[st.session_state["description_mapping"][category]])
                )
                description = code_to_description.get(classification, "Unknown")
                st.write(f"- **{category}:** {description} (Code: {classification})")
        else:
            st.error("Please enter a description to analyze.")

# Train Model Button for Uploaded Data
if uploaded_file and st.button("Train Model"):
    if text_column and category_columns:
        st.session_state["training_logs"] = []  # Reset logs for new training
        st.session_state["training_logs"].append(f"Training models using {model_choice}...")
        
        # Split Data
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

        # Prepare Data for Training
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(train_data[text_column])
        X_test = vectorizer.transform(test_data[text_column])

        # Train Models using the selected model
        models = {}
        for category in category_columns:
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
            elif model_choice == "Linear SVC":
                model = LinearSVC(class_weight="balanced", random_state=42)
            model.fit(X_train, train_data[category])
            
            # Evaluate the model
            classifications = model.predict(X_test)
            accuracy = accuracy_score(test_data[category], classifications)
            
            # Store the trained model and test data
            models[category] = {
                "model": model,
                "X_test": X_test,
                "y_test": test_data[category]
            }
            st.session_state["training_logs"].append(
                f"Model trained for **{category}**. (Accuracy: {accuracy:.2f})"
            )

        # Save trained models and vectorizer in session state
        st.session_state["models"] = models
        st.session_state["vectorizer"] = vectorizer
        st.session_state["training_logs"].append("Training completed!")

# Display Training Logs
if st.session_state["training_logs"]:
    st.subheader("Training Logs")
    for log in st.session_state["training_logs"]:
        st.write(log)
