import pandas as pd
import numpy as np
import re
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

# --- Configuration and Constants ---
# Model and Data Paths
EXCEL_PATH = 'data/oiics_201_code_list.xlsx'
DATA_PATH = 'data/January2015toFebruary2025.xlsx'
MODEL_SAVE_PATH = 'bert_multi_label_model.pth'
TOKENIZER_SAVE_PATH = 'bert_tokenizer'
LABEL_ENCODERS_SAVE_PATH = 'bert_label_encoders.pkl'
REPORT_SAVE_PATH = 'classification_reports_bert.txt'

# Model Hyperparameters
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

# Data Columns
TARGET_COLUMNS = ['Nature of Injury_Generalized', 'Part of Body_Generalized', 'Event Type_Generalized', 'Source of Injury_Generalized']
NARRATIVE_COLUMN = 'Final Narrative'
REQUIRED_COLUMNS = [NARRATIVE_COLUMN, 'NatureTitle', 'Part of Body Title', 'EventTitle', 'SourceTitle', 'Secondary Source Title']

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Data Loading and Preprocessing ---
def load_data(file_path):
    try:
        return pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit()

def map_generalized_code(code, mapping):
    code_str = str(int(code))
    while code_str.endswith('0') and len(code_str) > 1:
        code_str = code_str[:-1]
    general_code = code_str[:3] if len(code_str) >= 4 else code_str
    return mapping.get(int(general_code), 'Other')

def load_mapping_sheets(excel_path):
    sheet_mappings = pd.read_excel(excel_path, sheet_name=['Nature', 'Part', 'Event', 'Source'])
    return {
        'Nature': sheet_mappings['Nature'].set_index('CASE_CODE')['CASE_CODE_TITLE'].to_dict(),
        'Part': sheet_mappings['Part'].set_index('CASE_CODE')['CASE_CODE_TITLE'].to_dict(),
        'Event': sheet_mappings['Event'].set_index('CASE_CODE')['CASE_CODE_TITLE'].to_dict(),
        'Source': sheet_mappings['Source'].set_index('CASE_CODE')['CASE_CODE_TITLE'].to_dict()
    }

# --- PyTorch Dataset Class ---
class IncidentDataset(Dataset):
    def __init__(self, narratives, labels, tokenizer, max_len):
        self.narratives = narratives
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.narratives)

    def __getitem__(self, item):
        narrative = str(self.narratives[item])
        labels = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            narrative, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt',
        )
        return {
            'narrative_text': narrative,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# --- BERT Model Architecture ---
class BERTMultiLabelClassifier(torch.nn.Module):
    def __init__(self, n_classes_dict):
        super(BERTMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
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

# --- Training and Evaluation Functions ---
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    for d in tqdm(data_loader, desc="Training Epoch", unit="batch"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = 0
        for i, col in enumerate(TARGET_COLUMNS):
            loss += loss_fn(outputs[col], labels[:, i])
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return np.mean(losses)

def eval_model(model, data_loader, device):
    model = model.eval()
    predictions = {col: [] for col in TARGET_COLUMNS}
    real_values = {col: [] for col in TARGET_COLUMNS}
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating", unit="batch"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            for i, col in enumerate(TARGET_COLUMNS):
                _, preds = torch.max(outputs[col], dim=1)
                predictions[col].extend(preds.cpu().numpy())
                real_values[col].extend(labels[:, i].cpu().numpy())
    return predictions, real_values

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = load_data(DATA_PATH)
    mappings = load_mapping_sheets(EXCEL_PATH)
    df.dropna(subset=REQUIRED_COLUMNS, inplace=True)
    df['Nature of Injury_Generalized'] = df['Nature'].apply(lambda x: map_generalized_code(x, mappings['Nature']))
    df['Part of Body_Generalized'] = df['Part of Body'].apply(lambda x: map_generalized_code(x, mappings['Part']))
    df['Event Type_Generalized'] = df['Event'].apply(lambda x: map_generalized_code(x, mappings['Event']))
    df['Source of Injury_Generalized'] = df['Source'].apply(lambda x: map_generalized_code(x, mappings['Source']))
    for col in TARGET_COLUMNS:
        counts = df[col].value_counts()
        df = df[~df[col].isin(counts[counts < 2].index)]
    
    print("Encoding labels...")
    label_encoders = {}
    for col in TARGET_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

    print("Setting up tokenizer and dataloaders...")
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    def create_data_loader(df, tokenizer, max_len, batch_size):
        ds = IncidentDataset(
            narratives=df[NARRATIVE_COLUMN].to_numpy(),
            labels=df[TARGET_COLUMNS].to_numpy(),
            tokenizer=tokenizer, max_len=max_len
        )
        return DataLoader(ds, batch_size=batch_size, num_workers=0)
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    print("Initializing model...")
    n_classes_dict = {col: len(le.classes_) for col, le in label_encoders.items()}
    model = BERTMultiLabelClassifier(n_classes_dict).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
        print(f'Train loss {train_loss}')

    print("Evaluating model...")
    y_pred, y_test = eval_model(model, test_data_loader, device)
    with open(REPORT_SAVE_PATH, "w") as f:
        for col in TARGET_COLUMNS:
            all_labels = np.arange(len(label_encoders[col].classes_))
            report = classification_report(
                y_test[col], y_pred[col], labels=all_labels,
                target_names=label_encoders[col].classes_, zero_division=0
            )
            f.write(f"\n--- Classification Report for {col} ---\n")
            f.write(report + "\n")
    print(f"Classification reports saved to {REPORT_SAVE_PATH}")

    print("Saving artifacts...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)
    
    # FIX: Save the class lists instead of the encoder objects
    label_encoder_classes = {col: le.classes_.tolist() for col, le in label_encoders.items()}
    joblib.dump(label_encoder_classes, LABEL_ENCODERS_SAVE_PATH)
    
    print("Artifacts saved successfully.")
