from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import json
import os
import re
import spacy
import uvicorn

# ==== Setup ====
MODEL_PATH = "saved_intent_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load label map ====
with open(os.path.join(MODEL_PATH, "label2id.json"), "r") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

# ==== Load Tokenizer ====
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

# ==== Load spaCy ====
nlp = spacy.load("en_core_web_sm")

# ==== Define Services & Industries ====
services = [
    "Generative AI", "Intelligent Process Automation", "Custom Development", "Data Engineering",
    "Cloud", "Cloud Solutions", "Cybersecurity", "AI Agents", "Banking", "Metaverse and Web3",
    "GenAI in Tourism", "Finance", "Healthcare", "Marketing", "Education", "Construction",
    "LLM Chatbots", "Multi-Agent Systems"
]

industries = [
    "education", "health care", "financial services", "retail", "construction", "tourism", "marketing", "banking"
]

# ==== Regex ====
email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
datetime_regex = r"\b(?:on\s)?(?:\d{1,2}(?:st|nd|rd|th)?\s)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|" \
                 r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b"

# ==== Define Model ====
class IntentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(IntentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits

# ==== Load Model Weights ====
model = IntentClassifier(num_classes=len(label2id))
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "pytorch_model.bin"), map_location=device))
model.to(device)
model.eval()

# ==== FastAPI Setup ====
app = FastAPI(title="NLU System: Intent + Entity Extraction")

class QueryInput(BaseModel):
    text: str

# ==== Entity Extraction Function ====
def extract_entities(text):
    doc = nlp(text)
    entities = {
        "service_name": [],
        "project_type": None,
        "business_problem": None,
        "industry": None,
        "company_name": None,
        "contact_name": None,
        "email_address": None,
        "specific_page_topic": None,
        "desired_meeting_time": None
    }

    # service_name matching
    for s in services:
        if s.lower() in text.lower():
            entities["service_name"].append(s)

    # industry matching
    for i in industries:
        if i.lower() in text.lower():
            entities["industry"] = i.title()

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["contact_name"] = ent.text
        elif ent.label_ == "ORG":
            entities["company_name"] = ent.text
        elif ent.label_ == "DATE" and not entities["desired_meeting_time"]:
            entities["desired_meeting_time"] = ent.text

    # Regex-based
    email_match = re.search(email_regex, text)
    if email_match:
        entities["email_address"] = email_match.group()

    datetime_match = re.search(datetime_regex, text)
    if datetime_match and not entities["desired_meeting_time"]:
        entities["desired_meeting_time"] = datetime_match.group()

    # fallback
    if not entities["specific_page_topic"]:
        entities["specific_page_topic"] = text.split("about")[-1].strip() if "about" in text else None

    return entities

# ==== Main Prediction Route ====
@app.post("/predict")
def predict_intent_and_entities(data: QueryInput):
    text = data.text

    # --- Intent Classification ---
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred_id = torch.argmax(outputs, dim=1).item()
        intent = id2label[pred_id]

    # --- Entity Extraction ---
    entities = extract_entities(text)

    return {
        "intent": intent,
        "entities": entities
    }


# === Run server ===
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
