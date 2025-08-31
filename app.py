import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# --- 1. Load the Fine-Tuned Model and Tokenizer ---
@st.cache_resource
def load_model():
    """
    Loads the fine-tuned BERT model and tokenizer from the saved directory.
    This function is cached so the model is only loaded into memory once.
    """
    model_path = './bert_fake_news_model'
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading the AI model: {e}")
        st.info("Please make sure the 'bert_fake_news_model' directory exists in the same folder as this app.")
        st.info("You can create this by running the 'train_model.py' script first.")
        return None, None

print("Loading the AI model...")
tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model:
    model.to(device)
    print("Model loaded successfully.")

# --- 2. Prediction Function ---
def predict(text):
    """
    Takes a news article text, preprocesses it, and returns the predicted label
    and the confidence score.
    """
    if not tokenizer or not model:
        return "Error", 0.0

    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
    
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    
    probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
    confidence = probabilities[predicted_class_id]
    
    labels = ['REAL', 'FAKE']
    return labels[predicted_class_id], confidence

# --- 3. Streamlit User Interface ---
st.set_page_config(page_title="AI News Verifier", page_icon="📰", layout="wide")

st.title("📰 AI News Verifier")
st.markdown("An AI-powered tool to help you distinguish between real and fake news. Paste the text of an article below to get an instant analysis.")

user_input = st.text_area("Enter News Article Text Here", "", height=250, placeholder="Paste the full text of the news article you want to verify...")

if st.button("Analyze News", type="primary"):
    if user_input and model:
        with st.spinner('Analyzing... This may take a moment.'):
            prediction, confidence = predict(user_input)
        
        st.write("---")
        st.subheader("Analysis Result")
        
        if prediction == 'REAL':
            st.success(f"✅ This article appears to be REAL NEWS.")
        else:
            st.error(f"❌ This article appears to be FAKE NEWS.")
        
        # --- (THE FIX IS HERE) ---
        # We explicitly convert the NumPy float32 to a standard Python float.
        st.progress(float(confidence))
        # --- (END OF FIX) ---
        
        st.metric(label="Confidence Score", value=f"{confidence:.2%}")

    elif not model:
        st.error("The AI model is not loaded. Please check the console for errors and ensure the training script has been run.")
    else:
        st.warning("Please paste some text into the box above to analyze.")

# --- Sidebar ---
st.sidebar.header("About This App")
st.sidebar.info(
    "This application uses a fine-tuned BERT model, a state-of-the-art natural language processing "
    "architecture developed by Google. The model has been trained on a dataset of real and fake news articles to learn the patterns and nuances that distinguish them."
)
st.sidebar.header("Disclaimer")
st.sidebar.warning(
    "This AI tool is for educational purposes and is not a substitute for professional fact-checking. "
    "Its accuracy is limited by the dataset it was trained on. Always verify information from multiple reputable sources."
)

