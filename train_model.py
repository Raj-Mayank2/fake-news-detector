import pandas as pd
from sklearn.model_selection import train_test_split
import torch
# --- (FIXED IMPORTS) ---
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW 
# --- (END OF FIX) ---
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# --- 1. Load and Prepare the Dataset ---
print("Loading and preparing data from fake.csv and true.csv...")

try:
    df_fake = pd.read_csv('fake.csv')
    df_true = pd.read_csv('true.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure 'fake.csv' and 'true.csv' are in the same directory.")
    exit()

df_fake['label'] = 1
df_true['label'] = 0

df = pd.concat([df_fake, df_true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df = df[['full_text', 'label']].dropna()
df = df.sample(n=4000, random_state=42) 

X = df.full_text.values
y = df.label.values

# --- 2. Tokenization and Input Formatting ---
print("Tokenizing text...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def preprocess_data(texts):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,                      
                            add_special_tokens=True,
                            max_length=256,
                            padding='max_length',
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors='pt',
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    return input_ids, attention_masks

input_ids, attention_masks = preprocess_data(X)
labels = torch.tensor(y)

# --- 3. Splitting Data and Creating DataLoaders ---
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                            random_state=42, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                             random_state=42, test_size=0.1)

batch_size = 16
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# --- 4. Model Definition and Training ---
print("Loading pre-trained BERT model...")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 3
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# --- Training Loop ---
print("Starting training...")
for epoch_i in range(0, epochs):
    print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
    total_train_loss = 0
    model.train()

    for batch in train_dataloader:
        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
        model.zero_grad()
        
        result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = result.loss
        total_train_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"  Average training loss: {avg_train_loss:.2f}")

# --- 5. Saving the Model ---
print("Training complete. Saving model...")
model_save_path = './bert_fake_news_model'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to '{model_save_path}'")