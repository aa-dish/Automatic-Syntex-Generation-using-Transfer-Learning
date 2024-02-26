import openpyxl
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaForTokenClassification, AdamW, BertForTokenClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Load the Excel file
wb = openpyxl.load_workbook("/scratch/adesai/thesis2/Project/src/data/Dataset.xlsx")
ws = wb.active

# Extract the data from the Excel file
data = list(ws.iter_rows(values_only=True))

# Convert the data to a DataFrame
df = pd.DataFrame(data, columns=["text", "code"])

# Tokenize text and code columns
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_codebert = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

# Tokenize text and code columns
df['tokenized_text'] = df['text'].apply(lambda x: tokenizer_bert.encode(x, add_special_tokens=True, truncation=True, max_length=512))
df['tokenized_code'] = df['code'].apply(lambda x: tokenizer_codebert.encode(x, add_special_tokens=True, truncation=True, max_length=512))

# Define a dynamic max_length based on the input sequences in each batch
max_length_text = df['tokenized_text'].apply(len).max()
max_length_code = df['tokenized_code'].apply(len).max()

# Splitting the data into train, test, and validation sets
train_ratio = 0.7
test_ratio = 0.15
validation_ratio = 0.15

# Split into train and temp sets
train_df, temp_df = train_test_split(df, test_size=1 - train_ratio, random_state=42)

# Split the temp set into test and validation sets
test_df, validation_df = train_test_split(temp_df, test_size=test_ratio / (test_ratio + validation_ratio), random_state=42)

# Save train_df, test_df, and validation_df as CSV files
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
validation_df.to_csv('validation_data.csv', index=False)

# My Final Model
# Load the tokenizers
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_codebert = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

# Load data from CSVs
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')
validation_df = pd.read_csv('validation_data.csv')

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer_text, tokenizer_code, max_length_text=128, max_length_code=128):
        self.data = dataframe
        self.tokenizer_text = tokenizer_text
        self.tokenizer_code = tokenizer_code
        self.max_length_text = max_length_text
        self.max_length_code = max_length_code

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        code = str(self.data.iloc[idx]['code'])

        # Tokenize text and code
        inputs_text = self.tokenizer_text(text, add_special_tokens=True, truncation=True, max_length=self.max_length_text, padding='max_length', return_tensors='pt')
        inputs_code = self.tokenizer_code(code, add_special_tokens=True, truncation=True, max_length=self.max_length_code, padding='max_length', return_tensors='pt')

        return {
            'input_ids_text': inputs_text['input_ids'].squeeze(),
            'attention_mask_text': inputs_text['attention_mask'].squeeze(),
            'input_ids_code': inputs_code['input_ids'].squeeze(),
            'attention_mask_code': inputs_code['attention_mask'].squeeze(),
        }

# Create datasets and dataloaders
train_dataset = CustomDataset(train_df, tokenizer_bert, tokenizer_codebert)
test_dataset = CustomDataset(test_df, tokenizer_bert, tokenizer_codebert)
validation_dataset = CustomDataset(validation_df, tokenizer_bert, tokenizer_codebert)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# Define the BERT encoder
bert_encoder = BertModel.from_pretrained('bert-base-uncased')
bert_encoder.to(device)

# Define the CodeBERT decoder (you may need to fine-tune CodeBERT or implement a custom decoder)
codebert_decoder = RobertaForTokenClassification.from_pretrained('microsoft/codebert-base', num_labels=len(tokenizer_codebert.get_vocab()))
codebert_decoder.to(device)

# Define the overall sequence-to-sequence model
class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_ids_text, attention_mask_text, input_ids_code, attention_mask_code):
        # BERT encoder
        encoder_outputs = self.encoder(input_ids=input_ids_text, attention_mask=attention_mask_text)

        # CodeBERT decoder
        decoder_outputs = self.decoder(input_ids=input_ids_code, attention_mask=attention_mask_code)

        return decoder_outputs.logits

    def generate_output_without_special_tokens(self, input_ids, tokenizer):
        # Convert token IDs back to tokens using the tokenizer
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Remove special tokens and paddings
        tokens = [token for token in tokens if token not in tokenizer.all_special_tokens]
        
        # Convert tokens to string
        output_str = tokenizer.convert_tokens_to_string(tokens)

        return output_str

# Instantiate the model
model = Seq2SeqModel(bert_encoder, codebert_decoder)
model.to(device)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# Training loop with early stopping
num_epochs = 55
patience = 5
best_val_loss = float('inf')
counter_early_stop = 0

checkpoint_dir = '/scratch/adesai/thesis2/Project/src/notebooks/checkpoint.pth'
os.makedirs(checkpoint_dir, exist_ok=True)

# Check if there are saved checkpoints and load the latest one
latest_checkpoint = max([f for f in os.listdir(checkpoint_dir) if f.endswith('best_model2_32_55.pth')], default=None)
if latest_checkpoint is not None:
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, latest_checkpoint)))
    print(f"Resuming training from checkpoint: {latest_checkpoint}")

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    print(f"Epoch {epoch} started")
    epoch_train_loss = 0.0
    train_progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}/{num_epochs}', leave=False)
    for batch in train_progress_bar:
        input_ids_text = batch['input_ids_text'].to(device)
        attention_mask_text = batch['attention_mask_text'].to(device)
        input_ids_code = batch['input_ids_code'].to(device)
        attention_mask_code = batch['attention_mask_code'].to(device)

        optimizer.zero_grad()

        logits = model(input_ids_text, attention_mask_text, input_ids_code, attention_mask_code)
        
        # Reshape logits and labels for the CrossEntropyLoss
        logits = logits.view(-1, logits.shape[-1])
        labels = input_ids_code.view(-1)

        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()
        train_progress_bar.set_postfix({'Train Loss': loss.item()})
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # Validation loop
    model.eval()
    epoch_val_loss = 0.0
    val_progress_bar = tqdm(validation_dataloader, desc=f'Epoch {epoch}/{num_epochs} - Validation', leave=False)
    with torch.no_grad():
        for batch in val_progress_bar:
            input_ids_text = batch['input_ids_text'].to(device)
            attention_mask_text = batch['attention_mask_text'].to(device)
            input_ids_code = batch['input_ids_code'].to(device)
            attention_mask_code = batch['attention_mask_code'].to(device)

            logits = model(input_ids_text, attention_mask_text, input_ids_code, attention_mask_code)
            val_loss = loss_fn(logits.view(-1, logits.shape[-1]), input_ids_code.view(-1))
            
            val_progress_bar.set_postfix({'Validation Loss': val_loss.item()})
            epoch_val_loss += val_loss.item()

    avg_val_loss = epoch_val_loss / len(validation_dataloader)
    val_losses.append(avg_val_loss)

    # Print one message at the end of each epoch
    tqdm.write(f"Epoch {epoch}/{num_epochs}, Average Training Loss: {avg_train_loss}, Average Validation Loss: {avg_val_loss}")
    
    # Check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter_early_stop = 0
        # Save the best model
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model2_32_55.pth'))
    else:
        counter_early_stop += 1
        if counter_early_stop == patience:
            tqdm.write(f'Early stopping at epoch {epoch}')
            break

# Load the best model for testing
best_model_path = os.path.join(checkpoint_dir, 'best_model2_32_55.pth')
model.load_state_dict(torch.load(best_model_path))

# Plot the training and validation losses for each epoch
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Modify the testing loop
model.eval()
test_outputs = []
test_loss = 0.0
num_test_batches = 0

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc='Testing', leave=False):
        input_ids_text = batch['input_ids_text'].to(device)
        attention_mask_text = batch['attention_mask_text'].to(device)
        input_ids_code = batch['input_ids_code'].to(device)
        attention_mask_code = batch['attention_mask_code'].to(device)

        logits = model(input_ids_text, attention_mask_text, input_ids_code, attention_mask_code)
        predicted_ids = torch.argmax(logits, dim=-1)

        # Calculate the test loss excluding special tokens and paddings
        labels = input_ids_code.view(-1)
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels)
        test_loss += loss.item()
        num_test_batches += 1

        # Convert predicted_ids and input_ids_code to Python lists
        predicted_ids = predicted_ids.cpu().numpy().tolist()

        for pred, true in zip(predicted_ids, input_ids_code):
            # Generate output without special tokens and paddings
            predicted_str = model.generate_output_without_special_tokens(pred, tokenizer_codebert)
            true_str = model.generate_output_without_special_tokens(true, tokenizer_codebert)

            test_outputs.append({'input': true_str, 'generated_output': predicted_str})

# Calculate average test loss
avg_test_loss = test_loss / num_test_batches
print(f"Average Test Loss: {avg_test_loss}")

# Print some test inputs and their generated outputs
num_examples_to_print = 3
for example in test_outputs[:num_examples_to_print]:
    print(f"Original Output: {example['input']}")
    print(f"Generated Output: {example['generated_output']}")
    print("-" * 50)