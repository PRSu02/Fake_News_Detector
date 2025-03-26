import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


train_path = "train.csv"
val_path = "valid.csv"
test_path = "test.csv"



def load_data(file_path):
    df = pd.read_csv(file_path)
    return df["tweets"].tolist(), df["labels"].tolist()

train_texts, train_labels = load_data(train_path)
val_texts, val_labels = load_data(val_path)
test_texts, test_labels = load_data(test_path)


train_labels = torch.tensor(train_labels, dtype=torch.float32)
val_labels = torch.tensor(val_labels, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)


MAX_VOCAB_SIZE = 5000
MAX_LEN = 50

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

train_sequences = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=MAX_LEN, padding='post')
val_sequences = pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=MAX_LEN, padding='post')
test_sequences = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=MAX_LEN, padding='post')


train_data = torch.tensor(train_sequences, dtype=torch.long)
val_data = torch.tensor(val_sequences, dtype=torch.long)
test_data = torch.tensor(test_sequences, dtype=torch.long)


class CodeMixDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_dataset = CodeMixDataset(train_data, train_labels)
val_dataset = CodeMixDataset(val_data, val_labels)
test_dataset = CodeMixDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(vocab_size=MAX_VOCAB_SIZE).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



def train_model(model, train_loader, val_loader, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)  # Reshape for BCELoss
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = outputs > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}: Loss {total_loss:.4f}, Val Acc {val_acc:.4f}")



def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)

            preds = outputs > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


train_model(model, train_loader, val_loader)
test_acc = evaluate(model, test_loader)
print(f"Test Accuracy: {test_acc:.4f}")
