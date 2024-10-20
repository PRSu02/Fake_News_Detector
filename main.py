import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
df = pd.read_csv('fake_or_real_news.csv')
print(df.head())
print(df.info())
print(df['label'].value_counts())
df = df.dropna(subset=['text', 'label'])
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})
print(df['label'].unique())
print(df['label'].value_counts())
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.split()
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]
    return " ".join(text)

df['combined_text'] = df['title'] + " " + df['text']

df['clean_text'] = df['combined_text'].apply(clean_text)
df = df[df['clean_text'].str.strip().astype(bool)]
print(f'Number of valid rows after text cleaning: {len(df)}')


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['clean_text'])
X = tokenizer.texts_to_sequences(df['clean_text'])
max_len = 500
X_pad = pad_sequences(X, maxlen=max_len)
print(f'X_pad shape after padding: {X_pad.shape}')
y = df['label'].values
if X_pad.shape[0] > 0 and y.shape[0] > 0:
    X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)
    print(f'Training data shape: {X_train.shape}')
    print(f'Test data shape: {X_test.shape}')
    print(f'Training labels shape: {y_train.shape}')
    print(f'Test labels shape: {y_test.shape}')
else:
    print("Error: No data available for train-test split.")
    

#use bi-lstm model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
model.add(Bidirectional(LSTM(units=128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.build(input_shape=(None, max_len))
model.summary()
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# use cnn model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# Define CNN model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))  # Embedding layer
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))  # Convolutional layer
model.add(GlobalMaxPooling1D())  # Pooling layer
model.add(Dropout(0.5))  # Dropout layer
model.add(Dense(128, activation='relu'))  # Fully connected layer
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Build and summarize the model
model.build(input_shape=(None, max_len))
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')
