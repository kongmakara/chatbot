import json
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load and preprocess dataset
with open('intents.json', 'r') as f:
    data = json.load(f)

sentences = []
labels = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(intent['tag'])

# Encode labels
label_encoder = LabelEncoder()
training_labels = label_encoder.fit_transform(labels)
num_classes = len(np.unique(training_labels))

# Tokenization and Padding
vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# Build SimpleRNN Model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    SimpleRNN(32, return_sequences=False),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model
epochs = 200
model.fit(padded_sequences, np.array(training_labels), epochs=epochs, verbose=1)

# Save Model and Tokenizer
model.save('model.h5')

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Optional: Save configuration
config = {'max_len': max_len}
with open('config.pkl', 'wb') as f:
    pickle.dump(config, f)

print("Training complete and artifacts saved.")
