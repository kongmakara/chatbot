import json
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

train_nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Cambodia Tourism Chatbot - Training\n",
    "This notebook covers loading the dataset, preprocessing, training the SimpleRNN model, and saving the artifacts."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import numpy as np\n",
    "import pickle\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Embedding, SimpleRNN\n",
    "from tensorflow.keras.preprocessing.text import Tokenizer\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
    "from sklearn.preprocessing import LabelEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load and preprocess dataset\n",
    "with open('intents.json', 'r') as f:\n",
    "    data = json.load(f)\n",
    "\n",
    "sentences = []\n",
    "labels = []\n",
    "for intent in data['intents']:\n",
    "    for pattern in intent['patterns']:\n",
    "        sentences.append(pattern)\n",
    "        labels.append(intent['tag'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Encode labels\n",
    "label_encoder = LabelEncoder()\n",
    "training_labels = label_encoder.fit_transform(labels)\n",
    "num_classes = len(np.unique(training_labels))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Tokenization and Padding\n",
    "vocab_size = 1000\n",
    "embedding_dim = 16\n",
    "max_len = 20\n",
    "oov_token = \"<OOV>\"\n",
    "\n",
    "tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)\n",
    "tokenizer.fit_on_texts(sentences)\n",
    "word_index = tokenizer.word_index\n",
    "\n",
    "sequences = tokenizer.texts_to_sequences(sentences)\n",
    "padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Build SimpleRNN Model\n",
    "model = Sequential([\n",
    "    Embedding(vocab_size, embedding_dim, input_length=max_len),\n",
    "    SimpleRNN(32, return_sequences=False),\n",
    "    Dense(16, activation='relu'),\n",
    "    Dense(num_classes, activation='softmax')\n",
    "])\n",
    "\n",
    "model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train Model\n",
    "epochs = 200\n",
    "history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs, verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save Model and Tokenizer\n",
    "model.save('model.h5')\n",
    "\n",
    "with open('tokenizer.pkl', 'wb') as f:\n",
    "    pickle.dump(tokenizer, f)\n",
    "\n",
    "with open('label_encoder.pkl', 'wb') as f:\n",
    "    pickle.dump(label_encoder, f)\n",
    "\n",
    "# Optional: Save configuration\n",
    "config = {'max_len': max_len}\n",
    "with open('config.pkl', 'wb') as f:\n",
    "    pickle.dump(config, f)\n",
    "\n",
    "print(\"Training complete and artifacts saved.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

pred_nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Cambodia Tourism Chatbot - Prediction\n",
    "Load trained artifacts, test inputs, and perform error analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import pickle\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import load_model\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Artifacts\n",
    "model = load_model('model.h5')\n",
    "with open('tokenizer.pkl', 'rb') as f:\n",
    "    tokenizer = pickle.load(f)\n",
    "with open('label_encoder.pkl', 'rb') as f:\n",
    "    label_encoder = pickle.load(f)\n",
    "with open('config.pkl', 'rb') as f:\n",
    "    config = pickle.load(f)\n",
    "with open('intents.json', 'r') as f:\n",
    "    intents_data = json.load(f)\n",
    "max_len = config['max_len']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Inference function\n",
    "def predict_intent(text):\n",
    "    seq = tokenizer.texts_to_sequences([text])\n",
    "    padded = pad_sequences(seq, truncating='post', maxlen=max_len)\n",
    "    pred = model.predict(padded, verbose=0)\n",
    "    intent_idx = np.argmax(pred)\n",
    "    confidence = pred[0][intent_idx]\n",
    "    tag = label_encoder.inverse_transform([intent_idx])[0]\n",
    "    \n",
    "    # Get response\n",
    "    for intent in intents_data['intents']:\n",
    "        if intent['tag'] == tag:\n",
    "            response = np.random.choice(intent['responses'])\n",
    "            return tag, response, confidence\n",
    "    return tag, \"Sorry, I don't understand.\", confidence"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test at least 5 inputs\n",
    "test_inputs = [\n",
    "    \"Hello there!\",\n",
    "    \"Tell me about Angkor Wat\",\n",
    "    \"Do I need a visa?\",\n",
    "    \"What currency do you use?\",\n",
    "    \"What's the weather like?\",\n",
    "    \"Where can I eat pizza?\"\n",
    "]\n",
    "\n",
    "for text in test_inputs:\n",
    "    tag, response, conf = predict_intent(text)\n",
    "    print(f\"Input: {text}\\nPredicted Tag: {tag} (Conf: {conf:.2f})\\nResponse: {response}\\n\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Error Analysis\n",
    "- The model performs well on inputs similar to the training patterns.\n",
    "- For out-of-domain queries like 'Where can I eat pizza?', the model might predict an incorrect tag (e.g., 'food' or another class) because it lacks a fallback or 'unknown' class. \n",
    "- **Limitations:** SimpleRNN has limited memory for long sequences. A small dataset causes overconfidence in predictions even for unrelated queries."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('train.ipynb', 'w') as f:
    json.dump(train_nb, f, indent=1)
with open('prediction.ipynb', 'w') as f:
    json.dump(pred_nb, f, indent=1)
