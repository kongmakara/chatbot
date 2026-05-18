# Cambodia Tourism Chatbot Project Report

## Problem Description
The goal of this project is to build an AI-powered conversational agent (chatbot) dedicated to assisting tourists visiting Cambodia. Tourists often have common inquiries regarding visas, currency, transportation, popular attractions like Angkor Wat, and the best time to visit. Manually answering these repetitive queries is time-consuming. A chatbot streamlines this process, providing instant, accurate, and automated responses to user inputs, enhancing the overall tourist experience.

## Model Architecture
The chatbot relies on a **Simple Recurrent Neural Network (SimpleRNN)** built using TensorFlow/Keras.
- **Input Layer**: Text inputs are processed through a `Tokenizer` and padded to a fixed sequence length (`max_len = 20`) using `pad_sequences`.
- **Embedding Layer**: An embedding layer (`vocab_size=1000`, `output_dim=16`) converts word indices into dense vectors of fixed size, capturing semantic meaning.
- **RNN Layer**: A `SimpleRNN` layer with 32 units processes the sequence of word embeddings, maintaining a hidden state to capture temporal/sequential dependencies in the sentence.
- **Dense Layer (Hidden)**: A fully connected `Dense` layer with 16 units and a ReLU activation function extracts higher-level features.
- **Output Layer**: A final `Dense` layer with a Softmax activation outputs a probability distribution over the distinct intent classes. 
- **Loss Function**: `sparse_categorical_crossentropy` (since labels are integer encoded).

## Results
The model was trained on a custom intents dataset (`intents.json`) containing various categories such as greetings, farewells, food recommendations, and travel logistics.
- The model trained for 200 epochs and achieved a high training accuracy (near 100%), which is expected for a small, focused dataset.
- During inference (tested in `prediction.ipynb`), the model correctly maps queries like "Tell me about Angkor Wat" to the `angkor_wat` intent and "How to get a visa" to the `visa` intent with high confidence (>90%). 

## Error Analysis
During testing, the model was evaluated on both in-domain and out-of-domain queries:
- **In-Domain**: Inputs that are semantically similar to training patterns (e.g., "Do I need a visa?") are classified correctly.
- **Out-of-Domain**: Queries that are completely unrelated to Cambodia tourism (e.g., "Where can I eat pizza?") force the model to predict one of the existing intent classes. This results in misclassifications (e.g., assigning it to the `food` intent or `greeting` intent) because the model lacks a dedicated "fallback" or "unknown" class.
- **Typos and OOV Words**: Words not present in the training vocabulary are converted to an `<OOV>` token. If a sentence contains too many OOV tokens, the model's confidence drops significantly, leading to potential misclassifications.

## Chatbot Limitations
1. **Lack of Context/Memory**: The `SimpleRNN` processes each query independently and the Streamlit app does not feed previous chat history back into the model. Thus, the chatbot cannot handle multi-turn conversations or follow-up questions effectively.
2. **Vanishing Gradient**: `SimpleRNN` is prone to the vanishing gradient problem. While our sequence length is short (`max_len=20`), the model would struggle if extended to process much longer paragraphs. LSTM or GRU would be more suitable for longer texts.
3. **Static Responses**: The chatbot only selects a random predefined response from `intents.json`. It cannot generate novel text or fetch real-time information (e.g., current weather or live flight data).
4. **Limited Dataset Size**: The model is heavily restricted by the small number of training patterns. It may fail to understand complex phrasings or slang without a more extensive dataset.
