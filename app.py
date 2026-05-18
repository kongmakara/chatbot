import streamlit as st
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Set page config
st.set_page_config(page_title="Cambodia Tourism Chatbot", page_icon="🏖️", layout="centered")

@st.cache_resource
def load_artifacts():
    try:
        model = load_model('model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        with open('config.pkl', 'rb') as f:
            config = pickle.load(f)
        with open('intents.json', 'r') as f:
            intents_data = json.load(f)
        return model, tokenizer, label_encoder, config, intents_data
    except Exception as e:
        st.error(f"Error loading artifacts. Please ensure you have run the training notebook. {e}")
        return None, None, None, None, None

model, tokenizer, label_encoder, config, intents_data = load_artifacts()

def predict_intent(text, model, tokenizer, label_encoder, config, intents_data):
    max_len = config['max_len']
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, truncating='post', maxlen=max_len)
    pred = model.predict(padded, verbose=0)
    intent_idx = np.argmax(pred)
    tag = label_encoder.inverse_transform([intent_idx])[0]
    
    # Get random response for the predicted tag
    for intent in intents_data['intents']:
        if intent['tag'] == tag:
            response = np.random.choice(intent['responses'])
            return response
    return "Sorry, I don't understand."

st.title("🇰🇭 Cambodia Tourism Chatbot")
st.markdown("Welcome! Ask me anything about visiting Cambodia.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is the best time to visit Cambodia?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate chatbot response
    if model:
        response = predict_intent(prompt, model, tokenizer, label_encoder, config, intents_data)
    else:
        response = "Model is not trained yet."

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
