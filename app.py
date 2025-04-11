import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI

# Custom CSS for styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .chat-container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .user-message {
            background-color: #e3f2fd;
            padding: 10px 15px;
            border-radius: 18px 18px 0 18px;
            margin: 5px 0;
            max-width: 80%;
            margin-left: auto;
        }
        .bot-message {
            background-color: #f1f1f1;
            padding: 10px 15px;
            border-radius: 18px 18px 18px 0;
            margin: 5px 0;
            max-width: 80%;
        }
        .header {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: #4a90e2;
            color: white;
            border-radius: 20px;
            padding: 8px 20px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #3a7bc8;
        }
        .confidence-meter {
            height: 10px;
            background: linear-gradient(90deg, #ff4d4d, #f9cb28, #4CAF50);
            border-radius: 5px;
            margin: 5px 0;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize OpenAI client
client = AzureOpenAI(
    api_key="8Gkv0fHZNLZMrReXE3Ia8zXk322NGKosPcW1RxK6l5P70ZSeyPunJQQJ99BCACYeBjFXJ3w3AAABACOG6NV9",
    api_version="2023-05-15",
    azure_endpoint="https://chatbottestigp.openai.azure.com/"
)
# Session state initialization
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = []
if 'active' not in st.session_state:
    st.session_state.active = True

# Text normalization function
def normalize_text(text):
    text = str(text).lower().strip()
    replacements = {
        'lors': 'letter of recommendation',
        'lor': 'letter of recommendation',
        'recommendation letter': 'letter of recommendation',
        'rec letter': 'letter of recommendation',
        'reference letter': 'letter of recommendation',
        'letters of rec': 'letter of recommendation',
        'letters of recommendation': 'letter of recommendation'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# Embedding generation
def get_embedding(text, model="text-embedding-ada-002"):
    try:
        normalized_text = normalize_text(text)
        if not normalized_text or len(normalized_text) < 3:
            return None
        response = client.embeddings.create(
            input=normalized_text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding error: {str(e)[:200]}")
        return None

# Load QA pairs
def load_qa_pairs(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding='latin-1')
        df = df.dropna(subset=['prompt', 'response'])
        df['prompt'] = df['prompt'].apply(normalize_text)
        df['response'] = df['response'].apply(str)
        return df.to_dict('records')
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return []

# Initialize embeddings
def initialize_embeddings(qa_pairs):
    with st.spinner("Initializing embeddings..."):
        for pair in qa_pairs:
            pair['embedding'] = get_embedding(pair['prompt'])

# Find similar questions
def find_top_matches(user_input, qa_pairs, top_n=3):
    user_embedding = get_embedding(user_input)
    if not user_embedding:
        return []

    similarities = []
    for pair in qa_pairs:
        if not pair.get('embedding'):
            continue
        similarity = cosine_similarity([user_embedding], [pair['embedding']])[0][0]
        similarities.append((similarity, pair))

    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:top_n]

# Generate response
def generate_contextual_response(user_input, top_matches):
    if not top_matches:
        return None

    context = "Relevant knowledge snippets:\n"
    for i, (score, pair) in enumerate(top_matches):
        context += f"{i+1}. Question: {pair['prompt']}\n   Answer: {pair['response']}\n\n"

    prompt = f"""Synthesize a comprehensive answer by analyzing these relevant snippets:
    
    User Question: {user_input}
    
    {context}
    
    Create a well-structured response that:
    1. Directly answers the question
    2. Incorporates key points from all relevant snippets
    3. Maintains accuracy and completeness
    4. Uses natural, conversational language
    
    Final Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Contextual response error: {e}")
        return top_matches[0][1]['response']

# Process user input
def get_chat_response(user_input):
    qa_pairs = st.session_state.qa_pairs
    
    normalized_input = normalize_text(user_input)
    exact_matches = [pair for pair in qa_pairs 
                    if normalize_text(pair['prompt']) == normalized_input]
    
    if exact_matches:
        return {
            "answer": exact_matches[0]['response'],
            "confidence": 1.0,
            "source": "exact match"
        }

    top_matches = find_top_matches(user_input, qa_pairs, top_n=3)
    
    if not top_matches or top_matches[0][0] < 0.6:
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": f"Answer concisely: {user_input}. If about recommendations, mention 2-3 letters are typical."
                }],
                temperature=0.3,
                max_tokens=200
            )
            return {
                "answer": response.choices[0].message.content.strip(),
                "confidence": 0.5,
                "source": "general knowledge"
            }
        except:
            return {
                "answer": "Typically 2-3 letters of recommendation are required for most programs.",
                "confidence": 0.0,
                "source": "fallback"
            }

    answer = generate_contextual_response(user_input, top_matches)
    avg_confidence = sum(m[0] for m in top_matches) / len(top_matches)
    
    return {
        "answer": answer,
        "confidence": float(avg_confidence),
        "source": "contextual synthesis",
        "matched_questions": [m[1]['prompt'] for m in top_matches]
    }

# Chat control functions
def end_chat():
    st.session_state.active = False
    st.session_state.conversation = []
    st.success("Chat ended. Refresh the page to start a new conversation.")

def clear_chat_history():
    st.session_state.conversation = []
    st.rerun()

# Initialize chatbot
def initialize_chatbot():
    csv_path = 'Question_Answer.csv'
    with st.spinner("Initializing chatbot..."):
        st.session_state.qa_pairs = load_qa_pairs(csv_path)
        initialize_embeddings(st.session_state.qa_pairs)

# Main app function
def main():
    # Header with title
    st.markdown("<h1 class='header'>IGP Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; margin-bottom: 30px; color: #555;'>
            Your intelligent assistant for queries
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### Chat Controls")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear Chat", help="Clear conversation history"):
                clear_chat_history()
        with col2:
            if st.button("🚪 Exit Chat", help="End the current conversation"):
                end_chat()
        
        st.markdown("---")
        st.markdown("**About IGP Chatbot**")
        st.markdown("""
            - Powered by AI and knowledge base
            - Press Enter to send messages
        """)
    
    # Initialize chatbot if not already done
    if not st.session_state.qa_pairs:
        initialize_chatbot()
    
    # Handle chat inactive state
    if not st.session_state.active:
        st.info("Chat ended. Type a message below to start a new conversation.")
    
    # Chat container
    with st.container():
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        
        # Display conversation history
        for message in st.session_state.conversation:
            if message["role"] == "user":
                st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-message'>{message['content']}</div>", unsafe_allow_html=True)
                if message.get("details"):
                    with st.expander("Response Details"):
                        st.write(f"**Confidence:**")
                        st.markdown(f"""
                            <div class="confidence-meter" style="width: {message['details']['confidence']*100}%"></div>
                            {message['details']['confidence']:.2f}/1.00
                        """, unsafe_allow_html=True)
                        st.write(f"**Source:** {message['details']['source']}")
                        if message['details'].get('matched_questions'):
                            st.write("**Matched questions:**")
                            for q in message['details']['matched_questions']:
                                st.write(f"- {q}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat input (works with Enter key)
    if user_input := st.chat_input("Type your question here (press Enter to send)..."):
        if user_input.strip().lower() in ['exit', 'quit', 'end', 'bye']:
            end_chat()
            st.rerun()
        else:
            st.session_state.conversation.append({"role": "user", "content": user_input.strip()})
            
            with st.spinner("Thinking..."):
                response = get_chat_response(user_input)
                st.session_state.conversation.append({
                    "role": "assistant", 
                    "content": response["answer"],
                    "details": {
                        "confidence": response["confidence"],
                        "source": response["source"],
                        "matched_questions": response.get("matched_questions", [])
                    }
                })
            
            st.rerun()

if __name__ == '__main__':
    main()