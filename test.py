import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from groq import Groq

# Set API keys (use environment variables in production)
os.environ["PINECONE_API_KEY"] = "pcsk_5CsWGm_DTETbjaHK7ZP6P2eQaMNL2JdUTKitPSuGC3Ntx3nwJNjcWLGsjwopHmUrV58r5D"
os.environ["GROQ_API_KEY"] = "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"

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medibot",
    embedding=embeddings,
)

# Create retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Set up the Groq client
groq_client = Groq(api_key="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")

# Define prompt template for medical assistant
system_prompt = (
    "You are a medical assistant. Answer ONLY using retrieved context. "
    "If no relevant information is found, say 'I don't know'. "
    "Use 2 sentences each. Provide a page number when applicable."
)

# Streamlit UI
st.title("MediChat - AI Medical Assistant")
st.write("Ask me any medical question!")

# Initialize chat history if not already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your question...")

if user_input:
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Retrieve relevant documents directly using the retriever
    try:
        retrieved_docs = retriever.get_relevant_documents(user_input)
        
        # Extract content from documents
        if retrieved_docs:
            context = "\n".join([doc.page_content for doc in retrieved_docs])
        else:
            context = "No relevant documents found."
            
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        context = "Error retrieving context."
    
    # Build prompt with retrieved context
    full_prompt = f"{system_prompt}\n\nRetrieved Information:\n{context}\n\nUser Question: {user_input}"
    
    # Use Groq for the assistant's response
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": full_prompt}
        ],
        model="gemma2-9b-it",
        max_tokens=150
    )
    ai_response = response.choices[0].message.content
    
    # Display AI response
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)
