import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
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
    
    "For lists, processes, or step-by-step instructions: "
    "1. Clearly number each step or item "
    "2. Keep each point concise and focused on one action or piece of information "
    "3. Maintain the exact order as presented in the source material "
    "4. Include all relevant steps or items from the source material "
    "Provide the source page number when applicable using format (Page X). "
    "Always include the source page number for each piece of information you use. "
    "If multiple pages contain relevant information, cite each source clearly: (Page X), (Page Y). "
    "If information comes from different sections, organize your answer by topic and include all relevant page references."
)

# Streamlit UI
st.title("MediChat - AI Medical Assistant")
st.write("Ask me any medical question!")

# Initialize chat history if not already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize a session state for showing sources
if "show_sources" not in st.session_state:
    st.session_state.show_sources = False

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources if they exist and are meant to be shown
        if message.get("sources") and st.session_state.show_sources:
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:**\n{source}")

# Add a toggle for showing sources
st.sidebar.title("Settings")
st.session_state.show_sources = st.sidebar.checkbox("Show Sources", value=st.session_state.show_sources)

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
        
        # Extract content from documents and format with page numbers
        if retrieved_docs:
            formatted_docs = []
            sources = []
            
            for i, doc in enumerate(retrieved_docs):
                # Extract metadata - assuming metadata contains page info
                metadata = doc.metadata
                page_info = metadata.get('page', f'Document {i+1}')
                
                # Format the document with page info
                formatted_text = f"[Page {page_info}]: {doc.page_content}"
                formatted_docs.append(formatted_text)
                
                # Store source information for display
                source_info = f"Page {page_info}: {doc.page_content}"
                if metadata.get('source'):
                    source_info = f"Source: {metadata.get('source')}\n{source_info}"
                sources.append(source_info)
                
            context = "\n\n".join(formatted_docs)
        else:
            context = "No relevant documents found."
            sources = ["No sources found"]
            
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        context = "Error retrieving context."
        sources = ["Error retrieving sources"]
    
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
    message_with_sources = {
        "role": "assistant", 
        "content": ai_response,
        "sources": sources
    }
    st.session_state.chat_history.append(message_with_sources)
    
    with st.chat_message("assistant"):
        st.markdown(ai_response)
        if st.session_state.show_sources:
            with st.expander("Sources"):
                for i, source in enumerate(sources):
                    st.markdown(f"**Source {i+1}:**\n{source}")
