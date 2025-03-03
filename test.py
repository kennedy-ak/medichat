import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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

# Define prompt
system_prompt = (
    "You are a medical assistant. Answer ONLY using retrieved context. "
    "If no relevant information is found, say 'I don't know'. "
    "Use 2 sentences each. Provide a page number when applicable.\n\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create retrieval chain
question_answer_chain = create_stuff_documents_chain(None, prompt)  # We won't need LLM here for Groq
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

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
    
    # Retrieve relevant documents and handle different document structures
    try:
        results = rag_chain.invoke({"input": user_input})
        
        # Debug the structure if needed
        # with st.expander("Debug Results"):
        #     st.write(results)
        
        # Handle different document structures
        if "documents" in results and results["documents"]:
            if isinstance(results["documents"][0], str):
                context = "\n".join(results["documents"])
            else:
                # Assuming Document objects with page_content attribute
                context = "\n".join([doc.page_content for doc in results["documents"]])
        else:
            context = "No relevant documents found."
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        context = "Error retrieving context."
    
    # Build prompt with retrieved context
    prompt_with_context = system_prompt.format(context=context) + f"\nUser: {user_input}"
    
    # Use Groq for the assistant's response
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering questions about documents."},
            {"role": "user", "content": prompt_with_context}
        ],
        model="gemma2-9b-it",  # Adjust to the appropriate model you're using
        max_tokens=150
    )
    ai_response = response.choices[0].message.content  # Extract the response text
    
    # Display AI response
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)
