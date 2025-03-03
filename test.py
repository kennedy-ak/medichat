# # import streamlit as st
# # import os
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# # from langchain_pinecone import PineconeVectorStore
# # from groq import Groq

# # # Set API keys (use environment variables in production)
# # os.environ["PINECONE_API_KEY"] = "pcsk_5CsWGm_DTETbjaHK7ZP6P2eQaMNL2JdUTKitPSuGC3Ntx3nwJNjcWLGsjwopHmUrV58r5D"
# # os.environ["GROQ_API_KEY"] = "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"

# # # Load embedding model
# # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # # Load existing Pinecone index
# # docsearch = PineconeVectorStore.from_existing_index(
# #     index_name="medibot",
# #     embedding=embeddings,
# # )

# # # Create retriever
# # retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# # # Set up the Groq client
# # groq_client = Groq(api_key="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")

# # # Define prompt template for medical assistant
# # system_prompt = (
# #     "You are a medical assistant. Answer ONLY using retrieved context. "
# #     "If no relevant information is found, say 'I don't know'. "
# #     "Use 5 sentences each for general answers. "
# #     "For lists, processes, or step-by-step instructions: "
# #     "1. Clearly number each step or item "
# #     "2. Keep each point concise and focused on one action or piece of information "
# #     "3. Maintain the exact order as presented in the source material "
# #     "4. Include all relevant steps or items from the source material "
# #     "Provide the source page number when applicable using format (Page X). "
# #     "Always include the source page number for each piece of information you use. "
# #     "If multiple pages contain relevant information, cite each source clearly: (Page X), (Page Y). "
# #     "If information comes from different sections, organize your answer by topic and include all relevant page references."
# # )

# # # Streamlit UI
# # st.title("MediChat - AI Medical Assistant")
# # st.write("Ask me any medical question!")

# # # Initialize chat history if not already
# # if "chat_history" not in st.session_state:
# #     st.session_state.chat_history = []

# # # Initialize a session state for showing sources
# # if "show_sources" not in st.session_state:
# #     st.session_state.show_sources = False

# # # Display chat history
# # for message in st.session_state.chat_history:
# #     with st.chat_message(message["role"]):
# #         st.markdown(message["content"])
# #         # Display sources if they exist and are meant to be shown
# #         if message.get("sources") and st.session_state.show_sources:
# #             with st.expander("Sources"):
# #                 for i, source in enumerate(message["sources"]):
# #                     st.markdown(f"**Source {i+1}:**\n{source}")

# # # Add a toggle for showing sources
# # st.sidebar.title("Settings")
# # st.session_state.show_sources = st.sidebar.checkbox("Show Sources", value=st.session_state.show_sources)

# # # User input
# # user_input = st.chat_input("Type your question...")

# # if user_input:
# #     # Display user message
# #     st.session_state.chat_history.append({"role": "user", "content": user_input})
# #     with st.chat_message("user"):
# #         st.markdown(user_input)
    
# #     # Retrieve relevant documents directly using the retriever
# #     try:
# #         retrieved_docs = retriever.get_relevant_documents(user_input)
        
# #         # Extract content from documents and format with page numbers
# #         if retrieved_docs:
# #             formatted_docs = []
# #             sources = []
            
# #             for i, doc in enumerate(retrieved_docs):
# #                 # Extract metadata - assuming metadata contains page info
# #                 metadata = doc.metadata
# #                 page_info = metadata.get('page', f'Document {i+1}')
                
# #                 # Format the document with page info
# #                 formatted_text = f"[Page {page_info}]: {doc.page_content}"
# #                 formatted_docs.append(formatted_text)
                
# #                 # Store source information for display
# #                 source_info = f"Page {page_info}: {doc.page_content}"
# #                 if metadata.get('source'):
# #                     source_info = f"Source: {metadata.get('source')}\n{source_info}"
# #                 sources.append(source_info)
                
# #             context = "\n\n".join(formatted_docs)
# #         else:
# #             context = "No relevant documents found."
# #             sources = ["No sources found"]
            
# #     except Exception as e:
# #         st.error(f"Error retrieving context: {e}")
# #         context = "Error retrieving context."
# #         sources = ["Error retrieving sources"]
    
# #     # Build prompt with retrieved context
# #     full_prompt = f"{system_prompt}\n\nRetrieved Information:\n{context}\n\nUser Question: {user_input}"
    
# #     # Use Groq for the assistant's response
# #     response = groq_client.chat.completions.create(
# #         messages=[
# #             {"role": "system", "content": "You are a helpful medical assistant."},
# #             {"role": "user", "content": full_prompt}
# #         ],
# #         model="gemma2-9b-it",
# #         max_tokens=150
# #     )
# #     ai_response = response.choices[0].message.content
    
# #     # Display AI response
# #     message_with_sources = {
# #         "role": "assistant", 
# #         "content": ai_response,
# #         "sources": sources
# #     }
# #     st.session_state.chat_history.append(message_with_sources)
    
# #     with st.chat_message("assistant"):
# #         st.markdown(ai_response)
# #         if st.session_state.show_sources:
# #             with st.expander("Sources"):
# #                 for i, source in enumerate(sources):
# #                     st.markdown(f"**Source {i+1}:**\n{source}")


# import streamlit as st
# import os
# import base64
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from groq import Groq

# # Set API keys (use environment variables in production)
# os.environ["PINECONE_API_KEY"] = "pcsk_5CsWGm_DTETbjaHK7ZP6P2eQaMNL2JdUTKitPSuGC3Ntx3nwJNjcWLGsjwopHmUrV58r5D"
# os.environ["GROQ_API_KEY"] = "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"

# # Load embedding model
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Load existing Pinecone index
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name="medibot",
#     embedding=embeddings,
# )

# # Create retriever
# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# # Set up the Groq client
# groq_client = Groq(api_key="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")

# # Define improved system prompt
# system_prompt = (
#     "You are a medical assistant. Answer ONLY using retrieved context. "
#     "If no relevant information is found, say 'I don't know'. "
#     "Use 2 sentences each for general answers. "
#     "For lists, processes, or step-by-step instructions: "
#     "1. Clearly number each step or item "
#     "2. Keep each point concise and focused on one action or piece of information "
#     "3. Maintain the exact order as presented in the source material "
#     "4. Include all relevant steps or items from the source material "
#     "Provide the source page number when applicable using format (Page X). "
#     "Always include the source page number for each piece of information you use. "
#     "If multiple pages contain relevant information, cite each source clearly: (Page X), (Page Y). "
#     "If information comes from different sections, organize your answer by topic and include all relevant page references."
# )

# # Function to create a download link for a file
# def get_binary_file_downloader_html(bin_file, file_label='File'):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     bin_str = base64.b64encode(data).decode()
#     href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
#     return href

# # Dictionary of available reference materials (file path and display name)
# reference_materials = {
#     "medical_textbook": {
#         "path": "path/to/medical_textbook.pdf",
#         "display_name": "Medical Textbook",
#         "description": "Comprehensive medical reference covering general medicine topics."
#     },
#     "pharmacology_guide": {
#         "path": "path/to/pharmacology_guide.pdf",
#         "display_name": "Pharmacology Guide",
#         "description": "Detailed information about medications, interactions, and dosages."
#     },
#     "clinical_procedures": {
#         "path": "path/to/clinical_procedures.pdf",
#         "display_name": "Clinical Procedures Manual",
#         "description": "Step-by-step guides for common medical procedures."
#     }
# }

# # System information
# system_info = {
#     "name": "MediChat",
#     "version": "1.0.0",
#     "description": "An AI-powered medical assistant that provides evidence-based medical information from trusted reference materials.",
#     "capabilities": [
#         "Answer medical questions with citations to source materials",
#         "Provide step-by-step medical procedures when available",
#         "Explain medical terminology and concepts",
#         "Offer information about medications, treatments, and conditions"
#     ],
#     "limitations": [
#         "Not a substitute for professional medical advice",
#         "Limited to information in the reference materials",
#         "No diagnostic capabilities",
#         "Cannot prescribe medications "
#     ],
#     "data_sources": "Trained on trusted medical textbooks and reference materials that are available for download in the sidebar.",
#     "technology": "Powered by Retrieval-Augmented Generation (RAG) using Pinecone vector database and Groq's AI model."
# }

# # Streamlit UI
# st.title("MediChat - AI Medical Assistant")

# # App information section with expander
# with st.expander("About MediChat", expanded=True):
#     st.write(f"**{system_info['name']} v{system_info['version']}**")
#     st.write(system_info['description'])
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Capabilities")
#         for item in system_info['capabilities']:
#             st.write(f"✓ {item}")
    
#     with col2:
#         st.subheader("Limitations")
#         for item in system_info['limitations']:
#             st.write(f"⚠️ {item}")
    
#     st.subheader("How It Works")
#     st.write(system_info['technology'])
#     st.write(system_info['data_sources'])
    
#     st.warning("**Disclaimer**: This application is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")

# st.write("Ask me any medical question!")

# # Sidebar with settings and downloads
# st.sidebar.title("Settings")

# # Toggle for showing sources
# if "show_sources" not in st.session_state:
#     st.session_state.show_sources = False
# st.session_state.show_sources = st.sidebar.checkbox("Show Sources", value=st.session_state.show_sources)

# # Reference materials section
# st.sidebar.title("Reference Materials")
# st.sidebar.write("Download the reference materials used to train this system:")

# # Check if files exist before offering downloads
# for ref_id, ref_info in reference_materials.items():
#     if os.path.exists(ref_info["path"]):
#         st.sidebar.markdown(f"**{ref_info['display_name']}**")
#         st.sidebar.write(ref_info["description"])
#         st.sidebar.markdown(get_binary_file_downloader_html(ref_info["path"], f"Download {ref_info['display_name']}"), unsafe_allow_html=True)
#     else:
#         st.sidebar.markdown(f"**{ref_info['display_name']}** (Not available for download)")

# # Initialize chat history if not already
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Display chat history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         # Display sources if they exist and are meant to be shown
#         if message.get("sources") and st.session_state.show_sources:
#             with st.expander("Sources"):
#                 for i, source in enumerate(message["sources"]):
#                     st.markdown(f"**Source {i+1}:**\n{source}")

# # User input
# user_input = st.chat_input("Type your question...")

# if user_input:
#     # Display user message
#     st.session_state.chat_history.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.markdown(user_input)
    
#     # Retrieve relevant documents directly using the retriever
#     try:
#         retrieved_docs = retriever.get_relevant_documents(user_input)
        
#         # Extract content from documents and format with page numbers
#         if retrieved_docs:
#             formatted_docs = []
#             sources = []
            
#             for i, doc in enumerate(retrieved_docs):
#                 # Extract metadata - assuming metadata contains page info
#                 metadata = doc.metadata
#                 page_info = metadata.get('page', f'Document {i+1}')
#                 source_name = metadata.get('source', 'Unknown source')
                
#                 # Format the document with page info
#                 formatted_text = f"[Page {page_info} from {source_name}]: {doc.page_content}"
#                 formatted_docs.append(formatted_text)
                
#                 # Store source information for display
#                 source_info = f"Page {page_info} from {source_name}: {doc.page_content}"
#                 sources.append(source_info)
                
#             context = "\n\n".join(formatted_docs)
#         else:
#             context = "No relevant documents found."
#             sources = ["No sources found"]
            
#     except Exception as e:
#         st.error(f"Error retrieving context: {e}")
#         context = "Error retrieving context."
#         sources = ["Error retrieving sources"]
    
#     # Build prompt with retrieved context
#     full_prompt = f"{system_prompt}\n\nRetrieved Information:\n{context}\n\nUser Question: {user_input}"
    
#     # Use Groq for the assistant's response
#     response = groq_client.chat.completions.create(
#         messages=[
#             {"role": "system", "content": "You are a helpful medical assistant."},
#             {"role": "user", "content": full_prompt}
#         ],
#         model="gemma2-9b-it",
#         max_tokens=150
#     )
#     ai_response = response.choices[0].message.content
    
#     # Display AI response
#     message_with_sources = {
#         "role": "assistant", 
#         "content": ai_response,
#         "sources": sources
#     }
#     st.session_state.chat_history.append(message_with_sources)
    
#     with st.chat_message("assistant"):
#         st.markdown(ai_response)
#         if st.session_state.show_sources:
#             with st.expander("Sources"):
#                 for i, source in enumerate(sources):
#                     st.markdown(f"**Source {i+1}:**\n{source}")


import streamlit as st
import os
import base64
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

# Define improved system prompt
system_prompt = (
    "You are a medical assistant. Answer ONLY using retrieved context. "
    "If no relevant information is found, say 'I don't know'. "
    "Use 2 sentences each for general answers. "
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

# Updated reference materials - now using just one book with a HuggingFace link
reference_materials = {
    "medical_textbook": {
        "link": "https://huggingface.co/datasets/kennedy-ak/medichat",
        "display_name": "Medical Reference",
        "description": "Comprehensive medical reference covering general medicine topics."
    }
}

# System information
system_info = {
    "name": "MediChat",
    "version": "1.0.0",
    "description": "An AI-powered medical assistant that provides evidence-based medical information from trusted reference materials.",
    "capabilities": [
        "Answer medical questions with citations to source materials",
        "Provide step-by-step medical procedures when available",
        "Explain medical terminology and concepts",
        "Offer information about medications, treatments, and conditions"
    ],
    "limitations": [
        "Not a substitute for professional medical advice",
        "Limited to information in the reference materials",
        "No diagnostic capabilities",
        "Cannot prescribe medications "
    ],
    "data_sources": "Trained on trusted medical textbooks and reference materials that are available in the sidebar.",
    "technology": "Powered by Retrieval-Augmented Generation (RAG) using Pinecone vector database and Groq's AI model."
}

# Streamlit UI
st.title("MediChat - AI Medical Assistant")

# App information section with expander
with st.expander("About MediChat", expanded=True):
    st.write(f"**{system_info['name']} v{system_info['version']}**")
    st.write(system_info['description'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Capabilities")
        for item in system_info['capabilities']:
            st.write(f"✓ {item}")
    
    with col2:
        st.subheader("Limitations")
        for item in system_info['limitations']:
            st.write(f"⚠️ {item}")
    
    st.subheader("How It Works")
    st.write(system_info['technology'])
    st.write(system_info['data_sources'])
    
    st.warning("**Disclaimer**: This application is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")

st.write("Ask me any medical question!")

# Sidebar with settings and downloads
st.sidebar.title("Settings")

# Toggle for showing sources
if "show_sources" not in st.session_state:
    st.session_state.show_sources = False
st.session_state.show_sources = st.sidebar.checkbox("Show Sources", value=st.session_state.show_sources)

# Reference materials section
st.sidebar.title("Reference Materials")
st.sidebar.write("Access the reference materials used to train this system:")

# Display reference material with link to HuggingFace
for ref_id, ref_info in reference_materials.items():
    st.sidebar.markdown(f"**{ref_info['display_name']}**")
    st.sidebar.write(ref_info["description"])
    st.sidebar.markdown(f"[Access on HuggingFace]({ref_info['link']})")

# Initialize chat history if not already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources if they exist and are meant to be shown
        if message.get("sources") and st.session_state.show_sources:
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:**\n{source}")

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
                source_name = metadata.get('source', 'Unknown source')
                
                # Format the document with page info
                formatted_text = f"[Page {page_info} from {source_name}]: {doc.page_content}"
                formatted_docs.append(formatted_text)
                
                # Store source information for display
                source_info = f"Page {page_info} from {source_name}: {doc.page_content}"
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
