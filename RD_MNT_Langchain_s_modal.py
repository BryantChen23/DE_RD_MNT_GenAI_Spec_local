import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain


# Functions
def file_load(uploaded_files):
    """read document"""
    sup_filenames = []
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    for file in uploaded_files:
        if file.name.endswith(".docx"):
            sup_filenames.append(file.name)
            temp_path = os.path.join(temp_dir.name, file.name)
            with open(temp_path, "wb") as temp_file:
                temp_file.write(file.getvalue())
                loader = Docx2txtLoader(temp_path)
                docs.extend(loader.load())

    return sup_filenames, docs


def file_splitter(docs):
    """document split"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = splitter.split_documents(docs)

    return splits


def embedding_to_vector(document_splits):
    model = AzureOpenAIEmbeddings(
        api_key=os.getenv("AZURE_OPENAI_EMBD_KEY"),
        api_version="2023-03-15-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBD_ENDPOINT"),
        azure_deployment="xiaochiao-emd3-embeddings-000",
    )
    vectorstore = Chroma.from_documents(documents=document_splits, embedding=model)

    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
    )

    return retriever


# title setting
st.set_page_config(page_title="LangChain: Chat with your documents", page_icon="💻")
st.title(
    ":grinning_face_with_one_large_and_one_small_eye: Chat with Adorable Assistant"
)

# file selector
uploaded_files = st.sidebar.file_uploader(
    label="Select", type=["docx"], accept_multiple_files=True, label_visibility="hidden"
)

if not uploaded_files:
    st.info("Please upload documents to continue.")
    st.stop()

sup_filenames, docs = file_load(uploaded_files)
splits = file_splitter(docs)
retriever = embedding_to_vector(splits)


# AzureOpenAI
# llm = AzureChatOpenAI(
#     api_key=st.secrets["AZURE_OPENAI_KEY"],
#     api_version="2024-02-15-preview",
#     azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
#     azure_deployment="gpt-4-assistant",
# )

# OpenAI
llm = ChatOpenAI(model="gpt-4-1106-preview", api_key=os.getenv("OPENAI_API"))


qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, return_source_documents=True, verbose=True, retriever=retriever
)

# Initialize st_chat history and create message container
if "messages" not in st.session_state:
    st.session_state.messages = []
    with st.chat_message("assistant"):
        st.write("How can i help you?")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat history
chat_history = []

# Accept user input
if prompt := st.chat_input("Ask me anything!"):

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        result = qa_chain({"question": prompt, "chat_history": chat_history})
        response = result["answer"]
        st.markdown(response)

    # Add assistant message in chat message (streamlit)
    chat_history.append((prompt, response))
    st.session_state.messages.append({"role": "assistant", "content": response})
