import streamlit as st
import os
import tempfile
import random
import string

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


def get_file_type(filename):
    _, file_type = os.path.splitext(filename)
    return file_type.lower()


def file_load(uploaded_files):
    """read document"""
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    loader_map = {
        ".doc": Docx2txtLoader,
        ".docx": Docx2txtLoader,
        ".pdf": PyPDFLoader,
        ".csv": CSVLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".pptx": UnstructuredPowerPointLoader,
    }

    for file in uploaded_files:
        file_type = get_file_type(file.name)
        if file_type in loader_map:
            temp_path = os.path.join(temp_dir.name, file.name)
            with open(temp_path, "wb") as temp_file:
                temp_file.write(file.getvalue())
            loader = loader_map[file_type]
            docs.extend(loader(temp_path).load())

    return docs


def file_splitter(docs):
    """document split"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = splitter.split_documents(docs)

    return splits


def generate_random_code(length):
    characters = string.ascii_letters + string.digits
    code = "".join(random.choice(characters) for _ in range(length))
    return code


def embedding_to_vector(document_splits, docstore_id=None):
    # model = AzureOpenAIEmbeddings(
    #     api_key=os.getenv("AZURE_OPENAI_EMBD_KEY"),
    #     api_version="2023-03-15-preview",
    #     azure_endpoint=os.getenv("AZURE_OPENAI_EMBD_ENDPOINT"),
    #     azure_deployment="xiaochiao-emd3-embeddings-000",
    # )
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    model = HuggingFaceEmbeddings(model_name=model_name)
    persist_directory = f"./Chroma/{docstore_id}"
    vectorstore = Chroma.from_documents(
        documents=document_splits,
        embedding=model,
        persist_directory=persist_directory,
        collection_name="opengpt",
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5}
    )

    return retriever


# answer source extract
def source_extract(ai_response):
    sources = []
    for source in ai_response["source_documents"]:
        sources.append(os.path.split(source.metadata["source"])[1])
    unique_sources = list(set(sources))

    return unique_sources


# streamlit button status for button upload
def upload_initial_status_update():
    st.session_state.upload_initial = False


# Memory initial
document_chat_msgs = StreamlitChatMessageHistory()

# webpage title setting
st.set_page_config(page_title="OpenGPT Prototype", page_icon="📎")
st.title("Chat with Your Documents")


# 新對話視窗(新記憶位置)
if st.sidebar.button(label="New Chat", type="secondary"):
    document_chat_msgs.clear()
    del st.session_state.file_uploader
    del st.session_state.upload_initial
    del st.session_state.doc_store_id


if "upload_initial" not in st.session_state:
    st.session_state.upload_initial = True
    st.session_state.doc_store_id = generate_random_code(16)

# Files 在觸發 form_submit_button 才會改變
with st.sidebar.form(key="form_fileloader", clear_on_submit=True):
    files = st.file_uploader(
        label="File Loader",
        type=["docx", "pptx", "csv", "pdf", "xlsx"],
        accept_multiple_files=True,
        label_visibility="hidden",
        key="file_uploader",
    )
    form_btn_upload = st.form_submit_button(
        "Upload", type="primary", on_click=upload_initial_status_update
    )


if st.session_state.upload_initial:
    st.info("Please upload your documents to continue.")
    st.stop()

elif len(files) > 0:
    docs = file_load(files)
    splits = file_splitter(docs)
    st.session_state.retriever = embedding_to_vector(
        document_splits=splits, docstore_id=st.session_state.doc_store_id
    )
    st.info("Documents have already uploaded.")
    files = []

elif not files and form_btn_upload:
    st.warning("Oops, there are no documents.")


# ------------------------------
# LLM - Ollama(llama3)
# llm = ChatOllama(model="llama3")

# LLM-AzureOpenAI
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment="gpt-4-assistant",
)
# # LLM - OpenAI
# llm = ChatOpenAI(model="gpt-4-1106-preview", api_key=os.getenv("OPENAI_API"))

# Memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    chat_memory=document_chat_msgs,
    return_messages=True,
    output_key="answer",
)

# question_template
# question_template = """
# 1. You are a document assistant. You can reply based on my questions and refer to the documents.
# 2. Your response should be brief and concise.
# 3. If you don’t know, please don’t make it up or give arbitrary answers.
# {question}"""
# question_prompt = PromptTemplate(
#     template=question_template, input_variables=["question"]
# )

# system message template
sys_template = """
1. You are a document assistant. You can reply based on my questions and refer to the documents.
2. Your response should be brief and concise.
3. If you don’t know, please don’t make it up or give arbitrary answers.

The content of documents: {context}"""

sys_msg_template = SystemMessagePromptTemplate.from_template(template=sys_template)

# Chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=st.session_state.retriever,
    memory=memory,
    # condense_question_prompt=question_prompt,
    return_source_documents=True,
    verbose=True,
)

chain.combine_docs_chain.llm_chain.prompt.messages[0] = sys_msg_template


# Initialize st_chat history and create message container
if len(document_chat_msgs.messages) == 0:
    document_chat_msgs.add_ai_message("How can I help you?")

# Display history message
avatars = {"human": "user", "ai": "assistant"}
for msg in document_chat_msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

# User/AI Conversation
if prompt := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        response = chain.invoke(prompt)
        st.markdown(response["answer"])
        source = source_extract(response)
        st.sidebar.write(f"Reference sources: {source}")
