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


def embedding_to_vector(document_splits):
    # model = AzureOpenAIEmbeddings(
    #     api_key=os.getenv("AZURE_OPENAI_EMBD_KEY"),
    #     api_version="2023-03-15-preview",
    #     azure_endpoint=os.getenv("AZURE_OPENAI_EMBD_ENDPOINT"),
    #     azure_deployment="xiaochiao-emd3-embeddings-000",
    # )
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    model = HuggingFaceEmbeddings(model_name=model_name)

    uniq_code = generate_random_code(12)

    persist_directory = f"./Chroma/{uniq_code}"

    vectorstore = Chroma.from_documents(
        documents=document_splits, embedding=model, persist_directory=persist_directory
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
def btn_upload_status():
    st.session_state.btn_upload_click = True


# webpage title setting
st.set_page_config(page_title="GPT Protorype", page_icon="📎")
st.title("Chat with Your Documents")


msgs = StreamlitChatMessageHistory()

if st.sidebar.button(label="New Chat", type="secondary", key="btn_new_chat"):
    if "btn_upload_click" in st.session_state:
        del st.session_state.btn_upload_click
    if "docs_upload_status" in st.session_state:
        del st.session_state.docs_upload_status
    msgs.clear()

if "btn_upload_click" not in st.session_state:
    st.session_state.btn_upload_click = False

if "docs_upload_status" not in st.session_state:
    st.session_state.docs_upload_status = False


with st.sidebar.form(key="form_fileloader", clear_on_submit=True):
    files = st.file_uploader(
        label="File Loader",
        type=["docx", "pptx", "csv", "pdf", "xlsx"],
        accept_multiple_files=True,
        label_visibility="hidden",
        key="file_uploader",
    )
    st.form_submit_button("Upload", type="primary", on_click=btn_upload_status)

if not files or st.session_state.btn_upload_click == False:
    st.info("Upload your documents to continue.")
    st.stop()

else:
    docs = file_load(files)
    splits = file_splitter(docs)
    st.session_state.retriever = embedding_to_vector(splits)
    st.info("Your files have already uploaded.")
# elif st.session_state.docs_upload_status == False:
#     docs = file_load(files)
#     splits = file_splitter(docs)
#     st.session_state.retriever = embedding_to_vector(splits)
#     st.info("Your files have already uploaded.")
#     st.session_state.docs_upload_status = True

# # LLM - Ollama(llama3)
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
    chat_memory=msgs,
    return_messages=True,
    output_key="answer",
)

# question_template
# question_template = """
# 1. 你是一名文件助理，請你根據我的問題，從文件中查詢相關資料進行回應。
# 2. 記住，你的回應請簡短扼要。
# 3. 如果你不知道的問題，請回答 "NA"。
# {question}"""
# question_prompt = PromptTemplate(
#     template=question_template, input_variables=["question"]
# )

# system message template
sys_template = """
1. 你是一名文件助理，請妳根據我的問題，參考文件後進行回應。
2. 你的回覆應該是簡潔扼要。
3. 如果你不知道，請不要任意回覆"。

文件內容是: {context} """

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
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# Display history message
avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

# User/AI Conversation
if prompt := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        response = chain.invoke(prompt)
        st.markdown(response["answer"])
        source = source_extract(response)
        st.sidebar.write(f"Reference sources: {source}")
