import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain_experimental.tools import PythonREPLTool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ChatMessageHistory
from PyPDF2 import PdfReader
from PIL import Image
import docx
import tempfile
import streamlit as st
import requests

# ✅ Set the actual title of your app
st.set_page_config(page_title="AI File Analyst", page_icon="📂")
st.title("📂 AI File Analyst")
st.caption("Upload your data file\nDrag and drop file here\nLimit 200MB per file • CSV, XLSX, PDF, DOCX, TXT AND UPTO 1MB for JPG, JPEG, PNG")

# ✅ Set API key (either from secrets or fallback for local testing)
os.environ['GROQ_API_KEY'] = st.secrets.get("GROQ_API_KEY", "your_default_api_key_here")

# ✅ OCR.Space function

def resize_image_if_needed(file_path, max_size_kb=1024):
    size_kb = os.path.getsize(file_path) / 1024
    if size_kb <= max_size_kb:
        return file_path

    image = Image.open(file_path)
    quality = 95
    while size_kb > max_size_kb and quality > 10:
        resized_path = file_path.replace(".", "_resized.")
        image.save(resized_path, optimize=True, quality=quality)
        size_kb = os.path.getsize(resized_path) / 1024
        quality -= 5
    return resized_path


def ocr_space_image(file_path, api_key):
    file_path = resize_image_if_needed(file_path)

    payload = {
        'isOverlayRequired': False,
        'apikey': api_key,
        'language': 'eng',
    }
    with open(file_path, 'rb') as f:
        response = requests.post(
            'https://api.ocr.space/parse/image',
            files={file_path: f},
            data=payload,
        )

    result = response.json()

    if result.get('IsErroredOnProcessing'):
        error_msg = result.get('ErrorMessage', ['Unknown error'])[0]
        raise Exception(f"OCR failed: {error_msg}")

    if 'ParsedResults' not in result or not result['ParsedResults']:
        raise Exception("OCR failed: No ParsedResults in response")

    return result['ParsedResults'][0].get('ParsedText', '')



# ✅ Function to process different types of files
def process_file(file_path, original_filename=None):
    extension = original_filename.split('.')[-1].lower() if original_filename else file_path.split('.')[-1].lower()
    try:
        if extension == 'csv':
            df = pd.read_csv(file_path)
            return 'tabular', df, f"df = pd.read_csv('{file_path}')"
        elif extension == 'xlsx':
            df = pd.read_excel(file_path)
            return 'tabular', df, f"df = pd.read_excel('{file_path}')"
        elif extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return 'text', text, None
        elif extension == 'docx':
            doc = docx.Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs])
            return 'text', text, None
        elif extension == 'pdf':
            reader = PdfReader(file_path)
            text = ''.join([page.extract_text() or '' for page in reader.pages])
            return 'text', text, None
        elif extension in ['jpg', 'jpeg', 'png']:
            api_key = st.secrets.get("OCR_SPACE_API_KEY", "your_default_api_key_here")
            text = ocr_space_image(file_path, api_key)
            return 'text', text, None
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")

# ✅ Store chat history across sessions
session_histories = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

# ✅ Load LLM
llm = ChatGroq(model="llama3-70b-8192", api_key=os.environ['GROQ_API_KEY'])

# ✅ Global vars
executor = None
data_type = None
data = None

# ✅ File upload
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx", "pdf", "docx", "txt", "jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.success(f"Uploaded file: {uploaded_file.name}")

    try:
        data_type, data, load_instruction = process_file(tmp_file_path, uploaded_file.name)
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        st.stop()

    if data_type == 'tabular':
        st.subheader("📊 Data Preview")
        st.dataframe(data.head())

        repl_globals = {'pd': pd, 'np': np, 'plt': plt, 'df': data}
        python_repl = PythonREPLTool(globals_dict=repl_globals)
        tools = [python_repl]
        memory = ConversationBufferMemory(return_messages=True)

        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
            agent_kwargs={
                "prefix": (
                    "You are a smart data analyst who can read files and answer questions. "
                    f"Load the DataFrame using: {load_instruction}. "
                    "Use the Python_REPL tool to explore and answer all questions. "
                )
            }
        )
        executor = agent

    else:
        st.subheader("📄 Text successfully extracted from the file.")
        st.info("You're all set. Ask a question about your file below!")

        prompt = PromptTemplate.from_template(
            "You are a helpful data analyst. The following text was extracted from a file: {text}."
            "Answer the question: {question}"
        )
        chain = prompt | llm
        executor = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )

    st.info("You're all set. Ask a question below!")

    question = st.text_input("Ask a question about your file:")
    if st.button("Submit") and question:
        try:
            if data_type == 'tabular':
                response = executor.invoke({"input": question})['output']
            else:
                response = executor.invoke(
                    {"text": data, "question": question},
                    config={"configurable": {"session_id": "default"}}
                ).content
            st.success("Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"Agent error: {str(e)}")
