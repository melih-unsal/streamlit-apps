import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.document_loaders import UnstructuredPDFLoader
import tempfile

st.title("Ask to PDF")

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    placeholder="sk-...",
    type="password",
)

uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name
        st.session_state['file_path'] = file_path

def load_pdf(file_path):
    loader = UnstructuredPDFLoader(file_path, mode="elements", strategy="fast")
    docs = loader.load()
    return docs

pdf_doc = None
pdf_string = ""
answers = ""


if 'file_path' in st.session_state:
    pdf_doc = load_pdf(st.session_state['file_path'])
    pdf_string = "".join([doc.page_content for doc in pdf_doc])
    
question = st.text_input("Question")

def pdfQuestionAnswerer(question, pdf_string):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        openai_api_key=openai_api_key,
        temperature=0
    )
    system_template = """You are an assistant that can answer questions about a PDF file based on its content."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please provide answers to the following questions based on the content of the PDF file: '{pdf_string}'.
    
    Question : {question}"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(question=question, pdf_string=pdf_string)
    return result

if st.button("Submit"):
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API Key!", icon="⚠️")
    elif pdf_string and question:
        answers = pdfQuestionAnswerer(question,pdf_string)
        st.markdown(f"**Answer:** {answers}")