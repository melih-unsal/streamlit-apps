import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.document_loaders import WebBaseLoader

st.title("Web Blogger")

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    placeholder="sk-...",
    type="password",
)

# Initialize state variables
url = st.text_input("Enter website URL")
website_string = ""
blog_post = ""

# Load website content as Document object from the URL
def load_website(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

# Generate medium blog post from the website content
def mediumBlogPostGenerator(website_string):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        openai_api_key=openai_api_key,
        temperature=0.7
    )
    system_template = """You are an assistant designed to generate a medium blog post from the given website content."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please generate a medium blog post based on the content of the website: '{website_string}'."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(website_string=website_string)
    return result # returns string   

# Get input from the user
if not openai_api_key.startswith("sk-"):
    st.warning("Please enter your OpenAI API Key!", icon="⚠️")
else:
    if url:
        website_doc = load_website(url)
        website_string = "".join([doc.page_content for doc in website_doc])

    # Put a submit button with an appropriate title
    if st.button("Generate Blog Post"):
        if website_string:
            blog_post = mediumBlogPostGenerator(website_string)

    # Call functions only if all user inputs are taken and the button is clicked
    st.markdown(blog_post)