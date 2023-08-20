import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

st.title("Sentiment Analyzer")

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    placeholder="sk-...",
    type="password",
)

def sentimentAnalyzer(sentence):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        openai_api_key=openai_api_key,
        temperature=0
    )
    system_template = """You are a sentiment analysis tool. Your task is to analyze the sentiment of the given sentence."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please perform sentiment analysis on the following sentence: '{sentence}'."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(sentence=sentence)
    return result

sentence = st.text_input("Enter a sentence:")
sentiment_analysis = ""

if st.button("Analyze Sentiment"):
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API Key!", icon="⚠️")
    elif sentence:
        sentiment_analysis = sentimentAnalyzer(sentence)
    st.markdown(sentiment_analysis)