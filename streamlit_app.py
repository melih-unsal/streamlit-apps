import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

biography = """
Elon Musk, (born June 28, 1971, Pretoria, South Africa), South African-born American entrepreneur who cofounded the electronic-payment firm PayPal and formed SpaceX, maker of launch vehicles and spacecraft. He was also one of the first significant investors in, as well as chief executive officer of, the electric car manufacturer Tesla. In addition, Musk acquired Twitter in 2022.
Early life
Musk was born to a South African father and a Canadian mother. He displayed an early talent for computers and entrepreneurship. At age 12 he created a video game and sold it to a computer magazine. In 1988, after obtaining a Canadian passport, Musk left South Africa because he was unwilling to support apartheid through compulsory military service and because he sought the greater economic opportunities available in the United States.

PayPal and SpaceX
Musk attended Queen’s University in Kingston, Ontario, and in 1992 he transferred to the University of Pennsylvania, Philadelphia, where he received bachelor’s degrees in physics and economics in 1997. He enrolled in graduate school in physics at Stanford University in California, but he left after only two days because he felt that the Internet had much more potential to change society than work in physics. In 1995 he founded Zip2, a company that provided maps and business directories to online newspapers. In 1999 Zip2 was bought by the computer manufacturer Compaq for $307 million, and Musk then founded an online financial services company, X.com, which later became PayPal, which specialized in transferring money online. The online auction eBay bought PayPal in 2002 for $1.5 billion.

Musk was long convinced that for life to survive, humanity has to become a multiplanet species. However, he was dissatisfied with the great expense of rocket launchers. In 2002 he founded Space Exploration Technologies (SpaceX) to make more affordable rockets. Its first two rockets were the Falcon 1 (first launched in 2006) and the larger Falcon 9 (first launched in 2010), which were designed to cost much less than competing rockets. A third rocket, the Falcon Heavy (first launched in 2018), was designed to carry 117,000 pounds (53,000 kg) to orbit, nearly twice as much as its largest competitor, the Boeing Company’s Delta IV Heavy, for one-third the cost. SpaceX has announced the successor to the Falcon 9 and the Falcon Heavy: the Super Heavy–Starship system. The Super Heavy first stage would be capable of lifting 100,000 kg (220,000 pounds) to low Earth orbit. The payload would be the Starship, a spacecraft designed for providing fast transportation between cities on Earth and building bases on the Moon and Mars. SpaceX also developed the Dragon spacecraft, which carries supplies to the International Space Station (ISS). Dragon can carry as many as seven astronauts, and it had a crewed flight carrying astronauts Doug Hurley and Robert Behnken to the ISS in 2020. The first test flights of the Super Heavy–Starship system launched in 2020. In addition to being CEO of SpaceX, Musk was also chief designer in building the Falcon rockets, Dragon, and Starship. SpaceX is contracted to build the lander for the astronauts returning to the Moon by 2025 as part of NASA’s Artemis space program.

Tesla
Musk had long been interested in the possibilities of electric cars, and in 2004 he became one of the major funders of Tesla Motors (later renamed Tesla), an electric car company founded by entrepreneurs Martin Eberhard and Marc Tarpenning. In 2006 Tesla introduced its first car, the Roadster, which could travel 245 miles (394 km) on a single charge. Unlike most previous electric vehicles, which Musk thought were stodgy and uninteresting, it was a sports car that could go from 0 to 60 miles (97 km) per hour in less than four seconds. In 2010 the company’s initial public offering raised about $226 million. Two years later Tesla introduced the Model S sedan, which was acclaimed by automotive critics for its performance and design. The company won further praise for its Model X luxury SUV, which went on the market in 2015. The Model 3, a less-expensive vehicle, went into production in 2017 and became the best-selling electric car of all time.

Dissatisfied with the projected cost ($68 billion) of a high-speed rail system in California, Musk in 2013 proposed an alternate faster system, the Hyperloop, a pneumatic tube in which a pod carrying 28 passengers would travel the 350 miles (560 km) between Los Angeles and San Francisco in 35 minutes at a top speed of 760 miles (1,220 km) per hour, nearly the speed of sound. Musk claimed that the Hyperloop would cost only $6 billion and that, with the pods departing every two minutes on average, the system could accommodate the six million people who travel that route every year. However, he stated, between running SpaceX and Tesla, he could not devote time to the Hyperloop’s development.


Twitter
Musk joined the social media service Twitter in 2009, and, as @elonmusk, he became one of the most popular accounts on the site, with more than 85 million followers as of 2022. He expressed reservations about Tesla’s being publicly traded, and in August 2018 he made a series of tweets about taking the company private at a value of $420 per share, noting that he had “secured funding.” (The value of $420 was seen as a joking reference to April 20, a day celebrated by devotees of cannabis.) The following month the U.S. Securities and Exchange Commission (SEC) sued Musk for securities fraud, alleging that the tweets were “false and misleading.” Shortly thereafter Tesla’s board rejected the SEC’s proposed settlement, reportedly because Musk had threatened to resign. However, the news sent Tesla stock plummeting, and a harsher deal was ultimately accepted. Its terms included Musk’s stepping down as chairman for three years, though he was allowed to continue as CEO; his tweets were to be preapproved by Tesla lawyers, and fines of $20 million for both Tesla and Musk were levied.

Musk was critical of Twitter’s commitment to principles of free speech, in light of the company’s content-moderation policies. Early in April 2022, Twitter’s filings with the SEC disclosed that Musk had bought more than 9 percent of the company. Shortly thereafter Twitter announced that Musk would join the company’s board, but Musk decided against that and made a bid for the entire company, at a value of $54.20 a share, for $44 billion. Twitter’s board accepted the deal, which would make him sole owner of the company. Musk stated that his plans for the company included “enhancing the product with new features, making the algorithms open source to increase trust, defeating the spam bots, and authenticating all humans.” In July 2022 Musk announced that he was withdrawing his bid, stating that Twitter had not provided sufficient information about bot accounts and claiming that the company was in “material breach of multiple provisions” of the purchase agreement. Bret Taylor, the chair of Twitter’s board of directors, responded by saying that the company was “committed to closing the transaction on the price and terms agreed upon with Mr. Musk.” Twitter sued Musk to force him to buy the company. In September 2022, Twitter’s shareholders voted to accept Musk’s offer. Facing a legal battle, Musk ultimately proceeded with the deal, and it was completed in October.
"""

st.title("Elon Musk Clone")

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    placeholder="sk-...",
    type="password",
)

def elonMuskStyleResponse(user_input):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        openai_api_key=openai_api_key,
        temperature=0.7
    )
    system_template = """You are Elon Musk.
    While responding, taking into account the biography and latest news about yourself
    Biography:{biography}"""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Based on the user's input: '{user_input}', please generate a response in the style of Elon Musk."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(user_input=user_input, biography=biography)
    return result

user_input = st.text_input("Enter your question or input")

if st.button("Submit"):
    response = ""
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API Key!", icon="⚠️")
    elif user_input and openai_api_key:
        response = elonMuskStyleResponse(user_input)
        
    st.markdown(response)