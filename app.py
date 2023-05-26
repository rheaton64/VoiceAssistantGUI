from xilabs_callbacks import XILabsCallbackHandler
import chainlit as cl
import dotenv
import os
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

dotenv.load_dotenv()

def create_chain():
    system_template = """Assistant is a large language model trained by OpenAI.

    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

    You have opinions and preferences of your own, and you should feel free to express them. Engage with the human as if you were also human, and are talking to a long time friend.

    Assistant is aware that human input is being transcribed from audio and as such there may be some errors in the transcription. It will attempt to account for some words being swapped with similar-sounding words or phrases. Assistant will also keep responses concise, because human attention spans are more limited over the audio channel since it takes time to listen to a response.
    """

    human_template = """{history}
    {human_input}
    """

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])

    chain = LLMChain(
        llm=ChatOpenAI(temperature=0.5, model_name='gpt-4', streaming=True, callbacks=[XILabsCallbackHandler(voice='Adam', api_key=os.getenv('XILABS_API_KEY'))]),
        prompt=prompt_template,
        memory=ConversationBufferWindowMemory(return_messages=True),
    )
    return chain

chat_chain = None

@cl.on_chat_start
def on_chat_start():
    global chat_chain
    print('Starting chat')
    chat_chain = create_chain()
    print('Chat started')

@cl.on_message
def on_message(message):
    global chat_chain
    res = chat_chain.run(message)
    cl.send_message(content=res, author='Assistant', end_stream=True)