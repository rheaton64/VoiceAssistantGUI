from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
import pyperclip

def load_clip_chain():
    system_template = """You are an AI classification assistant that can help determine the intent of a user's query.
    This query is given to a larger and more powerful AI assistant, which will then respond to the query.
    You only respond with answers to questions, and only respond with the answer '<Yes>' or '<No>'.
    You will never respond with an answer that is not exactly '<Yes>' or '<No>'.
    You do not provide any other information other than the answer to the question."""

    clipboard_template = """In answering the following query, will the Assistant need to access the user's clipboard?
    Query: {query}"""

    clip_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessagePromptTemplate.from_template(clipboard_template),
    ])

    clip_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
        prompt=clip_prompt_template,
    )

    return clip_chain

def prep_input(query):
    clip_chain = load_clip_chain()
    clip = clip_chain.run(query)
    edits = {'clipboard': False}
    if clip in ['<Yes>', 'Yes', 'Yes.']:
        edits['clipboard'] = True
        return (query + '\nContent from user\'s clipboard:\n' + pyperclip.paste() + "\nDon't repeat my clipboard back to me unless I specifically ask you to, I'm just giving it to you for context.", edits)
    else:
        return (query, edits)

