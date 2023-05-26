from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
import pyperclip
import os

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

def load_file_chains():
    system_template = """You are an AI assistant that can help determine whether the user's query requires access to a local file.
    You only respond with answers to questions, and if asked a 'yes/no' question, only respond with the answer '<Yes>' or '<No>'.
    If asked an 'extract' question, please only respond with the requested information. Do not provide any other information or say anything else.
    You do not provide any other information other than the answer to the question."""

    file_template = """In answering the following query, will the Assistant need to access a local file?
    Question type: 'yes/no'
    Query: {query}"""

    filename_template = """In answering the following query, what is the name of the file that the Assistant will need to access?
    Question type: 'extract'
    Query: {query}"""

    file_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessagePromptTemplate.from_template(file_template),
    ])

    file_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
        prompt=file_prompt_template,
    )

    filename_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessagePromptTemplate.from_template(filename_template),
    ])

    filename_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
        prompt=filename_prompt_template,
    )

    return file_chain, filename_chain

def find_file(filename, search_path):
   result = []
   for root, dir, files in os.walk(search_path):
      if filename in files:
         result.append(os.path.join(root, filename))
   return result


def prep_all_inputs(query):
    clip_chain = load_clip_chain()
    file_chain, filename_chain = load_file_chains()
    clip_access = clip_chain.run(query)
    file_access = file_chain.run(query)
    filename = filename_chain.run(query)
    edits = {'clip_access': False, 'file_access': False}
    if clip_access in ['<Yes>', 'Yes', 'Yes.']:
        edits['clip_access'] = True
        query + '\nContent from user\'s clipboard:\n' + pyperclip.paste() + "\nDon't repeat my clipboard back to me unless I specifically ask you to, I'm just giving it to you for context.", edits
    if file_access in ['<Yes>', 'Yes', 'Yes.']:
        edits['file_access'] = True
        search_path = '~/code'  # Replace with the actual search path
        file_paths = find_file(filename, search_path)
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                file_content = file.read()
            query += '\nContent from specified file:\n' + file_content + "\nDon't repeat my file content back to me unless I specifically ask you to, I'm just giving it to you for context.", edits
    return (query, edits)
