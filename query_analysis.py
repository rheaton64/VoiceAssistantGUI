from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, messages_to_dict
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
import datetime
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
        llm=ChatOpenAI(temperature=0, model_name='gpt-4'),
        prompt=clip_prompt_template,
    )

    return clip_chain

def load_memory_chains():
    system_template = """You are an AI assistant that can help determine when a user is asking the assistant to perform an operation related to conversation memory.
    You will be given a query that the user has asked to the Assistant, and will be asked a question regarding the query.
    You only respond with answers to questions, and if asked a 'yes/no' question, only respond with the answer '<Yes>' or '<No>'.
    You do not provide any other information other than the answer to the question."""

    load_conversation_template = """In the following query, is the user asking the Assistant to load the previous conversation or pick up where they left off last time?
    Or, is the Assistant being asked to 'remember' the previous conversation, or use information that was saved from the previous conversation?
    Question type: 'yes/no'
    Query: {query}"""

    load_conversation_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessagePromptTemplate.from_template(load_conversation_template),
    ])

    load_conversation_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name='gpt-4'),
        prompt=load_conversation_prompt_template,
    )

    save_conversation_template = """In the following query, is the user asking the assistant to save their current conversation?
    Or, is the Assistant being asked to 'remember' the current conversation, or save some information, for later?
    Question type: 'yes/no'
    Query: {query}"""
    

    save_conversation_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessagePromptTemplate.from_template(save_conversation_template),
    ])

    save_conversation_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name='gpt-4'),
        prompt=save_conversation_prompt_template,
    )

    return save_conversation_chain, load_conversation_chain

def load_name_generation_chain():
    system_template = """You are an AI assistant that can help generate a relevant name for a file based on the content of a conversation.
    You only respond with answers to questions, and if asked a 'name' question, only respond with the suggested file name.
    You do not provide any other information other than the answer to the question."""

    name_generation_template = """Based on the content of the following conversation, what would be a relevant name for the file where it will be saved?
    Question type: 'name'
    Conversation: {conversation}"""

    name_generation_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessagePromptTemplate.from_template(name_generation_template),
    ])

    name_generation_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name='gpt-4'),
        prompt=name_generation_prompt_template,
    )

    return name_generation_chain



def load_file_chains():
    system_template = """You are an AI assistant that can help determine whether the user's query requires access to a local file.
    You only respond with answers to questions, and if asked a 'yes/no' question, only respond with the answer '<Yes>' or '<No>'.
    If asked an 'extract' question, please only respond with the requested information. Do not provide any other information or say anything else.
    You do not provide any other information other than the answer to the question."""

    file_template = """In answering the following query, will the Assistant need to access a local file? Note: the clipboard is not a local file, and this DOES NOT apply to saving and loading conversation history. That logic is handled elsewhere.
    Question type: 'yes/no'
    Query: {query}"""                                                                                                                                                                                                     

    filename_template = """In answering the following query, what is the name of the file that the Assistant will need to access?
    The Assistant is able to access these local files.
    ONLY respond with the filename, nothing else.
    Question type: 'extract'
    Query: {query}"""

    file_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessagePromptTemplate.from_template(file_template),
    ])

    file_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name='gpt-4'),
        prompt=file_prompt_template,
    )

    filename_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessagePromptTemplate.from_template(filename_template),
    ])

    filename_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name='gpt-4'),
        prompt=filename_prompt_template,
    )

    return file_chain, filename_chain

def find_file(filename, search_path):
    result = []
    for root, dir, files in os.walk(search_path):
        files = [f.lower() for f in files]
        if filename.strip().lower() in files:
            result.append(os.path.join(root, filename))
    return result

def save_conversation_history(memory: ConversationBufferWindowMemory, context):
    history = messages_to_dict(memory.chat_memory.messages)
    hist_str = ""
    for msg in history:
        hist_str += f"{msg['type']}: {msg['data']['content']}\n\n"
    name_chain = load_name_generation_chain()
    filename = name_chain.run(hist_str)
    filename = f"{filename}.txt"
    context['prev_conv_filename'] = filename
    with open(f'./saved_responses/{filename}', 'w') as file:
        file.write(hist_str)
    return filename

def load_conversation_history(filename):
    with open(f'./saved_responses/{filename}', 'r') as file:
        history = file.read()
    return history

class ConvoState:
    def __init__(self):
        self.loaded_prev = False


def prep_all_inputs(query, memory, context, state = None):
    if state is None:
        state = ConvoState()
    clip_chain = load_clip_chain()
    file_chain, filename_chain = load_file_chains()
    save_conversation_chain, load_conversation_chain = load_memory_chains()
    clip_access = clip_chain.run(query)
    print('Clip access?', clip_access)
    file_access = file_chain.run(query)
    print('File access?', file_access)
    save_conversation = save_conversation_chain.run(query)
    print('Save conversation?', save_conversation)
    if state.loaded_prev:
        load_conversation = '<No>'
    else:
        load_conversation = load_conversation_chain.run(query)
        print('Load conversation?', load_conversation)
    edits = {'clip_access': False, 'file_access': False}
    if clip_access in ['<Yes>', 'Yes', 'Yes.']:
        edits['clip_access'] = True
        query += '\n<SYSTEM>: Content from user\'s clipboard:\n-----\n' + pyperclip.paste() + "\n-----\nDon't repeat my clipboard back to me unless I specifically ask you to, I'm just giving it to you for context."
    if file_access in ['<Yes>', 'Yes', 'Yes.']:
        edits['file_access'] = True
        search_path = os.path.join(os.environ['USERPROFILE'], 'Documents')
        filename = filename_chain.run(query)
        print('Filename?', filename)
        file_paths = find_file(filename, search_path)
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                file_content = file.read()
            query += '\n<SYSTEM>: Content from specified file:\n' + file_content + "\nDon't repeat my file content back to me unless I specifically ask you to, I'm just giving it to you for context."
    if save_conversation in ['<Yes>', 'Yes', 'Yes.']:
        print('Saving conversation...')
        save_filename = save_conversation_history(memory, context)
        query += f'\n\n<SYSTEM> Conversation history successfully saved. Please let the user know that this is the case, and that this conversation has been saved in the file named {save_filename}.'
    if load_conversation in ['<Yes>', 'Yes', 'Yes.']:
        hist = load_conversation_history(context['prev_conv_filename'])
        query += f'\n\n<SYSTEM> The last conversation with the user has been loaded and can be found here:\n\n{hist}\n\n<SYSTEM> Let the user know that the conversation has been loaded, and do not repeat the conversation back to them unless they specifically ask you to.'
    return (query, edits, state)

