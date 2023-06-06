from langchain import LLMChain
from langchain.schema import SystemMessage
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.tools.python.tool import PythonREPLTool
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

ACTION_SYSTEM = SystemMessage(content="""You are a helpful assistant that writes a Python function to complete any task specified by me.
The function should always be able to run as-is, without any modifications for the user.
The environment where the function will be run is on my local machine, and it has access to the Internet and the Python standard library. To install any external libraries, you must specify them in your response.

At each round of conversation, I will provide you with 
Task: the task that you must complete by writing a Python function.

The only output that I can see is what the function returns. I cannot see any other output that you print to the console.

When given a task, you respond with
Explain: What is the user asking you to do? What outputs are the user expecting? Are there any ambiguities in the task? What are they? How can they be accounted for?
Plan: How to complete the task step by step.
Libraries (if applicable): What non-standard libraries, if any, need to be installed to complete the task?
Code: Write the code to complete the task.

You should only respond in the format as described below:
RESPONSE FORMAT:
Explain: ...
Plan:
1) ...
2) ...
3) ...
...
Code:
```python
// helper functions (only if needed, try to avoid them)
...
// main function after the helper functions
def yourMainFunctionName() {
  // ...
}
```
""")

ACTION_HUMAN = HumanMessagePromptTemplate.from_template("""Task: {task}""")

class ActionExecutor:
    def __init__(self, callback: BaseCallbackHandler):
        self.chain = LLMChain(
            prompt = ChatPromptTemplate.from_messages([ACTION_SYSTEM, ACTION_HUMAN]),
            llm = ChatOpenAI(model_name='gpt-4', temperature=0, streaming=True, callbacks=[StreamingStdOutCallbackHandler()]),
        )

    def send(self, message):
        self.chain.run(message)

