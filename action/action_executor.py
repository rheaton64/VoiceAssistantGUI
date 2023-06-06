from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.chat_models import ChatOpenAI

class ActionExecutor:
    def __init__(self):
        self.agent = create_python_agent(
            llm=ChatOpenAI(temperature=0),
            tool=PythonREPLTool(),
            verbose=True
        )

    def send(self, message):
        self.agent.run(message)

