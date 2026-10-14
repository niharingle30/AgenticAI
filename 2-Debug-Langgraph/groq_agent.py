from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv
from typing import Literal


load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

model = ChatGroq(
    model="qwen/qwen3.8-27b",
    max_tokens=200,
    temperature=0
)

def make_default_graph():
    graph_workflow=StateGraph(State)

    def call_model(state):
        return {"messages":[model.invoke(state['messages'])]}
    
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.add_edge(START, "agent")

    agent=graph_workflow.compile()
    return agent

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def make_alternative_graph():

    tools = [add]

    tool_node = ToolNode(tools)

    model_with_tools = model.bind_tools(tools)

    # Agent node
    def call_model(state: State):
        response = model_with_tools.invoke(
            state["messages"]
        )
        return {"messages": [response]}

    # Conditional routing
    def should_continue(
        state: State
    ) -> Literal["tools", "__end__"]:

        last_message = state["messages"][-1]

        if last_message.tool_calls:
            return "tools"

        return END

    # Build graph
    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)

    graph_workflow.add_edge(START, "agent")

    graph_workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "__end__": END
        }
    )

    graph_workflow.add_edge("tools", "agent")

    return graph_workflow.compile()


# Export graph for LangGraph Studio
agent = make_alternative_graph()

# use cmd langgraph dev it will redirect to smith.langchain and visual your agent where you can do converstaion