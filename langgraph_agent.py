from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, MessagesState, START, END


def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "Hello world!"}]}


def start_agent():
    graph = StateGraph(MessagesState)
    graph.add_node(mock_llm)
    graph.add_edge(START, "mock_llm")
    graph.add_edge("mock_llm", END)
    graph = graph.compile()

    result = graph.invoke({"messages": [{"role": "user", "content": "Hi! Hi! Hi!"}]})

    print("--- Полное состояние графа ---")
    print(result)

    print("\n--- Последний ответ модели ---")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    start_agent()
