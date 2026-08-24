from dotenv import load_dotenv

load_dotenv()

from deepagents import create_deep_agent


def start_assistant():
    print(f'Start process...')

    internet_search = {"type": "web_search"}
    research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.
    You have access to an internet search tool as your primary means of gathering information.
    ## `internet_search`
    Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
    """

    agent = create_deep_agent(
        # model="openai:gpt-5.6-luna",
        model="openai:gpt-5-nano",
        tools=[internet_search],
        system_prompt=research_instructions,
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "What is langgraph?"}]})

    print(result["messages"][-1].content)
    print(f'Process end!')


if __name__ == '__main__':
    start_assistant()
