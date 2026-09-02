from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def start_agent():
    agent = create_agent(
        model='openai:gpt-4o-mini',
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in Nalchik?"}]}
    )

    print(result["messages"][-1].content_blocks)


if __name__ == "__main__":
    start_agent()