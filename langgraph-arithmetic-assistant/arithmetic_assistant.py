from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from langchain.messages import SystemMessage
from langchain.messages import ToolMessage

from typing import Literal
from langgraph.graph import StateGraph, START, END

from IPython.display import Image, display
# Библиотека IPython подходит для визуализации схемы нашего ИИ-агента.
# Для показа изображения прямо внутри интерактивной консоли

# Image — класс, который принимает на вход картинку (в виде файла, ссылки или бинарных данных, например,
# сгенерированных самим LangGraph в формате PNG/JPEG) и превращает её в объект, понятный для отображения.

# display — специальная функция среды IPython, которая заставляет этот объект мгновенно отрендериться и появиться
# на экране прямо под ячейкой с кодом.

from langchain.messages import HumanMessage

model = init_chat_model(
    "openai:gpt-5-nano",
    temperature=0
)


@tool()
def multiply(a: int, b: int) -> int:
    """
        Multiply `a` and `b`.

        Args:
            a: First int
            b: Second int
    """
    return a * b


@tool()
def add(a: int, b: int) -> int:
    """
        Adds `a` and `b`.

        Args:
            a: First int
            b: Second int
    """
    return a + b


@tool()
def divide(a: int, b: int) -> float:
    """
        Divide `a` and `b`.

        Args:
            a: First int
            b: Second int
    """
    return a / b


tools = [multiply, add, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)  # привязать инструменты к модели


class MessagesState(TypedDict):
    # Создает класс состояния на основе словаря. Внутри графа узлы будут передавать друг другу и обновлять обычный Python-словарь,
    # но с строго заданными ключами
    messages: Annotated[list[AnyMessage], operator.add]
    # Ключ messages хранит список всех сообщений (от пользователя, от ИИ, системных)
    # operator.add (редьюсер) — это встроенная магия LangGraph. Она указывает графу: «Когда узел возвращает новое сообщение,
    # не перезаписывай весь список полностью, а просто допиши (добавь) новое сообщение в конец существующего списка».
    # Без этого узел ИИ просто стер бы сообщение пользователя и оставил только свой ответ.
    llm_calls: int
    # Новое кастомное поле для счетчика. Здесь будет храниться целое число. Так как у него нет редьюсера (как operator.add),
    # это поле при возврате из узла будет перезаписываться.


def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                        # Вы — полезный помощник, которому поручено выполнять арифметические действия с набором входных данных.
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


def tool_node(state: MessagesState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])  # observation - наблюдение
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# Функция условного ребра для маршрутизации к узлу инструмента или к конечному узлу в зависимости от того, выполнил ли LLM вызов инструмента.
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """
        Decide if we should continue the loop or stop based upon whether the LLM made a tool call
    """

    messages = state["messages"]
    last_message = messages[-1]

    # Если LLM вызывает инструмент, выполните действие
    if last_message.tool_calls:
        return "tool_node"

    # В противном случае мы останавливаемся (отвечаем пользователю)
    return END


# Сборка агента.

# Рабочий процесс сборки
agent_builder = StateGraph(MessagesState)

# Добавляем узлы
agent_builder.add_node("llm_call",
                       llm_call)  # тут добавляем узел где LLM будет решать, вызывать инструмент или не вызывать
agent_builder.add_node("tool_node", tool_node)  # тут добавляем узел где выполняется вызов инструмента

# Добавляем ребра для соединения узлов
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(  # добавляем условные ребра
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Компиляция агента
agent = agent_builder.compile()

# Вывод на экран детальной схемы структуры нашего ИИ-агента
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
# Это критически важный инструмент для отладки (debug) сложных мультиагентных систем
# Эта строка генерирует и мгновенно выводит на экран детальную схему структуры нашего ИИ-агента в Jupyter Notebook или Google Colab.
# Она делает то же самое, что и обычный вывод графа, но с одним ключевым отличием — параметром xray=True

# Аргумент xray=True (рентген) — это главная фишка. Если ваш агент сложный и содержит внутри себя подграфы (subgraphs)
# (когда один узел на самом деле является отдельной изолированной системой или другим полноценным агентом),
# то без этого флага подграф отобразится как один невзрачный кубик. С xray=True этот кубик «просвечивается рентгеном»,
# раскрывается, и вы видите все внутренние узлы и связи этого подграфа прямо на общей схеме

# .draw_mermaid_png() — берет эту структуру, переводит её в синтаксис диаграмм Mermaid и отправляет на специальный сервер,
# который превращает текст в бинарные данные картинки формата PNG

# Image(...) — оборачивает эти бинарные данные в графический объект, который среда разработки способна распознать

# display(...) — заставляет Jupyter Notebook отобразить готовую картинку прямо под текущей ячейкой кода

# Вызов
messages = [HumanMessage(content="Add 10 and 5.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
