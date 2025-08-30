from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(temperature=0, model="gpt-4")

# prompt template for ReAct from hub
react_prompt = hub.pull("hwchase17/react")

# output parser
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

# reAct prompt
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions=output_parser.get_format_instructions())

# creates chain
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions
)
# executes the tools as per reasoning by llm
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor


def main():
    result = chain.invoke(
        input={
            "input": "search for 3 job postings for a software engineer using python in India or remote on linkedin and list their details",
        }
    )
    print(result)


if __name__ == "__main__":
    main()
