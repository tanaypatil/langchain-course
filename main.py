from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(temperature=0, model="gpt-4")
# prompt template for ReAct from hub
react_prompt = hub.pull("hwchase17/react")
# creates chain
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
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
