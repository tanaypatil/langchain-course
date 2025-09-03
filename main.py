import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def main():
    print("Retrieving...")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    query = "What is pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print(result.content)

    vectorstore = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": query})
    print(result)



if __name__ == "__main__":
    main()
