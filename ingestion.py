import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


def main():
    print("Ingesting...")
    loader = TextLoader("mediumblog1.txt")
    documents = loader.load()

    print("Splitting...")
    character_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = character_text_splitter.split_documents(documents)

    print(f"Created {len(texts)} chunks")
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    print("Ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("PINECONE_INDEX_NAME"))
    print("Done")


if __name__ == "__main__":
    main()
