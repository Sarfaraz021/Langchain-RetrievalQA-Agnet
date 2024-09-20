import os
import time
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from prompt import template


def load_environment_variables():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def load_and_split_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def initialize_pinecone(docs):
    model_name = "multilingual-e5-large"
    embeddings = PineconeEmbeddings(
        model=model_name,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )

    return PineconeVectorStore.from_documents(
        documents=docs,
        index_name="airagapp",
        embedding=embeddings,
        namespace="wondervector5000"
    )


def setup_qa_system(docsearch):
    prompt_template = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

    llm = ChatOpenAI(
        model_name='gpt-4o-mini',
        temperature=0.7
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        chain_type_kwargs={
            "verbose": False,
            "prompt": prompt_template,
            "memory": ConversationBufferMemory(memory_key="history", input_key="question")
        }
    )


def main():
    load_environment_variables()

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the 'data' folder
    pdf_directory = os.path.join(current_dir, 'data_empt')

    # Ensure the directory exists
    if not os.path.exists(pdf_directory):
        raise FileNotFoundError(
            f"The 'data' directory does not exist in {current_dir}")

    docs = load_and_split_documents(pdf_directory)

    docsearch = initialize_pinecone(docs)
    time.sleep(1)

    qa_system = setup_qa_system(docsearch)

    while True:
        question = input("User> ")
        if question.lower() == "exit":
            break
        else:
            response = qa_system.invoke(question)
            print(f'AI> {response["result"]}')


if __name__ == "__main__":
    main()
