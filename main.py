from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import os
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from prompt import template
from langchain.memory import ConversationBufferMemory

# loading environment variables
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Load and Chunk the document
loader = PyPDFLoader(r"D:\yusuf-work\data\tyt18.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Initialize a LangChain embedding object.
model_name = "multilingual-e5-large"
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=docs,
    index_name="airagapp",
    embedding=embeddings,
    namespace="wondervector5000"
)

time.sleep(1)


# Initialize a LangChain object for chatting with the LLM
prompt_template = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

llm = ChatOpenAI(
    model_name='gpt-4o-mini',
    temperature=0.7
)

# Initialize a LangChain object for chatting with the LLM
# with knowledge from Pinecone.
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    chain_type_kwargs={
        "verbose": False,
        "prompt": prompt_template,
        "memory": ConversationBufferMemory(memory_key="history", input_key="question")
    }
)

while True:

    question = input("User> ")
    if question.lower() == "exit":
        break
    else:
        response = qa.invoke(question)
        print(f'AI> {response['result']}')
