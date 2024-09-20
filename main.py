from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import os
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# loading environment variables
load_dotenv()


# Load and Chunk the document
loader = PyPDFLoader("data/layout-parser-paper.pdf")
docs = loader.load()
split_docs = RecursiveCharacterTextSplitter(docs)

# Initialize a LangChain embedding object.
model_name = "multilingual-e5-large"
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=split_docs,
    index_name="airagapp",
    embedding=embeddings,
    namespace="wondervector5000"
)

time.sleep(1)


# Initialize a LangChain object for chatting with the LLM
# without knowledge from Pinecone.
llm = ChatOpenAI(
    openai_api_key=os.environ.get('OPENAI_API_KEY'),
    model_name='gpt-4o-mini',
    temperature=0.7
)

# Initialize a LangChain object for chatting with the LLM
# with knowledge from Pinecone.
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)


query1 = """What are the first 3 steps for getting started 
with the WonderVector5000?"""

print("Query 1\n")
print("Chat with knowledge:")
