from langchain.llms import GooglePalm
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.environ['GOOGLE_API_KEY']

llm = GooglePalm(google_api_key = google_api_key,temperature = 0.2)



embeddings = HuggingFaceInstructEmbeddings(
    query_instruction = "represent query for retrieval: "
)

file_path = "faiss_index"

def create_vector_db():
    loader = PyPDFLoader("Aniket_Surjuse_Resume.pdf")
    pages = loader.load_and_split()
    vectordb = FAISS.from_documents(documents=pages, embedding=embeddings)
    vectordb.save_local(file_path)

def get_qa_chain():
    vectordb = FAISS.load_local(file_path, embeddings)
    retriever = vectordb.as_retriever()
    chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        input_key = "query"
    )
    return chain

if __name__ == "__main__":

    # create_vector_db()
    chain = get_qa_chain()
    ans = chain("who is Aniket")
    print(ans['result'])
    # print(llm("how are you?"))



