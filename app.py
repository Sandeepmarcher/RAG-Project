import os
from flask import Flask, render_template, request
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# SET YOUR OPENAI KEY
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

app = Flask(__name__)

VECTOR_DB_PATH = "vectorstore"

def load_and_store_docs():
    loader = PyPDFLoader("data/sample.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DB_PATH)

def load_qa_chain():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever()
    )
    return qa

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        query = request.form["query"]
        qa = load_qa_chain()
        answer = qa.run(query)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    if not os.path.exists(VECTOR_DB_PATH):
        load_and_store_docs()
    app.run(debug=True)
