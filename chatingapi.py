from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from flask import Flask, request, jsonify
import os
from langchain_community.llms import HuggingFaceHub
import pickle
import zipfile
import shutil


def load_vector_db_from_zip(zip_file_path, extract_dir="temp_dir", pickle_file_name="vector_db.pkl"):
    # Step 1: Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Step 2: Load the pickle file from the extracted directory
    pickle_file_path = os.path.join(extract_dir, pickle_file_name)

    # Check if the pickle file exists
    if not os.path.exists(pickle_file_path):
        raise FileNotFoundError(
            f"{pickle_file_name} not found in the zip archive.")

    # Step 3: Load the vector_db.pkl using pickle
    with open(pickle_file_path, 'rb') as f:
        vector_db = pickle.load(f)

    # Optionally, clean up the extracted files
    # You can uncomment the following line if you want to delete the extracted files after loading
    shutil.rmtree(extract_dir)

    return vector_db


os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
hf = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.1, "max_length": 500}

)

prompt_template = """
Use the following piece of context to answer the question asked.
Please try to provide the answer only based on the context

{context}
Question:{question}

Helpful Answers:
 """

prompt = PromptTemplate(template=prompt_template,
                        input_variables=["context", "question"])

app = Flask(__name__)

# read the pdfs from the folder
# loader = PyPDFDirectoryLoader("hakimi")
# documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200)
# final_documents = text_splitter.split_documents(documents)

# hugging_face = HuggingFaceBgeEmbeddings(
#     model_name="BAAI/bge-small-en-v1.5",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )
# vector_db = FAISS.from_documents(final_documents[:100], hugging_face)

# with open('vector_db.pkl', 'wb') as f:
#     pickle.dump(vector_db, f)

vector_db = load_vector_db_from_zip('vector_db.zip')

# Load precomputed documents and embeddings
retriever = vector_db.as_retriever(
    search_type="similarity", search_kwargs={"k": 3})


retrievalQA = RetrievalQA.from_chain_type(
    llm=hf,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)


@app.route('/llm', methods=['POST'])
def llm_call():
    data = request.json
    query = data['query']
    print(query)
    result = retrievalQA.invoke({"query": query})
    print(result)
    return jsonify({"message": result['result']})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
