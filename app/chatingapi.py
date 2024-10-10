from flask import Flask, request, jsonify
import os
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from dotenv import load_dotenv
load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINFACE_KEY")
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 1.0, "max_length": 5000, "length_penalty": 1.2,
                  "repetition_penalty": 1.2, "do_sample": True}

)
# Define a prompt template
prompt_template = "Question: {question}"

# Create a PromptTemplate object for LangChain
prompt = PromptTemplate(
    input_variables=["question"],
    template=prompt_template,
)

# Create an LLMChain object which ties together the LLM and prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)


def generate_full_response(query):
    result = llm_chain.invoke({"question": query})
    cleaned_result = result['text'].replace(f"Question: {query}", "").strip()
    return cleaned_result


def create_app():

    app = Flask(__name__)

    @app.route('/llm', methods=['POST'])
    def llm_call():
        data = request.json
        query = data['query']
        print(query)
        # result = hf.invoke(query)
        # print(result)
        # return jsonify({"message": result})
        full_response = generate_full_response(query)
        print(full_response)
        return jsonify({"message": full_response})

    @app.route('/', methods=['GET'])
    def check():
        return jsonify({"Status": "ok"})

    return app
