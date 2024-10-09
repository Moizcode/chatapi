from flask import Flask, request, jsonify
import os
from langchain_community.llms import HuggingFaceHub

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
hf = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.1, "max_length": 5000}

)

def create_app():

	app = Flask(__name__)

	@app.route('/llm', methods=['POST'])
	def llm_call():
	    data = request.json
	    query = data['query']
	    print(query)
	    result = hf.invoke(query)
	    print(result)
	    return jsonify({"message": result})


	@app.route('/', methods=['GET'])
	def check():
	    return jsonify({"Status": "ok"})
	
	return app

