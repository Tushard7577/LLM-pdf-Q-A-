from flask import Flask, render_template, request, redirect, url_for
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
os.environ["OPENAI_API_KEY"] ="paste your openai key here"
app = Flask(__name__)

@app.route('/',methods=['POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Your existing code for processing the file and answering questions goes here
        reader = PdfReader(file_path)
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2500,
            chunk_overlap=10,
            length_function=len,
        )
        global texts
        texts = text_splitter.split_text(raw_text)
        return render_template('question.html')

@app.route('/answer', methods=['POST'])
def answer():
    global texts
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    query = request.form['query']
    docs = docsearch.similarity_search(query)
    answer = chain.run(input_documents=docs, question=query)

    return render_template('answer.html', query=query, answer=answer)
@app.route('/question',methods=['POST'])
def question():
    return render_template('question.html')
if __name__ == '__main__':
    app.run(debug=True)
