from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
import gradio as gr
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import ChatOpenAI ,ChatCohere , ChatVertexAI
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain_core.messages import AIMessage, HumanMessage
from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel

from operator import itemgetter
from langchain.memory import ConversationBufferMemory

import gradio as gr
import os
import shutil
import random
import openai
import pdfplumber
from httpx import stream
from langchain_community.chat_models import ChatCohere
from langchain_community.vectorstores import Chroma, Pinecone
from langchain_community.embeddings.openai import OpenAIEmbeddings
from numpy import vectorize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from sympy import false
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatOpenAI ,ChatCohere , ChatVertexAI
from posthog import flush
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
# from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
# package for the chatbot interface
import gradio as gr
import numpy as np
import openai
import pdfplumber
import time
from multi_rake import Rake
from sentence_transformers import util
from langchain.chains.question_answering import load_qa_chain
import random 
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
from langchain_core.prompts import ChatPromptTemplate



global_var = None
global_vector = None

def set_global_vector(value):
    global global_vector
    global_vector = value
    
def set_global_value(value):
    global global_var
    global_var = value


def file_selected(file_input):
    print("yes, file_selected is invoked")
    print(process_button)
    print(file_input.name)
    path = "/home/prince/Desktop/Ing projects/Private Tutor/knowledge/" + os.path.basename(file_input)
    shutil.copyfile(file_input.name, path)
    print(path)
    set_global_value(path)
    return gr.update(visible=True)

def process_docs():
    loader = PyPDFLoader(global_var)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    # load it into Chroma
    vectorstore = Chroma.from_documents(texts, embeddings)
    
    return set_global_vector(vectorstore)
    

# function to extract context
def parse_pdf():
    with pdfplumber.open(global_var) as pdf:
        pages = pdf.pages
        text = ""
        for page in pages:
            text += page.extract_text()
    context = text
    
    return context

def process_vector(docs_status,progress=gr.Progress()):
    process_docs()
    return "The document has been successfully processed!!"


book_path  = "./physics.pdf"

loader = PyPDFLoader(book_path)
docs = loader.load()
        
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
    
# cohera embeding
# embeddings = CohereEmbeddings(cohere_api_key=api_key)
    
#Openai Embeding 
embeddings = OpenAIEmbeddings(openai_api_key="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")

# storing the embeding in chroma
 
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
# Retrieve and generate using the relevant snippets of the blog.

retriever = vectorstore.as_retriever()


vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=embeddings
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)




# model = ChatGoogleGenerativeAI(google_api_key="AIzaSyCLHbw0jgv7vM0Xt-SCTyVWjNQSbEWXeuc", model="gemini-pro")
model = Ollama(model="mixtral")


from langchain.prompts.prompt import PromptTemplate
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)
memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
    )
# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)
# Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | model
    | StrOutputParser(),
}
# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | model,
    "docs": itemgetter("docs"),
}
# And now we put it all together!
final_chain = loaded_memory | standalone_question | retrieved_documents | answer    


def message_response(message,history):

    # vectorstore = global_vector
    # retriever = vectorstore.as_retriever()



    inputs = {"question": message}
    result = final_chain.invoke(inputs)
    memory.save_context(inputs, {"answer": result["answer"]})

    ai_response = result['answer']

    print(ai_response)

    return ai_response


# external components for the chatbot UI
chatbot1 = gr.Chatbot(height=650)
textbox1 = gr.Textbox(placeholder="Ask me questions...", container=True, scale=7)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload document")
            file_input = gr.File(label="Select File",height=20)
            process_button = gr.Button("Process", visible=False)
            docs_status = gr.Text()

        with gr.Column(scale=3):
            gr.ChatInterface(
            message_response,
            chatbot=chatbot1,
            textbox=textbox1,
            title="Private Tutor",
            theme="soft",
            cache_examples=False,
            retry_btn="Retry",
            undo_btn="Delete Previous",
            clear_btn="Clear",
            )

    # function for the file processing 
    file_input.change(fn=file_selected, inputs=file_input, outputs=process_button)
    process_button.click(fn=process_vector,inputs=docs_status , outputs=docs_status)  


# main function for the UI   
if __name__ == "__main__":
    demo.launch()
