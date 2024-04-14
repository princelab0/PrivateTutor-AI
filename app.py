import gradio as gr
import os
import shutil
import random
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
import numpy as np
import openai
import pdfplumber
import time
from sentence_transformers import util
from langchain.chains.question_answering import load_qa_chain
import random 
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnableParallel

from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate


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

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_together.embeddings import TogetherEmbeddings
from langchain_community.llms import Together

from langchain_cohere import CohereEmbeddings

# seting global variable
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
    path = "/Users/princesingh/Downloads/Github/PrivateTutor-AI/knowledge/" + os.path.basename(file_input)
    shutil.copyfile(file_input.name, path)
    print(path)
    set_global_value(path)
    return gr.update(visible=True)


# function to convert context to vector
def process_docs():
    loader = PyPDFLoader(global_var)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(data)
    
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key="KhkTTsI2G0zJ9Nzx5jsItCu1meU43KjWKk0tR9Xn")
    
    # embeddings = GoogleGenerativeAIEmbeddings(google_api_key="AIzaSyCLHbw0jgv7vM0Xt-SCTyVWjNQSbEWXeuc",model="models/embedding-001")
    
    # embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval",together_api_key="a11a0cd5f1ac457fdbdf3b75198225939104f564a61856f1b2bde4e97be328c4")

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


# function to check similarity
def answer_query(question):
    # using openai embeding
    # question_emb = openai.Embedding.create(input=question, engine="text-embedding-ada-002")
    # context_emb = openai.Embedding.create(input=CONTEXT, engine="text-embedding-ada-002")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    CONTEXT = parse_pdf()
    THRESHOLD = 0.00
    # using transformer embeding
    question_emb = model.encode(question)
    context_emb = model.encode(CONTEXT)

    # Reshape the arrays to 2D
    question_emb = question_emb.reshape(1, -1)
    context_emb = context_emb.reshape(1, -1)

    similarity_score = cosine_similarity(question_emb, context_emb)

    return similarity_score


# process the docs and convert to vector
def process_vector(docs_status,progress=gr.Progress()):
    process_docs()
    return "The document has been successfully processed!!"
    

# function for checking the query word exist or not in the list of words
def check_long(query):

    word_list = ["explain","describe","summarize","long"]

   # Split the query into individual words
    query_words = query.split()

    # Initialize a set to store matched words
    matched_words = []

    for word in word_list:
        if word in query_words:
            matched_words.append(word)


    if len(matched_words) == 0:
        return False
        
    else:
        return True

# assign the model for checking emotion
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion = pipeline('sentiment-analysis',  model='arpanghoshal/EmoRoBERTa')

# retriving the response from the llm
def message_response(message,history):

    emotion_labels = emotion(message)
    label = emotion_labels[0]['label']
    
    # assingning the constant
    vectorstore = global_vector
    THRESHOLD = 0.00
    response = ""
    n_response = ""
    ai_response = ""

    llm = ChatGoogleGenerativeAI(google_api_key="AIzaSyCLHbw0jgv7vM0Xt-SCTyVWjNQSbEWXeuc", model="gemini-pro",convert_system_message_to_human=True)

    # llm = Together(model="mistralai/Mixtral-8x7B-Instruct-v0.1",together_api_key="a11a0cd5f1ac457fdbdf3b75198225939104f564a61856f1b2bde4e97be328c4")

    # local LLM for the chat interface
    # model =  ChatOllama(model='qwen')
    
    
    # RAG chain for the long answer
    # Define the prompt template
    template1 = "give a precise, short and accurate response based on the input documents."

    # Create a prompt template object
    prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template1
    )

    # Ragchain for the context based answer
    retriever = vectorstore.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")

    template = "give a precise, short and accurate response based on the input documents.If you don't find the answer in the context then don't make your own anser."
    # Create a prompt template object
    prompt_t = PromptTemplate(
        input_variables=["query"],
        template=template
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

 
    chain = load_qa_chain(llm, chain_type="stuff")

    score = answer_query(message)
    score = score[0][0]
 
    return ai_response

    # checking the emotion about the sadness
    
    if label == "sadness":
        print(label)
        chunks = ["Please don't hesitate to share more about what you're struggling with. I'll do my best to provide the support and assistance you're looking for","I understand the answer is not what you expected. Let's work together to address your concerns and find the assistance you need","I’m sorry to upset you. We can work on this together. Please provide me with additional details about your question."]
        # response = "I'm sorry to hear that you're feeling this way, but I'm unable to provide the help that you need. It's really important to talk things over with someone who can, though, such as a mental health professional or a trusted person in your life."
        chunk = random.choice(chunks)
        for resp in chunk:
            time.sleep(0.01)
            ai_response = ai_response + resp
            yield ai_response

    elif label == "angry":
        print(label)
        chunks = ["I'm sorry if I've upset you in any way. If there's anything specific you'd like to discuss or if you need assistance with something, feel free to let me know.","I'm here to help and provide assistance to the best of my abilities. If there's something bothering you or if you need assistance with a particular issue, please let me know, and I'll do my best to assist you."]
        # response = "I'm sorry to hear that you're feeling this way, but I'm unable to provide the help that you need. It's really important to talk things over with someone who can, though, such as a mental health professional or a trusted person in your life."
        chunk = random.choice(chunks)
        for resp in chunk:
            time.sleep(0.01)
            ai_response = ai_response + resp
            yield ai_response

    elif label == "annoyance":
        print(label)
        chunks = ["I understand your frustration. Let me know how I can assist you better.","I apologise if the responses weren't helpful. Could you provide more details so that I can find a solution for you?"]
        # response = "I'm sorry to hear that you're feeling this way, but I'm unable to provide the help that you need. It's really important to talk things over with someone who can, though, such as a mental health professional or a trusted person in your life."
        chunk = random.choice(chunks)
        for resp in chunk:
            time.sleep(0.01)
            ai_response = ai_response + resp
            yield ai_response


    elif label == "disapproval":
        print(label)
        chunks = ["I'm sorry that the responses were not helpful to you. Could you assist me with in-depth details so that I can understand better?","I apologize if my response was not what you expected. Let me try it with a different approach."]
        # response = "I'm sorry to hear that you're feeling this way, but I'm unable to provide the help that you need. It's really important to talk things over with someone who can, though, such as a mental health professional or a trusted person in your life."
        chunk = random.choice(chunks)
        for resp in chunk:
            time.sleep(0.01)
            ai_response = ai_response + resp
            yield ai_response

    elif label == "realisation":
        print(label)
        chunks = ["No worries! Let’s focus on the next steps. Please provide the correct details and I'll do my best to assist you."]
        # response = "I'm sorry to hear that you're feeling this way, but I'm unable to provide the help that you need. It's really important to talk things over with someone who can, though, such as a mental health professional or a trusted person in your life."
        chunk = random.choice(chunks)
        for resp in chunk:
            time.sleep(0.01)
            ai_response = ai_response + resp
            yield ai_response

    elif label == "nervousness":
        print(label)
        chunks = ["Don’t worry even if you are not sure about your confusion. Asking questions is a crucial part of learning, and I'm here to provide guidance and support","I understand that receiving an answer can be nerve-wracking at times. Rest assured, I'll do my best to address your concerns."]
        # response = "I'm sorry to hear that you're feeling this way, but I'm unable to provide the help that you need. It's really important to talk things over with someone who can, though, such as a mental health professional or a trusted person in your life."
        chunk = random.choice(chunks)
        for resp in chunk:
            time.sleep(0.01)
            ai_response = ai_response + resp
            yield ai_response


    elif label == "joy":
        print(label)
        chunks = ["I'm delighted that I could assist you! If you have any more questions or need further assistance, just let me know. Keep up the great work!","That's fantastic to hear! If there's anything else you need help with, don't hesitate to ask","Wonderful! If you ever have more questions or need further clarification, feel free to reach out. Keep up the great work!"]
        # response = "I'm sorry to hear that you're feeling this way, but I'm unable to provide the help that you need. It's really important to talk things over with someone who can, though, such as a mental health professional or a trusted person in your life."
        chunk = random.choice(chunks)
        for resp in chunk:
            time.sleep(0.01)
            ai_response = ai_response + resp
            yield ai_response


    elif label == "embarrassment":
        print(label)
        chunks = ["I completely understand how frustrating it can be when things don't go as planned.It's okay to feel this way, but please know that mistakes happen, and it's all part of the learning process."]
        # response = "I'm sorry to hear that you're feeling this way, but I'm unable to provide the help that you need. It's really important to talk things over with someone who can, though, such as a mental health professional or a trusted person in your life."
        chunk = random.choice(chunks)
        for resp in chunk:
            time.sleep(0.01)
            ai_response = ai_response + resp
            yield ai_response

    elif label == "surprise":
        print(label)
        chunks = ["I understand that my response might not have been what you anticipated! However, my response is based on my access to a vast amount of information and knowledge about a wide range of topics.","Looks like you are not satisfied with the response. Let me try to clear the context with some additional details"]
        # response = "I'm sorry to hear that you're feeling this way, but I'm unable to provide the help that you need. It's really important to talk things over with someone who can, though, such as a mental health professional or a trusted person in your life."
        chunk = random.choice(chunks)
        for resp in chunk:
            time.sleep(0.01)
            ai_response = ai_response + resp
            yield ai_response


    elif label == "gratitude":
        print(label)
        chunks = ["I'm glad I could provide you with the answer. If you have any more questions, I'm here to help you learn bette","Thank you for your kind words! If you have any more questions, I'm here to help in any way I can"]
        # response = "I'm sorry to hear that you're feeling this way, but I'm unable to provide the help that you need. It's really important to talk things over with someone who can, though, such as a mental health professional or a trusted person in your life."
        chunk = random.choice(chunks)
        for resp in chunk:
            time.sleep(0.01)
            ai_response = ai_response + resp
            yield ai_response

    elif label == "amusement":
        print(label)
        chunks = ["I apologise if my previous response seemed humorous or confusing. Perhaps, we can try with a little more context on this topic"]
        # response = "I'm sorry to hear that you're feeling this way, but I'm unable to provide the help that you need. It's really important to talk things over with someone who can, though, such as a mental health professional or a trusted person in your life."
        chunk = random.choice(chunks)
        for resp in chunk:
            time.sleep(0.01)
            ai_response = ai_response + resp
            yield ai_response

    elif label == "fear":
        print(label)
        chunks = ["I apologise if the level of detail provided was not suitable for you. If you need a more specific description., please let me know, and I'll be happy to assist you accordingly.","I understand your concern. If you're unsure whether the explanation provided meets your curriculum, it's best to clarify by asking me more questions. I’ll try my best to answer them all."]
        # response = "I'm sorry to hear that you're feeling this way, but I'm unable to provide the help that you need. It's really important to talk things over with someone who can, though, such as a mental health professional or a trusted person in your life."
        chunk = random.choice(chunks)
        for resp in chunk:
            time.sleep(0.01)
            ai_response = ai_response + resp
            yield ai_response            

    elif label == "caring":
        print(label)
        chunks = ["Thank you for your understanding. It means a lot to me. Is there more you want to learn about.","I appreciate your gratitude. Your support motivates me to assist you even better. If you have any follow-up questions, please let me know."]
        # response = "I'm sorry to hear that you're feeling this way, but I'm unable to provide the help that you need. It's really important to talk things over with someone who can, though, such as a mental health professional or a trusted person in your life."
        chunk = random.choice(chunks)
        for resp in chunk:
            time.sleep(0.01)
            ai_response = ai_response + resp
            yield ai_response

    elif label == "love":
        print(label)
        chunks = ["I'm glad to hear that you're enjoying the assistance! If you have any more questions or need further assistance, please don't hesitate to ask. I'm here to help you.","Great! I'm glad to hear that you found the response helpful. If you have any more questions in the future or need further help, feel free to ask.","Thank you for your understanding. It means a lot to me. Is there more you want to learn about this."]
        # response = "I'm an artificial intelligence and don't have feelings, but I'm here to help you. Let's focus on our topic,"
        chunk = random.choice(chunks)
        for resp in chunks:
            time.sleep(0.01)
            ai_response = ai_response + resp
            yield ai_response

    else:
        print(label)

        if score < THRESHOLD:
            print(score)
            chunks = ["I am sorry, but i can only answer questions related to context.Please,try to ask about uploaded document.","It appears that the question you asked is not relevant to the context.Can you please ask a related question?","I am afraid, i can't assist you with this questionas. It is related to different subject.Could you please provide more information or help me understand the reevance?","Unfortunately, I am unable to provide a relevant to the subject we are discussing. Please ask a question related to context of the document.","I am afraid, the question you ask is unrelated to the context. Please provide a question that is more closely related to context.","I am sorry, but could you please clarify what you mean by " + message ]
            chunk = random.choice(chunks)
            for resp in chunk:
                n_response = n_response + resp
                time.sleep(0.01)
                yield n_response

        else:
            print(score) 
            output = {}
            curr_key = None

            check = check_long(query=message)

            if check is False:
                
                for chunk in rag_chain_with_source.stream(message):
                    for key in chunk:
                        if key not in output:
                            output[key] = chunk[key]
                        else:
                            output[key] += chunk[key]
                        if key != curr_key:
                            if (key=="answer"):
                                for resp in chunk[key]:
                                    response = response + resp
                                    yield response
                        else:
                            for resp in chunk[key]:
                                response = response + resp
                                print(resp)
                                yield response

                        # clearing the buffer memory
                        flush = True        

                        # print(chunk[key], end="", flush=True)
                        curr_key = key

            else:
                docs = vectorstore.similarity_search(message)
                response = chain.run(input_documents=docs, question=message,prompt=prompt_template)
                for resp in response:
                    n_response = n_response + resp
                    time.sleep(0.01)
                    yield n_response


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
