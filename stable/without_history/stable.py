from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel

from operator import itemgetter
from langchain.memory import ConversationBufferMemory

from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

from langchain import hub
import gradio as gr
import os
import shutil
import random
import time

# Assigning the global variables
global_var = None
global_vector = None
global_retriver = None

def set_global_retriver(value):
    global global_retriver
    global_retriver = value

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
    path = "/Users/princesingh/Downloads/Private Tutor/stable/without_history/knowledge/" + os.path.basename(file_input)
    shutil.copyfile(file_input.name, path)
    print(path)
    set_global_value(path)
    return gr.update(visible=True)


# function to convert context to vector

def process_docs():
    loader = PyPDFLoader(global_var)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True 
        )
    texts = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    # load it into Chroma
    vectorstore = Chroma.from_documents(texts, embeddings)
    
    return set_global_vector(vectorstore)


# process the docs and convert to vector
def process_vector(docs_status,progress=gr.Progress()):
    process_docs()
    return "The document has been successfully processed!!"


# assign the model for checking emotion
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion = pipeline('sentiment-analysis',  model='arpanghoshal/EmoRoBERTa')

# Creating the chain to contextualiz
def process_message(message,history):
    
    # Importing the LLM and embedings
    llm = ChatGoogleGenerativeAI(google_api_key="AIzaSyCLHbw0jgv7vM0Xt-SCTyVWjNQSbEWXeuc", model="gemini-pro",convert_system_message_to_human=True)

    # assingning the constant
    vectorstore = global_vector
    
    emotion_labels = emotion(message)
    
    label = emotion_labels[0]['label']
    
    THRESHOLD = 0.00
    response = ""
    n_response = ""
    ai_response = ""
    
    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k":6}
    )
    
    chat_history = [] 
    
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt= ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    # Setting the system prompt for the chatbot

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""


    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return contextualize_q_chain
        else:
            return input["question"]


    rag_chain = (
        RunnablePassthrough.assign(
            context = contextualized_question | retriever | format_docs
        )
        | qa_prompt 
        | llm
        )    
  
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
        # Generating the response from the content
        ai_msg = rag_chain.invoke(
            {
                "question": message,
                "chat_history": chat_history
            }
        )
    
        chat_history.extend(
            [
                HumanMessage(content=message), ai_msg
            ]
        )
        
        chunk = ai_msg.content

        for resp in chunk:
            ai_response = ai_response + resp
            yield ai_response


################################### Building the chatbot UI ########################################

# external components for the chatbot UI

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
            process_message,
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