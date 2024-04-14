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

from langchain import hub

from langchain import hub
import gradio as gr
import os
import shutil


llm = ChatGoogleGenerativeAI(google_api_key="AIzaSyCLHbw0jgv7vM0Xt-SCTyVWjNQSbEWXeuc", model="gemini-pro",convert_system_message_to_human=True)

embeddings = OpenAIEmbeddings(openai_api_key="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")

