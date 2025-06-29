import yaml
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing import List, TypedDict
from langchain_community.document_loaders import DirectoryLoader

load_dotenv(override=True)

# Classes devem ser definidas com letras maiúsculas no início
# Argumentos de funções e variáveis devem ser nomeados com letras minúsculas separadas por underscore
# Argumentos de funções ou de métodos devem ser tipados corretamente (Ex.: str, model_name: str = "gemini-2.0-flash")
# As importações devem ser organizadas de forma lógica, começando com as bibliotecas padrão, seguidas por bibliotecas de terceiros e, por último, importações locais.
# Os retornos de funções ou métodos devem ser tipados corretamente, se possível (Ex.: def eleva_quadrado(numero: float): -> float).

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class Lexa(object):
    def __init__(self, 
                 model_name: str = "gemini-2.0-flash", 
                 model_provider: str = "google_genai",
                 embeddings_model_name: str = "models/embedding-001",
                 config_path: str = "config.yaml") -> None:
        """ Initializes the RAG system with a chat model, embeddings, vector store, text splitter, and document loader. 
        Args:
            model_name (str): The name of the chat model to use.
            model_provider (str): The provider of the chat model.
        """

        print("Initializing Lexa...")
        # Load configuration from a YAML file
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # Define system prompt  
        system_prompt = config.get('system_prompt', "You are a helpful assistant that answers questions based on the provided context.")

        # Initialize chat model, embeddings, vector store, text splitter, and document loader
        self.llm = init_chat_model(model=model_name, model_provider=model_provider)
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model_name)
        self.vector_store = InMemoryVectorStore(self.embeddings)

        # Document text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        # Load documents from a directory
        self.loader = DirectoryLoader("documents", glob="**/*.txt", show_progress=True)#use_multithreading=True
        self.docs = self.loader.load()
        # Split documents into chunks
        self.all_splits = self.text_splitter.split_documents(self.docs)
        # Add document chunks to the vector store
        _ = self.vector_store.add_documents(documents=self.all_splits)
    
        # Define the prompt template for the chat model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Pergunta: {question}\nContexto: {context}")
        ])

        # Define user history
        self.user_states = {}
        # Build the state graph for the application
        self._build_graph()
        print("Lexa initialized.")

    def retrieve(self, state: State) -> dict:
        # Retrieve relevant documents based on the user's question
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State) -> dict:
        # Concatenate the content of the retrieved documents
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({
            "question": state["question"],
            "context": docs_content
        })
        # Generate the response using the chat model
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def _build_graph(self) -> None:
        #  Define the state graph for the application
        builder = StateGraph(State).add_sequence([
            self.retrieve,
            self.generate
        ])
        builder.add_edge(START, "retrieve")
        self.graph = builder.compile()

    def ask(self, user_id: int, question: str) -> str:
        # Invoke the state graph with the user's question
        # 
        previous_state = self.user_states.get(user_id, {})

        state = {"question": question}
        response = self.graph.invoke(state)
        return response["answer"]

