import os
from typing import Dict, List, Tuple, Optional
import logging
from functools import lru_cache

from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import yaml

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RAG')

class ConfigManager:
    """Gerenciador de configuração com cache"""
    _config = None
    
    @classmethod
    def get_config(cls, config_path="config.yaml"):
        if cls._config is None:
            try:
                with open(config_path, 'r') as file:
                    cls._config = yaml.safe_load(file)
                logger.info("Configuração carregada com sucesso")
            except Exception as e:
                logger.error(f"Erro ao carregar a configuração: {e}")
                # Configuração padrão
                cls._config = {
                    'retrieval_settings': {'chunk_size': 1000, 'chunk_overlap': 200, 'k': 4},
                    'documents': {'path': './docs'},
                    'prompt_template': {'system': 'Você é um assistente útil.'}
                }
        return cls._config

class ConversationHistoryManager:
    """Gerenciador de histórico de conversas com persistência opcional"""
    
    def __init__(self, max_history=5, persist_path=None):
        self.max_history = max_history
        self.history_dict = {}
        self.persist_path = persist_path
        
        # Carregar histórico persistente se existir
        if persist_path and os.path.exists(persist_path):
            try:
                import pickle
                with open(persist_path, 'rb') as f:
                    self.history_dict = pickle.load(f)
                logger.info(f"Histórico carregado de {persist_path}")
            except Exception as e:
                logger.warning(f"Falha ao carregar histórico: {e}")
    
    def get_user_history(self, user_id: str) -> List[Tuple[str, str]]:
        """Retorna o histórico do usuário ou uma lista vazia"""
        return self.history_dict.get(user_id, [])
    
    def update_history(self, user_id: str, query: str, response: str) -> None:
        """Atualiza o histórico de um usuário com uma nova entrada"""
        history = self.history_dict.get(user_id, [])
        history.append((query, response))
        
        # Manter apenas as últimas max_history entradas
        if len(history) > self.max_history:
            history = history[-self.max_history:]
            
        self.history_dict[user_id] = history
        
        # Persistir histórico se configurado
        if self.persist_path:
            try:
                import pickle
                with open(self.persist_path, 'wb') as f:
                    pickle.dump(self.history_dict, f)
            except Exception as e:
                logger.warning(f"Falha ao persistir histórico: {e}")
    
    def format_history(self, user_id: str) -> str:
        """Formata o histórico para inclusão no prompt"""
        history = self.get_user_history(user_id)
        
        if not history:
            return "Nenhum histórico de conversa anterior."
            
        formatted_entries = []
        for i, (query, response) in enumerate(history, 1):
            formatted_entries.append(f"Pergunta {i}: {query}")
            formatted_entries.append(f"Resposta {i}: {response}\n")
            
        return "\n".join(formatted_entries)
    
    def clear_user_history(self, user_id: str) -> None:
        """Limpa o histórico de um usuário específico"""
        if user_id in self.history_dict:
            del self.history_dict[user_id]

class RAG:
    def __init__(self, llm, persist_directory="./db", max_history=5, config_path="config.yaml"):
        """
        Inicializa RAG com modelo de linguagem e suporte a histórico de conversas.
        
        Args:
            llm: Instância do modelo de linguagem (ex: ChatGoogleGenerativeAI)
            persist_directory (str): Diretório para persistir o banco de dados vetorial
            max_history (int): Número máximo de turnos de conversa a manter
            config_path (str): Caminho para o arquivo de configuração
        """
        logger.info("Iniciando modelo RAG")
        self.llm = llm
        self.persist_directory = persist_directory
        
        # Carregar configuração
        self.config = ConfigManager.get_config(config_path)
        
        # Inicializar gerenciador de histórico
        history_persist_path = os.path.join(persist_directory, "history.pkl")
        self.history_manager = ConversationHistoryManager(
            max_history=max_history,
            persist_path=history_persist_path
        )
        
        # Inicializar embedding com cache para reutilização
        self._initialize_embedding()
        
        # Configurações de recuperação
        retrieval_settings = self.config['retrieval_settings']
        self.chunk_size = retrieval_settings.get('chunk_size', 1000)
        self.chunk_overlap = retrieval_settings.get('chunk_overlap', 200)
        self.retrieval_k = retrieval_settings.get('top_k', 4)
        self.files_dir = self.config['documents'].get('path', './docs')
        
        # Inicializar banco de dados vetorial
        self._vectordb = None
        
        # Configurar template do prompt
        system_prompt = self.config['prompt_template'].get('system')
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """
                **Histórico de Mensagens:**
                {conversation_history}

                **Pergunta:**
                {input}

                **Contexto:**
                {context}"""
            )
        ])
        
        logger.info("RAG inicializado e pronto para uso")
    
    @lru_cache(maxsize=1)
    def _initialize_embedding(self):
        """Inicializa o modelo de embedding com cache"""
        logger.info("Inicializando modelo de embedding")
        self.embedding = OllamaEmbeddings(model="all-minilm:33m")
        return self.embedding
    
    @property
    def vectordb(self):
        """Lazy loading do banco de dados vetorial"""
        if self._vectordb is None:
            logger.info("Inicializando banco de dados vetorial")
            self._vectordb = self._create_or_load_vector_db()
        return self._vectordb
    
    def _create_or_load_vector_db(self):
        """Cria ou carrega o banco de dados vetorial"""
        # Verificar se o banco de dados já existe
        if os.path.exists(self.persist_directory) and os.path.isdir(self.persist_directory):
            try:
                # Tentar carregar o banco existente
                logger.info(f"Carregando banco de dados vetorial de {self.persist_directory}")
                return Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding
                )
            except Exception as e:
                logger.warning(f"Falha ao carregar banco de dados existente: {e}")
        
        # Criar novo banco de dados
        logger.info(f"Criando novo banco de dados vetorial em {self.persist_directory}")
        return self._create_vector_db_from_documents()
    
    def _create_vector_db_from_documents(self):
        """Cria um novo banco de dados vetorial a partir dos documentos"""
        # Verificar se o diretório de documentos existe
        if not os.path.exists(self.files_dir):
            logger.error(f"Diretório de documentos não encontrado: {self.files_dir}")
            raise FileNotFoundError(f"Diretório não encontrado: {self.files_dir}")
        
        logger.info(f"Carregando documentos de {self.files_dir}")
        loader = DirectoryLoader(self.files_dir, glob="./*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        if not documents:
            logger.warning("Nenhum documento encontrado para indexação")
            # Criar um banco vazio
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding
            )
        
        # Dividir documentos em chunks
        logger.info(f"Dividindo {len(documents)} documentos em chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        
        # Criar e persistir banco de dados vetorial
        logger.info(f"Criando embeddings para {len(texts)} chunks")
        return Chroma.from_documents(
            documents=texts,
            persist_directory=self.persist_directory,
            embedding=self.embedding
        )
    
    def _get_retriever(self):
        """Obtém um retriever configurado com os parâmetros atuais"""
        return self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.retrieval_k}
        )
    
    def refresh_database(self):
        """Recarrega o banco de dados vetorial (útil após adicionar novos documentos)"""
        logger.info("Atualizando banco de dados vetorial")
        # Limpar cache do banco de dados
        self._vectordb = None
        # Acessar a propriedade para recriá-lo
        _ = self.vectordb
        logger.info("Banco de dados atualizado")
    
    def generate_response(self, query: str, user_id: str) -> Dict:
        """
        Gera uma resposta baseada na consulta usando RAG e histórico de conversas.
        
        Args:
            query (str): A consulta de entrada
            user_id (str): ID do usuário para rastrear histórico
            
        Returns:
            dict: A resposta da cadeia de recuperação
        """
        logger.info(f"Gerando resposta para usuário {user_id}")
        
        # Preparar entrada com histórico de conversas
        input_dict = {
            "input": query,
            "conversation_history": self.history_manager.format_history(user_id)
        }
        
        # Inicializar cadeias sob demanda para garantir configurações atualizadas
        logger.debug("Configurando retriever e cadeias de processamento")
        retriever = self._get_retriever()
        combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        # Gerar resposta
        logger.info("Executando cadeia de recuperação")
        result = retrieval_chain.invoke(input_dict)
        
        # Atualizar histórico de conversa
        if "answer" in result:
            logger.debug("Atualizando histórico de conversa")
            self.history_manager.update_history(user_id, query, result["answer"])
            
        return result
    
    def clear_user_history(self, user_id: str) -> None:
        """Limpa o histórico de conversa de um usuário específico"""
        self.history_manager.clear_user_history(user_id)