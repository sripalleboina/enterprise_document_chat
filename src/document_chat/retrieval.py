import sys
import os
from operator import itemgetter
from typing import Optional, List, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from exception.custom_exception import EnterpriseDocumentChatException
from logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5, index_name="index")
        answer = rag.invoke("What is ...?", chat_history=[])
    """
    
    def __init__(self, session_id: Optional[str], retriever=None):
        try:
            self.session_id = session_id
            
            #Load LLM
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]
            
            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()
                
            log.info("ConversationalRAG initialized", session_id=self.session_id)
            
        except Exception as e:
            log.error("FAiled to initialize ConversationalRAG", error=str(e))
            raise EnterpriseDocumentChatException("Initialization error in ConversationalRAG", sys)
        
    
    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """ 
        Load FAISS vectorstore from disk and build retriever + LCEL chain.
        """
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index path not found: {index_path}")
            
            embeddings = ModelLoader().load_embedding_model()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )
            
            if search_kwargs is None:
                search_kwargs = {"k": k}
                
            self.retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
            
            self._build_lcel_chain()
            
            log.info(
                "FAISS retriever loaded successfully", 
                index_path=index_path, 
                index_name=index_name, 
                k=k,
                session_id=self.session_id
            )
            return self.retriever
                    
        except Exception as e:
            log.error("Failed to load FAISS retriever", error=str(e))
            raise EnterpriseDocumentChatException("Error loading FAISS retriever", sys)
        
    def invoke(self, user_input:str, chat_history: Optional[List[BaseMessage]] = None)-> str:
        try:
            if self.chain is None:
                raise EnterpriseDocumentChatException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            response = self.chain.invoke(payload)
            if not response:
                log.warning("No answer generated", user_input=user_input, session_id=self.session_id)
                return "no answer generated"
            
            log.info(
                "Chain invoked successfully", 
                user_input=user_input,
                session_id=self.session_id,
                answer_preview=str(response)[:150],
            )
            return response
    
        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise EnterpriseDocumentChatException("Invocation error in ConversationalRAG", sys)
    
    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise EnterpriseDocumentChatException("LLM loading error in ConversationalRAG", sys)
        
    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
    
    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise EnterpriseDocumentChatException("No retriever set before building chain", sys)
            
            # 1) Rewriting the question based on chat history
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )
            # 2) Retrieve relevant documents based on rewritten question
            retrieve_docs = question_rewriter | self.retriever | self._format_docs
            
            # 3) Feed context + original input + chat history into answer prompt
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )
            
            log.info("LCEL chain built successfully", session_id=self.session_id)
            
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise EnterpriseDocumentChatException("LCEL chain building error in ConversationalRAG", sys)