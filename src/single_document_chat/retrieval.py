import sys
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils.model_loader import ModelLoader
from exception.custom_exception import EnterpriseDocumentChatException
from logger.custom_logger import CustomLogger
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType

class ConversationalRAG:
    def __init__(self, session_id: str, retriever):
        self.log = CustomLogger().get_logger(__name__)
        self.session_id = session_id
        self.retriever = retriever
        
        try:
            self.llm = self._load_llm()
            self.contextualize_prompt = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]
            
            self.history_aware_retriever = create_history_aware_retriever(
                self.llm, self.retriever, self.contextualize_prompt
            )
            self.log.info("ConversationalRAG initialized", session_id=session_id)
            self.qa_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
            self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.qa_chain)
            self.log.info("RAG chain created", session_id=session_id)
            
            self.chain = RunnableWithMessageHistory(
                self.rag_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            self.log.info("RunnableWithMessageHistory chain created", session_id=session_id)
            
        except Exception as e:
            self.log.error(f"Error initializing ConversationalRAG:", error=str(e), session_id=session_id)
            raise EnterpriseDocumentChatException("Initialization error in ConversationalRAG", sys)
        
    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            self.log.info("LLM loaded successfully", class_name=llm.__class__.__name__)
            return llm
            
        except Exception as e:
            self.log.error(f"Error loading LLM:", error=str(e))
            raise EnterpriseDocumentChatException("Error loading LLM", sys)
        
    def _get_session_history(self, session_id: str):
        try:
            if "store" not in st.session_state:
                st.session_state.store = {}

            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
                self.log.info("New chat session history created", session_id=session_id)

            return st.session_state.store[session_id]
        
        except Exception as e:
            self.log.error(f"Error retrieving session history:", error=str(e), session_id=session_id)
            raise EnterpriseDocumentChatException("Error retrieving session history", sys)
        
    def load_retriever_from_faiss(self, index_path: str):
        try:
            embeddings = ModelLoader().load_embedding_model()
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory does not exist at {index_path}", sys)
            
            vector_store = FAISS.load_local(index_path, embeddings)
            self.log.info("FAISS index loaded successfully", index_path=index_path)
            return vector_store.as_retriever(search_type="similarity", search_kwargs={"k":5})
            
        except Exception as e:
            self.log.error(f"Error loading retriever from FAISS:", error=str(e))
            raise EnterpriseDocumentChatException("Error loading retriever from FAISS", sys)
        
    def invoke(self, user_input:str)->str:
        try:
            response = self.chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": self.session_id}}
            )
            answer = response.get("answer", "No answer.")
            if not answer:
                self.log.warning("Empty answer received", session_id = self.session_id)
            self.log.info("ConversationalRAG invoked successfully", session_id=self.session_id, user_input=user_input, answer_preview=answer[:150])
            return answer   
                    
        except Exception as e:
            self.log.error(f"Error invoking ConversationalRAG:", error=str(e), session_id=self.session_id)
            raise EnterpriseDocumentChatException("Error invoking ConversationalRAG", sys)