import uuid
from pathlib import Path
import sys
from datetime import datetime, timezone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from logger.custom_logger import CustomLogger
from exception.custom_exception import EnterpriseDocumentChatException
from utils.model_loader import ModelLoader

class SingleDocIngestor:
    def __init__(self, data_dir: str = "data/single_document_chat", faiss_dir: str = "faiss_index"):
        try:
            self.log = CustomLogger().get_logger(__name__)            
            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)            
            self.faiss_dir = Path(faiss_dir)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)            
            self.model_loader = ModelLoader()
            self.log.info("SingleDocIngestor initialized", temp_path=str(self.data_dir), faiss_path=str(self.faiss_dir))            
        except Exception as e:
            self.log.error(f"Error initializing SingleDocIngestor:", error=str(e))
            raise EnterpriseDocumentChatException("Initialization error in SingleDDocIngestor", sys)
        
    def ingest_files(self, uploaded_files):
        try:
            documents = []
            
            for uploaded_file in uploaded_files:
                unique_filename = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
                temp_path = self.data_dir / unique_filename
                
                with open(temp_path, "wb") as f_out:
                    f_out.write(uploaded_file.read())
                self.log.info(f"File saved for ingestion", file_name=uploaded_file.name)
                loader = PyPDFLoader(str(temp_path))
                docs = loader.load()
                documents.extend(docs)
            self.log.info(f"File loaded and documents extracted", num_pages=len(documents))
            return self._create_retriever(documents)
                
        except Exception as e:
            self.log.error(f"Document ingestion failed:", error=str(e))
            raise EnterpriseDocumentChatException("Error during file ingestion", sys)
        
    def _create_retriever(self, documents):
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
            chunks = splitter.split_documents(documents) 
            self.log.info(f"Documents split into chunks", num_chunks=len(chunks))
            
            embeddings = self.model_loader.load_embedding_model()
            vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
            
            # save FAISS index to disk
            vector_store.save_local(str(self.faiss_dir))
            self.log.info(f"FAISS index created and saved", faiss_path=str(self.faiss_dir))
            
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":5})
            self.log.info(f"Retriever created from FAISS index", retriever_type=str(type(retriever)))
            return retriever
            
            
        except Exception as e:
            self.log.error(f"Retriever creation failed", error=str(e))
            raise EnterpriseDocumentChatException("Error creating retriever", sys)