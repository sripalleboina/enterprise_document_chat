import sys
import uuid
from pathlib import Path
import fitz
from datetime import datetime, timezone
from logger.custom_logger import CustomLogger
from exception.custom_exception import EnterpriseDocumentChatException

class DocumentIngestion:
    """
    Handles document ingestion, storage, and combination for comparison.
    """
    
    def __init__(self, base_dir:str="data\\doc_compare", session_id=None):
        self.log = CustomLogger().get_logger(__name__)
        self.base_dir = Path(base_dir)
        self.session_id = session_id or f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        
        self.log.info("DocumentIngestion initialized", session_path=str(self.session_path))
    
    def save_uploaded_files(self, reference_file, actual_file):
        """
        save uploaded reference and actual files in the session directory.
        """
        try:
            ref_path = self.session_path / reference_file.name
            act_path = self.session_path / actual_file.name
            
            if not reference_file.name.lower().endswith(".pdf") or not actual_file.name.lower().endswith(".pdf"):
                raise ValueError("Only PDF files are supported")
            
            with open(ref_path, "wb") as f:
                f.write(reference_file.getbuffer())
                
            with open(act_path, "wb") as f:
                f.write(actual_file.getbuffer())
                
            self.log.info("Uploaded files saved successfully", reference_file=str(ref_path), actual_file=str(act_path), session=self.session_id)
            return ref_path, act_path
            
        
        except Exception as e:
            self.log.error("Failed to save uploaded files", error=str(e), session=self.session_id)
            raise EnterpriseDocumentChatException("Error saving uploaded files", sys)
    
    def read_pdfs(self, pdf_path: Path) -> str:
        """
        Reads a PDF file and extracts text from each page.
        """
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise EnterpriseDocumentChatException("PDF is encrypted and cannot be read", sys)
                
                all_text = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    
                    if text.strip():
                        all_text.append(f"\n --- Page {page_num + 1} ---\n{text}")
                        
                self.log.info(f"PDF read successfully", file=str(pdf_path), pages=len(all_text))
                return "\n".join(all_text)
            
        except Exception as e:
            self.log.error("Failed to read PDF", error=str(e))
            raise EnterpriseDocumentChatException("Error reading PDF file", sys)
        
    
    def combine_documents(self) -> str:
        """
        Combines reference and actual document texts into a single string.
        """
        try:
            doc_parts = []
            
            for filename in sorted(self.base_dir.iterdir()):
                if filename.is_file() and filename.suffix.lower() == ".pdf":
                    content = self.read_pdfs(filename)
                    doc_parts.append(f"Document: {filename.name}\n{content}")
                
            combined_text = "\n\n".join(doc_parts)
            self.log.info("Documents combined successfully", count=len(doc_parts), session=self.session_id)
            return combined_text
        
        except Exception as e:
            self.log.error("Failed to combine documents", error=str(e), session=self.session_id) 
            raise EnterpriseDocumentChatException("Error combining documents", sys)
        
    def clean_old_sessions(self, keep_latest: int = 3):
        """
        Cleans up old session directories, keeping only the latest 'keep_latest' sessions.
        """
        try:
            session_folders = sorted(
                [f for f in self.base_dir.iterdir() if f.is_dir()],
                reverse=True
            )
            for folder in session_folders[keep_latest:]:
                for item in folder.iterdir():
                    if item.is_file():
                        item.unlink()
                folder.rmdir()
                self.log.info("Old session cleaned up", path=str(folder))
                
        except Exception as e:
            self.log.error("Failed to clean old sessions", error=str(e))
            raise EnterpriseDocumentChatException("Error cleaning old sessions", sys)
        