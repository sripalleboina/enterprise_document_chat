import sys
from pathlib import Path
import fitz
from logger.custom_logger import CustomLogger
from exception.custom_exception import EnterpriseDocumentChatException

class DocumentIngestion:
    def __init__(self, base_dir):
        self.log = CustomLogger().get_logger(__name__)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    
    def delete_existing_files(self):
        """
        Deletes existing files at the specified paths.
        """
        try:
            if self.base_dir.exists() and self.base_dir.is_dir():
                for file in self.base_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                        self.log.info("File deleted", path=str(file))
                self.log.info("Directory cleaned successfully", directory=str(self.base_dir))    
        
        except Exception as e:
            self.log.error("Failed to delete existing files", error=str(e))
            raise EnterpriseDocumentChatException("Error deleting existing files", sys)
    
    def save_uploaded_files(self, reference_file, actual_file):
        """
        Saves newly uploaded files to the specified paths.
        """
        try:
            self.delete_existing_files()
            self.log.info("Existing files deleted successfully")
            
            ref_path=self.base_dir / reference_file.name
            act_path=self.base_dir / actual_file.name
            
            if not reference_file.name.endswith(".pdf") or not actual_file.name.endswith(".pdf"):
                raise ValueError("Only PDF files are supported")
            
            with open(ref_path, "wb") as f:
                f.write(reference_file.getbuffer())
                
            with open(act_path, "wb") as f:
                f.write(actual_file.getbuffer())
                
            self.log.info("Uploaded files saved successfully", reference_file=str(ref_path), actual_file=str(act_path))
            return ref_path, act_path
            
        
        except Exception as e:
            self.log.error("Failed to save uploaded files", error=str(e))
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