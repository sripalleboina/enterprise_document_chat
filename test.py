# Test for Document Analyser
# import os
# from pathlib import Path
# from src.document_analyser.data_ingestion import DocumentHandler
# from src.document_analyser.data_analysis import DocumentAnalyzer

# DOC_PATH = r"C:\Users\sripa\enterprise_document_chat\data\doc_analysis\sample.pdf"

# class DummyFile:
#     def __init__(self, file_path):
#         self.name = Path(file_path).name
#         self._filepath = file_path
        
#     def getbuffer(self):
#         return open(self._filepath, 'rb').read()
    
# def main():
#     try:
#         #-----------------STEP 1: Document Ingestion & Text Extraction-----------------#
#         print(" Starting Document Ingestion...")
#         dummy_file = DummyFile(DOC_PATH)
        
#         handler = DocumentHandler(session_id="test_ingestion_analysis")
#         saved_path = handler.save_pdf(dummy_file)
#         print(f"PDF saved at: {saved_path}")
        
#         text_content = handler.read_pdf(saved_path)
#         print(f"Extracted Text length: {len(text_content)} characters\n")
        
        
#         #-----------------STEP 2: Document Analysis-----------------#        
#         print(" Starting Document Analysis...")
#         analyzer = DocumentAnalyzer()
#         analysis_result = analyzer.analyze_document(text_content)
        
        
#         #-----------------STEP 3: Display Results-----------------#
#         print("\n=== Document Analysis Result ===")
#         for key, value in analysis_result.items():
#             print(f"{key}: {value}\n")
        
#     except Exception as e:
#         print(f"Error during testing: {e}")

# if __name__ == "__main__":
#     main()

# Test for Document Comparator
# import io
# from pathlib import Path
# from src.document_compare.document_comparator import DocumentComparatorLLM
# from src.document_compare.data_ingestion import DocumentIngestion

# def load_fake_uploaded_file(file_path: Path):
#     return io.BytesIO(file_path.read_bytes())


# def test_compare_documents():
#     ref_path=Path("C:\\Users\\sripa\\enterprise_document_chat\\data\\doc_compare\\Long_Report_V1.pdf")
#     act_path=Path("C:\\Users\\sripa\\enterprise_document_chat\\data\\doc_compare\\Long_Report_V2.pdf")
    
#     class FakeUpload:
#         def __init__(self, fake_path: Path):
#             self.name = fake_path.name
#             self._buffer = fake_path.read_bytes()
            
#         def getbuffer(self):    
#             return self._buffer
        
#     comparator = DocumentIngestion()
#     ref_upload = FakeUpload(ref_path)
#     act_upload = FakeUpload(act_path)
    
#     ref_file, act_file = comparator.save_uploaded_files(ref_upload, act_upload)
#     combined_text = comparator.combine_documents()
#     comparator.clean_old_sessions(keep_latest=3)
    
#     print("\n Combined Text Preview: \n", combined_text[:1000])
    
    
#     llm_comparator = DocumentComparatorLLM()
#     comparison_df = llm_comparator.compare_documents(combined_text)
    
#     print("\n Document Comparison Result: \n", comparison_df)
    
    
# if __name__ == "__main__":
#     test_compare_documents()
    
    
# import sys
# from pathlib import Path
# from langchain_community.vectorstores import FAISS
# from src.single_document_chat.retrieval import ConversationalRAG
# from src.single_document_chat.data_ingestion import SingleDocIngestor
# from utils.model_loader import ModelLoader

# FAISS_INDEX_PATH = Path("faiss_index")

# def test_conversational_rag(pdf_path:str, question:str):
#     try:
#         model_loader = ModelLoader()
#         if FAISS_INDEX_PATH.exists():
#             print("Loading existing FAISS retriever...")
#             embeddings = model_loader.load_embedding_model()
#             vector_store = FAISS.load_local(folder_path=str(FAISS_INDEX_PATH),embeddings=embeddings,allow_dangerous_deserialization=True)
#             retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":5})
#         else:
#             print("Ingesting document and creating FAISS retriever...")
#             with open(pdf_path, "rb") as f:
#                 uploaded_file = [f]
#                 ingestor = SingleDocIngestor()
#                 retriever = ingestor.ingest_files(uploaded_file)
        
#         print("Initializing Conversational RAG...")
#         session_id = "test_conversational_rag"
#         rag = ConversationalRAG(session_id, retriever)
        
#         response = rag.invoke(question)
#         print(f"\nQuestion: {question}\nAnswer: {response}")
            
#     except Exception as e:
#         print(f"Test failed:, {str(e)}")
#         sys.exit(1)
        

# if __name__ == "__main__":
    
#     pdf_path = r"C:\\Users\\sripa\\enterprise_document_chat\\data\\single_document_chat\\NIPS-2017-attention-is-all-you-need-Paper.pdf"
#     question = "What is the main topic discussed in the document?"
    
#     if not Path(pdf_path).exists:
#         print(f"PDF file not found at {pdf_path}. Please provide a valid path.")
#         sys.exit(1)
    
#     test_conversational_rag(pdf_path, question)
    
    
    #testing for multidoc chat
import sys
from pathlib import Path
from src.multi_document_chat.retrieval import ConversationalRAG
from src.multi_document_chat.data_ingestion import DocumentIngestor
    
    
def test_document_ingestion_and_rag():
    try:
        test_files = [
            r"C:\\Users\\sripa\\enterprise_document_chat\\data\\multi_doc_chat\\market_analysis_report.docx",
            r"C:\\Users\\sripa\\enterprise_document_chat\\data\\multi_doc_chat\\NIPS-2017-attention-is-all-you-need-Paper.pdf",
            r"C:\\Users\\sripa\\enterprise_document_chat\\data\\multi_doc_chat\\sample.pdf",
            r"C:\\Users\\sripa\\enterprise_document_chat\\data\\multi_doc_chat\\state_of_the_union.txt"
        ]
        
        uploaded_files = []
        for file_path in test_files:
            if Path(file_path).exists():
                uploaded_files.append(open(file_path, "rb"))
            else:
                print(f"File not found: {file_path}")
                
        if not uploaded_files:
            print("No valid files to ingest. Exiting test.")
            sys.exit(1)
            
        ingestor = DocumentIngestor()
        retriever = ingestor.ingest_files(uploaded_files)
        
        for f in uploaded_files:
            f.close()
            
        session_id = "test_multi_doc_chat"
        
        rag = ConversationalRAG(session_id, retriever)
        question = "what is attention is all you need paper about?"
        answer=rag.invoke(question)
        print("\n Question:", question)
        print("Answer:", answer)
        
        
            
    except Exception as e:
        print(f"Test failed:, {str(e)}")
        sys.exit(1)
        
if __name__ == "__main__":
    test_document_ingestion_and_rag()