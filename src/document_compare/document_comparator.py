import sys
from dotenv import load_dotenv
import pandas as pd
from logger.custom_logger import CustomLogger
from exception.custom_exception import EnterpriseDocumentChatException
from model.models import SummaryResponse, PromptType
from prompt.prompt_library import PROMPT_REGISTRY
from utils.model_loader import ModelLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser


class DocumentComparatorLLM:
    def __init__(self):
        load_dotenv()
        self.log = CustomLogger().get_logger(__name__)
        self.loader = ModelLoader()
        self.llm = self.loader.load_llm()
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.prompt = PROMPT_REGISTRY[PromptType.DOCUMENT_COMPARISON.value]
        self.chain = self.prompt | self.llm | self.fixing_parser
        self.log.info("DocumentComparatorLLM initialized successfully")
    
    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        """
        compares two documents and returns a structured comparison
        """
        try:
            inputs = {
                "format_instruction": self.parser.get_format_instructions(),
                "combined_docs": combined_docs
            }
            self.log.info("Starting document comparison", inputs=inputs)
            response = self.chain.invoke(inputs)    
            self.log.info("Document comparison successful", response=response)
            return self._format_response(response)
        
        except Exception as e:
            self.log.error("Document comparison failed", error=str(e))
            raise EnterpriseDocumentChatException("Error comparing documents", sys)
    
    def _format_response(self, response: list[dict]) -> pd.DataFrame:
        """
        formats the LLM response into a structured format
        """
        try:
            df = pd.DataFrame(response)
            self.log.info("Response formatted successfully", dataframe=df)
            return df
        
        except Exception as e:
            self.log.error("Response formatting failed", error=str(e))
            raise EnterpriseDocumentChatException("Error formatting response", sys)