
import os
import sys
from dotenv import load_dotenv
from utils.config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
#from langchain_openai import ChatOpenAI
from logger.custom_logger import CustomLogger
from exception.custom_exception import EnterpriseDocumentChatException


log = CustomLogger().get_logger(__name__)


class ModelLoader:
    """
    A Utility class to load embedding models and LLM models.
    """
    
    def __init__(self):
        
        load_dotenv()
        self._validate_env()
        self.config = load_config()
        log.info("Configuration loaded successfully.", config_keys=list(self.config.keys()))
    
    def _validate_env(self):
        """
        Validate necessary environment variables.
        Ensure API keys exist.
        """
        required_vars = ["GOOGLE_API_KEY", "GROQ_API_KEY"]
        self.api_keys = {key:os.getenv(key) for key in required_vars}
        missing = [k for k, v in self.api_keys.items() if not v]
        if missing:
            log.error(f"Missing environment variables", missing_vars=missing)
            raise EnterpriseDocumentChatException(f"Missing environment variables", sys)
        log.info("Environment variables validated", available_keys=[k for k in self.api_keys if self.api_keys[k]])
        
    
    def load_embedding_model(self):
        """
        Load and return the embedding model based on configuration.
        Returns:
            An instance of the embedding model.
        """
        try:
            log.info("Loading embedding model...")
            model_name = self.config["embedding_model"]["model_name"]
            return GoogleGenerativeAIEmbeddings(model=model_name)
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise EnterpriseDocumentChatException("Failed to load embedding model", sys)
    
    def load_llm(self):
        """
        Load and return the LLm model.
        """
        
        llm_block = self.config["llm"]
        
        log.info("Loading LLM model...")
        
        provider_key = os.getenv("LLM_PROVIDER", "groq")
        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key)
            raise EnterpriseDocumentChatException(f"LLM provider '{provider_key}' not found in configuration", sys)  
        
        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_tokens", 2048)  
        
        log.info("Loading LLM", provider=provider, model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        
        if provider == "google":
            llm = ChatGoogleGenerativeAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return llm
        
        elif provider == "groq":
            llm = ChatGroq(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return llm
        
        # elif provider == "openai":
            # llm = ChatOpenAI(
            #     model_name=model_name,
            #     temperature=temperature,
            #     max_tokens=max_tokens
            # )
            # return llm
            
        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise EnterpriseDocumentChatException(f"Unsupported LLM provider '{provider}'", sys) 
        
if __name__ == "__main__":
    
    loader = ModelLoader()
    embedding_model = loader.load_embedding_model()
    print(f"Embedding model loaded: {embedding_model}")
    
    llm_model = loader.load_llm()
    print(f"LLM model loaded: {llm_model}")
    
    result = llm_model.invoke("Hello, how are you?")
    print(f"LLM response: {result.content}")
        