import yaml

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file. 
    """
    
    with open(config_path, 'r') as file:
        # Opens the YAML file in read mode ('r') [web:33]
        # "with" ensures the file is automatically closed after reading
        
        config = yaml.safe_load(file)
        # Reads the YAML file and converts it to a Python dictionary
        
    return config

