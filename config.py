

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for SteamNoodles Agent Framework"""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    
    # Model Configuration
    OPENAI_MODEL: str = "gpt-3.5-turbo-instruct"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 150
    
    # HuggingFace Model Configuration
    HF_SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # Directory Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    OUTPUTS_DIR: str = os.path.join(BASE_DIR, "outputs")
    LOGS_DIR: str = os.path.join(BASE_DIR, "logs")
    EXAMPLES_DIR: str = os.path.join(BASE_DIR, "examples")
    DOCS_DIR: str = os.path.join(BASE_DIR, "docs")
    
    # Visualization Settings
    PLOT_WIDTH: int = 12
    PLOT_HEIGHT: int = 8
    PLOT_DPI: int = 300
    
    # Performance Settings
    BATCH_SIZE: int = 10
    MAX_WORKERS: int = 4
    TIMEOUT_SECONDS: int = 30
    
    # Sentiment Analysis Settings
    CONFIDENCE_THRESHOLD: float = 0.5
    MIN_TEXT_LENGTH: int = 3
    MAX_TEXT_LENGTH: int = 1000
    
    # Response Generation Settings
    RESPONSE_MAX_LENGTH: int = 200
    RESPONSE_MIN_LENGTH: int = 20
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # Sample Data Configuration
    SAMPLE_DATA_SIZE: int = 50
    SAMPLE_DATE_RANGE_DAYS: int = 30
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Load environment variables
        self.load_from_environment()
        
        # Create necessary directories
        self.create_directories()
    
    def load_from_environment(self):
        """Load configuration from environment variables"""
        # API Keys
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', self.OPENAI_API_KEY)
        self.HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', self.HUGGINGFACE_API_KEY)
        
        # Model Configuration
        self.OPENAI_MODEL = os.getenv('OPENAI_MODEL', self.OPENAI_MODEL)
        self.OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', str(self.OPENAI_TEMPERATURE)))
        self.OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', str(self.OPENAI_MAX_TOKENS)))
        
        # HuggingFace Configuration
        self.HF_SENTIMENT_MODEL = os.getenv('HF_SENTIMENT_MODEL', self.HF_SENTIMENT_MODEL)
        
        # Directory Paths (allow custom paths via environment)
        self.DATA_DIR = os.getenv('STEAMNOODLES_DATA_DIR', self.DATA_DIR)
        self.OUTPUTS_DIR = os.getenv('STEAMNOODLES_OUTPUTS_DIR', self.OUTPUTS_DIR)
        self.LOGS_DIR = os.getenv('STEAMNOODLES_LOGS_DIR', self.LOGS_DIR)
        
        # Performance Settings
        self.BATCH_SIZE = int(os.getenv('BATCH_SIZE', str(self.BATCH_SIZE)))
        self.MAX_WORKERS = int(os.getenv('MAX_WORKERS', str(self.MAX_WORKERS)))
        self.TIMEOUT_SECONDS = int(os.getenv('TIMEOUT_SECONDS', str(self.TIMEOUT_SECONDS)))
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', self.LOG_LEVEL)
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.DATA_DIR,
            self.OUTPUTS_DIR,
            self.LOGS_DIR,
            self.EXAMPLES_DIR,
            self.DOCS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def validate_config(self) -> list:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check API key format (if provided)
        if self.OPENAI_API_KEY:
            if not self.OPENAI_API_KEY.startswith(('sk-', 'org-')):
                issues.append("OpenAI API key format appears invalid")
        
        # Check numeric ranges
        if not 0 <= self.OPENAI_TEMPERATURE <= 2:
            issues.append("OpenAI temperature should be between 0 and 2")
        
        if not 1 <= self.OPENAI_MAX_TOKENS <= 4096:
            issues.append("OpenAI max tokens should be between 1 and 4096")
        
        if not 0 <= self.CONFIDENCE_THRESHOLD <= 1:
            issues.append("Confidence threshold should be between 0 and 1")
        
        if self.MIN_TEXT_LENGTH >= self.MAX_TEXT_LENGTH:
            issues.append("Min text length should be less than max text length")
        
        if self.RESPONSE_MIN_LENGTH >= self.RESPONSE_MAX_LENGTH:
            issues.append("Min response length should be less than max response length")
        
        # Check directory permissions
        for directory in [self.OUTPUTS_DIR, self.LOGS_DIR]:
            if not os.access(directory, os.W_OK):
                issues.append(f"No write permission for directory: {directory}")
        
        return issues
    
    def get_model_config(self) -> dict:
        """Get model configuration dictionary"""
        return {
            'openai': {
                'api_key': self.OPENAI_API_KEY,
                'model': self.OPENAI_MODEL,
                'temperature': self.OPENAI_TEMPERATURE,
                'max_tokens': self.OPENAI_MAX_TOKENS
            },
            'huggingface': {
                'api_key': self.HUGGINGFACE_API_KEY,
                'sentiment_model': self.HF_SENTIMENT_MODEL
            }
        }
    
    def get_paths_config(self) -> dict:
        """Get paths configuration dictionary"""
        return {
            'base_dir': self.BASE_DIR,
            'data_dir': self.DATA_DIR,
            'outputs_dir': self.OUTPUTS_DIR,
            'logs_dir': self.LOGS_DIR,
            'examples_dir': self.EXAMPLES_DIR,
            'docs_dir': self.DOCS_DIR
        }
    
    def get_visualization_config(self) -> dict:
        """Get visualization configuration dictionary"""
        return {
            'width': self.PLOT_WIDTH,
            'height': self.PLOT_HEIGHT,
            'dpi': self.PLOT_DPI
        }
    
    def get_performance_config(self) -> dict:
        """Get performance configuration dictionary"""
        return {
            'batch_size': self.BATCH_SIZE,
            'max_workers': self.MAX_WORKERS,
            'timeout_seconds': self.TIMEOUT_SECONDS
        }
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            'api_keys': {
                'openai_api_key': '***' if self.OPENAI_API_KEY else None,
                'huggingface_api_key': '***' if self.HUGGINGFACE_API_KEY else None
            },
            'models': self.get_model_config(),
            'paths': self.get_paths_config(),
            'visualization': self.get_visualization_config(),
            'performance': self.get_performance_config(),
            'sentiment': {
                'confidence_threshold': self.CONFIDENCE_THRESHOLD,
                'min_text_length': self.MIN_TEXT_LENGTH,
                'max_text_length': self.MAX_TEXT_LENGTH
            },
            'response': {
                'max_length': self.RESPONSE_MAX_LENGTH,
                'min_length': self.RESPONSE_MIN_LENGTH
            },
            'logging': {
                'level': self.LOG_LEVEL,
                'format': self.LOG_FORMAT,
                'file_size': self.LOG_FILE_SIZE,
                'backup_count': self.LOG_BACKUP_COUNT
            }
        }
    
    @classmethod
    def from_file(cls, config_file: str) -> 'Config':
        """Load configuration from file"""
        import json
        
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        config = cls()
        
        # Update configuration from file data
        if 'api_keys' in config_data:
            api_keys = config_data['api_keys']
            if 'openai_api_key' in api_keys and api_keys['openai_api_key'] != '***':
                config.OPENAI_API_KEY = api_keys['openai_api_key']
            if 'huggingface_api_key' in api_keys and api_keys['huggingface_api_key'] != '***':
                config.HUGGINGFACE_API_KEY = api_keys['huggingface_api_key']
        
        if 'models' in config_data:
            models = config_data['models']
            if 'openai' in models:
                openai_config = models['openai']
                config.OPENAI_MODEL = openai_config.get('model', config.OPENAI_MODEL)
                config.OPENAI_TEMPERATURE = openai_config.get('temperature', config.OPENAI_TEMPERATURE)
                config.OPENAI_MAX_TOKENS = openai_config.get('max_tokens', config.OPENAI_MAX_TOKENS)
            if 'huggingface' in models:
                hf_config = models['huggingface']
                config.HF_SENTIMENT_MODEL = hf_config.get('sentiment_model', config.HF_SENTIMENT_MODEL)
        
        return config
    
    def save_to_file(self, config_file: str):
        """Save configuration to file"""
        import json
        
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def print_config(self):
        """Print current configuration"""
        print("🍜 SteamNoodles Agent Configuration")
        print("=" * 50)
        
        config_dict = self.to_dict()
        
        def print_section(name: str, data: dict, indent: int = 0):
            prefix = "  " * indent
            print(f"{prefix}{name.upper()}:")
            for key, value in data.items():
                if isinstance(value, dict):
                    print_section(key, value, indent + 1)
                else:
                    print(f"{prefix}  {key}: {value}")
            print()
        
        for section_name, section_data in config_dict.items():
            if isinstance(section_data, dict):
                print_section(section_name, section_data)
    
    def get_environment_template(self) -> str:
        """Get environment variable template"""
        template = """# SteamNoodles Feedback Agent - Environment Configuration
# Copy this file to .env and fill in your values

# API Keys (optional - system will use fallback methods if not provided)
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Model Configuration
OPENAI_MODEL=gpt-3.5-turbo-instruct
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=150

# HuggingFace Configuration
HF_SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest

# Custom Directory Paths (optional)
STEAMNOODLES_DATA_DIR=./data
STEAMNOODLES_OUTPUTS_DIR=./outputs
STEAMNOODLES_LOGS_DIR=./logs

# Performance Settings
BATCH_SIZE=10
MAX_WORKERS=4
TIMEOUT_SECONDS=30

# Logging
LOG_LEVEL=INFO
"""
        return template


def load_config_from_env_file(env_file: str = ".env") -> Config:
    """Load configuration from environment file"""
    if os.path.exists(env_file):
        # Load environment variables from file
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    return Config()


def create_sample_env_file(filename: str = "sample.env"):
    """Create a sample environment file"""
    config = Config()
    template = config.get_environment_template()
    
    with open(filename, 'w') as f:
        f.write(template)
    
    print(f"📄 Sample environment file created: {filename}")
    print("Copy this to '.env' and fill in your API keys to enable advanced features.")


if __name__ == "__main__":
    # Demo configuration usage
    print("🔧 Configuration Demo")
    print("=" * 30)
    
    # Create and display default configuration
    config = Config()
    config.print_config()
    
    # Validate configuration
    issues = config.validate_config()
    if issues:
        print("⚠️  Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration is valid!")
    
    # Create sample environment file
    create_sample_env_file()
    
    print("\n🎯 Configuration Tips:")
    print("1. Set OPENAI_API_KEY for enhanced LLM features")
    print("2. Set HUGGINGFACE_API_KEY for better sentiment analysis")
    print("3. Adjust BATCH_SIZE based on your system capabilities")
    print("4. Use custom directory paths for production deployment")