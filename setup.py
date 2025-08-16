"""
SteamNoodles Feedback Agent - Automated Setup Script
Sets up the project environment and dependencies
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_header():
    """Print setup header"""
    print("🍜 SteamNoodles Feedback Agent - Setup")
    print("=" * 50)
    print("Automated setup for the multi-agent feedback system")
    print()


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("⚠️  Python 3.8 or higher is required")
        print("Please upgrade Python and try again")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True


def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        
        # Install requirements
        print("Installing requirements...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        
        print("✅ All packages installed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print("You may need to install packages manually:")
        print("pip install -r requirements.txt")
        return False


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        'data',
        'outputs',
        'logs',
        'examples',
        'docs'
    ]
    
    created_dirs = []
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            created_dirs.append(directory)
    
    if created_dirs:
        print(f"✅ Created directories: {', '.join(created_dirs)}")
    else:
        print("✅ All directories already exist")


def create_sample_files():
    """Create sample configuration files"""
    print("\n📄 Creating sample configuration files...")
    
    # Create sample.env if it doesn't exist
    if not os.path.exists('sample.env'):
        env_content = """# SteamNoodles Feedback Agent - Environment Configuration
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
        with open('sample.env', 'w') as f:
            f.write(env_content)
        print("✅ Created sample.env")
    
    # Create .gitignore if it doesn't exist
    if not os.path.exists('.gitignore'):
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
logs/
outputs/
*.log
*.png
*.jpg
*.jpeg
*.gif
temp/

# API Keys and secrets
.env
config.json
secrets.json
"""
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print("✅ Created .gitignore")


def test_installation():
    """Test if the installation works"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test importing main modules
        print("Testing imports...")
        from main import SteamNoodlesAgentFramework
        from config import Config
        
        print("✅ Core modules import successfully")
        
        # Test basic functionality
        print("Testing basic functionality...")
        config = Config()
        framework = SteamNoodlesAgentFramework(config)
        
        # Test feedback processing
        test_feedback = "Test feedback for setup verification"
        result = framework.process_single_feedback(test_feedback)
        
        if result and hasattr(result, 'sentiment_result'):
            print("✅ Feedback processing works")
        
        # Test visualization (without saving)
        summary = framework.get_sentiment_summary("last 7 days")
        if summary and 'total_reviews' in summary:
            print("✅ Sentiment analysis works")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def display_next_steps():
    """Display next steps for the user"""
    print("\n🎯 SETUP COMPLETE!")
    print("=" * 30)
    
    print("\n📋 Next Steps:")
    print("1. 🚀 Try the quick demo:")
    print("   python examples/quick_start.py")
    
    print("\n2. 🎪 Run the interactive demo:")
    print("   python demo.py")
    
    print("\n3. 🧪 Run the test suite:")
    print("   python test_agents.py")
    
    print("\n4. ⚙️  Configure API keys (optional):")
    print("   - Copy sample.env to .env")
    print("   - Add your OpenAI API key for enhanced features")
    print("   - Add your HuggingFace API key for better sentiment analysis")
    
    print("\n5. 📚 Read the documentation:")
    print("   - README.md - Complete project documentation")
    print("   - docs/API.md - API documentation")
    print("   - docs/TROUBLESHOOTING.md - Troubleshooting guide")
    
    print("\n🔧 Advanced Usage:")
    print("- Customize configuration in config.py")
    print("- Add custom sentiment analysis models")
    print("- Extend visualization options")
    print("- Deploy as web service")


def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n⚠️  Package installation failed, but you can continue with manual installation")
    
    # Create directories
    create_directories()
    
    # Create sample files
    create_sample_files()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
    else:
        print("\n⚠️  Setup completed with warnings. Some features may not work correctly.")
        print("Please check the error messages above and install missing dependencies.")
    
    # Display next steps
    display_next_steps()


def quick_setup():
    """Quick setup without testing"""
    print("🚀 Quick Setup Mode")
    print("=" * 20)
    
    create_directories()
    create_sample_files()
    
    print("\n✅ Quick setup complete!")
    print("Run 'python setup.py' for full setup with testing.")


if __name__ == "__main__":
    # Check for quick setup mode
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_setup()
    else:
        main()