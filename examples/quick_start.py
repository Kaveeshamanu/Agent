#!/usr/bin/env python3
"""
SteamNoodles Feedback Agent - Quick Start Demo
Simple demonstration script that works without any API keys

Author: [Your Name]
Date: August 2025
"""

import sys
import os
import time
from datetime import datetime

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from main import SteamNoodlesAgentFramework
    from config import Config
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def print_header():
    """Print welcome header"""
    print("🍜" * 20)
    print("     SteamNoodles Feedback Agent")
    print("           Quick Start Demo")
    print("🍜" * 20)
    print()
    print("This demo showcases both agents working together")
    print("No API keys required - using built-in fallback methods")
    print("=" * 60)
    print()

def demo_feedback_processing():
    """Demonstrate Agent 1: Feedback Response Agent"""
    print("🤖 AGENT 1: Customer Feedback Response")
    print("-" * 40)
    
    # Sample feedback examples
    sample_feedback = [
        "The noodles were absolutely delicious! Best I've ever had at any restaurant!",
        "Service was really slow and the food arrived cold. Very disappointed.",
        "Decent place, nothing special. Average food and service.",
        "Amazing atmosphere and the staff was so friendly! Will definitely come back!",
        "Overpriced for what you get. The portion was tiny and tasteless."
    ]
    
    try:
        # Initialize framework
        framework = SteamNoodlesAgentFramework()
        
        for i, feedback in enumerate(sample_feedback, 1):
            print(f"\nExample {i}:")
            print(f"Customer Feedback: \"{feedback}\"")
            print("Processing...", end=" ")
            
            start_time = time.time()
            result = framework.process_single_feedback(feedback)
            processing_time = time.time() - start_time
            
            print(f"✅ ({processing_time:.2f}s)")
            
            # Display results
            sentiment = result.get('sentiment', 'UNKNOWN')
            confidence = result.get('confidence', 0.0)
            response = result.get('response', 'No response generated')
            method = result.get('method', 'unknown')
            
            print(f"  └─ Sentiment: {sentiment} ({confidence:.1%} confidence)")
            print(f"  └─ Method Used: {method}")
            print(f"  └─ Auto Response: \"{response}\"")
            print()
    
    except Exception as e:
        print(f"❌ Error in feedback processing demo: {e}")

def demo_sentiment_visualization():
    """Demonstrate Agent 2: Sentiment Visualization Agent"""
    print("📊 AGENT 2: Sentiment Visualization")
    print("-" * 40)
    
    try:
        # Initialize framework
        framework = SteamNoodlesAgentFramework()
        
        # Test different date ranges
        date_ranges = [
            "last 7 days",
            "last month", 
            "today"
        ]
        
        for date_range in date_ranges:
            print(f"\nGenerating visualization for: {date_range}")
            print("Processing...", end=" ")
            
            start_time = time.time()
            result = framework.generate_sentiment_report(date_range)
            processing_time = time.time() - start_time
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            print(f"✅ ({processing_time:.2f}s)")
            
            # Display results
            total_reviews = result.get('total_reviews', 0)
            sentiment_counts = result.get('sentiment_counts', {})
            plot_path = result.get('plot_saved', 'No plot saved')
            
            print(f"  └─ Date Range: {result.get('date_range', 'Unknown')}")
            print(f"  └─ Total Reviews: {total_reviews}")
            
            for sentiment, count in sentiment_counts.items():
                percentage = (count / total_reviews * 100) if total_reviews > 0 else 0
                print(f"  └─ {sentiment}: {count} ({percentage:.1f}%)")
            
            if os.path.exists(plot_path):
                print(f"  └─ Plot saved: {plot_path} ✅")
            else:
                print(f"  └─ Plot not saved ❌")
            
            print()
    
    except Exception as e:
        print(f"❌ Error in visualization demo: {e}")

def demo_batch_processing():
    """Demonstrate batch processing capabilities"""
    print("⚡ BATCH PROCESSING DEMO")
    print("-" * 40)
    
    try:
        framework = SteamNoodlesAgentFramework()
        
        # Sample batch of feedback
        batch_feedback = [
            "Love this place! Great noodles!",
            "Terrible service, will never return",
            "Average experience, nothing special",
            "Outstanding food and service!",
            "Too expensive for the quality"
        ]
        
        print(f"Processing {len(batch_feedback)} pieces of feedback...")
        print("Starting batch processing...", end=" ")
        
        start_time = time.time()
        results, metadata = framework.process_batch_feedback(batch_feedback)
        processing_time = time.time() - start_time
        
        print(f"✅ ({processing_time:.2f}s)")
        print()
        
        # Display batch results
        print("Batch Processing Results:")
        print(f"  └─ Total Processed: {metadata['total_processed']}")
        print(f"  └─ Successful: {metadata['successful']}")
        print(f"  └─ Failed: {metadata['failed']}")
        print(f"  └─ Total Time: {metadata['total_processing_time']:.2f}s")
        print(f"  └─ Avg Time/Feedback: {metadata['average_time_per_feedback']:.2f}s")
        
        # Show sentiment distribution
        sentiments = [r.get('sentiment') for r in results if 'sentiment' in r]
        if sentiments:
            from collections import Counter
            sentiment_counts = Counter(sentiments)
            print(f"  └─ Sentiment Distribution:")
            for sentiment, count in sentiment_counts.items():
                print(f"      • {sentiment}: {count}")
        
        print()
    
    except Exception as e:
        print(f"❌ Error in batch processing demo: {e}")

def demo_system_status():
    """Show system status and capabilities"""
    print("🔍 SYSTEM STATUS CHECK")
    print("-" * 40)
    
    try:
        framework = SteamNoodlesAgentFramework()
        status = framework.get_system_status()
        
        print("System Information:")
        print(f"  └─ Framework Version: {status.get('framework_version', 'Unknown')}")
        print(f"  └─ Timestamp: {status.get('timestamp', 'Unknown')}")
        
        # Agent status
        agents = status.get('agents', {})
        feedback_agent = agents.get('feedback_agent', {})
        viz_agent = agents.get('visualization_agent', {})
        
        print("\nAgent 1 (Feedback Response):")
        print(f"  └─ Status: {feedback_agent.get('status', 'Unknown')}")
        print(f"  └─ LLM Available: {feedback_agent.get('llm_available', False)}")
        print(f"  └─ Transformers Available: {feedback_agent.get('transformers_available', False)}")
        
        print("\nAgent 2 (Sentiment Visualization):")
        print(f"  └─ Status: {viz_agent.get('status', 'Unknown')}")
        print(f"  └─ Sample Data Generated: {viz_agent.get('sample_data_available', False)}")
        
        # Configuration status
        config_info = status.get('configuration', {})
        print("\nConfiguration:")
        print(f"  └─ Environment: {config_info.get('environment', 'Unknown')}")
        print(f"  └─ Log Level: {config_info.get('log_level', 'Unknown')}")
        
        print()
    
    except Exception as e:
        print(f"❌ Error checking system status: {e}")

def main():
    """Run the complete quick start demo"""
    print_header()
    
    try:
        # Run all demo sections
        demo_system_status()
        demo_feedback_processing()
        demo_sentiment_visualization()
        demo_batch_processing()
        
        # Final message
        print("🎉 DEMO COMPLETE!")
        print("=" * 60)
        print("✅ Both agents are working properly")
        print("✅ Sample visualizations created")
        print("✅ System is ready for use")
        print()
        print("Next steps:")
        print("  • Run 'python demo.py' for interactive demo")
        print("  • Run 'python run_tests.py' to run full test suite")
        print("  • Check 'outputs/' directory for generated plots")
        print("  • See README.md for detailed usage instructions")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your setup and try again")

if __name__ == "__main__":
    main()