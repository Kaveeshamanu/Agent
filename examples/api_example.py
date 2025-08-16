#!/usr/bin/env python3
"""
SteamNoodles Feedback Agent - API Usage Examples
Demonstrates how to use the framework programmatically

Author: [Your Name]
Date: August 2025
"""

import sys
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from main import SteamNoodlesAgentFramework
    from config import Config
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    sys.exit(1)

class SteamNoodlesAPI:
    """
    API wrapper for SteamNoodles Feedback Agent Framework
    Provides clean interface for programmatic usage
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the API wrapper"""
        self.framework = SteamNoodlesAgentFramework(config_path)
        self.session_data = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'processed_feedback': [],
            'generated_reports': []
        }
    
    def analyze_feedback(self, feedback: str) -> Dict[str, Any]:
        """
        Analyze a single piece of customer feedback
        
        Args:
            feedback: Customer feedback text
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            result = self.framework.process_single_feedback(feedback)
            
            # Store in session data
            self.session_data['processed_feedback'].append({
                'timestamp': datetime.now().isoformat(),
                'feedback': feedback,
                'result': result
            })
            
            return {
                'success': True,
                'data': result,
                'session_id': self.session_data['session_id']
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'session_id': self.session_data['session_id']
            }
    
    def analyze_feedback_batch(self, feedback_list: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple pieces of feedback at once
        
        Args:
            feedback_list: List of customer feedback texts
            
        Returns:
            Dictionary containing batch analysis results
        """
        try:
            results, metadata = self.framework.process_batch_feedback(feedback_list)
            
            # Store in session data
            self.session_data['processed_feedback'].extend([
                {
                    'timestamp': datetime.now().isoformat(),
                    'feedback': feedback_list[i],
                    'result': result
                }
                for i, result in enumerate(results) if 'error' not in result
            ])
            
            return {
                'success': True,
                'data': {
                    'results': results,
                    'metadata': metadata
                },
                'session_id': self.session_data['session_id']
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'session_id': self.session_data['session_id']
            }
    
    def generate_sentiment_report(self, date_range: str) -> Dict[str, Any]:
        """
        Generate sentiment visualization and report
        
        Args:
            date_range: Natural language date range (e.g., "last 7 days")
            
        Returns:
            Dictionary containing report data and plot information
        """
        try:
            result = self.framework.generate_sentiment_report(date_range)
            
            # Store in session data
            if 'error' not in result:
                self.session_data['generated_reports'].append({
                    'timestamp': datetime.now().isoformat(),
                    'date_range': date_range,
                    'result': result
                })
            
            return {
                'success': 'error' not in result,
                'data': result,
                'session_id': self.session_data['session_id']
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'session_id': self.session_data['session_id']
            }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        processed_count = len(self.session_data['processed_feedback'])
        reports_count = len(self.session_data['generated_reports'])
        
        # Calculate sentiment distribution
        sentiments = []
        for item in self.session_data['processed_feedback']:
            sentiment = item.get('result', {}).get('sentiment')
            if sentiment:
                sentiments.append(sentiment)
        
        from collections import Counter
        sentiment_distribution = dict(Counter(sentiments))
        
        return {
            'session_id': self.session_data['session_id'],
            'processed_feedback_count': processed_count,
            'generated_reports_count': reports_count,
            'sentiment_distribution': sentiment_distribution,
            'session_data': self.session_data
        }
    
    def save_session_data(self, filepath: str = None) -> str:
        """Save session data to file"""
        if not filepath:
            filepath = f"outputs/session_{self.session_data['session_id']}.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        
        return filepath

def example_single_feedback():
    """Example: Analyze single piece of feedback"""
    print("📝 Example 1: Single Feedback Analysis")
    print("-" * 50)
    
    # Initialize API
    api = SteamNoodlesAPI()
    
    # Sample feedback
    feedback = "The ramen was absolutely incredible! Best noodles I've ever had, and the service was amazing too!"
    
    print(f"Analyzing feedback: \"{feedback}\"")
    
    # Analyze feedback
    result = api.analyze_feedback(feedback)
    
    if result['success']:
        data = result['data']
        print(f"✅ Analysis completed successfully")
        print(f"   Sentiment: {data.get('sentiment', 'Unknown')}")
        print(f"   Confidence: {data.get('confidence', 0):.1%}")
        print(f"   Method: {data.get('method', 'Unknown')}")
        print(f"   Response: \"{data.get('response', 'No response')}\"")
    else:
        print(f"❌ Analysis failed: {result['error']}")
    
    print()

def example_batch_processing():
    """Example: Batch process multiple feedback"""
    print("📦 Example 2: Batch Processing")
    print("-" * 50)
    
    # Initialize API
    api = SteamNoodlesAPI()
    
    # Sample feedback batch
    feedback_batch = [
        "Excellent food and service! Highly recommend!",
        "Food was cold and service was terrible",
        "Average meal, nothing special",
        "Love the atmosphere here, great noodles!",
        "Way too expensive for the portion size"
    ]
    
    print(f"Processing batch of {len(feedback_batch)} feedback items...")
    
    # Process batch
    result = api.analyze_feedback_batch(feedback_batch)
    
    if result['success']:
        data = result['data']
        metadata = data['metadata']
        
        print(f"✅ Batch processing completed")
        print(f"   Total processed: {metadata['total_processed']}")
        print(f"   Successful: {metadata['successful']}")
        print(f"   Failed: {metadata['failed']}")
        print(f"   Processing time: {metadata['total_processing_time']:.2f}s")
        
        # Show sentiment distribution
        sentiments = [r.get('sentiment') for r in data['results'] if 'sentiment' in r]
        if sentiments:
            from collections import Counter
            distribution = Counter(sentiments)
            print(f"   Sentiment distribution: {dict(distribution)}")
    else:
        print(f"❌ Batch processing failed: {result['error']}")
    
    print()

def example_sentiment_report():
    """Example: Generate sentiment report"""
    print("📊 Example 3: Sentiment Report Generation")
    print("-" * 50)
    
    # Initialize API
    api = SteamNoodlesAPI()
    
    # Generate report for different time periods
    time_periods = ["last 7 days", "last month"]
    
    for period in time_periods:
        print(f"Generating report for: {period}")
        
        result = api.generate_sentiment_report(period)
        
        if result['success']:
            data = result['data']
            print(f"✅ Report generated successfully")
            print(f"   Date range: {data.get('date_range', 'Unknown')}")
            print(f"   Total reviews: {data.get('total_reviews', 0)}")
            
            sentiment_counts = data.get('sentiment_counts', {})
            for sentiment, count in sentiment_counts.items():
                total = data.get('total_reviews', 1)
                percentage = (count / total * 100) if total > 0 else 0
                print(f"   {sentiment}: {count} ({percentage:.1f}%)")
            
            plot_path = data.get('plot_saved')
            if plot_path and os.path.exists(plot_path):
                print(f"   Plot saved: {plot_path}")
        else:
            print(f"❌ Report generation failed: {result.get('error', 'Unknown error')}")
        
        print()

def example_session_management():
    """Example: Session data management"""
    print("💾 Example 4: Session Management")
    print("-" * 50)
    
    # Initialize API
    api = SteamNoodlesAPI()
    
    # Process some feedback to generate session data
    sample_feedback = [
        "Great food!",
        "Poor service",
        "Average experience"
    ]
    
    for feedback in sample_feedback:
        api.analyze_feedback(feedback)
    
    # Generate a report
    api.generate_sentiment_report("last 7 days")
    
    # Get session summary
    summary = api.get_session_summary()
    
    print(f"Session Summary:")
    print(f"   Session ID: {summary['session_id']}")
    print(f"   Processed feedback: {summary['processed_feedback_count']}")
    print(f"   Generated reports: {summary['generated_reports_count']}")
    print(f"   Sentiment distribution: {summary['sentiment_distribution']}")
    
    # Save session data
    filepath = api.save_session_data()
    print(f"   Session data saved to: {filepath}")
    
    print()

def example_error_handling():
    """Example: Error handling and edge cases"""
    print("⚠️  Example 5: Error Handling")
    print("-" * 50)
    
    # Initialize API
    api = SteamNoodlesAPI()
    
    # Test with empty feedback
    print("Testing with empty feedback...")
    result = api.analyze_feedback("")
    if not result['success']:
        print(f"   Handled gracefully: {result['error']}")
    else:
        print(f"   Processed empty feedback: {result['data']['sentiment']}")
    
    # Test with very long feedback
    print("Testing with very long feedback...")
    long_feedback = "Amazing food! " * 100  # Very long text
    result = api.analyze_feedback(long_feedback)
    if result['success']:
        print(f"   Processed long feedback: {result['data']['sentiment']}")
    else:
        print(f"   Error with long feedback: {result['error']}")
    
    # Test with invalid date range
    print("Testing with invalid date range...")
    result = api.generate_sentiment_report("invalid date range")
    if result['success']:
        print(f"   Handled invalid date: {result['data']['date_range']}")
    else:
        print(f"   Error handled: {result['error']}")
    
    print()

def example_advanced_usage():
    """Example: Advanced API usage patterns"""
    print("🚀 Example 6: Advanced Usage")
    print("-" * 50)
    
    # Initialize API with custom configuration
    api = SteamNoodlesAPI()
    
    # Simulate real-time feedback processing
    print("Simulating real-time feedback stream...")
    
    feedback_stream = [
        "Just ordered - excited to try!",
        "Food arrived quickly, smells great!",
        "Finished eating - absolutely delicious!",
        "Paid and leaving - will definitely return!"
    ]
    
    for i, feedback in enumerate(feedback_stream, 1):
        print(f"   Processing feedback {i}/{len(feedback_stream)}...")
        result = api.analyze_feedback(feedback)
        
        if result['success']:
            sentiment = result['data']['sentiment']
            confidence = result['data']['confidence']
            print(f"   └─ {sentiment} ({confidence:.1%})")
        else:
            print(f"   └─ Error: {result['error']}")
    
    # Generate comprehensive report
    print("\nGenerating comprehensive session report...")
    summary = api.get_session_summary()
    
    print(f"Final Session Statistics:")
    print(f"   Total feedback processed: {summary['processed_feedback_count']}")
    
    if summary['sentiment_distribution']:
        total_feedback = sum(summary['sentiment_distribution'].values())
        print(f"   Sentiment breakdown:")
        for sentiment, count in summary['sentiment_distribution'].items():
            percentage = (count / total_feedback * 100) if total_feedback > 0 else 0
            print(f"     • {sentiment}: {count} ({percentage:.1f}%)")
    
    print()

def main():
    """Run all API examples"""
    print("🍜 SteamNoodles Feedback Agent - API Examples")
    print("=" * 60)
    print()
    
    try:
        # Run all examples
        example_single_feedback()
        example_batch_processing()
        example_sentiment_report()
        example_session_management()
        example_error_handling()
        example_advanced_usage()
        
        print("✅ All API examples completed successfully!")
        print("=" * 60)
        print("These examples demonstrate:")
        print("  • Single feedback analysis")
        print("  • Batch processing capabilities") 
        print("  • Report generation")
        print("  • Session management")
        print("  • Error handling")
        print("  • Advanced usage patterns")
        print()
        print("Check the 'outputs/' directory for generated files.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")

if __name__ == "__main__":
    main()