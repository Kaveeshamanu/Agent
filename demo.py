"""
SteamNoodles Feedback Agent - Interactive Demo Interface
Provides a menu-driven interface to test both agents
"""

import os
import sys
import time
from datetime import datetime
from typing import List

from main import SteamNoodlesAgentFramework, FeedbackResponse
from config import Config


class SteamNoodlesDemo:
    """Interactive demonstration interface for the SteamNoodles Agent Framework"""
    
    def __init__(self):
        self.framework = SteamNoodlesAgentFramework()
        self.config = Config()
        
        # Sample feedback for testing
        self.sample_feedbacks = [
            "The noodles were absolutely delicious! Great service too!",
            "Food was okay, nothing special. Service was a bit slow.",
            "Terrible experience. Food was cold and staff was rude. Never coming back!",
            "Amazing flavors and perfect texture! Will definitely recommend to friends!",
            "Average meal for the price. Nothing to write home about.",
            "Worst noodles I've ever had. Overpriced and tasteless.",
            "Fantastic restaurant! Loved the atmosphere and the food was incredible!",
            "The food was fine but it took forever to arrive. Staff seemed overwhelmed.",
            "Excellent quality ingredients and very friendly staff. Great experience!",
            "Not impressed at all. Overpriced for what you get and poor service."
        ]
    
    def print_header(self):
        """Print demo header"""
        print("\n" + "="*60)
        print("🍜 SteamNoodles Feedback Agent - Interactive Demo")
        print("="*60)
        print("Multi-Agent System for Customer Feedback Processing")
        print("Agent 1: Customer Feedback Response Generator")
        print("Agent 2: Sentiment Visualization Creator")
        print("="*60)
    
    def print_menu(self):
        """Print main menu options"""
        print("\n📋 DEMO OPTIONS:")
        print("1. 🔍 Test Single Feedback Analysis")
        print("2. 📊 Create Sentiment Visualization")
        print("3. 🎯 Complete Demo (Both Agents)")
        print("4. 📦 Batch Feedback Processing")
        print("5. 📈 Sentiment Summary Report")
        print("6. ⚡ Performance Benchmark")
        print("7. 🎲 Random Feedback Generator")
        print("8. 📋 View Sample Data")
        print("0. 🚪 Exit")
        print("-"*40)
    
    def get_user_input(self, prompt: str) -> str:
        """Get user input with error handling"""
        try:
            return input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting demo...")
            sys.exit(0)
    
    def demo_single_feedback(self):
        """Demo single feedback processing"""
        print("\n🔍 SINGLE FEEDBACK ANALYSIS DEMO")
        print("-"*40)
        
        print("Choose feedback source:")
        print("1. Enter your own feedback")
        print("2. Use sample feedback")
        
        choice = self.get_user_input("Your choice (1-2): ")
        
        if choice == "1":
            feedback = self.get_user_input("Enter customer feedback: ")
            if not feedback:
                print("❌ No feedback entered!")
                return
        else:
            import random
            feedback = random.choice(self.sample_feedbacks)
            print(f"Using sample feedback: {feedback}")
        
        print("\n⏳ Processing feedback...")
        start_time = time.time()
        
        result = self.framework.process_single_feedback(feedback)
        
        processing_time = time.time() - start_time
        
        # Display results
        print("\n✅ ANALYSIS RESULTS:")
        print("-"*40)
        print(f"📝 Original Feedback: {result.original_feedback}")
        print(f"😊 Sentiment: {result.sentiment_result.sentiment}")
        print(f"📊 Confidence: {result.sentiment_result.confidence:.1%}")
        print(f"🔧 Analysis Method: {result.sentiment_result.method}")
        print(f"⏱️  Processing Time: {processing_time:.2f}s")
        print(f"📅 Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n🤖 GENERATED RESPONSE:")
        print("-"*40)
        print(result.generated_response)
        
        # Save result to file
        self.save_result_to_file(result)
    
    def demo_sentiment_visualization(self):
        """Demo sentiment visualization creation"""
        print("\n📊 SENTIMENT VISUALIZATION DEMO")
        print("-"*40)
        
        print("Enter date range (examples):")
        print("• 'today' - Today's data")
        print("• 'last 7 days' - Past week")
        print("• 'last 30 days' - Past month")
        print("• 'yesterday' - Yesterday's data")
        print("• '2025-08-01 to 2025-08-07' - Specific range")
        
        date_input = self.get_user_input("Date range: ") or "last 7 days"
        
        print("\nChoose chart type:")
        print("1. Bar chart (default)")
        print("2. Line chart")
        
        chart_choice = self.get_user_input("Chart type (1-2): ") or "1"
        chart_type = "line" if chart_choice == "2" else "bar"
        
        print(f"\n⏳ Creating {chart_type} chart for '{date_input}'...")
        
        result = self.framework.create_sentiment_visualization(date_input, chart_type)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print("\n✅ VISUALIZATION CREATED:")
        print("-"*40)
        print(f"📁 File saved: {result['filename']}")
        print(f"📂 Full path: {result['filepath']}")
        print(f"📊 Total reviews: {result['total_reviews']}")
        print(f"📅 Date range: {result['start_date'].strftime('%Y-%m-%d')} to {result['end_date'].strftime('%Y-%m-%d')}")
        print(f"📈 Chart type: {result['chart_type'].title()}")
        
        print("\n📊 SENTIMENT BREAKDOWN:")
        print("-"*40)
        for sentiment, stats in result['sentiment_stats'].items():
            print(f"{sentiment:8}: {stats['count']:3} reviews ({stats['percentage']:5.1f}%)")
    
    def demo_complete_workflow(self):
        """Demo complete workflow using both agents"""
        print("\n🎯 COMPLETE WORKFLOW DEMO")
        print("-"*40)
        print("Processing sample feedback and creating visualization...")
        
        # Process multiple feedbacks
        print("\n📝 Processing customer feedback...")
        results = []
        
        for i, feedback in enumerate(self.sample_feedbacks[:5]):  # Process first 5 samples
            print(f"Processing feedback {i+1}/5...")
            result = self.framework.process_single_feedback(feedback)
            results.append(result)
            time.sleep(0.5)  # Small delay for demo effect
        
        # Display processing results
        print("\n✅ FEEDBACK PROCESSING RESULTS:")
        print("-"*50)
        sentiment_summary = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
        
        for i, result in enumerate(results):
            sentiment_summary[result.sentiment_result.sentiment] += 1
            print(f"\n{i+1}. {result.original_feedback[:50]}{'...' if len(result.original_feedback) > 50 else ''}")
            print(f"   → {result.sentiment_result.sentiment} ({result.sentiment_result.confidence:.1%})")
            print(f"   → {result.generated_response[:80]}{'...' if len(result.generated_response) > 80 else ''}")
        
        print(f"\n📊 PROCESSING SUMMARY:")
        print("-"*30)
        total = len(results)
        for sentiment, count in sentiment_summary.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{sentiment:8}: {count} ({percentage:.1f}%)")
        
        # Create visualization
        print("\n📈 Creating sentiment visualization...")
        viz_result = self.framework.create_sentiment_visualization("last 7 days", "bar")
        
        if 'error' not in viz_result:
            print(f"✅ Visualization saved: {viz_result['filename']}")
            print(f"📊 Total reviews analyzed: {viz_result['total_reviews']}")
        else:
            print(f"❌ Visualization error: {viz_result['error']}")
    
    def demo_batch_processing(self):
        """Demo batch feedback processing"""
        print("\n📦 BATCH PROCESSING DEMO")
        print("-"*40)
        
        print("Choose batch size:")
        print("1. Small batch (5 items)")
        print("2. Medium batch (10 items)")
        print("3. All sample data")
        
        choice = self.get_user_input("Your choice (1-3): ") or "1"
        
        if choice == "1":
            feedbacks = self.sample_feedbacks[:5]
        elif choice == "2":
            feedbacks = self.sample_feedbacks
        else:
            # Duplicate sample data for larger batch
            feedbacks = self.sample_feedbacks * 3
        
        print(f"\n⏳ Processing {len(feedbacks)} feedback items...")
        start_time = time.time()
        
        results = self.framework.process_batch_feedback(feedbacks)
        
        processing_time = time.time() - start_time
        
        # Analyze results
        sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
        confidence_sum = 0
        method_counts = {}
        
        for result in results:
            sentiment_counts[result.sentiment_result.sentiment] += 1
            confidence_sum += result.sentiment_result.confidence
            method = result.sentiment_result.method
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print("\n✅ BATCH PROCESSING RESULTS:")
        print("-"*40)
        print(f"📊 Total processed: {len(results)} items")
        print(f"⏱️  Total time: {processing_time:.2f}s")
        print(f"🚀 Average time per item: {processing_time/len(results):.3f}s")
        print(f"📈 Average confidence: {confidence_sum/len(results):.1%}")
        
        print(f"\n📊 SENTIMENT DISTRIBUTION:")
        print("-"*30)
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(results) * 100)
            print(f"{sentiment:8}: {count:3} ({percentage:5.1f}%)")
        
        print(f"\n🔧 ANALYSIS METHODS USED:")
        print("-"*30)
        for method, count in method_counts.items():
            percentage = (count / len(results) * 100)
            print(f"{method:12}: {count:3} ({percentage:5.1f}%)")
    
    def demo_sentiment_summary(self):
        """Demo sentiment summary report"""
        print("\n📈 SENTIMENT SUMMARY REPORT")
        print("-"*40)
        
        date_input = self.get_user_input("Enter date range (or press Enter for 'last 7 days'): ") or "last 7 days"
        
        print(f"\n⏳ Generating summary for '{date_input}'...")
        
        summary = self.framework.get_sentiment_summary(date_input)
        
        if 'error' in summary:
            print(f"❌ Error: {summary['error']}")
            return
        
        print("\n✅ SENTIMENT SUMMARY REPORT:")
        print("="*50)
        print(f"📅 Date Range: {summary['date_range']}")
        print(f"📊 Total Reviews: {summary['total_reviews']}")
        
        print(f"\n📊 SENTIMENT BREAKDOWN:")
        print("-"*30)
        for sentiment, stats in summary['sentiment_breakdown'].items():
            bar_length = int(stats['percentage'] / 5)  # Scale for display
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"{sentiment:8}: {stats['count']:3} ({stats['percentage']:5.1f}%) {bar}")
        
        # Additional insights
        pos_pct = summary['sentiment_breakdown']['POSITIVE']['percentage']
        neg_pct = summary['sentiment_breakdown']['NEGATIVE']['percentage']
        
        print(f"\n💡 INSIGHTS:")
        print("-"*20)
        if pos_pct > 60:
            print("🟢 Excellent customer satisfaction!")
        elif pos_pct > 40:
            print("🟡 Good customer satisfaction with room for improvement")
        else:
            print("🔴 Customer satisfaction needs attention")
        
        if neg_pct > 30:
            print("⚠️  High negative feedback - investigate issues")
        elif neg_pct < 10:
            print("✅ Low negative feedback - good service quality")
    
    def demo_performance_benchmark(self):
        """Demo performance benchmarking"""
        print("\n⚡ PERFORMANCE BENCHMARK")
        print("-"*40)
        
        print("Running performance tests...")
        
        # Test single item processing speed
        single_times = []
        test_feedback = "The food was good but service could be better."
        
        for i in range(5):
            start = time.time()
            self.framework.process_single_feedback(test_feedback)
            single_times.append(time.time() - start)
        
        avg_single = sum(single_times) / len(single_times)
        
        # Test batch processing speed
        batch_start = time.time()
        self.framework.process_batch_feedback(self.sample_feedbacks[:10])
        batch_time = time.time() - batch_start
        avg_batch = batch_time / 10
        
        # Test visualization creation speed
        viz_start = time.time()
        self.framework.create_sentiment_visualization("last 7 days")
        viz_time = time.time() - viz_start
        
        print("\n✅ PERFORMANCE RESULTS:")
        print("-"*40)
        print(f"🔍 Single feedback processing:")
        print(f"   Average time: {avg_single:.3f}s")
        print(f"   Throughput: {1/avg_single:.1f} items/second")
        
        print(f"\n📦 Batch processing (10 items):")
        print(f"   Total time: {batch_time:.3f}s")
        print(f"   Average per item: {avg_batch:.3f}s")
        print(f"   Throughput: {10/batch_time:.1f} items/second")
        
        print(f"\n📊 Visualization creation:")
        print(f"   Time: {viz_time:.3f}s")
        
        print(f"\n💻 SYSTEM INFO:")
        print(f"   Python version: {sys.version.split()[0]}")
        print(f"   LangChain available: {'✅' if hasattr(self.framework.feedback_agent, 'llm') and self.framework.feedback_agent.llm else '❌'}")
        print(f"   Transformers available: {'✅' if self.framework.feedback_agent.sentiment_pipeline else '❌'}")
    
    def demo_random_generator(self):
        """Demo random feedback generation and processing"""
        print("\n🎲 RANDOM FEEDBACK GENERATOR")
        print("-"*40)
        
        # Generate random feedback
        import random
        
        positive_templates = [
            "The {food} was {positive_adj}! {positive_comment}",
            "{positive_adj} {food} and {positive_service}",
            "Really enjoyed the {food}. {positive_overall}"
        ]
        
        negative_templates = [
            "The {food} was {negative_adj}. {negative_comment}",
            "{negative_adj} experience with {food}. {negative_service}",
            "Not happy with the {food}. {negative_overall}"
        ]
        
        neutral_templates = [
            "The {food} was okay. {neutral_comment}",
            "Average {food}. {neutral_service}",
            "Nothing special about the {food}. {neutral_overall}"
        ]
        
        words = {
            'food': ['noodles', 'ramen', 'soup', 'broth', 'meal', 'dish'],
            'positive_adj': ['amazing', 'delicious', 'fantastic', 'excellent', 'wonderful', 'perfect'],
            'negative_adj': ['terrible', 'awful', 'disappointing', 'horrible', 'bland', 'overpriced'],
            'positive_comment': ['Great service too!', 'Will definitely come back!', 'Highly recommend!'],
            'negative_comment': ['Never coming back!', 'Waste of money!', 'Very disappointed!'],
            'neutral_comment': ['Nothing to complain about.', 'It was fine.', 'Average for the price.'],
            'positive_service': ['excellent service', 'friendly staff', 'quick delivery'],
            'negative_service': ['poor service', 'rude staff', 'long wait time'],
            'neutral_service': ['decent service', 'average staff', 'reasonable wait time'],
            'positive_overall': ['Great experience overall!', 'Perfect meal!', 'Exceeded expectations!'],
            'negative_overall': ['Poor experience overall.', 'Would not recommend.', 'Below expectations.'],
            'neutral_overall': ['It was okay.', 'Nothing special.', 'Average experience.']
        }
        
        def generate_feedback(sentiment_type):
            if sentiment_type == 'positive':
                template = random.choice(positive_templates)
            elif sentiment_type == 'negative':
                template = random.choice(negative_templates)
            else:
                template = random.choice(neutral_templates)
            
            # Fill in template with random words
            feedback = template
            for key, word_list in words.items():
                if '{' + key + '}' in feedback:
                    feedback = feedback.replace('{' + key + '}', random.choice(word_list))
            
            return feedback
        
        num_feedback = int(self.get_user_input("Number of random feedback to generate (1-20): ") or "5")
        num_feedback = max(1, min(20, num_feedback))
        
        print(f"\n🎲 Generating {num_feedback} random feedback items...")
        
        # Generate mix of sentiments
        sentiment_types = ['positive'] * 3 + ['negative'] * 2 + ['neutral'] * 2
        random.shuffle(sentiment_types)
        
        generated_feedbacks = []
        for i in range(num_feedback):
            sentiment_type = sentiment_types[i % len(sentiment_types)]
            feedback = generate_feedback(sentiment_type)
            generated_feedbacks.append((feedback, sentiment_type))
        
        # Process generated feedback
        print(f"\n⏳ Processing generated feedback...")
        results = []
        
        for feedback, expected_sentiment in generated_feedbacks:
            result = self.framework.process_single_feedback(feedback)
            results.append((result, expected_sentiment))
        
        # Display results
        print(f"\n✅ RANDOM FEEDBACK RESULTS:")
        print("-"*50)
        
        correct_predictions = 0
        for i, (result, expected) in enumerate(results):
            predicted = result.sentiment_result.sentiment
            is_correct = predicted.lower() == expected.upper()
            if is_correct:
                correct_predictions += 1
            
            status = "✅" if is_correct else "❌"
            print(f"\n{i+1}. {result.original_feedback}")
            print(f"   Expected: {expected.upper()}, Predicted: {predicted} {status}")
            print(f"   Confidence: {result.sentiment_result.confidence:.1%}")
        
        accuracy = (correct_predictions / len(results) * 100) if results else 0
        print(f"\n📊 ACCURACY: {correct_predictions}/{len(results)} ({accuracy:.1f}%)")
    
    def demo_view_sample_data(self):
        """Display sample data used by the system"""
        print("\n📋 SAMPLE DATA VIEWER")
        print("-"*40)
        
        print("Sample feedback data used for demonstrations:")
        print()
        
        for i, feedback in enumerate(self.sample_feedbacks, 1):
            # Quick sentiment analysis for display
            result = self.framework.process_single_feedback(feedback)
            sentiment = result.sentiment_result.sentiment
            confidence = result.sentiment_result.confidence
            
            # Color coding for terminal
            if sentiment == 'POSITIVE':
                color = '🟢'
            elif sentiment == 'NEGATIVE':
                color = '🔴'
            else:
                color = '🟡'
            
            print(f"{i:2d}. {color} {feedback}")
            print(f"     → {sentiment} ({confidence:.1%})")
            print()
    
    def save_result_to_file(self, result: FeedbackResponse):
        """Save processing result to file"""
        try:
            os.makedirs(self.config.OUTPUTS_DIR, exist_ok=True)
            filename = f"feedback_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = os.path.join(self.config.OUTPUTS_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("SteamNoodles Feedback Analysis Result\n")
                f.write("="*50 + "\n\n")
                f.write(f"Timestamp: {result.timestamp}\n")
                f.write(f"Original Feedback: {result.original_feedback}\n")
                f.write(f"Sentiment: {result.sentiment_result.sentiment}\n")
                f.write(f"Confidence: {result.sentiment_result.confidence:.1%}\n")
                f.write(f"Analysis Method: {result.sentiment_result.method}\n\n")
                f.write("Generated Response:\n")
                f.write("-" * 20 + "\n")
                f.write(result.generated_response)
            
            print(f"💾 Result saved to: {filename}")
        except Exception as e:
            print(f"⚠️  Could not save result: {e}")
    
    def run(self):
        """Main demo loop"""
        self.print_header()
        
        while True:
            self.print_menu()
            choice = self.get_user_input("Select option (0-8): ")
            
            try:
                if choice == "1":
                    self.demo_single_feedback()
                elif choice == "2":
                    self.demo_sentiment_visualization()
                elif choice == "3":
                    self.demo_complete_workflow()
                elif choice == "4":
                    self.demo_batch_processing()
                elif choice == "5":
                    self.demo_sentiment_summary()
                elif choice == "6":
                    self.demo_performance_benchmark()
                elif choice == "7":
                    self.demo_random_generator()
                elif choice == "8":
                    self.demo_view_sample_data()
                elif choice == "0":
                    print("\n👋 Thank you for trying the SteamNoodles Agent Framework!")
                    print("🍜 Visit us at SteamNoodles for the best noodle experience!")
                    break
                else:
                    print("❌ Invalid option. Please choose 0-8.")
                
                if choice != "0":
                    input("\n⏸️  Press Enter to continue...")
                    
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                print("Please try again or contact support.")
                input("\n⏸️  Press Enter to continue...")


if __name__ == "__main__":
    try:
        demo = SteamNoodlesDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("Please check your setup and try again.")