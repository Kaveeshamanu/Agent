"""
SteamNoodles Feedback Agent - Core Implementation
Multi-agent system for customer feedback processing and sentiment visualization
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dateutil import parser as date_parser
import re

# LLM and ML imports with fallbacks
try:
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available, using fallback sentiment analysis")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available, using rule-based sentiment analysis")

from config import Config


@dataclass
class SentimentResult:
    """Represents sentiment analysis result"""
    sentiment: str
    confidence: float
    method: str


@dataclass
class FeedbackResponse:
    """Represents generated response to feedback"""
    original_feedback: str
    sentiment_result: SentimentResult
    generated_response: str
    timestamp: datetime


class CustomerFeedbackAgent:
    """Agent 1: Processes customer feedback and generates responses"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM if available
        self.llm = None
        if LANGCHAIN_AVAILABLE and config.OPENAI_API_KEY:
            try:
                self.llm = OpenAI(
                    openai_api_key=config.OPENAI_API_KEY,
                    temperature=0.7,
                    max_tokens=150
                )
                self.logger.info("OpenAI LLM initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI LLM: {e}")
        
        # Initialize HuggingFace sentiment pipeline if available
        self.sentiment_pipeline = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
                self.logger.info("HuggingFace sentiment pipeline initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize HuggingFace pipeline: {e}")
        
        # Sentiment keywords for rule-based fallback
        self.positive_keywords = [
            'excellent', 'amazing', 'great', 'wonderful', 'fantastic', 'delicious',
            'perfect', 'love', 'best', 'awesome', 'incredible', 'outstanding',
            'superb', 'brilliant', 'magnificent', 'terrific', 'good', 'nice',
            'tasty', 'fresh', 'clean', 'friendly', 'fast', 'quick'
        ]
        self.positive_keywords.extend(['😊', '👍', '🔥', '💯', '❤️', '😍', '🤤', '🌟', '✨', '👌'])
        
        self.negative_keywords = [
            'terrible', 'awful', 'horrible', 'disgusting', 'worst', 'bad',
            'poor', 'slow', 'rude', 'dirty', 'cold', 'stale', 'expensive',
            'disappointing', 'hate', 'never', 'overpriced', 'bland', 'soggy',
            'burnt', 'undercooked', 'salty', 'spicy', 'bitter'
        ]

        
        self.negative_keywords.extend(['😠', '😡', '🤮', '👎', '💩', '😤', '😞', '🤢'])
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment using multi-layer approach"""
        
        # Method 1: Try LLM-based analysis
        if self.llm:
            try:
                result = self._analyze_with_llm(text)
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"LLM sentiment analysis failed: {e}")
        
        # Method 2: Try HuggingFace transformers
        if self.sentiment_pipeline:
            try:
                result = self._analyze_with_transformers(text)
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"Transformers sentiment analysis failed: {e}")
        
        # Method 3: Rule-based fallback
        return self._analyze_with_rules(text)
    
    def _analyze_with_llm(self, text: str) -> Optional[SentimentResult]:
        """Analyze sentiment using LLM"""
        try:
            prompt = PromptTemplate(
                input_variables=["feedback"],
                template="""
                Analyze the sentiment of this restaurant feedback and provide a confidence score.
                
                Feedback: {feedback}
                
                Respond with only: SENTIMENT|CONFIDENCE
                Where SENTIMENT is POSITIVE, NEGATIVE, or NEUTRAL
                And CONFIDENCE is a number between 0 and 1
                
                Example: POSITIVE|0.85
                """
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.run(feedback=text)
            
            # Parse response
            parts = response.strip().split('|')
            if len(parts) == 2:
                sentiment = parts[0].strip().upper()
                confidence = float(parts[1].strip())
                
                if sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                    return SentimentResult(sentiment, confidence, "LLM")
        except Exception as e:
            self.logger.error(f"LLM analysis error: {e}")
        
        return None
    
    def _analyze_with_transformers(self, text: str) -> Optional[SentimentResult]:
        """Analyze sentiment using HuggingFace transformers"""
        try:
            result = self.sentiment_pipeline(text)[0]
            
            # Map labels to our format
            label_mapping = {
                'LABEL_0': 'NEGATIVE',
                'LABEL_1': 'NEUTRAL', 
                'LABEL_2': 'POSITIVE',
                'NEGATIVE': 'NEGATIVE',
                'NEUTRAL': 'NEUTRAL',
                'POSITIVE': 'POSITIVE'
            }
            
            sentiment = label_mapping.get(result['label'], 'NEUTRAL')
            confidence = result['score']
            
            return SentimentResult(sentiment, confidence, "Transformers")
        except Exception as e:
            self.logger.error(f"Transformers analysis error: {e}")
        
        return None
    
    def _analyze_with_rules(self, text: str) -> SentimentResult:
        """Rule-based sentiment analysis fallback"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        # Calculate sentiment based on keyword counts
        if positive_count > negative_count:
            sentiment = "POSITIVE"
            confidence = min(0.6 + (positive_count - negative_count) * 0.1, 0.9)
        elif negative_count > positive_count:
            sentiment = "NEGATIVE"
            confidence = min(0.6 + (negative_count - positive_count) * 0.1, 0.9)
        else:
            sentiment = "NEUTRAL"
            confidence = 0.5
        
        return SentimentResult(sentiment, confidence, "Rule-based")
    
    def generate_response(self, feedback: str, sentiment_result: SentimentResult) -> str:
        """Generate appropriate response based on sentiment"""
        
        # Try LLM-based response generation first
        if self.llm:
            try:
                llm_response = self._generate_with_llm(feedback, sentiment_result)
                if llm_response:
                    return llm_response
            except Exception as e:
                self.logger.warning(f"LLM response generation failed: {e}")
        
        # Fallback to template-based responses
        return self._generate_with_template(sentiment_result)
    
    def _generate_with_llm(self, feedback: str, sentiment_result: SentimentResult) -> Optional[str]:
        """Generate response using LLM"""
        try:
            prompt = PromptTemplate(
                input_variables=["feedback", "sentiment"],
                template="""
                Generate a professional restaurant response to this customer feedback.
                
                Customer Feedback: {feedback}
                Sentiment: {sentiment}
                
                Response should be:
                - Professional and courteous
                - Appropriate for the sentiment
                - Brief (1-2 sentences)
                - Signed as "SteamNoodles Team"
                
                Response:
                """
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.run(feedback=feedback, sentiment=sentiment_result.sentiment)
            
            return response.strip()
        except Exception as e:
            self.logger.error(f"LLM response generation error: {e}")
        
        return None
    
    def _generate_with_template(self, sentiment_result: SentimentResult) -> str:
        """Generate response using templates"""
        templates = {
            "POSITIVE": [
                "Thank you so much for your wonderful feedback! We're thrilled you enjoyed your experience at SteamNoodles. We look forward to serving you again soon!",
                "We're delighted to hear you had a great experience! Your positive feedback means the world to us. See you again soon at SteamNoodles!",
                "Thank you for taking the time to share your positive experience! We're so glad you enjoyed our food and service."
            ],
            "NEGATIVE": [
                "We sincerely apologize for not meeting your expectations. Your feedback is valuable to us, and we're working to improve. Please give us another chance to serve you better.",
                "Thank you for bringing this to our attention. We take all feedback seriously and are committed to improving our service. We'd love the opportunity to make it right.",
                "We're sorry to hear about your disappointing experience. Please contact our manager directly so we can address your concerns and improve."
            ],
            "NEUTRAL": [
                "Thank you for your feedback. We appreciate you taking the time to share your experience with us. We're always working to improve our service.",
                "We appreciate your honest feedback. If there's anything specific we can do to improve your next visit, please let us know.",
                "Thank you for visiting SteamNoodles. Your feedback helps us serve you better. We hope to see you again soon."
            ]
        }
        
        import random
        responses = templates.get(sentiment_result.sentiment, templates["NEUTRAL"])
        return random.choice(responses) + "\n\n- SteamNoodles Team"
    
    def process_feedback(self, feedback: str) -> FeedbackResponse:
        """Process customer feedback end-to-end"""
        sentiment_result = self.analyze_sentiment(feedback)
        response = self.generate_response(feedback, sentiment_result)
        
        return FeedbackResponse(
            original_feedback=feedback,
            sentiment_result=sentiment_result,
            generated_response=response,
            timestamp=datetime.now()
        )


class SentimentVisualizationAgent:
    """Agent 2: Creates sentiment visualizations from data"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Sample data for demonstration
        self.sample_data = self._generate_sample_data()
    
    def _generate_sample_data(self) -> List[Dict]:
        """Generate sample feedback data for demonstration"""
        import random
        
        sample_feedbacks = [
            "The noodles were absolutely delicious! Great service too!",
            "Food was okay, nothing special. Service was slow.",
            "Terrible experience. Food was cold and staff was rude.",
            "Amazing flavors! Will definitely come back!",
            "Average meal, decent price.",
            "Worst noodles I've ever had. Never coming back.",
            "Fantastic restaurant! Loved everything about it!",
            "Food was fine but took forever to arrive.",
            "Excellent quality ingredients and friendly staff!",
            "Not impressed. Overpriced for what you get."
        ]
        
        sentiments = ["POSITIVE", "NEUTRAL", "NEGATIVE", "POSITIVE", "NEUTRAL", 
                     "NEGATIVE", "POSITIVE", "NEUTRAL", "POSITIVE", "NEGATIVE"]
        
        data = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(50):  # Generate 50 sample reviews
            date = base_date + timedelta(days=random.randint(0, 30))
            feedback_idx = i % len(sample_feedbacks)
            
            data.append({
                'date': date,
                'feedback': sample_feedbacks[feedback_idx],
                'sentiment': sentiments[feedback_idx],
                'confidence': random.uniform(0.6, 0.95)
            })
        
        return data
    
    def parse_date_range(self, date_input: str) -> Tuple[datetime, datetime]:
        """Parse natural language date input"""
        date_input = date_input.lower().strip()
        now = datetime.now()
        
        # Handle relative dates
        if "today" in date_input:
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now
        elif "yesterday" in date_input:
            start_date = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=1)
        elif "last week" in date_input or "past week" in date_input:
            start_date = now - timedelta(days=7)
            end_date = now
        elif "last month" in date_input or "past month" in date_input:
            start_date = now - timedelta(days=30)
            end_date = now
        elif "last" in date_input and "days" in date_input:
            # Extract number of days
            match = re.search(r'last (\d+) days?', date_input)
            if match:
                days = int(match.group(1))
                start_date = now - timedelta(days=days)
                end_date = now
            else:
                start_date = now - timedelta(days=7)  # Default to 7 days
                end_date = now
        else:
            # Try to parse as specific dates
            try:
                if "to" in date_input or "-" in date_input:
                    parts = date_input.replace(" to ", "|").replace("-", "|").split("|")
                    start_date = date_parser.parse(parts[0].strip())
                    end_date = date_parser.parse(parts[1].strip())
                else:
                    # Single date - show that day
                    start_date = date_parser.parse(date_input)
                    end_date = start_date + timedelta(days=1)
            except:
                # Default to last 7 days
                start_date = now - timedelta(days=7)
                end_date = now
        
        return start_date, end_date
    
    def filter_data_by_date(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Filter sample data by date range"""
        return [
            item for item in self.sample_data
            if start_date <= item['date'] <= end_date
        ]
    
    def create_sentiment_visualization(self, date_input: str, chart_type: str = "bar") -> Dict[str, Any]:
        """Create sentiment visualization for given date range"""
        
        # Parse date range
        start_date, end_date = self.parse_date_range(date_input)
        
        # Filter data
        filtered_data = self.filter_data_by_date(start_date, end_date)
        
        if not filtered_data:
            return {
                'error': 'No data found for the specified date range',
                'start_date': start_date,
                'end_date': end_date
            }
        
        # Create DataFrame for analysis
        df = pd.DataFrame(filtered_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate sentiment counts
        sentiment_counts = df['sentiment'].value_counts()
        total_reviews = len(filtered_data)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        if chart_type.lower() == "line":
            # Line chart showing sentiment over time
            df['date_only'] = df['date'].dt.date
            daily_sentiment = df.groupby(['date_only', 'sentiment']).size().unstack(fill_value=0)
            
            for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                if sentiment in daily_sentiment.columns:
                    plt.plot(daily_sentiment.index, daily_sentiment[sentiment], 
                           marker='o', label=sentiment, linewidth=2, markersize=6)
            
            plt.title(f'Sentiment Trends: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Number of Reviews', fontsize=12)
            plt.xticks(rotation=45)
            
        else:
            # Bar chart showing sentiment distribution
            colors = {'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c', 'NEUTRAL': '#f39c12'}
            bars = plt.bar(sentiment_counts.index, sentiment_counts.values, 
                          color=[colors.get(s, '#95a5a6') for s in sentiment_counts.index])
            
            plt.title(f'Sentiment Distribution: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Sentiment', fontsize=12)
            plt.ylabel('Number of Reviews', fontsize=12)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        filename = f"sentiment_plot_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
        filepath = os.path.join(self.config.OUTPUTS_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate statistics
        stats = {}
        for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            count = sentiment_counts.get(sentiment, 0)
            percentage = (count / total_reviews * 100) if total_reviews > 0 else 0
            stats[sentiment] = {'count': count, 'percentage': percentage}
        
        return {
            'filename': filename,
            'filepath': filepath,
            'total_reviews': total_reviews,
            'sentiment_stats': stats,
            'start_date': start_date,
            'end_date': end_date,
            'chart_type': chart_type
        }


class SteamNoodlesAgentFramework:
    """Main framework coordinating both agents"""
    
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        
        self.config = config
        self.setup_logging()
        
        # Initialize agents
        self.feedback_agent = CustomerFeedbackAgent(config)
        self.visualization_agent = SentimentVisualizationAgent(config)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("SteamNoodles Agent Framework initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.config.LOGS_DIR, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.LOGS_DIR, 'steamnoodles.log')),
                logging.StreamHandler()
            ]
        )
    
    def process_single_feedback(self, feedback: str) -> FeedbackResponse:
        """Process a single piece of customer feedback"""
        return self.feedback_agent.process_feedback(feedback)
    
    def process_batch_feedback(self, feedbacks: List[str]) -> List[FeedbackResponse]:
        """Process multiple pieces of feedback"""
        results = []
        for feedback in feedbacks:
            result = self.feedback_agent.process_feedback(feedback)
            results.append(result)
        
        self.logger.info(f"Processed {len(feedbacks)} feedback items")
        return results
    
    def create_sentiment_visualization(self, date_input: str, chart_type: str = "bar") -> Dict[str, Any]:
        """Create sentiment visualization"""
        return self.visualization_agent.create_sentiment_visualization(date_input, chart_type)
    
    def get_sentiment_summary(self, date_input: str) -> Dict[str, Any]:
        """Get sentiment summary for date range"""
        start_date, end_date = self.visualization_agent.parse_date_range(date_input)
        filtered_data = self.visualization_agent.filter_data_by_date(start_date, end_date)
        
        if not filtered_data:
            return {'error': 'No data found for the specified date range'}
        
        # Calculate summary statistics
        df = pd.DataFrame(filtered_data)
        sentiment_counts = df['sentiment'].value_counts()
        total_reviews = len(filtered_data)
        
        summary = {
            'total_reviews': total_reviews,
            'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'sentiment_breakdown': {}
        }
        
        for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            count = sentiment_counts.get(sentiment, 0)
            percentage = (count / total_reviews * 100) if total_reviews > 0 else 0
            summary['sentiment_breakdown'][sentiment] = {
                'count': count,
                'percentage': round(percentage, 1)
            }
        
        return summary


if __name__ == "__main__":
    # Quick test
    framework = SteamNoodlesAgentFramework()
    
    # Test feedback processing
    test_feedback = "The noodles were amazing! Great service and atmosphere."
    result = framework.process_single_feedback(test_feedback)
    
    print(f"Feedback: {result.original_feedback}")
    print(f"Sentiment: {result.sentiment_result.sentiment} ({result.sentiment_result.confidence:.2f})")
    print(f"Response: {result.generated_response}")
    
    # Test visualization
    viz_result = framework.create_sentiment_visualization("last 7 days")
    print(f"\nVisualization created: {viz_result.get('filename', 'Error creating visualization')}")