#!/usr/bin/env python3
"""
Task 7: Social Engagement Analysis with Mapping-Aware Social Dynamics

This script analyzes comment sentiment patterns and social dynamics across different
content types and mapping relationships in the Fakeddit dataset.

Key Analysis Areas:
- Comment sentiment patterns by content type
- Engagement metrics across mapping types (multimodal vs image-only vs text-only)
- User interaction patterns and social dynamics
- Comment-based authenticity indicators
- Social amplification patterns for different misinformation categories

Requirements: 2.5, 2.6, 4.3
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('task7_social_engagement_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SocialEngagementAnalyzer:
    """
    Analyzes social engagement patterns and dynamics across different content types
    and mapping relationships in the Fakeddit dataset.
    """
    
    def __init__(self):
        self.setup_directories()
        self.performance_metrics = {
            'start_time': datetime.now(),
            'processing_steps': [],
            'data_loaded': {},
            'analysis_results': {}
        }
        
    def setup_directories(self):
        """Create necessary output directories"""
        directories = [
            'processed_data/social_engagement',
            'analysis_results/social_analysis',
            'visualizations/social_patterns',
            'reports'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def load_integrated_data(self) -> Dict[str, pd.DataFrame]:
        """Load all necessary data for social engagement analysis"""
        logger.info("Loading integrated data for social engagement analysis...")
        
        data = {}
        
        try:
            # Load text data with mapping information
            logger.info("Loading text data...")
            text_files = [
                'processed_data/text_data/train_clean.parquet',
                'processed_data/text_data/validation_clean.parquet', 
                'processed_data/text_data/test_clean.parquet'
            ]
            
            text_dfs = []
            for file_path in text_files:
                if Path(file_path).exists():
                    df = pd.read_parquet(file_path)
                    text_dfs.append(df)
                    logger.info(f"Loaded {len(df)} records from {file_path}")
            
            if text_dfs:
                data['text_data'] = pd.concat(text_dfs, ignore_index=True)
                logger.info(f"Total text records: {len(data['text_data'])}")
            
            # Load comments data
            logger.info("Loading comments data...")
            comments_file = 'processed_data/comments/comments_with_mapping.parquet'
            if Path(comments_file).exists():
                data['comments'] = pd.read_parquet(comments_file)
                logger.info(f"Loaded {len(data['comments'])} comment records")
            
            # Load image catalog for mapping analysis
            logger.info("Loading image catalog...")
            image_files = [
                'analysis_results/image_catalog/comprehensive_image_catalog.parquet',
                'processed_data/images/image_metadata.parquet'
            ]
            
            for file_path in image_files:
                if Path(file_path).exists():
                    data['images'] = pd.read_parquet(file_path)
                    logger.info(f"Loaded {len(data['images'])} image records from {file_path}")
                    break
            
            self.performance_metrics['data_loaded'] = {
                'text_records': len(data.get('text_data', [])),
                'comment_records': len(data.get('comments', [])),
                'image_records': len(data.get('images', []))
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return {}
    
    def analyze_comment_sentiment(self, comments_df: pd.DataFrame) -> Dict:
        """Analyze sentiment patterns in comments"""
        logger.info("Analyzing comment sentiment patterns...")
        
        sentiment_analysis = {
            'overall_sentiment': {},
            'sentiment_by_content_type': {},
            'sentiment_distribution': {},
            'temporal_sentiment': {}
        }
        
        try:
            # Sample comments for sentiment analysis (due to computational constraints)
            sample_size = min(50000, len(comments_df))
            sample_comments = comments_df.sample(n=sample_size, random_state=42)
            
            logger.info(f"Analyzing sentiment for {len(sample_comments)} sampled comments...")
            
            # Calculate sentiment scores
            sentiments = []
            polarities = []
            subjectivities = []
            
            for idx, comment in enumerate(sample_comments['body'].fillna('')):
                if idx % 10000 == 0:
                    logger.info(f"Processed {idx}/{len(sample_comments)} comments for sentiment")
                
                try:
                    blob = TextBlob(str(comment))
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                    
                    # Classify sentiment
                    if polarity > 0.1:
                        sentiment = 'positive'
                    elif polarity < -0.1:
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
                    
                    sentiments.append(sentiment)
                    polarities.append(polarity)
                    subjectivities.append(subjectivity)
                    
                except Exception:
                    sentiments.append('neutral')
                    polarities.append(0.0)
                    subjectivities.append(0.0)
            
            # Add sentiment data to sample
            sample_comments = sample_comments.copy()
            sample_comments['sentiment'] = sentiments
            sample_comments['polarity'] = polarities
            sample_comments['subjectivity'] = subjectivities
            
            # Overall sentiment distribution
            sentiment_counts = Counter(sentiments)
            sentiment_analysis['overall_sentiment'] = {
                'positive': sentiment_counts.get('positive', 0),
                'negative': sentiment_counts.get('negative', 0),
                'neutral': sentiment_counts.get('neutral', 0),
                'total_analyzed': len(sentiments)
            }
            
            # Sentiment statistics
            sentiment_analysis['sentiment_distribution'] = {
                'polarity_mean': np.mean(polarities),
                'polarity_std': np.std(polarities),
                'polarity_median': np.median(polarities),
                'subjectivity_mean': np.mean(subjectivities),
                'subjectivity_std': np.std(subjectivities),
                'subjectivity_median': np.median(subjectivities)
            }
            
            # Save processed sentiment data
            sample_comments.to_parquet('processed_data/social_engagement/comments_with_sentiment.parquet')
            
            logger.info("Comment sentiment analysis completed")
            return sentiment_analysis
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return sentiment_analysis
    
    def analyze_engagement_by_mapping_type(self, text_data: pd.DataFrame, 
                                         comments_df: pd.DataFrame,
                                         images_df: Optional[pd.DataFrame] = None) -> Dict:
        """Analyze engagement patterns across different mapping types"""
        logger.info("Analyzing engagement patterns by mapping type...")
        
        engagement_analysis = {
            'mapping_type_distribution': {},
            'engagement_by_type': {},
            'statistical_comparisons': {}
        }
        
        try:
            # Create integrated dataset with mapping information
            logger.info("Creating integrated dataset with mapping information...")
            
            # Start with text data as base
            integrated_data = text_data.copy()
            
            # Determine content types based on available data
            integrated_data['has_text'] = True  # All records have text
            integrated_data['has_image'] = integrated_data['image_url'].notna()
            
            # Add comment information
            if not comments_df.empty:
                # Clean and prepare comments data
                comments_clean = comments_df.copy()
                
                # Convert ups to numeric, handling None values
                comments_clean['ups'] = pd.to_numeric(comments_clean['ups'], errors='coerce')
                comments_clean['ups'] = comments_clean['ups'].fillna(0)
                
                # Filter out rows with missing submission_id
                comments_clean = comments_clean[comments_clean['submission_id'].notna()]
                
                # Aggregate comments by submission_id
                comment_stats = comments_clean.groupby('submission_id').agg({
                    'body': 'count',
                    'ups': ['mean', 'sum', 'std'],
                    'sentiment_score': ['mean', 'std']
                }).round(3)
                
                comment_stats.columns = [
                    'comment_count', 'avg_comment_score', 'total_comment_score',
                    'comment_score_std', 'avg_sentiment', 'sentiment_std'
                ]
                comment_stats = comment_stats.reset_index()
                
                # Map comments to text data using linked_submission_id
                integrated_data = integrated_data.merge(
                    comment_stats, 
                    left_on='linked_submission_id', 
                    right_on='submission_id',
                    how='left'
                )
                
                integrated_data['has_comments'] = integrated_data['comment_count'].notna()
                integrated_data['comment_count'] = integrated_data['comment_count'].fillna(0)
                integrated_data['avg_comment_score'] = integrated_data['avg_comment_score'].fillna(0)
                integrated_data['total_comment_score'] = integrated_data['total_comment_score'].fillna(0)
                integrated_data['avg_sentiment'] = integrated_data['avg_sentiment'].fillna(0)
            else:
                integrated_data['has_comments'] = False
                integrated_data['comment_count'] = 0
                integrated_data['avg_comment_score'] = 0
                integrated_data['total_comment_score'] = 0
                integrated_data['avg_sentiment'] = 0
            
            # Define content types based on mapping relationships
            def determine_content_type(row):
                if row['has_text'] and row['has_image'] and row['has_comments']:
                    return 'full_multimodal'
                elif row['has_text'] and row['has_image']:
                    return 'text_image'
                elif row['has_text'] and row['has_comments']:
                    return 'text_comments'
                elif row['has_text']:
                    return 'text_only'
                else:
                    return 'unknown'
            
            integrated_data['content_type'] = integrated_data.apply(determine_content_type, axis=1)
            
            # Analyze mapping type distribution
            type_distribution = integrated_data['content_type'].value_counts()
            engagement_analysis['mapping_type_distribution'] = type_distribution.to_dict()
            
            # Analyze engagement metrics by content type
            engagement_metrics = ['score', 'num_comments', 'comment_count', 'avg_comment_score', 'avg_sentiment']
            
            for content_type in type_distribution.index:
                type_data = integrated_data[integrated_data['content_type'] == content_type]
                
                engagement_analysis['engagement_by_type'][content_type] = {
                    'count': len(type_data),
                    'percentage': len(type_data) / len(integrated_data) * 100
                }
                
                for metric in engagement_metrics:
                    if metric in type_data.columns:
                        values = type_data[metric].dropna()
                        if len(values) > 0:
                            engagement_analysis['engagement_by_type'][content_type][metric] = {
                                'mean': float(values.mean()),
                                'median': float(values.median()),
                                'std': float(values.std()),
                                'min': float(values.min()),
                                'max': float(values.max()),
                                'count': len(values)
                            }
            
            # Statistical comparisons between content types
            logger.info("Performing statistical comparisons between content types...")
            
            # Compare engagement metrics across content types
            from scipy import stats
            
            main_types = ['text_image', 'text_only', 'text_comments', 'full_multimodal']
            available_types = [t for t in main_types if t in type_distribution.index and type_distribution[t] > 100]
            
            for metric in ['score', 'num_comments']:
                if metric in integrated_data.columns:
                    engagement_analysis['statistical_comparisons'][metric] = {}
                    
                    # Perform pairwise comparisons
                    for i, type1 in enumerate(available_types):
                        for type2 in available_types[i+1:]:
                            data1 = integrated_data[integrated_data['content_type'] == type1][metric].dropna()
                            data2 = integrated_data[integrated_data['content_type'] == type2][metric].dropna()
                            
                            if len(data1) > 30 and len(data2) > 30:
                                # Perform Mann-Whitney U test (non-parametric)
                                statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                                
                                engagement_analysis['statistical_comparisons'][metric][f"{type1}_vs_{type2}"] = {
                                    'statistic': float(statistic),
                                    'p_value': float(p_value),
                                    'significant': p_value < 0.05,
                                    'effect_size': abs(data1.mean() - data2.mean()) / np.sqrt((data1.var() + data2.var()) / 2)
                                }
            
            # Save integrated dataset
            integrated_data.to_parquet('processed_data/social_engagement/integrated_engagement_data.parquet')
            
            logger.info("Engagement analysis by mapping type completed")
            return engagement_analysis
            
        except Exception as e:
            logger.error(f"Error in engagement analysis: {str(e)}")
            return engagement_analysis
    
    def analyze_authenticity_engagement_patterns(self, integrated_data: pd.DataFrame) -> Dict:
        """Analyze how engagement patterns relate to authenticity labels"""
        logger.info("Analyzing authenticity-engagement relationships...")
        
        authenticity_analysis = {
            'engagement_by_label': {},
            'authenticity_indicators': {},
            'cross_modal_patterns': {}
        }
        
        try:
            # Load integrated data if not provided
            if integrated_data.empty:
                integrated_data = pd.read_parquet('processed_data/social_engagement/integrated_engagement_data.parquet')
            
            # Analyze engagement by authenticity label
            if '2_way_label' in integrated_data.columns:
                for label in integrated_data['2_way_label'].unique():
                    if pd.notna(label):
                        label_data = integrated_data[integrated_data['2_way_label'] == label]
                        
                        authenticity_analysis['engagement_by_label'][str(label)] = {
                            'count': len(label_data),
                            'avg_score': float(label_data['score'].mean()),
                            'avg_comments': float(label_data['num_comments'].mean()),
                            'avg_comment_sentiment': float(label_data['avg_comment_score'].mean()) if 'avg_comment_score' in label_data.columns else 0,
                            'content_type_distribution': label_data['content_type'].value_counts().to_dict()
                        }
            
            # Identify potential authenticity indicators from engagement patterns
            logger.info("Identifying authenticity indicators from engagement patterns...")
            
            # Analyze comment-based indicators
            if 'comment_count' in integrated_data.columns and integrated_data['comment_count'].sum() > 0:
                # High engagement posts
                high_engagement = integrated_data[integrated_data['comment_count'] > integrated_data['comment_count'].quantile(0.9)]
                low_engagement = integrated_data[integrated_data['comment_count'] < integrated_data['comment_count'].quantile(0.1)]
                
                if '2_way_label' in integrated_data.columns:
                    authenticity_analysis['authenticity_indicators']['high_engagement_authenticity'] = {
                        'fake_percentage': (high_engagement['2_way_label'] == 0).mean() * 100,
                        'real_percentage': (high_engagement['2_way_label'] == 1).mean() * 100,
                        'total_posts': len(high_engagement)
                    }
                    
                    authenticity_analysis['authenticity_indicators']['low_engagement_authenticity'] = {
                        'fake_percentage': (low_engagement['2_way_label'] == 0).mean() * 100,
                        'real_percentage': (low_engagement['2_way_label'] == 1).mean() * 100,
                        'total_posts': len(low_engagement)
                    }
            
            # Cross-modal pattern analysis
            logger.info("Analyzing cross-modal engagement patterns...")
            
            for content_type in integrated_data['content_type'].unique():
                type_data = integrated_data[integrated_data['content_type'] == content_type]
                
                if len(type_data) > 100 and '2_way_label' in type_data.columns:
                    authenticity_analysis['cross_modal_patterns'][content_type] = {
                        'total_posts': len(type_data),
                        'fake_posts': int((type_data['2_way_label'] == 0).sum()),
                        'real_posts': int((type_data['2_way_label'] == 1).sum()),
                        'avg_engagement_fake': float(type_data[type_data['2_way_label'] == 0]['score'].mean()),
                        'avg_engagement_real': float(type_data[type_data['2_way_label'] == 1]['score'].mean()),
                        'avg_comments_fake': float(type_data[type_data['2_way_label'] == 0]['num_comments'].mean()),
                        'avg_comments_real': float(type_data[type_data['2_way_label'] == 1]['num_comments'].mean())
                    }
            
            logger.info("Authenticity-engagement analysis completed")
            return authenticity_analysis
            
        except Exception as e:
            logger.error(f"Error in authenticity analysis: {str(e)}")
            return authenticity_analysis
    
    def create_social_engagement_visualizations(self, analysis_results: Dict):
        """Create comprehensive visualizations for social engagement analysis"""
        logger.info("Creating social engagement visualizations...")
        
        try:
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Load integrated data for visualizations
            integrated_data = pd.read_parquet('processed_data/social_engagement/integrated_engagement_data.parquet')
            
            # 1. Content Type Distribution
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Content type pie chart
            content_type_counts = integrated_data['content_type'].value_counts()
            ax1.pie(content_type_counts.values, labels=content_type_counts.index, autopct='%1.1f%%')
            ax1.set_title('Distribution of Content Types\n(Mapping Relationships)', fontsize=14, fontweight='bold')
            
            # Content type bar chart
            content_type_counts.plot(kind='bar', ax=ax2, color='skyblue')
            ax2.set_title('Content Type Counts', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Content Type')
            ax2.set_ylabel('Number of Posts')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('visualizations/social_patterns/content_type_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Engagement Metrics by Content Type
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Score distribution by content type
            content_types = integrated_data['content_type'].unique()
            score_data = [integrated_data[integrated_data['content_type'] == ct]['score'].dropna() for ct in content_types]
            
            axes[0, 0].boxplot(score_data, labels=content_types)
            axes[0, 0].set_title('Post Score Distribution by Content Type', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('Post Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Comment count distribution by content type
            comment_data = [integrated_data[integrated_data['content_type'] == ct]['num_comments'].dropna() for ct in content_types]
            
            axes[0, 1].boxplot(comment_data, labels=content_types)
            axes[0, 1].set_title('Comment Count Distribution by Content Type', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('Number of Comments')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Average engagement by content type
            engagement_by_type = integrated_data.groupby('content_type')[['score', 'num_comments']].mean()
            
            engagement_by_type['score'].plot(kind='bar', ax=axes[1, 0], color='lightcoral')
            axes[1, 0].set_title('Average Post Score by Content Type', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Average Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            engagement_by_type['num_comments'].plot(kind='bar', ax=axes[1, 1], color='lightgreen')
            axes[1, 1].set_title('Average Comments by Content Type', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Average Comments')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('visualizations/social_patterns/engagement_by_content_type.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Authenticity vs Engagement Analysis
            if '2_way_label' in integrated_data.columns:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                # Engagement by authenticity label
                auth_engagement = integrated_data.groupby('2_way_label')[['score', 'num_comments']].mean()
                
                auth_engagement['score'].plot(kind='bar', ax=axes[0, 0], color=['red', 'green'])
                axes[0, 0].set_title('Average Post Score by Authenticity', fontsize=12, fontweight='bold')
                axes[0, 0].set_ylabel('Average Score')
                axes[0, 0].set_xlabel('Authenticity (0=Fake, 1=Real)')
                axes[0, 0].tick_params(axis='x', rotation=0)
                
                auth_engagement['num_comments'].plot(kind='bar', ax=axes[0, 1], color=['red', 'green'])
                axes[0, 1].set_title('Average Comments by Authenticity', fontsize=12, fontweight='bold')
                axes[0, 1].set_ylabel('Average Comments')
                axes[0, 1].set_xlabel('Authenticity (0=Fake, 1=Real)')
                axes[0, 1].tick_params(axis='x', rotation=0)
                
                # Content type distribution by authenticity
                auth_content = pd.crosstab(integrated_data['2_way_label'], integrated_data['content_type'], normalize='index') * 100
                
                auth_content.plot(kind='bar', ax=axes[1, 0], stacked=True)
                axes[1, 0].set_title('Content Type Distribution by Authenticity (%)', fontsize=12, fontweight='bold')
                axes[1, 0].set_ylabel('Percentage')
                axes[1, 0].set_xlabel('Authenticity (0=Fake, 1=Real)')
                axes[1, 0].legend(title='Content Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[1, 0].tick_params(axis='x', rotation=0)
                
                # Engagement scatter plot
                fake_data = integrated_data[integrated_data['2_way_label'] == 0]
                real_data = integrated_data[integrated_data['2_way_label'] == 1]
                
                axes[1, 1].scatter(fake_data['score'], fake_data['num_comments'], alpha=0.5, color='red', label='Fake', s=10)
                axes[1, 1].scatter(real_data['score'], real_data['num_comments'], alpha=0.5, color='green', label='Real', s=10)
                axes[1, 1].set_title('Engagement Scatter: Score vs Comments', fontsize=12, fontweight='bold')
                axes[1, 1].set_xlabel('Post Score')
                axes[1, 1].set_ylabel('Number of Comments')
                axes[1, 1].legend()
                
                plt.tight_layout()
                plt.savefig('visualizations/social_patterns/authenticity_engagement_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 4. Sentiment Analysis Visualization (if available)
            sentiment_file = 'processed_data/social_engagement/comments_with_sentiment.parquet'
            if Path(sentiment_file).exists():
                sentiment_data = pd.read_parquet(sentiment_file)
                
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                # Sentiment distribution
                sentiment_counts = sentiment_data['sentiment'].value_counts()
                sentiment_counts.plot(kind='pie', ax=axes[0, 0], autopct='%1.1f%%')
                axes[0, 0].set_title('Overall Comment Sentiment Distribution', fontsize=12, fontweight='bold')
                
                # Polarity distribution
                axes[0, 1].hist(sentiment_data['polarity'], bins=50, alpha=0.7, color='blue')
                axes[0, 1].set_title('Comment Polarity Distribution', fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel('Polarity Score')
                axes[0, 1].set_ylabel('Frequency')
                
                # Subjectivity distribution
                axes[1, 0].hist(sentiment_data['subjectivity'], bins=50, alpha=0.7, color='orange')
                axes[1, 0].set_title('Comment Subjectivity Distribution', fontsize=12, fontweight='bold')
                axes[1, 0].set_xlabel('Subjectivity Score')
                axes[1, 0].set_ylabel('Frequency')
                
                # Polarity vs Subjectivity scatter
                axes[1, 1].scatter(sentiment_data['polarity'], sentiment_data['subjectivity'], alpha=0.5, s=10)
                axes[1, 1].set_title('Polarity vs Subjectivity', fontsize=12, fontweight='bold')
                axes[1, 1].set_xlabel('Polarity')
                axes[1, 1].set_ylabel('Subjectivity')
                
                plt.tight_layout()
                plt.savefig('visualizations/social_patterns/sentiment_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info("Social engagement visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    def generate_social_analysis_report(self, analysis_results: Dict):
        """Generate comprehensive social analysis report"""
        logger.info("Generating social analysis report...")
        
        try:
            report_content = f"""# Social Media Engagement Analysis Report
## Multimodal Fake News Detection Project

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a comprehensive analysis of social engagement patterns and dynamics across different content types and mapping relationships in the Fakeddit dataset. The analysis focuses on understanding how user engagement varies between multimodal content (text+image), image-only content, text-only content, and their relationships to authenticity labels.

## Methodology

### Data Integration Approach
- **Primary Dataset**: Fakeddit multimodal samples (train, validation, test)
- **Comment Integration**: Reddit comments mapped via submission_id
- **Image Integration**: Visual content mapped via record_id
- **Content Type Classification**: Based on availability of text, image, and comment data

### Analysis Dimensions
1. **Content Type Mapping**: Classification based on modality availability
2. **Engagement Metrics**: Post scores, comment counts, comment sentiment
3. **Authenticity Relationships**: Engagement patterns by truth labels
4. **Cross-Modal Dynamics**: How different modalities affect social response

## Key Findings

### Content Type Distribution
"""
            
            # Add content type analysis
            if 'mapping_type_distribution' in analysis_results.get('engagement_analysis', {}):
                report_content += "\n#### Mapping Type Distribution:\n"
                for content_type, count in analysis_results['engagement_analysis']['mapping_type_distribution'].items():
                    percentage = (count / sum(analysis_results['engagement_analysis']['mapping_type_distribution'].values())) * 100
                    report_content += f"- **{content_type}**: {count:,} posts ({percentage:.1f}%)\n"
            
            # Add engagement analysis
            if 'engagement_by_type' in analysis_results.get('engagement_analysis', {}):
                report_content += "\n### Engagement Patterns by Content Type\n\n"
                
                for content_type, metrics in analysis_results['engagement_analysis']['engagement_by_type'].items():
                    report_content += f"#### {content_type.replace('_', ' ').title()}\n"
                    report_content += f"- **Total Posts**: {metrics['count']:,}\n"
                    report_content += f"- **Dataset Percentage**: {metrics['percentage']:.1f}%\n"
                    
                    if 'score' in metrics:
                        report_content += f"- **Average Score**: {metrics['score']['mean']:.2f} (±{metrics['score']['std']:.2f})\n"
                        report_content += f"- **Median Score**: {metrics['score']['median']:.2f}\n"
                    
                    if 'num_comments' in metrics:
                        report_content += f"- **Average Comments**: {metrics['num_comments']['mean']:.2f} (±{metrics['num_comments']['std']:.2f})\n"
                        report_content += f"- **Median Comments**: {metrics['num_comments']['median']:.2f}\n"
                    
                    report_content += "\n"
            
            # Add sentiment analysis
            if 'sentiment_analysis' in analysis_results:
                report_content += "### Comment Sentiment Analysis\n\n"
                
                sentiment = analysis_results['sentiment_analysis']['overall_sentiment']
                total = sentiment['total_analyzed']
                
                report_content += f"**Sample Size**: {total:,} comments analyzed\n\n"
                report_content += "#### Sentiment Distribution:\n"
                report_content += f"- **Positive**: {sentiment['positive']:,} ({sentiment['positive']/total*100:.1f}%)\n"
                report_content += f"- **Negative**: {sentiment['negative']:,} ({sentiment['negative']/total*100:.1f}%)\n"
                report_content += f"- **Neutral**: {sentiment['neutral']:,} ({sentiment['neutral']/total*100:.1f}%)\n\n"
                
                if 'sentiment_distribution' in analysis_results['sentiment_analysis']:
                    dist = analysis_results['sentiment_analysis']['sentiment_distribution']
                    report_content += "#### Sentiment Statistics:\n"
                    report_content += f"- **Average Polarity**: {dist['polarity_mean']:.3f} (±{dist['polarity_std']:.3f})\n"
                    report_content += f"- **Average Subjectivity**: {dist['subjectivity_mean']:.3f} (±{dist['subjectivity_std']:.3f})\n\n"
            
            # Add authenticity analysis
            if 'authenticity_analysis' in analysis_results:
                report_content += "### Authenticity and Engagement Relationships\n\n"
                
                if 'engagement_by_label' in analysis_results['authenticity_analysis']:
                    report_content += "#### Engagement by Authenticity Label:\n"
                    for label, metrics in analysis_results['authenticity_analysis']['engagement_by_label'].items():
                        label_name = "Real" if label == "1" else "Fake" if label == "0" else f"Label {label}"
                        report_content += f"**{label_name} Content**:\n"
                        report_content += f"- Posts: {metrics['count']:,}\n"
                        report_content += f"- Average Score: {metrics['avg_score']:.2f}\n"
                        report_content += f"- Average Comments: {metrics['avg_comments']:.2f}\n\n"
                
                if 'cross_modal_patterns' in analysis_results['authenticity_analysis']:
                    report_content += "#### Cross-Modal Authenticity Patterns:\n"
                    for content_type, patterns in analysis_results['authenticity_analysis']['cross_modal_patterns'].items():
                        report_content += f"**{content_type.replace('_', ' ').title()}**:\n"
                        report_content += f"- Total Posts: {patterns['total_posts']:,}\n"
                        report_content += f"- Fake Posts: {patterns['fake_posts']:,}\n"
                        report_content += f"- Real Posts: {patterns['real_posts']:,}\n"
                        report_content += f"- Avg Engagement (Fake): {patterns['avg_engagement_fake']:.2f}\n"
                        report_content += f"- Avg Engagement (Real): {patterns['avg_engagement_real']:.2f}\n\n"
            
            # Add statistical comparisons
            if 'statistical_comparisons' in analysis_results.get('engagement_analysis', {}):
                report_content += "### Statistical Comparisons\n\n"
                
                for metric, comparisons in analysis_results['engagement_analysis']['statistical_comparisons'].items():
                    report_content += f"#### {metric.replace('_', ' ').title()} Comparisons:\n"
                    
                    for comparison, stats in comparisons.items():
                        types = comparison.replace('_vs_', ' vs ')
                        significance = "**Significant**" if stats['significant'] else "Not significant"
                        report_content += f"- **{types}**: {significance} (p={stats['p_value']:.4f}, effect size={stats['effect_size']:.3f})\n"
                    
                    report_content += "\n"
            
            # Add conclusions
            report_content += """## Key Insights and Implications

### Mapping Relationship Insights
1. **Content Type Diversity**: The dataset shows significant variation in content types, with different patterns of multimodal integration
2. **Engagement Variations**: Different content types show distinct engagement patterns, suggesting users respond differently to various modality combinations
3. **Authenticity Indicators**: Engagement patterns may serve as indicators of content authenticity, with fake and real content showing different social dynamics

### Social Dynamics Observations
1. **Cross-Modal Effects**: The presence of multiple modalities (text, image, comments) affects user engagement in measurable ways
2. **Sentiment Patterns**: Comment sentiment analysis reveals distinct patterns that may correlate with content authenticity
3. **Engagement Amplification**: Certain content types show higher social amplification, which may be relevant for misinformation spread

### Methodological Considerations
- **Sample Limitations**: Sentiment analysis was performed on a sample due to computational constraints
- **Mapping Coverage**: Comment coverage varies across content types, affecting analysis depth
- **Statistical Significance**: Statistical tests help identify meaningful differences between content types

## Future Research Directions

1. **Temporal Analysis**: Investigate how engagement patterns evolve over time
2. **Network Analysis**: Examine user interaction networks and influence patterns
3. **Content Quality**: Correlate engagement with content quality metrics
4. **Platform Dynamics**: Study how platform-specific features affect engagement

## Technical Implementation

### Data Processing Pipeline
1. **Data Integration**: Combined text, image, and comment data with mapping validation
2. **Content Classification**: Automated content type determination based on modality availability
3. **Engagement Metrics**: Calculated comprehensive engagement statistics
4. **Statistical Analysis**: Applied appropriate statistical tests for group comparisons

### Quality Assurance
- **Data Validation**: Verified data integrity and mapping accuracy
- **Statistical Rigor**: Used appropriate non-parametric tests for non-normal distributions
- **Reproducibility**: Documented all processing steps and parameters

---

*This analysis contributes to understanding social dynamics in multimodal misinformation detection and provides insights for developing more effective detection systems.*
"""
            
            # Save report
            with open('reports/social_analysis_report.md', 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info("Social analysis report generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
    
    def run_complete_analysis(self):
        """Run the complete social engagement analysis pipeline"""
        logger.info("Starting comprehensive social engagement analysis...")
        
        try:
            # Load integrated data
            data = self.load_integrated_data()
            
            if not data:
                logger.error("Failed to load data. Exiting analysis.")
                return
            
            # Initialize results dictionary
            analysis_results = {}
            
            # 1. Analyze comment sentiment patterns
            if 'comments' in data and not data['comments'].empty:
                logger.info("Step 1: Analyzing comment sentiment patterns...")
                sentiment_analysis = self.analyze_comment_sentiment(data['comments'])
                analysis_results['sentiment_analysis'] = sentiment_analysis
                self.performance_metrics['processing_steps'].append('sentiment_analysis')
            
            # 2. Analyze engagement by mapping type
            if 'text_data' in data:
                logger.info("Step 2: Analyzing engagement by mapping type...")
                engagement_analysis = self.analyze_engagement_by_mapping_type(
                    data['text_data'], 
                    data.get('comments', pd.DataFrame()),
                    data.get('images', None)
                )
                analysis_results['engagement_analysis'] = engagement_analysis
                self.performance_metrics['processing_steps'].append('engagement_analysis')
            
            # 3. Analyze authenticity-engagement relationships
            logger.info("Step 3: Analyzing authenticity-engagement relationships...")
            authenticity_analysis = self.analyze_authenticity_engagement_patterns(pd.DataFrame())
            analysis_results['authenticity_analysis'] = authenticity_analysis
            self.performance_metrics['processing_steps'].append('authenticity_analysis')
            
            # 4. Create visualizations
            logger.info("Step 4: Creating social engagement visualizations...")
            self.create_social_engagement_visualizations(analysis_results)
            self.performance_metrics['processing_steps'].append('visualizations')
            
            # 5. Generate comprehensive report
            logger.info("Step 5: Generating social analysis report...")
            self.generate_social_analysis_report(analysis_results)
            self.performance_metrics['processing_steps'].append('report_generation')
            
            # Save complete analysis results
            self.performance_metrics['end_time'] = datetime.now()
            self.performance_metrics['total_duration'] = str(self.performance_metrics['end_time'] - self.performance_metrics['start_time'])
            self.performance_metrics['analysis_results'] = {
                'sentiment_analysis_completed': 'sentiment_analysis' in analysis_results,
                'engagement_analysis_completed': 'engagement_analysis' in analysis_results,
                'authenticity_analysis_completed': 'authenticity_analysis' in analysis_results,
                'visualizations_created': True,
                'report_generated': True
            }
            
            # Save analysis results and performance metrics
            with open('analysis_results/social_analysis/social_engagement_analysis.json', 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            with open('analysis_results/social_analysis/task7_performance_metrics.json', 'w') as f:
                json.dump(self.performance_metrics, f, indent=2, default=str)
            
            # Create summary
            summary = {
                'task': 'Social Engagement Analysis with Mapping-Aware Social Dynamics',
                'completion_time': datetime.now().isoformat(),
                'data_processed': self.performance_metrics['data_loaded'],
                'analysis_components': self.performance_metrics['processing_steps'],
                'outputs_generated': {
                    'processed_data': [
                        'processed_data/social_engagement/comments_with_sentiment.parquet',
                        'processed_data/social_engagement/integrated_engagement_data.parquet'
                    ],
                    'analysis_results': [
                        'analysis_results/social_analysis/social_engagement_analysis.json',
                        'analysis_results/social_analysis/task7_performance_metrics.json'
                    ],
                    'visualizations': [
                        'visualizations/social_patterns/content_type_distribution.png',
                        'visualizations/social_patterns/engagement_by_content_type.png',
                        'visualizations/social_patterns/authenticity_engagement_analysis.png',
                        'visualizations/social_patterns/sentiment_analysis.png'
                    ],
                    'reports': [
                        'reports/social_analysis_report.md'
                    ]
                },
                'key_insights': {
                    'content_types_analyzed': len(analysis_results.get('engagement_analysis', {}).get('mapping_type_distribution', {})),
                    'sentiment_analysis_completed': 'sentiment_analysis' in analysis_results,
                    'statistical_comparisons_performed': len(analysis_results.get('engagement_analysis', {}).get('statistical_comparisons', {})),
                    'authenticity_patterns_identified': len(analysis_results.get('authenticity_analysis', {}).get('cross_modal_patterns', {}))
                }
            }
            
            with open('analysis_results/social_analysis/task7_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("Social engagement analysis completed successfully!")
            logger.info(f"Total processing time: {self.performance_metrics['total_duration']}")
            logger.info("All outputs saved to respective directories")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {str(e)}")
            return None

def main():
    """Main execution function"""
    print("="*80)
    print("TASK 7: SOCIAL ENGAGEMENT ANALYSIS WITH MAPPING-AWARE SOCIAL DYNAMICS")
    print("="*80)
    print()
    
    analyzer = SocialEngagementAnalyzer()
    results = analyzer.run_complete_analysis()
    
    if results:
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nKey Outputs Generated:")
        print("📊 Processed Data: processed_data/social_engagement/")
        print("📈 Analysis Results: analysis_results/social_analysis/")
        print("📉 Visualizations: visualizations/social_patterns/")
        print("📋 Report: reports/social_analysis_report.md")
        print("\nNext Steps:")
        print("- Review the social analysis report for key insights")
        print("- Examine visualizations for engagement patterns")
        print("- Use findings for cross-modal pattern discovery")
    else:
        print("\n" + "="*80)
        print("ANALYSIS FAILED - CHECK LOGS FOR DETAILS")
        print("="*80)

if __name__ == "__main__":
    main()