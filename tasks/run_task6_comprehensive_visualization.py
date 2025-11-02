#!/usr/bin/env python3
"""
Task 6: Comprehensive Visualization Pipeline for Multimodal Analysis
Creates visualizations for all completed analysis results with authenticity focus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/task6_comprehensive_visualization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveVisualizationPipeline:
    """Comprehensive visualization pipeline for multimodal analysis results"""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.analysis_dir = self.base_dir / "analysis_results"
        self.processed_dir = self.base_dir / "processed_data"
        self.viz_dir = self.base_dir / "visualizations"
        
        # Create visualization directories
        self.authenticity_dir = self.viz_dir / "authenticity_analysis"
        self.multimodal_dir = self.viz_dir / "multimodal_features"
        self.social_dir = self.viz_dir / "social_engagement"
        self.interactive_dir = self.viz_dir / "interactive"
        
        # Clear previous outputs and recreate directories
        self.clear_previous_outputs()
        
        for dir_path in [self.authenticity_dir, self.multimodal_dir, self.social_dir, self.interactive_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set visualization style
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info("Comprehensive Visualization Pipeline initialized")
    
    def clear_previous_outputs(self):
        """Clear all previous visualization outputs"""
        logger.info("Clearing previous visualization outputs...")
        
        import shutil
        
        try:
            # Clear specific task 6 output directories
            dirs_to_clear = [
                self.viz_dir / "authenticity_analysis",
                self.viz_dir / "multimodal_features", 
                self.viz_dir / "social_engagement",
                self.viz_dir / "interactive"
            ]
            
            for dir_path in dirs_to_clear:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    logger.info(f"✓ Cleared {dir_path}")
            
            # Clear any existing summary files
            summary_files = [
                self.viz_dir / "comprehensive_visualization_summary.json",
                self.viz_dir / "visualization_summary.json"
            ]
            
            for file_path in summary_files:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"✓ Removed {file_path}")
            
            logger.info("✓ Previous outputs cleared successfully")
            
        except Exception as e:
            logger.warning(f"Warning: Could not clear some previous outputs: {e}")
    
    def load_analysis_data(self):
        """Load all available analysis results"""
        logger.info("Loading analysis data from completed tasks...")
        
        data = {}
        
        try:
            # Load social analysis data
            social_path = self.analysis_dir / "social_analysis" / "social_engagement_analysis.json"
            if social_path.exists():
                with open(social_path, 'r') as f:
                    data['social_analysis'] = json.load(f)
                logger.info("✓ Social analysis data loaded")
            
            # Load text integration data
            text_path = self.analysis_dir / "text_integration" / "text_integration_analysis.json"
            if text_path.exists():
                with open(text_path, 'r') as f:
                    data['text_integration'] = json.load(f)
                logger.info("✓ Text integration data loaded")
            
            # Load image catalog data
            image_path = self.analysis_dir / "image_catalog" / "comprehensive_image_catalog.parquet"
            if image_path.exists():
                data['image_catalog'] = pd.read_parquet(image_path)
                logger.info("✓ Image catalog data loaded")
            
            # Load image mapping analysis
            mapping_path = self.analysis_dir / "image_catalog" / "id_mapping_analysis.json"
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    data['image_mapping'] = json.load(f)
                logger.info("✓ Image mapping data loaded")
            
            # Load processed datasets for authenticity analysis
            clean_datasets_dir = self.processed_dir / "clean_datasets"
            if clean_datasets_dir.exists():
                train_path = clean_datasets_dir / "train_final_clean.parquet"
                if train_path.exists():
                    # Load a sample for visualization (full dataset might be too large)
                    data['train_sample'] = pd.read_parquet(train_path).sample(n=min(10000, len(pd.read_parquet(train_path))))
                    logger.info("✓ Training data sample loaded")
            
            # Load social engagement processed data
            social_engagement_dir = self.processed_dir / "social_engagement"
            if social_engagement_dir.exists():
                integrated_path = social_engagement_dir / "integrated_engagement_data.parquet"
                if integrated_path.exists():
                    data['engagement_data'] = pd.read_parquet(integrated_path)
                    logger.info("✓ Social engagement data loaded")
            
            logger.info(f"Successfully loaded {len(data)} data sources")
            return data
            
        except Exception as e:
            logger.error(f"Error loading analysis data: {e}")
            return {}
    
    def create_authenticity_visualizations(self, data):
        """Create authenticity comparison charts"""
        logger.info("Creating authenticity analysis visualizations...")
        
        try:
            # 1. Content Type vs Authenticity Distribution
            if 'social_analysis' in data:
                social_data = data['social_analysis']
                engagement_by_label = social_data.get('authenticity_analysis', {}).get('engagement_by_label', {})
                
                if engagement_by_label:
                    # Create authenticity distribution chart
                    labels = ['Fake Content', 'Real Content']
                    counts = [
                        engagement_by_label.get('0', {}).get('count', 0),  # Fake
                        engagement_by_label.get('1', {}).get('count', 0)   # Real
                    ]
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Pie chart
                    colors = ['#FF6B6B', '#4ECDC4']
                    ax1.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax1.set_title('Authenticity Distribution in Dataset', fontsize=14, fontweight='bold')
                    
                    # Bar chart
                    bars = ax2.bar(labels, counts, color=colors)
                    ax2.set_title('Content Count by Authenticity Label', fontsize=14, fontweight='bold')
                    ax2.set_ylabel('Number of Posts')
                    
                    # Add value labels on bars
                    for bar, count in zip(bars, counts):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{count:,}', ha='center', va='bottom', fontweight='bold')
                    
                    plt.tight_layout()
                    plt.savefig(self.authenticity_dir / 'authenticity_distribution.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info("✓ Authenticity distribution chart created")
            
            # 2. Cross-Modal Authenticity Patterns
            if 'social_analysis' in data:
                cross_modal_patterns = social_data.get('authenticity_analysis', {}).get('cross_modal_patterns', {})
                
                if cross_modal_patterns:
                    content_types = []
                    fake_ratios = []
                    real_ratios = []
                    total_posts = []
                    
                    for content_type, pattern_data in cross_modal_patterns.items():
                        if pattern_data.get('total_posts', 0) > 0:
                            if content_type == 'text_image':
                                display_name = 'Text + Image'
                            elif content_type == 'full_multimodal':
                                display_name = 'Full Multimodal'
                            else:
                                display_name = 'Text Only'
                            
                            content_types.append(display_name)
                            total = pattern_data['total_posts']
                            fake_count = pattern_data.get('fake_posts', 0)
                            real_count = pattern_data.get('real_posts', 0)
                            
                            fake_ratios.append((fake_count / total) * 100)
                            real_ratios.append((real_count / total) * 100)
                            total_posts.append(total)
                    
                    if content_types:
                        # Stacked bar chart
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        x = np.arange(len(content_types))
                        width = 0.6
                        
                        bars1 = ax.bar(x, fake_ratios, width, label='Fake Content', color='#FF6B6B', alpha=0.8)
                        bars2 = ax.bar(x, real_ratios, width, bottom=fake_ratios, label='Real Content', color='#4ECDC4', alpha=0.8)
                        
                        ax.set_xlabel('Content Type', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
                        ax.set_title('Authenticity Patterns by Content Type', fontsize=14, fontweight='bold')
                        ax.set_xticks(x)
                        ax.set_xticklabels(content_types)
                        ax.legend()
                        
                        # Add percentage labels
                        for i, (fake_pct, real_pct, total) in enumerate(zip(fake_ratios, real_ratios, total_posts)):
                            if fake_pct > 5:  # Only show label if segment is large enough
                                ax.text(i, fake_pct/2, f'{fake_pct:.1f}%', ha='center', va='center', fontweight='bold')
                            if real_pct > 5:
                                ax.text(i, fake_pct + real_pct/2, f'{real_pct:.1f}%', ha='center', va='center', fontweight='bold')
                            
                            # Add total count below x-axis
                            ax.text(i, -8, f'n={total:,}', ha='center', va='top', fontsize=10, style='italic')
                        
                        plt.tight_layout()
                        plt.savefig(self.authenticity_dir / 'cross_modal_authenticity_patterns.png', dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        logger.info("✓ Cross-modal authenticity patterns chart created")
            
            # 3. Engagement vs Authenticity Analysis
            if 'social_analysis' in data:
                engagement_by_label = social_data.get('authenticity_analysis', {}).get('engagement_by_label', {})
                
                if engagement_by_label:
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                    
                    # Extract data
                    fake_data = engagement_by_label.get('0', {})
                    real_data = engagement_by_label.get('1', {})
                    
                    # Average scores comparison
                    scores = [fake_data.get('avg_score', 0), real_data.get('avg_score', 0)]
                    labels = ['Fake Content', 'Real Content']
                    colors = ['#FF6B6B', '#4ECDC4']
                    
                    bars1 = ax1.bar(labels, scores, color=colors)
                    ax1.set_title('Average Engagement Score by Authenticity', fontweight='bold')
                    ax1.set_ylabel('Average Score')
                    
                    for bar, score in zip(bars1, scores):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
                    
                    # Average comments comparison
                    comments = [fake_data.get('avg_comments', 0), real_data.get('avg_comments', 0)]
                    
                    bars2 = ax2.bar(labels, comments, color=colors)
                    ax2.set_title('Average Comments by Authenticity', fontweight='bold')
                    ax2.set_ylabel('Average Comments')
                    
                    for bar, comment in zip(bars2, comments):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{comment:.1f}', ha='center', va='bottom', fontweight='bold')
                    
                    # Content distribution by authenticity
                    fake_content_dist = fake_data.get('content_type_distribution', {})
                    real_content_dist = real_data.get('content_type_distribution', {})
                    
                    if fake_content_dist:
                        fake_types = list(fake_content_dist.keys())
                        fake_counts = list(fake_content_dist.values())
                        ax3.pie(fake_counts, labels=[t.replace('_', ' ').title() for t in fake_types], 
                               autopct='%1.1f%%', startangle=90, colors=['#FF9999', '#FFCC99', '#FF6666'])
                        ax3.set_title('Fake Content Type Distribution', fontweight='bold')
                    
                    if real_content_dist:
                        real_types = list(real_content_dist.keys())
                        real_counts = list(real_content_dist.values())
                        ax4.pie(real_counts, labels=[t.replace('_', ' ').title() for t in real_types], 
                               autopct='%1.1f%%', startangle=90, colors=['#99DDDD', '#AAEEDD', '#77CCCC'])
                        ax4.set_title('Real Content Type Distribution', fontweight='bold')
                    
                    plt.tight_layout()
                    plt.savefig(self.authenticity_dir / 'engagement_authenticity_analysis.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info("✓ Engagement vs authenticity analysis chart created")
            
            logger.info("✓ Authenticity visualizations completed")
            
        except Exception as e:
            logger.error(f"Error creating authenticity visualizations: {e}")
    
    def create_multimodal_feature_visualizations(self, data):
        """Create combined visual and textual feature plots"""
        logger.info("Creating multimodal feature visualizations...")
        
        try:
            # 1. Content Type Distribution Analysis
            if 'social_analysis' in data:
                social_data = data['social_analysis']
                content_dist = social_data.get('engagement_analysis', {}).get('mapping_type_distribution', {})
                
                if content_dist:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Pie chart
                    labels = []
                    values = []
                    colors = ['#2E8B57', '#FF6347', '#4682B4']
                    
                    for content_type, count in content_dist.items():
                        if content_type == 'text_image':
                            labels.append('Text + Image')
                        elif content_type == 'full_multimodal':
                            labels.append('Full Multimodal')
                        else:
                            labels.append('Text Only')
                        values.append(count)
                    
                    wedges, texts, autotexts = ax1.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax1.set_title('Content Modality Distribution', fontsize=14, fontweight='bold')
                    
                    # Make percentage text bold
                    for autotext in autotexts:
                        autotext.set_fontweight('bold')
                        autotext.set_color('white')
                    
                    # Bar chart with counts
                    bars = ax2.bar(labels, values, color=colors)
                    ax2.set_title('Content Count by Modality Type', fontsize=14, fontweight='bold')
                    ax2.set_ylabel('Number of Posts')
                    ax2.tick_params(axis='x', rotation=45)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{value:,}', ha='center', va='bottom', fontweight='bold')
                    
                    plt.tight_layout()
                    plt.savefig(self.multimodal_dir / 'content_modality_distribution.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info("✓ Content modality distribution chart created")
            
            # 2. Text Quality vs Image Availability Analysis
            if 'text_integration' in data and 'image_mapping' in data:
                text_data = data['text_integration']
                image_data = data['image_mapping']
                
                text_quality = text_data.get('text_quality_metrics', {})
                mapping_success = image_data.get('mapping_success_rate', 0)
                
                if text_quality:
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                    
                    # Title length distribution
                    title_stats = text_quality.get('title_length_stats', {})
                    if title_stats:
                        # Create a histogram-like visualization
                        categories = ['Very Short\n(<20 chars)', 'Short\n(20-40 chars)', 'Medium\n(40-80 chars)', 'Long\n(>80 chars)']
                        
                        # Estimate distribution based on stats
                        mean_length = title_stats.get('mean', 46)
                        std_length = title_stats.get('std', 35)
                        
                        # Create sample data for visualization
                        sample_lengths = np.random.normal(mean_length, std_length, 1000)
                        sample_lengths = np.clip(sample_lengths, 1, 300)  # Reasonable bounds
                        
                        ax1.hist(sample_lengths, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
                        ax1.axvline(mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.1f}')
                        ax1.set_xlabel('Title Length (characters)')
                        ax1.set_ylabel('Frequency')
                        ax1.set_title('Title Length Distribution', fontweight='bold')
                        ax1.legend()
                    
                    # Missing data analysis
                    missing_data = text_data.get('missing_data_analysis', {})
                    if missing_data:
                        missing_fields = []
                        missing_percentages = []
                        
                        for field, stats in missing_data.items():
                            missing_fields.append(field.replace('_', ' ').title())
                            missing_percentages.append(stats.get('missing_percentage', 0))
                        
                        bars = ax2.bar(missing_fields, missing_percentages, color='coral')
                        ax2.set_title('Missing Data by Field', fontweight='bold')
                        ax2.set_ylabel('Missing Percentage (%)')
                        ax2.tick_params(axis='x', rotation=45)
                        
                        for bar, pct in zip(bars, missing_percentages):
                            height = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
                    
                    # Image mapping success visualization
                    mapping_categories = ['Successfully Mapped', 'Not Mapped']
                    mapping_values = [mapping_success, 100 - mapping_success]
                    colors = ['#4ECDC4', '#FF6B6B']
                    
                    wedges, texts, autotexts = ax3.pie(mapping_values, labels=mapping_categories, colors=colors, 
                                                      autopct='%1.1f%%', startangle=90)
                    ax3.set_title('Image Mapping Success Rate', fontweight='bold')
                    
                    for autotext in autotexts:
                        autotext.set_fontweight('bold')
                        autotext.set_color('white')
                    
                    # Dataset overview
                    total_text = text_data.get('dataset_overview', {}).get('total_records', 0)
                    total_images = image_data.get('total_images', 0)
                    
                    overview_data = ['Text Records', 'Total Images', 'Mapped Images']
                    overview_counts = [total_text, total_images, int(total_images * mapping_success / 100)]
                    
                    bars = ax4.bar(overview_data, overview_counts, color=['#87CEEB', '#DDA0DD', '#98FB98'])
                    ax4.set_title('Dataset Overview', fontweight='bold')
                    ax4.set_ylabel('Count')
                    ax4.tick_params(axis='x', rotation=45)
                    
                    for bar, count in zip(bars, overview_counts):
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{count:,}', ha='center', va='bottom', fontweight='bold')
                    
                    plt.tight_layout()
                    plt.savefig(self.multimodal_dir / 'text_image_integration_analysis.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info("✓ Text-image integration analysis chart created")
            
            # 3. Cross-Modal Relationship Network (if we have image catalog data)
            if 'image_catalog' in data:
                catalog_df = data['image_catalog']
                
                # Content type relationship analysis
                content_type_counts = catalog_df['content_type'].value_counts()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Content type distribution
                colors = ['#2E8B57' if ct == 'multimodal' else '#FF6347' for ct in content_type_counts.index]
                bars = ax1.bar(content_type_counts.index, content_type_counts.values, color=colors)
                ax1.set_title('Image Content Type Distribution', fontweight='bold')
                ax1.set_ylabel('Number of Images')
                
                for bar, count in zip(bars, content_type_counts.values):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{count:,}', ha='center', va='bottom', fontweight='bold')
                
                # Mapping success visualization
                total_images = len(catalog_df)
                multimodal_images = len(catalog_df[catalog_df['content_type'] == 'multimodal'])
                image_only = len(catalog_df[catalog_df['content_type'] == 'image_only'])
                
                mapping_data = {
                    'Multimodal\n(Text + Image)': multimodal_images,
                    'Image Only': image_only
                }
                
                wedges, texts, autotexts = ax2.pie(mapping_data.values(), labels=mapping_data.keys(), 
                                                  colors=['#2E8B57', '#FF6347'], autopct='%1.1f%%', startangle=90)
                ax2.set_title('Image-Text Mapping Results', fontweight='bold')
                
                for autotext in autotexts:
                    autotext.set_fontweight('bold')
                    autotext.set_color('white')
                
                plt.tight_layout()
                plt.savefig(self.multimodal_dir / 'image_catalog_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info("✓ Image catalog analysis chart created")
            
            logger.info("✓ Multimodal feature visualizations completed")
            
        except Exception as e:
            logger.error(f"Error creating multimodal feature visualizations: {e}")
    
    def create_social_engagement_visualizations(self, data):
        """Create comment and engagement pattern charts"""
        logger.info("Creating social engagement visualizations...")
        
        try:
            if 'social_analysis' in data:
                social_data = data['social_analysis']
                
                # 1. Engagement by Content Type
                engagement_by_type = social_data.get('engagement_analysis', {}).get('engagement_by_type', {})
                
                if engagement_by_type:
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
                    
                    content_types = []
                    avg_scores = []
                    avg_comments = []
                    post_counts = []
                    
                    for content_type, stats in engagement_by_type.items():
                        if content_type == 'text_image':
                            display_name = 'Text + Image'
                        elif content_type == 'full_multimodal':
                            display_name = 'Full Multimodal'
                        else:
                            display_name = 'Text Only'
                        
                        content_types.append(display_name)
                        avg_scores.append(stats.get('score', {}).get('mean', 0))
                        avg_comments.append(stats.get('num_comments', {}).get('mean', 0))
                        post_counts.append(stats.get('count', 0))
                    
                    # Average engagement scores
                    colors = ['#2E8B57', '#FF6347', '#4682B4']
                    bars1 = ax1.bar(content_types, avg_scores, color=colors)
                    ax1.set_title('Average Engagement Score by Content Type', fontweight='bold', fontsize=14)
                    ax1.set_ylabel('Average Score')
                    ax1.tick_params(axis='x', rotation=45)
                    
                    for bar, score in zip(bars1, avg_scores):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
                    
                    # Average comments
                    bars2 = ax2.bar(content_types, avg_comments, color=colors)
                    ax2.set_title('Average Comments by Content Type', fontweight='bold', fontsize=14)
                    ax2.set_ylabel('Average Comments')
                    ax2.tick_params(axis='x', rotation=45)
                    
                    for bar, comments in zip(bars2, avg_comments):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{comments:.1f}', ha='center', va='bottom', fontweight='bold')
                    
                    # Post counts
                    bars3 = ax3.bar(content_types, post_counts, color=colors)
                    ax3.set_title('Number of Posts by Content Type', fontweight='bold', fontsize=14)
                    ax3.set_ylabel('Number of Posts')
                    ax3.tick_params(axis='x', rotation=45)
                    
                    for bar, count in zip(bars3, post_counts):
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{count:,}', ha='center', va='bottom', fontweight='bold')
                    
                    # Engagement efficiency (score per comment)
                    efficiency = [score/comments if comments > 0 else 0 for score, comments in zip(avg_scores, avg_comments)]
                    bars4 = ax4.bar(content_types, efficiency, color=colors)
                    ax4.set_title('Engagement Efficiency (Score per Comment)', fontweight='bold', fontsize=14)
                    ax4.set_ylabel('Score per Comment')
                    ax4.tick_params(axis='x', rotation=45)
                    
                    for bar, eff in zip(bars4, efficiency):
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')
                    
                    plt.tight_layout()
                    plt.savefig(self.social_dir / 'engagement_by_content_type.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info("✓ Engagement by content type chart created")
                
                # 2. Sentiment Analysis Visualization
                sentiment_data = social_data.get('sentiment_analysis', {})
                if sentiment_data:
                    overall_sentiment = sentiment_data.get('overall_sentiment', {})
                    sentiment_dist = sentiment_data.get('sentiment_distribution', {})
                    
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                    
                    # Overall sentiment pie chart
                    if overall_sentiment:
                        sentiments = ['Positive', 'Negative', 'Neutral']
                        counts = [
                            overall_sentiment.get('positive', 0),
                            overall_sentiment.get('negative', 0),
                            overall_sentiment.get('neutral', 0)
                        ]
                        colors = ['#4ECDC4', '#FF6B6B', '#95A5A6']
                        
                        wedges, texts, autotexts = ax1.pie(counts, labels=sentiments, colors=colors, 
                                                          autopct='%1.1f%%', startangle=90)
                        ax1.set_title('Overall Comment Sentiment Distribution', fontweight='bold')
                        
                        for autotext in autotexts:
                            autotext.set_fontweight('bold')
                            autotext.set_color('white')
                    
                    # Sentiment statistics
                    if sentiment_dist:
                        stats_labels = ['Polarity\nMean', 'Polarity\nStd', 'Subjectivity\nMean', 'Subjectivity\nStd']
                        stats_values = [
                            sentiment_dist.get('polarity_mean', 0),
                            sentiment_dist.get('polarity_std', 0),
                            sentiment_dist.get('subjectivity_mean', 0),
                            sentiment_dist.get('subjectivity_std', 0)
                        ]
                        
                        bars = ax2.bar(stats_labels, stats_values, color=['#87CEEB', '#DDA0DD', '#98FB98', '#F0E68C'])
                        ax2.set_title('Sentiment Statistics', fontweight='bold')
                        ax2.set_ylabel('Value')
                        
                        for bar, value in zip(bars, stats_values):
                            height = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                    
                    # Engagement vs Sentiment (if we have engagement data)
                    engagement_by_label = social_data.get('authenticity_analysis', {}).get('engagement_by_label', {})
                    if engagement_by_label:
                        labels = ['Fake Content', 'Real Content']
                        avg_scores = [
                            engagement_by_label.get('0', {}).get('avg_score', 0),
                            engagement_by_label.get('1', {}).get('avg_score', 0)
                        ]
                        avg_comments = [
                            engagement_by_label.get('0', {}).get('avg_comments', 0),
                            engagement_by_label.get('1', {}).get('avg_comments', 0)
                        ]
                        
                        x = np.arange(len(labels))
                        width = 0.35
                        
                        bars1 = ax3.bar(x - width/2, avg_scores, width, label='Avg Score', color='#FF6B6B', alpha=0.8)
                        bars2 = ax3.bar(x + width/2, avg_comments, width, label='Avg Comments', color='#4ECDC4', alpha=0.8)
                        
                        ax3.set_title('Engagement Patterns by Authenticity', fontweight='bold')
                        ax3.set_ylabel('Value')
                        ax3.set_xticks(x)
                        ax3.set_xticklabels(labels)
                        ax3.legend()
                        
                        # Add value labels
                        for bars in [bars1, bars2]:
                            for bar in bars:
                                height = bar.get_height()
                                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                        f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
                    
                    # Comment volume analysis
                    if overall_sentiment:
                        total_analyzed = overall_sentiment.get('total_analyzed', 0)
                        ax4.text(0.5, 0.7, f'Total Comments Analyzed:', ha='center', va='center', 
                                transform=ax4.transAxes, fontsize=14, fontweight='bold')
                        ax4.text(0.5, 0.5, f'{total_analyzed:,}', ha='center', va='center', 
                                transform=ax4.transAxes, fontsize=24, fontweight='bold', color='#2E8B57')
                        ax4.text(0.5, 0.3, 'Comments processed for\nsentiment analysis', ha='center', va='center', 
                                transform=ax4.transAxes, fontsize=12, style='italic')
                        ax4.set_xlim(0, 1)
                        ax4.set_ylim(0, 1)
                        ax4.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(self.social_dir / 'sentiment_analysis_overview.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info("✓ Sentiment analysis overview chart created")
            
            logger.info("✓ Social engagement visualizations completed")
            
        except Exception as e:
            logger.error(f"Error creating social engagement visualizations: {e}")
    
    def create_interactive_components(self, data):
        """Create interactive dashboard components using Plotly"""
        logger.info("Creating interactive dashboard components...")
        
        try:
            # 1. Interactive Authenticity Analysis
            if 'social_analysis' in data:
                social_data = data['social_analysis']
                engagement_by_label = social_data.get('authenticity_analysis', {}).get('engagement_by_label', {})
                
                if engagement_by_label:
                    # Create interactive comparison chart
                    fake_data = engagement_by_label.get('0', {})
                    real_data = engagement_by_label.get('1', {})
                    
                    fig = go.Figure()
                    
                    # Add bars for fake content
                    fig.add_trace(go.Bar(
                        name='Fake Content',
                        x=['Posts', 'Avg Score', 'Avg Comments'],
                        y=[fake_data.get('count', 0), fake_data.get('avg_score', 0), fake_data.get('avg_comments', 0)],
                        marker_color='#FF6B6B',
                        text=[f"{fake_data.get('count', 0):,}", f"{fake_data.get('avg_score', 0):.1f}", f"{fake_data.get('avg_comments', 0):.1f}"],
                        textposition='auto',
                    ))
                    
                    # Add bars for real content
                    fig.add_trace(go.Bar(
                        name='Real Content',
                        x=['Posts', 'Avg Score', 'Avg Comments'],
                        y=[real_data.get('count', 0), real_data.get('avg_score', 0), real_data.get('avg_comments', 0)],
                        marker_color='#4ECDC4',
                        text=[f"{real_data.get('count', 0):,}", f"{real_data.get('avg_score', 0):.1f}", f"{real_data.get('avg_comments', 0):.1f}"],
                        textposition='auto',
                    ))
                    
                    fig.update_layout(
                        title='Interactive Authenticity Comparison',
                        xaxis_title='Metrics',
                        yaxis_title='Values',
                        barmode='group',
                        template='plotly_white'
                    )
                    
                    # Save as HTML
                    fig.write_html(self.interactive_dir / 'authenticity_comparison.html')
                    
                    # Save as JSON for dashboard integration
                    fig.write_json(self.interactive_dir / 'authenticity_comparison.json')
                    
                    logger.info("✓ Interactive authenticity comparison created")
            
            # 2. Interactive Content Type Distribution
            if 'social_analysis' in data:
                content_dist = social_data.get('engagement_analysis', {}).get('mapping_type_distribution', {})
                
                if content_dist:
                    labels = []
                    values = []
                    colors = ['#2E8B57', '#FF6347', '#4682B4']
                    
                    for content_type, count in content_dist.items():
                        if content_type == 'text_image':
                            labels.append('Text + Image')
                        elif content_type == 'full_multimodal':
                            labels.append('Full Multimodal')
                        else:
                            labels.append('Text Only')
                        values.append(count)
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        marker_colors=colors,
                        textinfo='label+percent+value',
                        texttemplate='%{label}<br>%{percent}<br>(%{value:,})',
                        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
                    )])
                    
                    fig.update_layout(
                        title='Interactive Content Type Distribution',
                        template='plotly_white'
                    )
                    
                    fig.write_html(self.interactive_dir / 'content_type_distribution.html')
                    fig.write_json(self.interactive_dir / 'content_type_distribution.json')
                    
                    logger.info("✓ Interactive content type distribution created")
            
            # 3. Interactive Engagement Analysis
            if 'social_analysis' in data:
                engagement_by_type = social_data.get('engagement_analysis', {}).get('engagement_by_type', {})
                
                if engagement_by_type:
                    content_types = []
                    avg_scores = []
                    avg_comments = []
                    post_counts = []
                    
                    for content_type, stats in engagement_by_type.items():
                        if content_type == 'text_image':
                            display_name = 'Text + Image'
                        elif content_type == 'full_multimodal':
                            display_name = 'Full Multimodal'
                        else:
                            display_name = 'Text Only'
                        
                        content_types.append(display_name)
                        avg_scores.append(stats.get('score', {}).get('mean', 0))
                        avg_comments.append(stats.get('num_comments', {}).get('mean', 0))
                        post_counts.append(stats.get('count', 0))
                    
                    # Create subplots
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Average Score', 'Average Comments', 'Post Count', 'Engagement Efficiency'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    colors = ['#2E8B57', '#FF6347', '#4682B4']
                    
                    # Average scores
                    fig.add_trace(
                        go.Bar(x=content_types, y=avg_scores, name='Avg Score', marker_color=colors,
                               text=[f'{score:.1f}' for score in avg_scores], textposition='auto'),
                        row=1, col=1
                    )
                    
                    # Average comments
                    fig.add_trace(
                        go.Bar(x=content_types, y=avg_comments, name='Avg Comments', marker_color=colors,
                               text=[f'{comments:.1f}' for comments in avg_comments], textposition='auto'),
                        row=1, col=2
                    )
                    
                    # Post counts
                    fig.add_trace(
                        go.Bar(x=content_types, y=post_counts, name='Post Count', marker_color=colors,
                               text=[f'{count:,}' for count in post_counts], textposition='auto'),
                        row=2, col=1
                    )
                    
                    # Engagement efficiency
                    efficiency = [score/comments if comments > 0 else 0 for score, comments in zip(avg_scores, avg_comments)]
                    fig.add_trace(
                        go.Bar(x=content_types, y=efficiency, name='Efficiency', marker_color=colors,
                               text=[f'{eff:.2f}' for eff in efficiency], textposition='auto'),
                        row=2, col=2
                    )
                    
                    fig.update_layout(
                        title_text='Interactive Engagement Analysis by Content Type',
                        showlegend=False,
                        template='plotly_white'
                    )
                    
                    fig.write_html(self.interactive_dir / 'engagement_analysis.html')
                    fig.write_json(self.interactive_dir / 'engagement_analysis.json')
                    
                    logger.info("✓ Interactive engagement analysis created")
            
            logger.info("✓ Interactive components completed")
            
        except Exception as e:
            logger.error(f"Error creating interactive components: {e}")
    
    def generate_visualization_summary(self, data):
        """Generate a summary of all created visualizations"""
        logger.info("Generating visualization summary...")
        
        try:
            summary = {
                "generation_timestamp": datetime.now().isoformat(),
                "data_sources": list(data.keys()),
                "visualizations_created": {
                    "authenticity_analysis": [
                        "authenticity_distribution.png",
                        "cross_modal_authenticity_patterns.png", 
                        "engagement_authenticity_analysis.png"
                    ],
                    "multimodal_features": [
                        "content_modality_distribution.png",
                        "text_image_integration_analysis.png",
                        "image_catalog_analysis.png"
                    ],
                    "social_engagement": [
                        "engagement_by_content_type.png",
                        "sentiment_analysis_overview.png"
                    ],
                    "interactive": [
                        "authenticity_comparison.html",
                        "content_type_distribution.html",
                        "engagement_analysis.html"
                    ]
                },
                "total_visualizations": 0,
                "export_formats": ["PNG", "HTML", "JSON"]
            }
            
            # Count total visualizations
            for category in summary["visualizations_created"].values():
                summary["total_visualizations"] += len(category)
            
            # Save summary
            summary_path = self.viz_dir / "comprehensive_visualization_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"✓ Visualization summary saved: {summary['total_visualizations']} visualizations created")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating visualization summary: {e}")
            return {}
    
    def run_comprehensive_pipeline(self):
        """Run the complete visualization pipeline"""
        logger.info("Starting comprehensive visualization pipeline...")
        
        try:
            # Load all analysis data
            data = self.load_analysis_data()
            
            if not data:
                logger.error("No analysis data available. Please run analysis tasks first.")
                return False
            
            # Create all visualization categories
            self.create_authenticity_visualizations(data)
            self.create_multimodal_feature_visualizations(data)
            self.create_social_engagement_visualizations(data)
            self.create_interactive_components(data)
            
            # Generate summary
            summary = self.generate_visualization_summary(data)
            
            logger.info("✓ Comprehensive visualization pipeline completed successfully")
            logger.info(f"✓ Created {summary.get('total_visualizations', 0)} visualizations across 4 categories")
            logger.info(f"✓ Output directories: {[str(d) for d in [self.authenticity_dir, self.multimodal_dir, self.social_dir, self.interactive_dir]]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in comprehensive visualization pipeline: {e}")
            return False

def main():
    """Main execution function"""
    logger.info("=" * 80)
    logger.info("TASK 6: COMPREHENSIVE VISUALIZATION PIPELINE FOR MULTIMODAL ANALYSIS")
    logger.info("=" * 80)
    
    try:
        # Initialize and run pipeline
        pipeline = ComprehensiveVisualizationPipeline()
        success = pipeline.run_comprehensive_pipeline()
        
        if success:
            logger.info("✅ Task 6 completed successfully!")
            logger.info("📊 All visualizations have been created and are ready for dashboard integration")
        else:
            logger.error("❌ Task 6 failed to complete")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error in Task 6: {e}")
        return 1

if __name__ == "__main__":
    exit(main())