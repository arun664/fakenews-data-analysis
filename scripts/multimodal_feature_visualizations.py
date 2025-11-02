#!/usr/bin/env python3
"""
Multimodal Feature Visualizations
Creates combined visual and textual feature plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MultimodalFeatureVisualizer:
    def __init__(self):
        self.base_path = Path(".")
        self.clean_data = self._load_clean_datasets()
        
    def _load_clean_datasets(self):
        """Load clean datasets"""
        datasets = {}
        try:
            datasets['train'] = pd.read_parquet("processed_data/clean_datasets/train_final_clean.parquet")
            datasets['test'] = pd.read_parquet("processed_data/clean_datasets/test_final_clean.parquet")
            datasets['validation'] = pd.read_parquet("processed_data/clean_datasets/validation_final_clean.parquet")
            return datasets
        except FileNotFoundError:
            print("Warning: Clean datasets not found")
            return {}
    
    def create_multimodal_feature_visualizations(self):
        """Create combined visual and textual feature visualizations"""
        print("Creating multimodal feature visualizations...")
        
        if not self.clean_data:
            print("No clean datasets available")
            return
        
        # Combine all datasets
        all_data = pd.concat([self.clean_data['train'], self.clean_data['test'], 
                             self.clean_data['validation']], ignore_index=True)
        
        print(f"Combined dataset shape: {all_data.shape}")
        print(f"Available columns: {list(all_data.columns)}")
        
        # Create multimodal feature analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multimodal Feature Analysis', fontsize=16, fontweight='bold')
        
        # Title length distribution by authenticity
        if 'title' in all_data.columns and '2_way_label' in all_data.columns:
            fake_titles = all_data[all_data['2_way_label'] == 0]['title'].str.len()
            real_titles = all_data[all_data['2_way_label'] == 1]['title'].str.len()
            
            axes[0, 0].hist([fake_titles.dropna(), real_titles.dropna()], 
                           bins=50, alpha=0.7, label=['Fake', 'Real'], color=['#ff6b6b', '#4ecdc4'])
            axes[0, 0].set_title('Title Length Distribution by Authenticity')
            axes[0, 0].set_xlabel('Title Length (characters)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
        
        # Content type distribution
        if 'mapping_type' in all_data.columns:
            mapping_counts = all_data['mapping_type'].value_counts()
            axes[0, 1].pie(mapping_counts.values, labels=mapping_counts.index, autopct='%1.1f%%')
            axes[0, 1].set_title('Content Type Distribution')
        
        # Authenticity distribution by content type
        if 'mapping_type' in all_data.columns and '2_way_label' in all_data.columns:
            cross_tab = pd.crosstab(all_data['mapping_type'], all_data['2_way_label'])
            cross_tab.plot(kind='bar', ax=axes[1, 0], color=['#ff6b6b', '#4ecdc4'])
            axes[1, 0].set_title('Authenticity Distribution by Content Type')
            axes[1, 0].set_xlabel('Content Type')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].legend(['Fake', 'Real'])
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Score distribution by authenticity
        if 'score' in all_data.columns and '2_way_label' in all_data.columns:
            fake_scores = all_data[all_data['2_way_label'] == 0]['score']
            real_scores = all_data[all_data['2_way_label'] == 1]['score']
            
            axes[1, 1].boxplot([fake_scores.dropna(), real_scores.dropna()], 
                              labels=['Fake', 'Real'])
            axes[1, 1].set_title('Score Distribution by Authenticity')
            axes[1, 1].set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig('visualizations/multimodal_features/multimodal_feature_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create correlation heatmap
        self._create_feature_correlation_heatmap(all_data)
        
        # Create detailed feature analysis
        self._create_detailed_feature_analysis(all_data)
        
    def _create_feature_correlation_heatmap(self, data):
        """Create correlation heatmap for numerical features"""
        # Select numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return
        
        # Calculate correlation matrix
        corr_matrix = data[numerical_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('visualizations/multimodal_features/feature_correlation_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_detailed_feature_analysis(self, data):
        """Create detailed feature analysis charts"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Detailed Multimodal Feature Analysis', fontsize=16, fontweight='bold')
        
        # 1. Title length vs Score relationship
        if 'title' in data.columns and 'score' in data.columns:
            title_lengths = data['title'].str.len()
            sample_indices = np.random.choice(len(data), min(5000, len(data)), replace=False)
            sample_data = data.iloc[sample_indices]
            sample_lengths = title_lengths.iloc[sample_indices]
            
            axes[0, 0].scatter(sample_lengths, sample_data['score'], alpha=0.6)
            axes[0, 0].set_xlabel('Title Length')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_title('Title Length vs Score')
        
        # 2. Comments distribution by authenticity
        if 'num_comments' in data.columns and '2_way_label' in data.columns:
            fake_comments = data[data['2_way_label'] == 0]['num_comments']
            real_comments = data[data['2_way_label'] == 1]['num_comments']
            
            # Use log scale for better visualization
            fake_comments_log = np.log1p(fake_comments.dropna())
            real_comments_log = np.log1p(real_comments.dropna())
            
            axes[0, 1].hist([fake_comments_log, real_comments_log], 
                           bins=50, alpha=0.7, label=['Fake', 'Real'], color=['#ff6b6b', '#4ecdc4'])
            axes[0, 1].set_title('Log(Comments + 1) Distribution by Authenticity')
            axes[0, 1].set_xlabel('Log(Comments + 1)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
        
        # 3. Score vs Comments relationship
        if 'score' in data.columns and 'num_comments' in data.columns:
            sample_indices = np.random.choice(len(data), min(5000, len(data)), replace=False)
            sample_data = data.iloc[sample_indices]
            
            axes[0, 2].scatter(sample_data['score'], sample_data['num_comments'], alpha=0.6)
            axes[0, 2].set_xlabel('Score')
            axes[0, 2].set_ylabel('Number of Comments')
            axes[0, 2].set_title('Score vs Comments Relationship')
        
        # 4. Authenticity by dataset split
        if '2_way_label' in data.columns and 'split' in data.columns:
            split_auth = pd.crosstab(data['split'], data['2_way_label'])
            split_auth.plot(kind='bar', ax=axes[1, 0], color=['#ff6b6b', '#4ecdc4'])
            axes[1, 0].set_title('Authenticity Distribution by Dataset Split')
            axes[1, 0].set_xlabel('Dataset Split')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].legend(['Fake', 'Real'])
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Content type by authenticity (stacked bar)
        if 'mapping_type' in data.columns and '2_way_label' in data.columns:
            content_auth = pd.crosstab(data['mapping_type'], data['2_way_label'], normalize='index') * 100
            content_auth.plot(kind='bar', stacked=True, ax=axes[1, 1], color=['#ff6b6b', '#4ecdc4'])
            axes[1, 1].set_title('Authenticity Percentage by Content Type')
            axes[1, 1].set_xlabel('Content Type')
            axes[1, 1].set_ylabel('Percentage')
            axes[1, 1].legend(['Fake', 'Real'])
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Feature summary statistics
        if '2_way_label' in data.columns:
            # Calculate summary statistics by authenticity
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 1:
                fake_data = data[data['2_way_label'] == 0][numerical_cols].mean()
                real_data = data[data['2_way_label'] == 1][numerical_cols].mean()
                
                # Select top features for visualization
                top_features = ['score', 'num_comments'] if all(col in numerical_cols for col in ['score', 'num_comments']) else numerical_cols[:5]
                
                x = np.arange(len(top_features))
                width = 0.35
                
                fake_values = [fake_data.get(feat, 0) for feat in top_features]
                real_values = [real_data.get(feat, 0) for feat in top_features]
                
                axes[1, 2].bar(x - width/2, fake_values, width, label='Fake', color='#ff6b6b')
                axes[1, 2].bar(x + width/2, real_values, width, label='Real', color='#4ecdc4')
                axes[1, 2].set_title('Average Feature Values by Authenticity')
                axes[1, 2].set_ylabel('Average Value')
                axes[1, 2].set_xticks(x)
                axes[1, 2].set_xticklabels(top_features, rotation=45)
                axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/multimodal_features/detailed_feature_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_feature_dashboard(self):
        """Create interactive feature dashboard"""
        print("Creating interactive feature dashboard...")
        
        if not self.clean_data:
            return
        
        # Combine all datasets
        all_data = pd.concat([self.clean_data['train'], self.clean_data['test'], 
                             self.clean_data['validation']], ignore_index=True)
        
        # Sample data for performance
        sample_size = min(10000, len(all_data))
        sample_data = all_data.sample(n=sample_size, random_state=42)
        
        # Create interactive scatter plot
        if 'score' in sample_data.columns and 'num_comments' in sample_data.columns and '2_way_label' in sample_data.columns:
            fig = px.scatter(sample_data, 
                           x='score', 
                           y='num_comments',
                           color='2_way_label',
                           color_discrete_map={0: '#ff6b6b', 1: '#4ecdc4'},
                           labels={'2_way_label': 'Authenticity', 'score': 'Engagement Score', 'num_comments': 'Number of Comments'},
                           title='Interactive Feature Analysis: Score vs Comments by Authenticity',
                           hover_data=['mapping_type'] if 'mapping_type' in sample_data.columns else None)
            
            fig.update_layout(height=600)
            fig.write_html('visualizations/interactive/feature_scatter_dashboard.html')
        
        # Create interactive feature distribution
        if 'title' in sample_data.columns and '2_way_label' in sample_data.columns:
            sample_data['title_length'] = sample_data['title'].str.len()
            
            fig = px.histogram(sample_data, 
                             x='title_length', 
                             color='2_way_label',
                             color_discrete_map={0: '#ff6b6b', 1: '#4ecdc4'},
                             labels={'2_way_label': 'Authenticity', 'title_length': 'Title Length'},
                             title='Interactive Title Length Distribution by Authenticity',
                             nbins=50,
                             opacity=0.7)
            
            fig.update_layout(height=500)
            fig.write_html('visualizations/interactive/title_length_distribution.html')
    
    def run_pipeline(self):
        """Run the multimodal feature visualization pipeline"""
        print("Starting Multimodal Feature Visualization Pipeline...")
        print("=" * 60)
        
        try:
            # Create multimodal feature visualizations
            self.create_multimodal_feature_visualizations()
            
            # Create interactive dashboard
            self.create_interactive_feature_dashboard()
            
            print("=" * 60)
            print("Multimodal Feature Visualization Pipeline completed successfully!")
            print("\nGenerated visualizations:")
            print("- Multimodal feature analysis charts")
            print("- Feature correlation heatmap")
            print("- Detailed feature analysis")
            print("- Interactive feature dashboards")
            
        except Exception as e:
            print(f"Error in multimodal feature visualization pipeline: {str(e)}")
            raise

def main():
    """Main execution function"""
    visualizer = MultimodalFeatureVisualizer()
    visualizer.run_pipeline()

if __name__ == "__main__":
    main()