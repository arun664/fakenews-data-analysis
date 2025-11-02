#!/usr/bin/env python3
"""
Comprehensive Visualization Pipeline for Multimodal Analysis
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
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveVisualizationPipeline:
    def __init__(self):
        self.base_path = Path(".")
        
        # Load analysis results
        self.social_analysis = self._load_json("analysis_results/social_analysis/social_engagement_analysis.json")
        self.text_analysis = self._load_json("analysis_results/text_integration/text_integration_analysis.json")
        self.image_analysis = self._load_json("analysis_results/image_catalog/id_mapping_analysis.json")
        
    def _load_json(self, filepath):
        """Load JSON analysis results"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found")
            return {}
    
    def create_authenticity_comparison_charts(self):
        """Generate authenticity comparison charts showing fake vs real content patterns"""
        print("Creating authenticity comparison charts...")
        
        if not self.social_analysis:
            print("No social analysis data available")
            return
        
        # Extract authenticity data
        auth_data = self.social_analysis.get('authenticity_analysis', {})
        engagement_by_label = auth_data.get('engagement_by_label', {})
        
        if not engagement_by_label:
            print("No engagement by label data available")
            return
        
        # Create authenticity comparison charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Authenticity Analysis: Fake vs Real Content Patterns', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        labels = ['Fake (0)', 'Real (1)']
        counts = [engagement_by_label.get('0', {}).get('count', 0), 
                 engagement_by_label.get('1', {}).get('count', 0)]
        avg_scores = [engagement_by_label.get('0', {}).get('avg_score', 0),
                     engagement_by_label.get('1', {}).get('avg_score', 0)]
        avg_comments = [engagement_by_label.get('0', {}).get('avg_comments', 0),
                       engagement_by_label.get('1', {}).get('avg_comments', 0)]
        
        # Content distribution
        axes[0, 0].bar(labels, counts, color=['#ff6b6b', '#4ecdc4'])
        axes[0, 0].set_title('Content Distribution by Authenticity')
        axes[0, 0].set_ylabel('Number of Posts')
        for i, v in enumerate(counts):
            axes[0, 0].text(i, v + max(counts)*0.01, f'{v:,}', ha='center', fontweight='bold')
        
        # Average engagement scores
        axes[0, 1].bar(labels, avg_scores, color=['#ff6b6b', '#4ecdc4'])
        axes[0, 1].set_title('Average Engagement Score by Authenticity')
        axes[0, 1].set_ylabel('Average Score')
        for i, v in enumerate(avg_scores):
            axes[0, 1].text(i, v + max(avg_scores)*0.01, f'{v:.1f}', ha='center', fontweight='bold')
        
        # Average comments
        axes[1, 0].bar(labels, avg_comments, color=['#ff6b6b', '#4ecdc4'])
        axes[1, 0].set_title('Average Comments by Authenticity')
        axes[1, 0].set_ylabel('Average Comments')
        for i, v in enumerate(avg_comments):
            axes[1, 0].text(i, v + max(avg_comments)*0.01, f'{v:.1f}', ha='center', fontweight='bold')
        
        # Content type distribution by authenticity
        cross_modal = auth_data.get('cross_modal_patterns', {})
        if cross_modal:
            content_types = list(cross_modal.keys())
            fake_counts = [cross_modal[ct].get('fake_posts', 0) for ct in content_types]
            real_counts = [cross_modal[ct].get('real_posts', 0) for ct in content_types]
            
            x = np.arange(len(content_types))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, fake_counts, width, label='Fake', color='#ff6b6b')
            axes[1, 1].bar(x + width/2, real_counts, width, label='Real', color='#4ecdc4')
            axes[1, 1].set_title('Content Type Distribution by Authenticity')
            axes[1, 1].set_ylabel('Number of Posts')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels([ct.replace('_', ' ').title() for ct in content_types], rotation=45)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/authenticity_analysis/authenticity_comparison_overview.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create interactive authenticity comparison
        self._create_interactive_authenticity_chart(engagement_by_label, cross_modal)
        
    def _create_interactive_authenticity_chart(self, engagement_by_label, cross_modal):
        """Create interactive authenticity comparison chart"""
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Content Distribution', 'Engagement Scores', 
                          'Comment Activity', 'Cross-Modal Patterns'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Content distribution
        labels = ['Fake', 'Real']
        counts = [engagement_by_label.get('0', {}).get('count', 0), 
                 engagement_by_label.get('1', {}).get('count', 0)]
        
        fig.add_trace(go.Bar(x=labels, y=counts, name='Posts', 
                           marker_color=['#ff6b6b', '#4ecdc4']), row=1, col=1)
        
        # Engagement scores
        avg_scores = [engagement_by_label.get('0', {}).get('avg_score', 0),
                     engagement_by_label.get('1', {}).get('avg_score', 0)]
        
        fig.add_trace(go.Bar(x=labels, y=avg_scores, name='Avg Score',
                           marker_color=['#ff6b6b', '#4ecdc4']), row=1, col=2)
        
        # Comments
        avg_comments = [engagement_by_label.get('0', {}).get('avg_comments', 0),
                       engagement_by_label.get('1', {}).get('avg_comments', 0)]
        
        fig.add_trace(go.Bar(x=labels, y=avg_comments, name='Avg Comments',
                           marker_color=['#ff6b6b', '#4ecdc4']), row=2, col=1)
        
        # Cross-modal patterns
        if cross_modal:
            content_types = list(cross_modal.keys())
            fake_counts = [cross_modal[ct].get('fake_posts', 0) for ct in content_types]
            real_counts = [cross_modal[ct].get('real_posts', 0) for ct in content_types]
            
            fig.add_trace(go.Bar(x=content_types, y=fake_counts, name='Fake',
                               marker_color='#ff6b6b'), row=2, col=2)
            fig.add_trace(go.Bar(x=content_types, y=real_counts, name='Real',
                               marker_color='#4ecdc4'), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False,
                         title_text="Interactive Authenticity Analysis Dashboard")
        
        fig.write_html('visualizations/interactive/authenticity_dashboard.html')
        
    def create_social_engagement_visualizations(self):
        """Create social engagement pattern visualizations"""
        print("Creating social engagement visualizations...")
        
        if not self.social_analysis:
            return
        
        engagement_data = self.social_analysis.get('engagement_analysis', {})
        engagement_by_type = engagement_data.get('engagement_by_type', {})
        
        if not engagement_by_type:
            return
        
        # Create engagement comparison charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Social Engagement Patterns Across Content Types', fontsize=16, fontweight='bold')
        
        content_types = list(engagement_by_type.keys())
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        # Engagement scores distribution
        scores = [engagement_by_type[ct]['score']['mean'] for ct in content_types]
        axes[0, 0].bar(content_types, scores, color=colors)
        axes[0, 0].set_title('Average Engagement Score by Content Type')
        axes[0, 0].set_ylabel('Average Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Content type distribution
        counts = [engagement_by_type[ct]['count'] for ct in content_types]
        
        axes[0, 1].pie(counts, labels=[ct.replace('_', ' ').title() for ct in content_types], 
                      autopct='%1.1f%%', colors=colors)
        axes[0, 1].set_title('Content Type Distribution')
        
        # Cross-modal engagement comparison
        cross_modal = self.social_analysis.get('authenticity_analysis', {}).get('cross_modal_patterns', {})
        if cross_modal:
            ct_names = list(cross_modal.keys())
            fake_eng = [cross_modal[ct].get('avg_engagement_fake', 0) for ct in ct_names]
            real_eng = [cross_modal[ct].get('avg_engagement_real', 0) for ct in ct_names if not np.isnan(cross_modal[ct].get('avg_engagement_real', np.nan))]
            
            x = np.arange(len(ct_names))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, fake_eng, width, label='Fake', color='#ff6b6b')
            
            # Only plot real engagement for types that have real content
            real_types = [ct for ct in ct_names if not np.isnan(cross_modal[ct].get('avg_engagement_real', np.nan))]
            real_x = [i for i, ct in enumerate(ct_names) if ct in real_types]
            real_values = [cross_modal[ct]['avg_engagement_real'] for ct in real_types]
            
            if real_values:
                axes[1, 0].bar([x + width/2 for x in real_x], real_values, width, label='Real', color='#4ecdc4')
            
            axes[1, 0].set_title('Engagement by Content Type and Authenticity')
            axes[1, 0].set_ylabel('Average Engagement')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([ct.replace('_', ' ').title() for ct in ct_names], rotation=45)
            axes[1, 0].legend()
        
        # Sentiment distribution
        sentiment_data = self.social_analysis.get('sentiment_analysis', {})
        overall_sentiment = sentiment_data.get('overall_sentiment', {})
        if overall_sentiment:
            sentiments = ['Positive', 'Negative', 'Neutral']
            sent_counts = [overall_sentiment.get('positive', 0),
                          overall_sentiment.get('negative', 0),
                          overall_sentiment.get('neutral', 0)]
            sent_colors = ['#2ecc71', '#e74c3c', '#95a5a6']
            
            axes[1, 1].pie(sent_counts, labels=sentiments, autopct='%1.1f%%', colors=sent_colors)
            axes[1, 1].set_title('Overall Sentiment Distribution in Comments')
        
        plt.tight_layout()
        plt.savefig('visualizations/social_engagement/engagement_patterns_overview.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_cross_modal_relationship_visualizations(self):
        """Create visualizations showing text-image-comment interconnections"""
        print("Creating cross-modal relationship visualizations...")
        
        # Create network-style visualization of relationships
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cross-Modal Relationship Analysis', fontsize=16, fontweight='bold')
        
        # Mapping type distribution from social analysis
        if self.social_analysis:
            engagement_data = self.social_analysis.get('engagement_analysis', {})
            mapping_dist = engagement_data.get('mapping_type_distribution', {})
            
            if mapping_dist:
                types = list(mapping_dist.keys())
                counts = list(mapping_dist.values())
                colors = ['#ff9999', '#66b3ff', '#99ff99']
                
                axes[0, 0].bar(types, counts, color=colors)
                axes[0, 0].set_title('Cross-Modal Content Distribution')
                axes[0, 0].set_ylabel('Number of Posts')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Add percentage labels
                total = sum(counts)
                for i, (type_name, count) in enumerate(zip(types, counts)):
                    percentage = (count / total) * 100
                    axes[0, 0].text(i, count + max(counts)*0.01, 
                                   f'{percentage:.1f}%', ha='center', fontweight='bold')
        
        # Cross-modal authenticity patterns
        if self.social_analysis:
            cross_modal = self.social_analysis.get('authenticity_analysis', {}).get('cross_modal_patterns', {})
            if cross_modal:
                content_types = list(cross_modal.keys())
                fake_engagement = [cross_modal[ct].get('avg_engagement_fake', 0) for ct in content_types]
                real_engagement = [cross_modal[ct].get('avg_engagement_real', 0) for ct in content_types if not np.isnan(cross_modal[ct].get('avg_engagement_real', np.nan))]
                
                if len(real_engagement) > 0:
                    x = np.arange(len(content_types))
                    width = 0.35
                    
                    axes[0, 1].bar(x - width/2, fake_engagement, width, label='Fake', color='#ff6b6b')
                    # Only plot real engagement for types that have real content
                    real_types = [ct for ct in content_types if not np.isnan(cross_modal[ct].get('avg_engagement_real', np.nan))]
                    real_x = [i for i, ct in enumerate(content_types) if ct in real_types]
                    real_values = [cross_modal[ct]['avg_engagement_real'] for ct in real_types]
                    
                    axes[0, 1].bar([x + width/2 for x in real_x], real_values, width, label='Real', color='#4ecdc4')
                    axes[0, 1].set_title('Cross-Modal Engagement by Authenticity')
                    axes[0, 1].set_ylabel('Average Engagement')
                    axes[0, 1].set_xticks(x)
                    axes[0, 1].set_xticklabels([ct.replace('_', ' ').title() for ct in content_types], rotation=45)
                    axes[0, 1].legend()
        
        # Image-text mapping success visualization
        if self.image_analysis:
            mapping_stats = {
                'Images with Text Match': int(self.image_analysis.get('images_with_text_match', 0)),
                'Images without Text Match': int(self.image_analysis.get('images_without_text_match', 0))
            }
            
            axes[1, 0].pie(mapping_stats.values(), labels=mapping_stats.keys(), 
                          autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
            axes[1, 0].set_title('Image-Text Mapping Success Rate')
        
        # Content type vs authenticity heatmap
        if self.social_analysis:
            cross_modal = self.social_analysis.get('authenticity_analysis', {}).get('cross_modal_patterns', {})
            if cross_modal:
                # Create data for heatmap
                content_types = list(cross_modal.keys())
                fake_posts = [cross_modal[ct].get('fake_posts', 0) for ct in content_types]
                real_posts = [cross_modal[ct].get('real_posts', 0) for ct in content_types]
                
                # Create matrix for heatmap
                data_matrix = np.array([fake_posts, real_posts])
                
                im = axes[1, 1].imshow(data_matrix, cmap='YlOrRd', aspect='auto')
                axes[1, 1].set_xticks(range(len(content_types)))
                axes[1, 1].set_xticklabels([ct.replace('_', ' ').title() for ct in content_types], rotation=45)
                axes[1, 1].set_yticks([0, 1])
                axes[1, 1].set_yticklabels(['Fake', 'Real'])
                axes[1, 1].set_title('Content Type vs Authenticity Heatmap')
                
                # Add text annotations
                for i in range(2):
                    for j in range(len(content_types)):
                        text = axes[1, 1].text(j, i, f'{data_matrix[i, j]:,}',
                                             ha="center", va="center", color="black", fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('visualizations/interactive/cross_modal_relationships.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create interactive cross-modal visualization
        self._create_interactive_cross_modal_chart()
    
    def _create_interactive_cross_modal_chart(self):
        """Create interactive cross-modal relationship chart"""
        if not self.social_analysis:
            return
        
        cross_modal = self.social_analysis.get('authenticity_analysis', {}).get('cross_modal_patterns', {})
        if not cross_modal:
            return
        
        # Create interactive scatter plot
        content_types = []
        fake_posts = []
        real_posts = []
        fake_engagement = []
        real_engagement = []
        
        for ct, data in cross_modal.items():
            content_types.append(ct.replace('_', ' ').title())
            fake_posts.append(data.get('fake_posts', 0))
            real_posts.append(data.get('real_posts', 0))
            fake_engagement.append(data.get('avg_engagement_fake', 0))
            real_engagement.append(data.get('avg_engagement_real', 0) if not np.isnan(data.get('avg_engagement_real', np.nan)) else 0)
        
        fig = go.Figure()
        
        # Add fake content scatter
        fig.add_trace(go.Scatter(
            x=fake_posts,
            y=fake_engagement,
            mode='markers+text',
            text=content_types,
            textposition="top center",
            marker=dict(size=15, color='#ff6b6b'),
            name='Fake Content'
        ))
        
        # Add real content scatter (only for types with real content)
        real_indices = [i for i, val in enumerate(real_posts) if val > 0]
        if real_indices:
            fig.add_trace(go.Scatter(
                x=[real_posts[i] for i in real_indices],
                y=[real_engagement[i] for i in real_indices],
                mode='markers+text',
                text=[content_types[i] for i in real_indices],
                textposition="bottom center",
                marker=dict(size=15, color='#4ecdc4'),
                name='Real Content'
            ))
        
        fig.update_layout(
            title='Cross-Modal Content: Posts vs Engagement by Authenticity',
            xaxis_title='Number of Posts',
            yaxis_title='Average Engagement',
            height=600
        )
        
        fig.write_html('visualizations/interactive/cross_modal_scatter.html')
    
    def export_visualizations_multiple_formats(self):
        """Export all visualizations in multiple formats for dashboard integration"""
        print("Exporting visualizations in multiple formats...")
        
        # Create summary statistics for dashboard
        summary_stats = {
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'total_visualizations_created': 0,
            'authenticity_charts': [],
            'social_engagement_charts': [],
            'multimodal_feature_charts': [],
            'interactive_components': []
        }
        
        # Count created visualizations
        viz_dirs = ['authenticity_analysis', 'social_engagement', 'multimodal_features', 'interactive']
        for viz_dir in viz_dirs:
            viz_path = Path(f'visualizations/{viz_dir}')
            if viz_path.exists():
                files = list(viz_path.glob('*'))
                summary_stats[f'{viz_dir}_charts'] = [f.name for f in files]
                summary_stats['total_visualizations_created'] += len(files)
        
        # Save summary
        with open('visualizations/visualization_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Created {summary_stats['total_visualizations_created']} visualizations")
        
    def run_comprehensive_pipeline(self):
        """Run the complete visualization pipeline"""
        print("Starting Comprehensive Visualization Pipeline...")
        print("=" * 60)
        
        try:
            # Create authenticity comparison charts
            self.create_authenticity_comparison_charts()
            
            # Create social engagement visualizations
            self.create_social_engagement_visualizations()
            
            # Create cross-modal relationship visualizations
            self.create_cross_modal_relationship_visualizations()
            
            # Export in multiple formats
            self.export_visualizations_multiple_formats()
            
            print("=" * 60)
            print("Comprehensive Visualization Pipeline completed successfully!")
            print("\nGenerated visualizations:")
            print("- Authenticity comparison charts")
            print("- Social engagement pattern visualizations")
            print("- Cross-modal relationship visualizations")
            print("- Interactive dashboard components")
            
        except Exception as e:
            print(f"Error in visualization pipeline: {str(e)}")
            raise

def main():
    """Main execution function"""
    pipeline = ComprehensiveVisualizationPipeline()
    pipeline.run_comprehensive_pipeline()

if __name__ == "__main__":
    main()