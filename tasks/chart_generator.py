#!/usr/bin/env python3
"""
Dashboard Chart Generator
Creates optimized charts and plots for Streamlit dashboard display
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

class DashboardChartGenerator:
    """Generates optimized charts for dashboard display"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.analysis_dir = self.base_dir / "analysis_results"
        self.viz_dir = self.base_dir / "visualizations" / "dashboard_charts"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def generate_content_type_charts(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content type distribution charts"""
        try:
            overview = dashboard_data.get("dataset_overview", {})
            content_dist = overview.get("content_type_distribution", {})
            
            if not content_dist:
                return {}
            
            # Pie chart for content distribution
            labels = []
            values = []
            colors = ['#2E8B57', '#FF6347', '#4682B4']
            
            for content_type, count in content_dist.items():
                if content_type == "text_image":
                    labels.append("Text + Image")
                elif content_type == "full_multimodal":
                    labels.append("Full Multimodal")
                else:
                    labels.append("Text Only")
                values.append(count)
            
            pie_fig = px.pie(
                values=values,
                names=labels,
                title="Content Type Distribution",
                color_discrete_sequence=colors
            )
            pie_fig.update_traces(textposition='inside', textinfo='percent+label')
            
            # Bar chart for content counts
            bar_fig = px.bar(
                x=labels,
                y=values,
                title="Content Type Counts",
                color=labels,
                color_discrete_sequence=colors
            )
            bar_fig.update_layout(showlegend=False)
            
            return {
                "content_pie_chart": pie_fig.to_json(),
                "content_bar_chart": bar_fig.to_json()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating content type charts: {e}")
            return {}
    
    def generate_authenticity_charts(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate authenticity analysis charts"""
        try:
            overview = dashboard_data.get("dataset_overview", {})
            auth_dist = overview.get("authenticity_distribution", {})
            
            if not auth_dist:
                return {}
            
            fake_count = auth_dist.get("fake", 0)
            real_count = auth_dist.get("real", 0)
            
            # Authenticity pie chart
            auth_pie = px.pie(
                values=[fake_count, real_count],
                names=["Fake Content", "Real Content"],
                title="Authenticity Distribution",
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
            
            # Authenticity bar chart
            auth_bar = px.bar(
                x=["Fake Content", "Real Content"],
                y=[fake_count, real_count],
                title="Authenticity Counts",
                color=["Fake Content", "Real Content"],
                color_discrete_map={
                    "Fake Content": "#FF6B6B",
                    "Real Content": "#4ECDC4"
                }
            )
            auth_bar.update_layout(showlegend=False)
            
            return {
                "authenticity_pie_chart": auth_pie.to_json(),
                "authenticity_bar_chart": auth_bar.to_json()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating authenticity charts: {e}")
            return {}
    
    def generate_engagement_charts(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate social engagement charts"""
        try:
            social_data = dashboard_data.get("social_analysis", {})
            engagement_by_type = social_data.get("engagement_by_type", {})
            
            if not engagement_by_type:
                return {}
            
            # Prepare data for engagement charts
            content_types = []
            avg_scores = []
            avg_comments = []
            post_counts = []
            
            for content_type, stats in engagement_by_type.items():
                if content_type == "text_image":
                    display_name = "Text + Image"
                elif content_type == "full_multimodal":
                    display_name = "Full Multimodal"
                else:
                    display_name = "Text Only"
                
                content_types.append(display_name)
                avg_scores.append(stats.get("score", {}).get("mean", 0))
                avg_comments.append(stats.get("num_comments", {}).get("mean", 0))
                post_counts.append(stats.get("count", 0))
            
            # Average engagement score chart
            score_fig = px.bar(
                x=content_types,
                y=avg_scores,
                title="Average Engagement Score by Content Type",
                color=content_types,
                color_discrete_sequence=['#2E8B57', '#FF6347', '#4682B4']
            )
            score_fig.update_layout(showlegend=False)
            
            # Average comments chart
            comments_fig = px.bar(
                x=content_types,
                y=avg_comments,
                title="Average Comments by Content Type",
                color=content_types,
                color_discrete_sequence=['#2E8B57', '#FF6347', '#4682B4']
            )
            comments_fig.update_layout(showlegend=False)
            
            # Combined engagement chart
            combined_fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Engagement Score", "Comments"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            combined_fig.add_trace(
                go.Bar(x=content_types, y=avg_scores, name="Avg Score", marker_color='#2E8B57'),
                row=1, col=1
            )
            
            combined_fig.add_trace(
                go.Bar(x=content_types, y=avg_comments, name="Avg Comments", marker_color='#FF6347'),
                row=1, col=2
            )
            
            combined_fig.update_layout(title_text="Engagement Metrics by Content Type")
            
            return {
                "engagement_score_chart": score_fig.to_json(),
                "engagement_comments_chart": comments_fig.to_json(),
                "engagement_combined_chart": combined_fig.to_json()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating engagement charts: {e}")
            return {}
    
    def generate_cross_modal_charts(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cross-modal analysis charts"""
        try:
            cross_modal_data = dashboard_data.get("cross_modal_analysis", {})
            cross_modal_auth = cross_modal_data.get("cross_modal_authenticity", {})
            
            if not cross_modal_auth:
                return {}
            
            # Cross-modal authenticity stacked bar chart
            content_types = []
            fake_percentages = []
            real_percentages = []
            
            for content_type, data in cross_modal_auth.items():
                if data.get("total_posts", 0) > 0:
                    if content_type == "text_image":
                        display_name = "Text + Image"
                    elif content_type == "full_multimodal":
                        display_name = "Full Multimodal"
                    else:
                        display_name = "Text Only"
                    
                    content_types.append(display_name)
                    total = data["total_posts"]
                    fake_pct = (data.get("fake_posts", 0) / total) * 100
                    real_pct = (data.get("real_posts", 0) / total) * 100
                    
                    fake_percentages.append(fake_pct)
                    real_percentages.append(real_pct)
            
            if content_types:
                stacked_fig = go.Figure()
                stacked_fig.add_trace(go.Bar(
                    name='Fake Content',
                    x=content_types,
                    y=fake_percentages,
                    marker_color='#FF6B6B'
                ))
                stacked_fig.add_trace(go.Bar(
                    name='Real Content',
                    x=content_types,
                    y=real_percentages,
                    marker_color='#4ECDC4'
                ))
                
                stacked_fig.update_layout(
                    title="Authenticity Distribution by Content Type (%)",
                    barmode='stack',
                    yaxis_title="Percentage"
                )
                
                return {
                    "cross_modal_authenticity_chart": stacked_fig.to_json()
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error generating cross-modal charts: {e}")
            return {}
    
    def generate_sentiment_charts(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sentiment analysis charts"""
        try:
            social_data = dashboard_data.get("social_analysis", {})
            sentiment_data = social_data.get("sentiment_analysis", {})
            overall_sentiment = sentiment_data.get("overall_sentiment", {})
            
            if not overall_sentiment:
                return {}
            
            # Sentiment pie chart
            sentiments = ["Positive", "Negative", "Neutral"]
            counts = [
                overall_sentiment.get("positive", 0),
                overall_sentiment.get("negative", 0),
                overall_sentiment.get("neutral", 0)
            ]
            
            sentiment_pie = px.pie(
                values=counts,
                names=sentiments,
                title="Comment Sentiment Distribution",
                color_discrete_sequence=['#4ECDC4', '#FF6B6B', '#95A5A6']
            )
            
            return {
                "sentiment_pie_chart": sentiment_pie.to_json()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating sentiment charts: {e}")
            return {}
    
    def generate_all_charts(self) -> bool:
        """Generate all dashboard charts and save to files"""
        try:
            # Load dashboard data
            dashboard_data_path = self.analysis_dir / "dashboard_data" / "processed_dashboard_data.json"
            with open(dashboard_data_path, 'r') as f:
                dashboard_data = json.load(f)
            
            # Generate all chart types
            all_charts = {}
            
            all_charts.update(self.generate_content_type_charts(dashboard_data))
            all_charts.update(self.generate_authenticity_charts(dashboard_data))
            all_charts.update(self.generate_engagement_charts(dashboard_data))
            all_charts.update(self.generate_cross_modal_charts(dashboard_data))
            all_charts.update(self.generate_sentiment_charts(dashboard_data))
            
            # Save charts to file
            charts_output_path = self.viz_dir / "dashboard_charts.json"
            with open(charts_output_path, 'w') as f:
                json.dump(all_charts, f, indent=2)
            
            self.logger.info(f"Generated {len(all_charts)} charts saved to {charts_output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating all charts: {e}")
            return False

if __name__ == "__main__":
    generator = DashboardChartGenerator()
    success = generator.generate_all_charts()
    print(f"Chart generation: {'Success' if success else 'Failed'}")