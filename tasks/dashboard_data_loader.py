#!/usr/bin/env python3
"""
Dashboard Data Loader
Processes analysis results for optimized dashboard consumption
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

class DashboardDataLoader:
    """Loads and processes data from completed analysis tasks for dashboard display"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.analysis_dir = self.base_dir / "analysis_results"
        self.processed_dir = self.base_dir / "processed_data"
        self.logger = logging.getLogger(__name__)
        
    def load_dataset_overview(self) -> Dict[str, Any]:
        """Load comprehensive dataset overview statistics"""
        try:
            # Load text integration analysis
            text_analysis_path = self.analysis_dir / "text_integration" / "text_integration_analysis.json"
            with open(text_analysis_path, 'r') as f:
                text_data = json.load(f)
            
            # Load image mapping analysis
            image_analysis_path = self.analysis_dir / "image_catalog" / "id_mapping_analysis.json"
            with open(image_analysis_path, 'r') as f:
                image_data = json.load(f)
            
            # Load social engagement analysis
            social_analysis_path = self.analysis_dir / "social_analysis" / "social_engagement_analysis.json"
            with open(social_analysis_path, 'r') as f:
                social_data = json.load(f)
            
            return {
                "total_text_records": text_data["dataset_overview"]["total_records"],
                "total_images": image_data["total_images"],
                "mapping_success_rate": image_data["mapping_success_rate"],
                "content_type_distribution": social_data["engagement_analysis"]["mapping_type_distribution"],
                "authenticity_distribution": {
                    "fake": social_data["authenticity_analysis"]["engagement_by_label"]["0"]["count"],
                    "real": social_data["authenticity_analysis"]["engagement_by_label"]["1"]["count"]
                },
                "text_quality": text_data["text_quality_metrics"],
                "missing_data": text_data["missing_data_analysis"]
            }
        except Exception as e:
            self.logger.error(f"Error loading dataset overview: {e}")
            return {}
    
    def load_social_analysis_data(self) -> Dict[str, Any]:
        """Load social engagement and comment analysis data"""
        try:
            # Load main social analysis
            social_analysis_path = self.analysis_dir / "social_analysis" / "social_engagement_analysis.json"
            with open(social_analysis_path, 'r') as f:
                social_data = json.load(f)
            
            # Load processed comment data
            comments_path = self.processed_dir / "social_engagement" / "comments_with_sentiment.parquet"
            if comments_path.exists():
                comments_df = pd.read_parquet(comments_path)
                comment_stats = {
                    "total_comments": len(comments_df),
                    "avg_sentiment": comments_df['sentiment_polarity'].mean() if 'sentiment_polarity' in comments_df.columns else 0,
                    "sentiment_distribution": comments_df['sentiment_polarity'].describe().to_dict() if 'sentiment_polarity' in comments_df.columns else {}
                }
            else:
                comment_stats = {}
            
            return {
                "engagement_by_type": social_data["engagement_analysis"]["engagement_by_type"],
                "authenticity_patterns": social_data["authenticity_analysis"],
                "sentiment_analysis": social_data["sentiment_analysis"],
                "comment_statistics": comment_stats,
                "statistical_comparisons": social_data["engagement_analysis"]["statistical_comparisons"]
            }
        except Exception as e:
            self.logger.error(f"Error loading social analysis data: {e}")
            return {}
    
    def load_cross_modal_data(self) -> Dict[str, Any]:
        """Load cross-modal relationship and consistency data"""
        try:
            # Load image catalog for cross-modal analysis
            image_catalog_path = self.analysis_dir / "image_catalog" / "comprehensive_image_catalog.parquet"
            if image_catalog_path.exists():
                catalog_df = pd.read_parquet(image_catalog_path)
                
                # Calculate cross-modal statistics
                content_type_stats = catalog_df['content_type'].value_counts().to_dict()
                
                # Load social data for cross-modal patterns
                social_analysis_path = self.analysis_dir / "social_analysis" / "social_engagement_analysis.json"
                with open(social_analysis_path, 'r') as f:
                    social_data = json.load(f)
                
                cross_modal_patterns = social_data["authenticity_analysis"]["cross_modal_patterns"]
                
                return {
                    "content_type_distribution": content_type_stats,
                    "cross_modal_authenticity": cross_modal_patterns,
                    "multimodal_consistency": self._calculate_consistency_metrics(catalog_df, social_data),
                    "mapping_relationships": self._analyze_mapping_relationships(catalog_df)
                }
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error loading cross-modal data: {e}")
            return {}
    
    def _calculate_consistency_metrics(self, catalog_df: pd.DataFrame, social_data: Dict) -> Dict[str, Any]:
        """Calculate multimodal consistency metrics"""
        try:
            # Analyze consistency between text and image authenticity
            consistency_metrics = {}
            
            # Get authenticity patterns by content type
            cross_modal = social_data["authenticity_analysis"]["cross_modal_patterns"]
            
            for content_type, data in cross_modal.items():
                if data["total_posts"] > 0:
                    fake_ratio = data["fake_posts"] / data["total_posts"]
                    consistency_metrics[content_type] = {
                        "total_posts": data["total_posts"],
                        "fake_ratio": fake_ratio,
                        "real_ratio": 1 - fake_ratio,
                        "avg_engagement_fake": data.get("avg_engagement_fake", 0),
                        "avg_engagement_real": data.get("avg_engagement_real", 0)
                    }
            
            return consistency_metrics
        except Exception as e:
            self.logger.error(f"Error calculating consistency metrics: {e}")
            return {}
    
    def _analyze_mapping_relationships(self, catalog_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze ID mapping relationships and patterns"""
        try:
            mapping_stats = {}
            
            # Content type distribution
            content_types = catalog_df['content_type'].value_counts()
            mapping_stats["content_distribution"] = content_types.to_dict()
            
            # Mapping success patterns
            total_images = len(catalog_df)
            multimodal_count = len(catalog_df[catalog_df['content_type'] == 'multimodal'])
            image_only_count = len(catalog_df[catalog_df['content_type'] == 'image_only'])
            
            mapping_stats["mapping_success"] = {
                "total_images": total_images,
                "multimodal_images": multimodal_count,
                "image_only": image_only_count,
                "mapping_rate": (multimodal_count / total_images) * 100 if total_images > 0 else 0
            }
            
            return mapping_stats
        except Exception as e:
            self.logger.error(f"Error analyzing mapping relationships: {e}")
            return {}
    
    def export_dashboard_data(self) -> bool:
        """Export all processed data for dashboard consumption"""
        try:
            dashboard_data = {
                "dataset_overview": self.load_dataset_overview(),
                "social_analysis": self.load_social_analysis_data(),
                "cross_modal_analysis": self.load_cross_modal_data(),
                "generation_timestamp": pd.Timestamp.now().isoformat()
            }
            
            # Save to dashboard data directory
            output_path = self.analysis_dir / "dashboard_data" / "processed_dashboard_data.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            self.logger.info(f"Dashboard data exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting dashboard data: {e}")
            return False

if __name__ == "__main__":
    loader = DashboardDataLoader()
    success = loader.export_dashboard_data()
    print(f"Dashboard data export: {'Success' if success else 'Failed'}")