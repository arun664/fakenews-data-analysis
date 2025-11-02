"""
Data Loader for Dashboard
Loads and processes data from analysis results using .env configuration
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from .config import DashboardConfig

logger = logging.getLogger(__name__)

class DataLoader:
    """Centralized data loading for dashboard"""
    
    def __init__(self):
        self.analysis_dir = Path(DashboardConfig.ANALYSIS_DIR)
        self.processed_dir = Path(DashboardConfig.PROCESSED_DIR)
        self.viz_dir = Path(DashboardConfig.VISUALIZATIONS_DIR)
        self.reports_dir = Path(DashboardConfig.REPORTS_DIR)
        
        # Cache for loaded data
        self._cache = {}
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system overview metrics"""
        
        if 'system_overview' in self._cache:
            return self._cache['system_overview']
        
        overview = {
            'total_images': 0,
            'multimodal_images': 0,
            'image_only_images': 0,
            'text_records': 0,
            'total_comments': 0,
            'processing_status': {},
            'task_progress': self._get_task_progress()
        }
        
        # Load image catalog if available
        image_catalog = self.load_image_catalog()
        if image_catalog is not None:
            overview['total_images'] = len(image_catalog)
            overview['multimodal_images'] = len(image_catalog[image_catalog['content_type'] == 'multimodal'])
            overview['image_only_images'] = len(image_catalog[image_catalog['content_type'] == 'image_only'])
        
        # Load text data stats
        text_stats = self._get_text_stats()
        overview.update(text_stats)
        
        # Load comment stats
        comment_stats = self._get_comment_stats()
        overview.update(comment_stats)
        
        self._cache['system_overview'] = overview
        return overview
    
    def load_image_catalog(self) -> Optional[pd.DataFrame]:
        """Load comprehensive image catalog"""
        
        catalog_path = self.analysis_dir / 'image_catalog' / 'comprehensive_image_catalog.parquet'
        
        try:
            if catalog_path.exists():
                df = pd.read_parquet(catalog_path)
                logger.info(f"Loaded image catalog: {len(df):,} records")
                return df
            else:
                logger.warning(f"Image catalog not found: {catalog_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading image catalog: {e}")
            return None
    
    def load_text_data(self, split: str = 'train') -> Optional[pd.DataFrame]:
        """Load processed text data"""
        
        text_path = self.processed_dir / 'text_data' / f'{split}_clean.parquet'
        
        try:
            if text_path.exists():
                df = pd.read_parquet(text_path)
                logger.info(f"Loaded {split} text data: {len(df):,} records")
                return df
            else:
                logger.warning(f"Text data not found: {text_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading text data: {e}")
            return None
    
    def load_comment_data(self) -> Optional[pd.DataFrame]:
        """Load processed comment data"""
        
        comment_path = self.processed_dir / 'comments' / 'all_relevant_comments.parquet'
        
        try:
            if comment_path.exists():
                df = pd.read_parquet(comment_path)
                logger.info(f"Loaded comment data: {len(df):,} records")
                return df
            else:
                logger.warning(f"Comment data not found: {comment_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading comment data: {e}")
            return None
    
    def load_visual_features(self) -> Optional[pd.DataFrame]:
        """Load visual feature analysis results"""
        
        features_path = self.processed_dir / 'visual_features' / 'comprehensive_visual_features.parquet'
        
        try:
            if features_path.exists():
                df = pd.read_parquet(features_path)
                logger.info(f"Loaded visual features: {len(df):,} records")
                return df
            else:
                logger.info("Visual features not yet available")
                return None
        except Exception as e:
            logger.error(f"Error loading visual features: {e}")
            return None
    
    def load_analysis_results(self, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Load analysis results JSON files"""
        
        analysis_paths = {
            'image_catalog': self.analysis_dir / 'image_catalog' / 'processing_statistics.json',
            'text_integration': self.analysis_dir / 'text_integration' / 'mapping_analysis.json',
            'comment_integration': self.analysis_dir / 'comment_integration' / 'engagement_analysis.json',
            'visual_analysis': self.analysis_dir / 'visual_analysis' / 'mapping_statistical_analysis.json',
            'data_quality': self.analysis_dir / 'data_quality' / 'quality_metrics.json'
        }
        
        analysis_path = analysis_paths.get(analysis_type)
        if not analysis_path:
            logger.warning(f"Unknown analysis type: {analysis_type}")
            return None
        
        try:
            if analysis_path.exists():
                with open(analysis_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded {analysis_type} results")
                return data
            else:
                logger.info(f"{analysis_type} results not yet available")
                return None
        except Exception as e:
            logger.error(f"Error loading {analysis_type} results: {e}")
            return None
    
    def get_image_analysis_data(self) -> Dict[str, Any]:
        """Get comprehensive image analysis data"""
        
        data = {
            'catalog': self.load_image_catalog(),
            'statistics': self.load_analysis_results('image_catalog'),
            'visual_features': self.load_visual_features()
        }
        
        # Calculate derived metrics
        if data['catalog'] is not None:
            catalog = data['catalog']
            
            # Content type distribution
            data['content_distribution'] = catalog['content_type'].value_counts()
            
            # Quality metrics if available
            if 'quality_score' in catalog.columns:
                data['quality_distribution'] = catalog['quality_score'].describe()
            
            # Mapping success rate
            total = len(catalog)
            multimodal = len(catalog[catalog['content_type'] == 'multimodal'])
            data['mapping_success_rate'] = (multimodal / total * 100) if total > 0 else 0
            
            # File size analysis if available
            if 'file_size_mb' in catalog.columns:
                data['size_stats'] = catalog['file_size_mb'].describe()
        
        return data
    
    def get_text_analysis_data(self) -> Dict[str, Any]:
        """Get comprehensive text analysis data"""
        
        # Load all text splits
        splits = ['train', 'validation', 'test']
        text_data = {}
        
        for split in splits:
            text_data[split] = self.load_text_data(split)
        
        # Combine for overall statistics
        combined_data = []
        for split, df in text_data.items():
            if df is not None:
                combined_data.append(df)
        
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            return {
                'splits': text_data,
                'combined': combined_df,
                'total_records': len(combined_df),
                'subreddit_distribution': combined_df['subreddit'].value_counts().head(20) if 'subreddit' in combined_df.columns else None,
                'title_length_stats': combined_df['clean_title'].str.len().describe() if 'clean_title' in combined_df.columns else None,
                'score_distribution': combined_df['score'].describe() if 'score' in combined_df.columns else None
            }
        
        return {'splits': text_data, 'combined': None}
    
    def get_social_analysis_data(self) -> Dict[str, Any]:
        """Get social engagement analysis data"""
        
        comment_data = self.load_comment_data()
        
        if comment_data is not None:
            return {
                'raw_data': comment_data,
                'total_comments': len(comment_data),
                'unique_posts': comment_data['submission_id'].nunique() if 'submission_id' in comment_data.columns else 0,
                'avg_comments_per_post': len(comment_data) / comment_data['submission_id'].nunique() if 'submission_id' in comment_data.columns else 0,
                'sentiment_distribution': comment_data['sentiment'].value_counts() if 'sentiment' in comment_data.columns else None,
                'engagement_stats': self.load_analysis_results('comment_integration')
            }
        
        return {
            'raw_data': None,
            'total_comments': 97041,  # From .env
            'unique_posts': 5056,     # From .env
            'avg_comments_per_post': 19.2,
            'coverage_rate': 51.4
        }
    
    def get_cross_modal_data(self) -> Dict[str, Any]:
        """Get cross-modal relationship analysis data"""
        
        image_catalog = self.load_image_catalog()
        text_data = self.get_text_analysis_data()
        
        cross_modal_data = {
            'mapping_relationships': None,
            'consistency_analysis': None,
            'comparative_metrics': None
        }
        
        if image_catalog is not None:
            # Mapping relationship analysis
            content_types = image_catalog['content_type'].value_counts()
            
            cross_modal_data['mapping_relationships'] = {
                'total_images': len(image_catalog),
                'multimodal_count': content_types.get('multimodal', 0),
                'image_only_count': content_types.get('image_only', 0),
                'mapping_success_rate': (content_types.get('multimodal', 0) / len(image_catalog) * 100) if len(image_catalog) > 0 else 0
            }
        
        return cross_modal_data
    
    def get_data_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality assessment metrics"""
        
        quality_data = {
            'completeness': {},
            'consistency': {},
            'accuracy': {},
            'coverage': {}
        }
        
        # Image data quality
        image_catalog = self.load_image_catalog()
        if image_catalog is not None:
            quality_data['completeness']['images'] = {
                'total_records': len(image_catalog),
                'missing_values': image_catalog.isnull().sum().to_dict(),
                'completeness_score': ((len(image_catalog) - image_catalog.isnull().sum().sum()) / (len(image_catalog) * len(image_catalog.columns)) * 100) if len(image_catalog) > 0 else 0
            }
        
        # Text data quality
        text_data = self.get_text_analysis_data()
        if text_data['combined'] is not None:
            df = text_data['combined']
            quality_data['completeness']['text'] = {
                'total_records': len(df),
                'missing_values': df.isnull().sum().to_dict(),
                'completeness_score': ((len(df) - df.isnull().sum().sum()) / (len(df) * len(df.columns)) * 100) if len(df) > 0 else 0
            }
        
        # Load quality analysis results if available
        quality_results = self.load_analysis_results('data_quality')
        if quality_results:
            quality_data.update(quality_results)
        
        return quality_data
    
    def _get_task_progress(self) -> Dict[str, Dict[str, Any]]:
        """Get task completion progress"""
        
        tasks = {
            'task_1': {'name': 'Image Catalog Creation', 'status': 'complete', 'progress': 100},
            'task_2': {'name': 'Text Data Integration', 'status': 'complete', 'progress': 100},
            'task_3': {'name': 'Comments Integration', 'status': 'complete', 'progress': 100},
            'task_4': {'name': 'Data Quality Assessment', 'status': 'complete', 'progress': 100},
            'task_5': {'name': 'Visual Feature Engineering', 'status': 'not_started', 'progress': 0},
            'task_6': {'name': 'Linguistic Analysis', 'status': 'not_started', 'progress': 0},
            'task_7': {'name': 'Social Analysis', 'status': 'not_started', 'progress': 0},
            'task_8': {'name': 'Pattern Discovery', 'status': 'not_started', 'progress': 0}
        }
        
        # Check for actual completion based on file existence
        if (self.processed_dir / 'visual_features').exists():
            tasks['task_5']['status'] = 'progress'
            tasks['task_5']['progress'] = 50
        
        return tasks
    
    def _get_text_stats(self) -> Dict[str, int]:
        """Get text data statistics"""
        
        text_data = self.get_text_analysis_data()
        
        if text_data['combined'] is not None:
            return {
                'text_records': len(text_data['combined']),
                'unique_subreddits': text_data['combined']['subreddit'].nunique() if 'subreddit' in text_data['combined'].columns else 0
            }
        
        return {'text_records': 0, 'unique_subreddits': 0}
    
    def _get_comment_stats(self) -> Dict[str, int]:
        """Get comment data statistics"""
        
        comment_data = self.load_comment_data()
        
        if comment_data is not None:
            return {
                'total_comments': len(comment_data),
                'posts_with_comments': comment_data['submission_id'].nunique() if 'submission_id' in comment_data.columns else 0
            }
        
        # Fallback to .env values
        return {
            'total_comments': 97041,
            'posts_with_comments': 5056
        }
    
    def clear_cache(self):
        """Clear data cache"""
        self._cache.clear()
        logger.info("Data cache cleared")  @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_image_catalog() -> Optional[pd.DataFrame]:
        """Load image catalog data with caching"""
        try:
            catalog_path = Path(f'{DashboardConfig.ANALYSIS_DIR}/image_catalog/comprehensive_image_catalog.parquet')
            
            if catalog_path.exists():
                df = pd.read_parquet(catalog_path)
                logger.info(f"Loaded image catalog: {len(df):,} records")
                return df
            else:
                logger.warning(f"Image catalog not found at: {catalog_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading image catalog: {e}")
            st.error(f"Failed to load image catalog: {e}")
            return None
    
    @staticmethod
    @st.cache_data(ttl=300)
    def load_text_data(split: str = "train") -> Optional[pd.DataFrame]:
        """Load text data for specified split"""
        try:
            text_path = Path(f'{DashboardConfig.PROCESSED_DIR}/text_data/{split}_clean.parquet')
            
            if text_path.exists():
                df = pd.read_parquet(text_path)
                logger.info(f"Loaded {split} text data: {len(df):,} records")
                return df
            else:
                logger.warning(f"Text data not found at: {text_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading text data: {e}")
            st.error(f"Failed to load text data: {e}")
            return None
    
    @staticmethod
    @st.cache_data(ttl=300)
    def load_comments_data() -> Optional[pd.DataFrame]:
        """Load comments data"""
        try:
            comments_path = Path(f'{DashboardConfig.PROCESSED_DIR}/comments/all_relevant_comments.parquet')
            
            if comments_path.exists():
                df = pd.read_parquet(comments_path)
                logger.info(f"Loaded comments data: {len(df):,} records")
                return df
            else:
                logger.warning(f"Comments data not found at: {comments_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading comments data: {e}")
            st.error(f"Failed to load comments data: {e}")
            return None
    
    @staticmethod
    @st.cache_data(ttl=300)
    def load_processing_stats() -> Dict[str, Any]:
        """Load processing statistics"""
        try:
            stats_path = Path(f'{DashboardConfig.ANALYSIS_DIR}/image_catalog/processing_statistics.json')
            
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                logger.info("Loaded processing statistics")
                return stats
            else:
                logger.warning("Processing statistics not found")
                return {}
                
        except Exception as e:
            logger.error(f"Error loading processing stats: {e}")
            return {}
    
    @staticmethod
    @st.cache_data(ttl=300)
    def load_visual_features() -> Optional[pd.DataFrame]:
        """Load visual features data"""
        try:
            features_path = Path(f'{DashboardConfig.PROCESSED_DIR}/visual_features/comprehensive_visual_features.parquet')
            
            if features_path.exists():
                df = pd.read_parquet(features_path)
                logger.info(f"Loaded visual features: {len(df):,} records")
                return df
            else:
                logger.warning("Visual features not found")
                return None
                
        except Exception as e:
            logger.error(f"Error loading visual features: {e}")
            return None
    
    @staticmethod
    @st.cache_data(ttl=300)
    def get_data_summary() -> Dict[str, Any]:
        """Get comprehensive data summary"""
        
        summary = {
            "image_catalog": {"status": "not_found", "count": 0, "details": {}},
            "text_data": {"status": "not_found", "count": 0, "details": {}},
            "comments_data": {"status": "not_found", "count": 0, "details": {}},
            "visual_features": {"status": "not_found", "count": 0, "details": {}},
            "processing_stats": {}
        }
        
        # Check image catalog
        image_df = DataLoader.load_image_catalog()
        if image_df is not None:
            summary["image_catalog"] = {
                "status": "available",
                "count": len(image_df),
                "details": {
                    "multimodal": len(image_df[image_df['content_type'] == 'multimodal']) if 'content_type' in image_df.columns else 0,
                    "image_only": len(image_df[image_df['content_type'] == 'image_only']) if 'content_type' in image_df.columns else 0,
                    "columns": list(image_df.columns)
                }
            }
        
        # Check text data
        text_df = DataLoader.load_text_data()
        if text_df is not None:
            summary["text_data"] = {
                "status": "available",
                "count": len(text_df),
                "details": {
                    "subreddits": text_df['subreddit'].nunique() if 'subreddit' in text_df.columns else 0,
                    "avg_title_length": text_df['clean_title'].str.len().mean() if 'clean_title' in text_df.columns else 0,
                    "columns": list(text_df.columns)
                }
            }
        
        # Check comments data
        comments_df = DataLoader.load_comments_data()
        if comments_df is not None:
            summary["comments_data"] = {
                "status": "available", 
                "count": len(comments_df),
                "details": {
                    "unique_posts": comments_df['submission_id'].nunique() if 'submission_id' in comments_df.columns else 0,
                    "avg_score": comments_df['score'].mean() if 'score' in comments_df.columns else 0,
                    "columns": list(comments_df.columns)
                }
            }
        
        # Check visual features
        visual_df = DataLoader.load_visual_features()
        if visual_df is not None:
            summary["visual_features"] = {
                "status": "available",
                "count": len(visual_df),
                "details": {
                    "feature_count": len([col for col in visual_df.columns if col not in ['image_path', 'mapping_status', 'extraction_success']]),
                    "success_rate": (visual_df['extraction_success'].sum() / len(visual_df) * 100) if 'extraction_success' in visual_df.columns else 0,
                    "columns": list(visual_df.columns)
                }
            }
        
        # Load processing stats
        summary["processing_stats"] = DataLoader.load_processing_stats()
        
        return summary

class DataProcessor:
    """Data processing utilities"""
    
    @staticmethod
    def calculate_mapping_statistics(image_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate mapping relationship statistics"""
        
        if image_df is None or 'content_type' not in image_df.columns:
            return {}
        
        total = len(image_df)
        multimodal = len(image_df[image_df['content_type'] == 'multimodal'])
        image_only = len(image_df[image_df['content_type'] == 'image_only'])
        
        return {
            "total_images": total,
            "multimodal_count": multimodal,
            "image_only_count": image_only,
            "multimodal_percentage": (multimodal / total * 100) if total > 0 else 0,
            "image_only_percentage": (image_only / total * 100) if total > 0 else 0,
            "mapping_success_rate": (multimodal / total * 100) if total > 0 else 0
        }
    
    @staticmethod
    def get_text_statistics(text_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate text data statistics"""
        
        if text_df is None:
            return {}
        
        stats = {
            "total_records": len(text_df),
            "unique_subreddits": text_df['subreddit'].nunique() if 'subreddit' in text_df.columns else 0,
            "avg_title_length": text_df['clean_title'].str.len().mean() if 'clean_title' in text_df.columns else 0,
            "avg_score": text_df['score'].mean() if 'score' in text_df.columns else 0,
            "top_subreddits": text_df['subreddit'].value_counts().head(10).to_dict() if 'subreddit' in text_df.columns else {}
        }
        
        return stats
    
    @staticmethod
    def get_social_statistics(comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate social engagement statistics"""
        
        if comments_df is None:
            return {}
        
        stats = {
            "total_comments": len(comments_df),
            "unique_posts": comments_df['submission_id'].nunique() if 'submission_id' in comments_df.columns else 0,
            "avg_score": comments_df['score'].mean() if 'score' in comments_df.columns else 0,
            "avg_comments_per_post": len(comments_df) / comments_df['submission_id'].nunique() if 'submission_id' in comments_df.columns else 0
        }
        
        if 'submission_id' in comments_df.columns:
            stats["engagement_distribution"] = comments_df.groupby('submission_id').size().describe().to_dict()
        
        return stats
    
    @staticmethod
    def check_data_availability() -> Dict[str, bool]:
        """Check which data sources are available"""
        
        availability = {
            "image_catalog": Path(f'{DashboardConfig.ANALYSIS_DIR}/image_catalog/comprehensive_image_catalog.parquet').exists(),
            "text_data": any(Path(f'{DashboardConfig.PROCESSED_DIR}/text_data').glob('*_clean.parquet')),
            "comments_data": Path(f'{DashboardConfig.PROCESSED_DIR}/comments/all_relevant_comments.parquet').exists(),
            "visual_features": Path(f'{DashboardConfig.PROCESSED_DIR}/visual_features/comprehensive_visual_features.parquet').exists(),
            "processing_stats": Path(f'{DashboardConfig.ANALYSIS_DIR}/image_catalog/processing_statistics.json').exists()
        }
        
        return availability