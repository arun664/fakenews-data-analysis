"""
Data Loader Functions for Dashboard Visualizations

This module provides cached data loading functions with error handling,
sampling, and preprocessing for dashboard visualizations.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data sampling thresholds
SAMPLING_THRESHOLDS = {
    'sentiment_data': 50000,
    'visual_features': 50000,
    'linguistic_features': 100000,
    'temporal_points': 10000,
    'clustering_points': 10000
}

# Base paths
ANALYSIS_RESULTS_PATH = Path('analysis_results')
PROCESSED_DATA_PATH = Path('processed_data')


def sample_for_visualization(
    data: pd.DataFrame,
    threshold: int,
    stratify_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Sample data if it exceeds threshold while maintaining distribution.
    
    Args:
        data: DataFrame to sample
        threshold: Maximum number of rows
        stratify_column: Column to stratify sampling (e.g., 'authenticity_label')
    
    Returns:
        Sampled DataFrame
    """
    if len(data) <= threshold:
        return data
    
    logger.info(f"Sampling data from {len(data)} to {threshold} rows")
    
    if stratify_column and stratify_column in data.columns:
        # Stratified sampling to maintain distribution
        sampled = data.groupby(stratify_column).apply(
            lambda x: x.sample(n=min(len(x), threshold // 2), random_state=42)
        ).reset_index(drop=True)
    else:
        # Simple random sampling
        sampled = data.sample(n=threshold, random_state=42)
    
    return sampled


@st.cache_data(ttl=600)
def load_sentiment_data() -> Optional[Dict[str, Any]]:
    """
    Load sentiment analysis data with caching.
    
    Returns:
        Dictionary with sentiment data or None if not found
    """
    file_path = ANALYSIS_RESULTS_PATH / 'sentiment_analysis' / 'comprehensive_sentiment_analysis.json'
    
    try:
        if not file_path.exists():
            logger.warning(f"Sentiment data not found at {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded sentiment data from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading sentiment data: {e}")
        return None


@st.cache_data(ttl=600)
def load_visual_features(sample: bool = True) -> Optional[pd.DataFrame]:
    """
    Load visual features data with caching.
    
    Args:
        sample: Whether to sample large datasets
    
    Returns:
        DataFrame with visual features or None if not found
    """
    file_path = PROCESSED_DATA_PATH / 'visual_features' / 'visual_features_with_authenticity.parquet'
    
    try:
        if not file_path.exists():
            logger.warning(f"Visual features not found at {file_path}")
            return None
        
        data = pd.read_parquet(file_path)
        
        if sample:
            data = sample_for_visualization(
                data,
                SAMPLING_THRESHOLDS['visual_features'],
                stratify_column='2_way_label' if '2_way_label' in data.columns else None
            )
        
        logger.info(f"Loaded visual features from {file_path} ({len(data)} rows)")
        return data
    
    except Exception as e:
        logger.error(f"Error loading visual features: {e}")
        return None


@st.cache_data(ttl=600)
def load_linguistic_data() -> Optional[Dict[str, Any]]:
    """
    Load linguistic analysis data with caching.
    
    Returns:
        Dictionary with linguistic data or None if not found
    """
    # Try dashboard-specific file first
    file_path = ANALYSIS_RESULTS_PATH / 'dashboard_data' / 'linguistic_analysis_dashboard.json'
    
    if not file_path.exists():
        # Fallback to authenticity patterns file
        file_path = ANALYSIS_RESULTS_PATH / 'linguistic_analysis' / 'authenticity_patterns.json'
    
    try:
        if not file_path.exists():
            logger.warning(f"Linguistic data not found at {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded linguistic data from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading linguistic data: {e}")
        return None


@st.cache_data(ttl=600)
def load_temporal_data() -> Optional[Dict[str, Any]]:
    """
    Load temporal analysis data with caching.
    
    Returns:
        Dictionary with temporal data or None if not found
    """
    file_path = ANALYSIS_RESULTS_PATH / 'temporal_patterns' / 'temporal_analysis_results.json'
    
    try:
        if not file_path.exists():
            logger.warning(f"Temporal data not found at {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded temporal data from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading temporal data: {e}")
        return None


@st.cache_data(ttl=600)
def load_clustering_data() -> Optional[Dict[str, Any]]:
    """
    Load clustering analysis data with caching.
    
    Returns:
        Dictionary with clustering data or None if not found
    """
    file_path = ANALYSIS_RESULTS_PATH / 'dashboard_data' / 'clustering_dashboard_data.json'
    
    try:
        if not file_path.exists():
            logger.warning(f"Clustering data not found at {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded clustering data from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading clustering data: {e}")
        return None


@st.cache_data(ttl=600)
def load_association_rules() -> Optional[Dict[str, Any]]:
    """
    Load association rules data with caching.
    
    Returns:
        Dictionary with association rules or None if not found
    """
    file_path = ANALYSIS_RESULTS_PATH / 'dashboard_data' / 'association_mining_dashboard_data.json'
    
    try:
        if not file_path.exists():
            logger.warning(f"Association rules not found at {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded association rules from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading association rules: {e}")
        return None


@st.cache_data(ttl=600)
def load_cross_modal_data() -> Optional[Dict[str, Any]]:
    """
    Load cross-modal analysis data with caching.
    
    Returns:
        Dictionary with cross-modal data or None if not found
    """
    file_path = ANALYSIS_RESULTS_PATH / 'cross_modal_comparison' / 'cross_modal_analysis.json'
    
    try:
        if not file_path.exists():
            logger.warning(f"Cross-modal data not found at {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded cross-modal data from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading cross-modal data: {e}")
        return None


@st.cache_data(ttl=600)
def load_image_catalog() -> Optional[Dict[str, Any]]:
    """
    Load image catalog data with caching.
    
    Returns:
        Dictionary with image catalog or None if not found
    """
    file_path = ANALYSIS_RESULTS_PATH / 'image_catalog' / 'image_catalog.json'
    
    try:
        if not file_path.exists():
            logger.warning(f"Image catalog not found at {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded image catalog from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading image catalog: {e}")
        return None


def validate_data(data: Any, required_keys: list) -> bool:
    """
    Validate that data contains required keys.
    
    Args:
        data: Data to validate (dict or DataFrame)
        required_keys: List of required keys/columns
    
    Returns:
        True if valid, False otherwise
    """
    if data is None:
        return False
    
    if isinstance(data, dict):
        return all(key in data for key in required_keys)
    elif isinstance(data, pd.DataFrame):
        return all(key in data.columns for key in required_keys)
    
    return False


def preprocess_for_viz(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data for visualization (handle missing values, types, etc.).
    
    Args:
        data: DataFrame to preprocess
    
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying original
    data = data.copy()
    
    # Handle missing values
    numeric_columns = data.select_dtypes(include=['number']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    
    # Ensure authenticity labels are consistent
    if '2_way_label' in data.columns:
        data['authenticity'] = data['2_way_label'].map({0: 'fake', 1: 'real'})
    
    return data


def handle_missing_data_error(data_name: str, file_path: str):
    """
    Display informative error message for missing data.
    
    Args:
        data_name: Name of the data (e.g., "Sentiment Analysis")
        file_path: Path where data should be located
    """
    st.warning(f"📊 {data_name} data not available")
    st.info(f"""
    **Data file not found:** `{file_path}`
    
    To generate this data, run the appropriate analysis script from the `tasks/` directory.
    """)



@st.cache_data(ttl=600)
def load_linguistic_features_summary() -> Optional[Dict[str, Any]]:
    """
    Load linguistic features summary (lightweight version for deployment).
    
    Returns:
        Dictionary with linguistic features summary or None if not found
    """
    file_path = ANALYSIS_RESULTS_PATH / 'dashboard_data' / 'linguistic_features_summary.json'
    
    try:
        if not file_path.exists():
            logger.warning(f"Linguistic features summary not found at {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded linguistic features summary from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading linguistic features summary: {e}")
        return None


@st.cache_data(ttl=600)
def load_social_engagement_summary() -> Optional[Dict[str, Any]]:
    """
    Load social engagement summary (lightweight version for deployment).
    
    Returns:
        Dictionary with social engagement summary or None if not found
    """
    file_path = ANALYSIS_RESULTS_PATH / 'dashboard_data' / 'social_engagement_summary.json'
    
    try:
        if not file_path.exists():
            logger.warning(f"Social engagement summary not found at {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded social engagement summary from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading social engagement summary: {e}")
        return None


@st.cache_data(ttl=600)
def load_dataset_overview_summary() -> Optional[Dict[str, Any]]:
    """
    Load dataset overview summary (lightweight version for deployment).
    
    Returns:
        Dictionary with dataset overview summary or None if not found
    """
    file_path = ANALYSIS_RESULTS_PATH / 'dashboard_data' / 'dataset_overview_summary.json'
    
    try:
        if not file_path.exists():
            logger.warning(f"Dataset overview summary not found at {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded dataset overview summary from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading dataset overview summary: {e}")
        return None


@st.cache_data(ttl=600)
def load_authenticity_analysis_summary() -> Optional[Dict[str, Any]]:
    """
    Load authenticity analysis summary (lightweight version for deployment).
    
    Returns:
        Dictionary with authenticity analysis summary or None if not found
    """
    file_path = ANALYSIS_RESULTS_PATH / 'dashboard_data' / 'authenticity_analysis_summary.json'
    
    try:
        if not file_path.exists():
            logger.warning(f"Authenticity analysis summary not found at {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded authenticity analysis summary from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading authenticity analysis summary: {e}")
        return None
