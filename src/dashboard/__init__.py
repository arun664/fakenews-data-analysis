"""
Dashboard Module for Multimodal Fake News Detection

This module contains all components for the Streamlit dashboard including:
- Styles and CSS components
- Data loading and caching functions
- Visualization components
- Page rendering functions
- Analysis result displays
"""

from .styles import load_custom_css, create_metric_card, create_insight_box
from .data_loader import load_eda_data, load_sample_data, detect_completed_analyses
from .visualizations import (
    create_category_distribution_chart,
    create_split_distribution_chart,
    create_text_analysis_chart,
    create_multimodal_comparison_chart
)
from .pages import show_system_overview, show_data_explorer
from .analysis_pages import show_task1_results, show_task2_results, show_text_analysis_results

__all__ = [
    'load_custom_css',
    'create_metric_card',
    'create_insight_box',
    'load_eda_data',
    'load_sample_data',
    'detect_completed_analyses',
    'create_category_distribution_chart',
    'create_split_distribution_chart',
    'create_text_analysis_chart',
    'create_multimodal_comparison_chart',
    'show_system_overview',
    'show_data_explorer',
    'show_task1_results',
    'show_task2_results',
    'show_text_analysis_results'
]