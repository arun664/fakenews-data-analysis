"""
Dashboard pages module
Each file contains the rendering logic for one analysis view
"""

from .dataset_overview import render_dataset_overview
from .sentiment_analysis import render_sentiment_analysis
from .visual_patterns import render_visual_patterns
from .text_patterns import render_text_patterns
from .social_patterns import render_social_patterns
from .cross_modal_insights import render_cross_modal_insights
from .temporal_trends import render_temporal_trends
from .advanced_analytics import render_advanced_analytics
from .authenticity_analysis import render_authenticity_analysis

__all__ = [
    'render_dataset_overview',
    'render_sentiment_analysis',
    'render_visual_patterns',
    'render_text_patterns',
    'render_social_patterns',
    'render_cross_modal_insights',
    'render_temporal_trends',
    'render_advanced_analytics',
    'render_authenticity_analysis',
]
