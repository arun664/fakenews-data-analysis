"""
Dashboard Pages
Individual page components for the dashboard
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any
from .components import MetricCard, StatusIndicator, ProgressBar, ChartFactory, DataTable, Layout
from .data_loader import DataLoader, DataProcessor
from .config import DashboardConfig

class OverviewPage:
    """System overview page"""
    
    @staticmethod
    def render():
        """Render overview page"""
        
        Layout.create_header("System Overview", "Comprehensive analysis pipeline status and metrics")
        
        # Load data summary
        data_summary = DataLoader.get_data_summary()
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            image_count = data_summary["image_catalog"]["count"]
            MetricCard.render(
                "Total Images", 
                f"{image_count:,}" if image_count > 0 else "Loading...",
                icon="🖼️"
            )
        
        with col2:
            text_count = data_summary["text_data"]["count"]
            MetricCard.render(
                "Text Records",
                f"{text_count:,}" if text_count > 0 else "Loading...",
                icon="📝"
            )
        
        with col3:
            comments_count = data_summary["comments_data"]["count"]
            MetricCard.render(
                "Comments",
                f"{comments_count:,}" if comments_count > 0 else "Loading...",
                icon="💬"
            )
        
        with col4:
            visual_count = data_summary["visual_features"]["count"]
            MetricCard.render(
                "Visual Features",
                f"{visual_count:,}" if visual_count > 0 else "Pending",
                icon="🎨"
            )
        
        Layout.add_spacing(30)
        
        # Progress