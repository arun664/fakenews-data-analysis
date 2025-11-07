#!/usr/bin/env python3
"""
Enhanced Multimodal Fake News Detection - Interactive Dashboard
Modular architecture with separate page components
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import os
import json
from dotenv import load_dotenv
import sys

# Import sklearn for data preprocessing
try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    st.error("scikit-learn not found. Please install: pip install scikit-learn")
    st.stop()

# Add tasks folder to path for dashboard data loader
sys.path.append(str(Path(__file__).parent / "tasks"))

# Import modular page components
from src.pages import (
    render_dataset_overview,
    render_sentiment_analysis,
    render_visual_patterns,
    render_text_patterns,
    render_social_patterns,
    render_cross_modal_insights,
    render_temporal_trends,
    render_advanced_analytics,
    render_authenticity_analysis
)

try:
    from dashboard_data_loader import DashboardDataLoader
except ImportError:
    st.error("Dashboard data loader not found. Please ensure analysis is complete.")
    st.stop()

# Load environment
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Enhanced Multimodal Fake News Analysis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lazy Loading Framework
class LazyLoader:
    """Lazy loading framework for heavy content"""
    
    @staticmethod
    def show_section_loading(section_name):
        """Show loading screen for a section"""
        st.markdown(f"""
        <script>
            showGlobalLoading('Loading {section_name}...', true);
        </script>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def hide_section_loading():
        """Hide loading screen"""
        st.markdown("""
        <script>
            hideGlobalLoading();
        </script>
        """, unsafe_allow_html=True)
    
    @staticmethod
    @st.cache_data(ttl=300)
    def load_heavy_data(data_path, sample_size=None):
        """Load heavy data with optional sampling"""
        try:
            data = pd.read_parquet(data_path)
            if sample_size and len(data) > sample_size:
                data = data.sample(n=sample_size, random_state=42)
            return data
        except Exception as e:
            st.error(f"Error loading data from {data_path}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def lazy_component(component_func, *args, **kwargs):
        """Wrapper for lazy loading components"""
        try:
            return component_func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error loading component: {e}")
            return None

# Initialize lazy loader
lazy_loader = LazyLoader()

# Custom CSS and JavaScript
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    /* Custom Loading Spinner */
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.98);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        backdrop-filter: blur(1px);
    }
    
    .loading-overlay-right {
        position: fixed;
        top: 0;
        left: 21rem;
        width: calc(100% - 21rem);
        height: 100%;
        background: rgba(255, 255, 255, 0.98);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        backdrop-filter: blur(1px);
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #1f77b4;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    
    .loading-content {
        text-align: center;
        color: #1f77b4;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    .loading-text {
        font-size: 18px;
        font-weight: 600;
        margin-top: 15px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
</style>

<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>
    mermaid.initialize({
        startOnLoad: true,
        theme: 'default',
        flowchart: {
            useMaxWidth: true,
            htmlLabels: true
        }
    });
    
    function showGlobalLoading(message = 'Loading...', rightSideOnly = false) {
        const existingOverlay = document.getElementById('global-loading-overlay');
        if (existingOverlay) {
            existingOverlay.remove();
        }
        
        const overlay = document.createElement('div');
        overlay.id = 'global-loading-overlay';
        overlay.className = rightSideOnly ? 'loading-overlay-right' : 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-content">
                <div class="loading-spinner"></div>
                <div class="loading-text">${message}</div>
            </div>
        `;
        
        document.body.appendChild(overlay);
    }
    
    function hideGlobalLoading() {
        const overlay = document.getElementById('global-loading-overlay');
        if (overlay) {
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.remove();
            }, 300);
        }
    }
    
    document.addEventListener('DOMContentLoaded', function() {
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList') {
                    const streamlitElements = document.querySelectorAll('[data-testid="stAppViewContainer"]');
                    if (streamlitElements.length > 0) {
                        setTimeout(hideGlobalLoading, 500);
                    }
                }
            });
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    });
</script>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Multimodal Fake News Detection Analysis</h1>', unsafe_allow_html=True)

# Initialize dashboard data loader with error handling
@st.cache_resource
def get_dashboard_loader():
    """Initialize and cache the dashboard data loader"""
    try:
        return DashboardDataLoader()
    except Exception as e:
        st.error(f"❌ Error initializing dashboard data loader: {e}")
        return None

loader = get_dashboard_loader()

# Load dashboard data with performance optimization
@st.cache_data(ttl=600)  # 10 minutes cache for dashboard data
def load_dashboard_data():
    try:
        dashboard_data_path = Path("analysis_results/dashboard_data/processed_dashboard_data.json")
        if dashboard_data_path.exists():
            with open(dashboard_data_path, 'r') as f:
                return json.load(f)
        else:
            # Try to generate dashboard data if not exists
            loader.export_dashboard_data()
            with open(dashboard_data_path, 'r') as f:
                return json.load(f)
    except FileNotFoundError as e:
        st.warning(f"📂 Dashboard data not found: {e}")
        st.info("""
        **To generate dashboard data:**
        ```bash
        python tasks/create_dashboard_json_data.py
        ```
        This will create the processed dashboard data from analysis results.
        """)
        return {}
    except Exception as e:
        st.error(f"❌ Error loading dashboard data: {e}")
        return {}

dashboard_data = load_dashboard_data()

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

tabs = [
    "Dataset Overview", 
    "Authenticity Analysis",
    "Sentiment Analysis",
    "Visual Patterns", 
    "Text Patterns",
    "Social Patterns",
    "Cross-Modal Insights",
    "Temporal Trends",
    "Advanced Analytics"
]

selected_tab = st.sidebar.selectbox("Select Analysis View", tabs)

# Main content rendering function
def render_content_section(selected_tab):
    """Render content section with lazy loading"""
    
    # Show loading screen immediately
    lazy_loader.show_section_loading(selected_tab)
    
    # Create content container
    content_container = st.empty()
    
    try:
        if selected_tab == "Dataset Overview":
            render_dataset_overview(content_container)
        elif selected_tab == "Sentiment Analysis":
            render_sentiment_analysis(content_container)
        elif selected_tab == "Visual Patterns":
            render_visual_patterns(content_container)
        elif selected_tab == "Text Patterns":
            render_text_patterns(content_container)
        elif selected_tab == "Social Patterns":
            render_social_patterns(content_container)
        elif selected_tab == "Cross-Modal Insights":
            render_cross_modal_insights(content_container)
        elif selected_tab == "Temporal Trends":
            render_temporal_trends(content_container)
        elif selected_tab == "Advanced Analytics":
            render_advanced_analytics(content_container)
        elif selected_tab == "Authenticity Analysis":
            render_authenticity_analysis(content_container)
        else:
            content_container.error(f"Unknown tab: {selected_tab}")
            
    except FileNotFoundError as e:
        lazy_loader.hide_section_loading()
        content_container.error(f"📂 Data file not found: {e}")
        content_container.info("""
        **Data files are missing.** Please run the required analysis tasks to generate the necessary data.
        Check the specific section's error message for detailed instructions.
        """)
    except Exception as e:
        lazy_loader.hide_section_loading()
        content_container.error(f"❌ Error loading {selected_tab}: {e}")
        with content_container.expander("🔍 Debug Information"):
            import traceback
            st.code(traceback.format_exc())

# Call the main render function
render_content_section(selected_tab)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Dashboard Status:** Enhanced with Modular Architecture")

with col2:
    st.markdown("**🔄 Last Updated:** " + (dashboard_data.get("generation_timestamp", "Unknown")[:19] if dashboard_data else "Unknown"))

with col3:
    st.markdown("**⚡ Performance:** Optimized for large datasets")
