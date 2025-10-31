"""
Dashboard Styles and CSS Components

Contains all CSS styling for the Streamlit dashboard.
"""

import streamlit as st

def load_custom_css():
    """Load custom CSS styles for the dashboard."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white !important;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-card h3, .metric-card p {
        color: white !important;
        margin: 0.2rem 0;
    }
    
    .insight-box {
        background: #f8f9fa !important;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: #212529 !important;
    }
    
    .insight-box h4, .insight-box p, .insight-box ul, .insight-box li {
        color: #212529 !important;
        margin: 0.5rem 0;
    }
    
    .warning-box {
        background: #fff3cd !important;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: #856404 !important;
    }
    
    .warning-box h4, .warning-box p, .warning-box ul, .warning-box li {
        color: #856404 !important;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: #d4edda !important;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: #155724 !important;
    }
    
    .success-box h4, .success-box p, .success-box ul, .success-box li {
        color: #155724 !important;
        margin: 0.5rem 0;
    }
    
    /* Fix for dark mode compatibility */
    [data-testid="stMarkdownContainer"] .insight-box,
    [data-testid="stMarkdownContainer"] .warning-box,
    [data-testid="stMarkdownContainer"] .success-box {
        background-color: inherit;
    }
    
    [data-testid="stMarkdownContainer"] .insight-box {
        background: #f8f9fa !important;
    }
    
    [data-testid="stMarkdownContainer"] .warning-box {
        background: #fff3cd !important;
    }
    
    [data-testid="stMarkdownContainer"] .success-box {
        background: #d4edda !important;
    }
    </style>
    """, unsafe_allow_html=True)

def create_metric_card(title: str, value: str) -> str:
    """Create a styled metric card."""
    return f"""
    <div class="metric-card">
        <h3>{value}</h3>
        <p>{title}</p>
    </div>
    """

def create_insight_box(content: str) -> str:
    """Create a styled insight box."""
    return f"""
    <div class="insight-box">
        <p>{content}</p>
    </div>
    """

def create_warning_box(title: str, content: str) -> str:
    """Create a styled warning box."""
    return f"""
    <div class="warning-box">
        <h4>{title}</h4>
        <p>{content}</p>
    </div>
    """

def create_success_box(title: str, content: str) -> str:
    """Create a styled success box."""
    return f"""
    <div class="success-box">
        <h4>{title}</h4>
        <p>{content}</p>
    </div>
    """