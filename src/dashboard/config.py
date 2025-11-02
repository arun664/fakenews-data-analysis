"""
Dashboard Configuration
Centralized configuration for the Streamlit dashboard
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class DashboardConfig:
    """Dashboard configuration settings"""
    
    # Page Configuration
    PAGE_TITLE = "Multimodal Fake News Detection"
    PAGE_ICON = "🔍"
    LAYOUT = "wide"
    INITIAL_SIDEBAR_STATE = "expanded"
    
    # Theme Configuration (Monochromatic)
    PRIMARY_COLOR = "#2E3440"      # Dark blue-gray
    SECONDARY_COLOR = "#3B4252"    # Medium blue-gray  
    ACCENT_COLOR = "#5E81AC"       # Light blue
    SUCCESS_COLOR = "#A3BE8C"      # Muted green
    WARNING_COLOR = "#EBCB8B"      # Muted yellow
    ERROR_COLOR = "#BF616A"        # Muted red
    BACKGROUND_COLOR = "#ECEFF4"   # Light gray
    TEXT_COLOR = "#2E3440"         # Dark text
    
    # Data Paths (from .env)
    ANALYSIS_DIR = os.getenv('ANALYSIS_OUTPUT_DIR', 'analysis_results')
    PROCESSED_DIR = os.getenv('PROCESSED_DATA_DIR', 'processed_data')
    VISUALIZATIONS_DIR = os.getenv('VISUALIZATIONS_DIR', 'visualizations')
    REPORTS_DIR = os.getenv('REPORTS_DIR', 'reports')
    
    # Navigation Menu
    MENU_ITEMS = [
        {"icon": "📊", "label": "Overview", "key": "overview"},
        {"icon": "🖼️", "label": "Images", "key": "images"},
        {"icon": "📝", "label": "Text", "key": "text"},
        {"icon": "👥", "label": "Social", "key": "social"},
        {"icon": "🔗", "label": "Cross-Modal", "key": "cross_modal"},
        {"icon": "⚙️", "label": "Quality", "key": "quality"},
    ]
    
    # Chart Configuration
    CHART_HEIGHT = 400
    CHART_TEMPLATE = "plotly_white"
    
    @classmethod
    def get_custom_css(cls) -> str:
        """Get custom CSS for monochromatic theme"""
        return f"""
        <style>
        /* Main theme colors */
        :root {{
            --primary-color: {cls.PRIMARY_COLOR};
            --secondary-color: {cls.SECONDARY_COLOR};
            --accent-color: {cls.ACCENT_COLOR};
            --success-color: {cls.SUCCESS_COLOR};
            --warning-color: {cls.WARNING_COLOR};
            --error-color: {cls.ERROR_COLOR};
            --background-color: {cls.BACKGROUND_COLOR};
            --text-color: {cls.TEXT_COLOR};
        }}
        
        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        /* Custom sidebar */
        .css-1d391kg {{
            background-color: var(--primary-color);
        }}
        
        /* Main content area */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }}
        
        /* Custom metric cards */
        .metric-card {{
            background: linear-gradient(135deg, var(--background-color) 0%, #f8f9fa 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 8px rgba(46, 52, 64, 0.1);
            margin-bottom: 1rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(46, 52, 64, 0.15);
        }}
        
        /* Custom buttons */
        .stButton > button {{
            background-color: var(--accent-color);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        
        .stButton > button:hover {{
            background-color: var(--primary-color);
            transform: translateY(-1px);
        }}
        
        /* Custom headers */
        .custom-header {{
            color: var(--primary-color);
            font-weight: 600;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--accent-color);
        }}
        
        /* Status indicators */
        .status-complete {{ color: var(--success-color); }}
        .status-progress {{ color: var(--warning-color); }}
        .status-pending {{ color: var(--secondary-color); }}
        
        /* Compact layout */
        .compact-metric {{
            text-align: center;
            padding: 1rem;
            background-color: var(--background-color);
            border-radius: 8px;
            border-left: 4px solid var(--accent-color);
        }}
        
        /* Modal styling */
        .modal-content {{
            background-color: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(46, 52, 64, 0.2);
        }}
        </style>
        """