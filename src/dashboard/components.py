"""
Reusable Dashboard Components
Clean, modular components for the Streamlit dashboard
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional
from .config import DashboardConfig

class MetricCard:
    """Reusable metric card component"""
    
    @staticmethod
    def render(title: str, value: str, delta: Optional[str] = None, 
               delta_color: str = "normal", icon: str = "📊"):
        """Render a metric card with custom styling"""
        
        delta_html = ""
        if delta:
            color = DashboardConfig.SUCCESS_COLOR if delta_color == "normal" else DashboardConfig.ERROR_COLOR
            delta_html = f'<div style="color: {color}; font-size: 0.9rem; margin-top: 0.5rem;">{delta}</div>'
        
        card_html = f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                <h4 style="margin: 0; color: {DashboardConfig.TEXT_COLOR};">{title}</h4>
            </div>
            <div style="font-size: 2rem; font-weight: 600; color: {DashboardConfig.PRIMARY_COLOR};">{value}</div>
            {delta_html}
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)

class StatusIndicator:
    """Status indicator component"""
    
    @staticmethod
    def render(status: str, label: str):
        """Render status indicator"""
        
        status_config = {
            "complete": {"color": DashboardConfig.SUCCESS_COLOR, "icon": "✅"},
            "progress": {"color": DashboardConfig.WARNING_COLOR, "icon": "🔄"},
            "pending": {"color": DashboardConfig.SECONDARY_COLOR, "icon": "⏳"},
            "error": {"color": DashboardConfig.ERROR_COLOR, "icon": "❌"}
        }
        
        config = status_config.get(status, status_config["pending"])
        
        status_html = f"""
        <div style="display: flex; align-items: center; padding: 0.5rem; margin: 0.25rem 0;">
            <span style="margin-right: 0.5rem;">{config["icon"]}</span>
            <span style="color: {config["color"]}; font-weight: 500;">{label}</span>
        </div>
        """
        
        st.markdown(status_html, unsafe_allow_html=True)

class ProgressBar:
    """Custom progress bar component"""
    
    @staticmethod
    def render(progress: float, label: str = "", height: int = 20):
        """Render custom progress bar"""
        
        progress_html = f"""
        <div style="margin: 1rem 0;">
            {f'<div style="margin-bottom: 0.5rem; font-weight: 500;">{label}</div>' if label else ''}
            <div style="background-color: #e9ecef; border-radius: {height//2}px; height: {height}px; overflow: hidden;">
                <div style="
                    background: linear-gradient(90deg, {DashboardConfig.ACCENT_COLOR} 0%, {DashboardConfig.PRIMARY_COLOR} 100%);
                    height: 100%;
                    width: {progress}%;
                    transition: width 0.3s ease;
                    border-radius: {height//2}px;
                "></div>
            </div>
            <div style="text-align: right; font-size: 0.9rem; margin-top: 0.25rem; color: {DashboardConfig.SECONDARY_COLOR};">
                {progress:.1f}%
            </div>
        </div>
        """
        
        st.markdown(progress_html, unsafe_allow_html=True)

class ChartFactory:
    """Factory for creating consistent charts"""
    
    @staticmethod
    def create_pie_chart(data: pd.Series, title: str, colors: Optional[List[str]] = None) -> go.Figure:
        """Create a styled pie chart"""
        
        if colors is None:
            colors = [DashboardConfig.ACCENT_COLOR, DashboardConfig.SECONDARY_COLOR, 
                     DashboardConfig.SUCCESS_COLOR, DashboardConfig.WARNING_COLOR]
        
        fig = px.pie(
            values=data.values,
            names=data.index,
            title=title,
            color_discrete_sequence=colors
        )
        
        fig.update_layout(
            template=DashboardConfig.CHART_TEMPLATE,
            height=DashboardConfig.CHART_HEIGHT,
            showlegend=True,
            font=dict(color=DashboardConfig.TEXT_COLOR),
            title_font_size=16,
            title_font_color=DashboardConfig.PRIMARY_COLOR
        )
        
        return fig
    
    @staticmethod
    def create_bar_chart(x: List, y: List, title: str, orientation: str = 'v') -> go.Figure:
        """Create a styled bar chart"""
        
        fig = px.bar(
            x=x if orientation == 'v' else y,
            y=y if orientation == 'v' else x,
            orientation=orientation,
            title=title,
            color_discrete_sequence=[DashboardConfig.ACCENT_COLOR]
        )
        
        fig.update_layout(
            template=DashboardConfig.CHART_TEMPLATE,
            height=DashboardConfig.CHART_HEIGHT,
            showlegend=False,
            font=dict(color=DashboardConfig.TEXT_COLOR),
            title_font_size=16,
            title_font_color=DashboardConfig.PRIMARY_COLOR
        )
        
        return fig
    
    @staticmethod
    def create_histogram(data: pd.Series, title: str, bins: int = 30) -> go.Figure:
        """Create a styled histogram"""
        
        fig = px.histogram(
            x=data,
            title=title,
            nbins=bins,
            color_discrete_sequence=[DashboardConfig.ACCENT_COLOR]
        )
        
        fig.update_layout(
            template=DashboardConfig.CHART_TEMPLATE,
            height=DashboardConfig.CHART_HEIGHT,
            showlegend=False,
            font=dict(color=DashboardConfig.TEXT_COLOR),
            title_font_size=16,
            title_font_color=DashboardConfig.PRIMARY_COLOR
        )
        
        return fig
    
    @staticmethod
    def create_line_chart(x: List, y: List, title: str) -> go.Figure:
        """Create a styled line chart"""
        
        fig = px.line(
            x=x,
            y=y,
            title=title,
            color_discrete_sequence=[DashboardConfig.ACCENT_COLOR]
        )
        
        fig.update_layout(
            template=DashboardConfig.CHART_TEMPLATE,
            height=DashboardConfig.CHART_HEIGHT,
            showlegend=False,
            font=dict(color=DashboardConfig.TEXT_COLOR),
            title_font_size=16,
            title_font_color=DashboardConfig.PRIMARY_COLOR
        )
        
        return fig

class DataTable:
    """Styled data table component"""
    
    @staticmethod
    def render(df: pd.DataFrame, title: str = "", max_rows: int = 10, 
               key: Optional[str] = None):
        """Render a styled data table"""
        
        if title:
            st.markdown(f'<h4 class="custom-header">{title}</h4>', unsafe_allow_html=True)
        
        # Limit rows for better UX
        display_df = df.head(max_rows) if len(df) > max_rows else df
        
        st.dataframe(
            display_df,
            use_container_width=True,
            key=key,
            height=min(400, len(display_df) * 35 + 50)  # Dynamic height
        )
        
        if len(df) > max_rows:
            st.caption(f"Showing {max_rows} of {len(df):,} rows")

class Modal:
    """Modal dialog component"""
    
    @staticmethod
    def render(title: str, content: str, button_text: str = "Show Details"):
        """Render a modal dialog"""
        
        if st.button(button_text, key=f"modal_{title.replace(' ', '_')}"):
            with st.expander(title, expanded=True):
                st.markdown(content)

class Sidebar:
    """Custom sidebar component"""
    
    @staticmethod
    def render_navigation() -> str:
        """Render navigation menu"""
        
        st.sidebar.markdown(f"""
        <div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
            <h2 style="color: white; margin: 0;">{DashboardConfig.PAGE_ICON} Analysis Dashboard</h2>
            <p style="color: #d8dee9; margin: 0.5rem 0 0 0;">Multimodal Fake News Detection</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation menu
        selected_page = st.sidebar.selectbox(
            "Navigate to:",
            options=[item["key"] for item in DashboardConfig.MENU_ITEMS],
            format_func=lambda x: next(item["icon"] + " " + item["label"] 
                                     for item in DashboardConfig.MENU_ITEMS 
                                     if item["key"] == x),
            key="navigation"
        )
        
        return selected_page
    
    @staticmethod
    def render_system_info():
        """Render system information in sidebar"""
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🖥️ System Status**")
        
        # System metrics
        system_info = {
            "GPU": "NVIDIA MX130",
            "RAM": "32GB Available", 
            "Processing": "Optimized",
            "Status": "Ready"
        }
        
        for key, value in system_info.items():
            st.sidebar.markdown(f"**{key}:** {value}")
    
    @staticmethod
    def render_quick_actions():
        """Render quick action buttons"""
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**⚡ Quick Actions**")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("📊 Export", use_container_width=True):
                st.sidebar.success("Export initiated!")

class Layout:
    """Layout utilities"""
    
    @staticmethod
    def create_header(title: str, subtitle: str = ""):
        """Create page header"""
        
        header_html = f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: {DashboardConfig.PRIMARY_COLOR}; margin-bottom: 0.5rem;">{title}</h1>
            {f'<p style="color: {DashboardConfig.SECONDARY_COLOR}; font-size: 1.1rem; margin: 0;">{subtitle}</p>' if subtitle else ''}
        </div>
        """
        
        st.markdown(header_html, unsafe_allow_html=True)
    
    @staticmethod
    def create_section(title: str, content_func, collapsible: bool = False):
        """Create a section with optional collapsible content"""
        
        if collapsible:
            with st.expander(f"📋 {title}", expanded=True):
                content_func()
        else:
            st.markdown(f'<h3 class="custom-header">{title}</h3>', unsafe_allow_html=True)
            content_func()
    
    @staticmethod
    def create_columns(ratios: List[int]):
        """Create columns with specified ratios"""
        return st.columns(ratios)
    
    @staticmethod
    def add_spacing(height: int = 20):
        """Add vertical spacing"""
        st.markdown(f'<div style="height: {height}px;"></div>', unsafe_allow_html=True)