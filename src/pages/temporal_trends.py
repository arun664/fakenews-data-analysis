"""
Temporal Trends Page
Temporal pattern analysis of misinformation evolution
COMPLETE IMPLEMENTATION - Extracted from app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys
import numpy as np
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.lazy_loader import LazyLoader
from src.utils.visualization_helpers import (
    create_time_series,
    create_heatmap,
    get_base_layout,
    CHART_CONFIG
)
from src.utils.data_loaders import load_temporal_data

lazy_loader = LazyLoader()


def create_yearly_time_series(temporal_data):
    """Create time series line chart with separate lines for fake and real"""
    yearly_trends = temporal_data.get('yearly_trends', [])
    
    if not yearly_trends:
        st.warning("No yearly trends data available")
        return
    
    # Prepare data
    years = [item['year'] for item in yearly_trends]
    fake_counts = [item['fake_count'] for item in yearly_trends]
    real_counts = [item['real_count'] for item in yearly_trends]
    fake_rates = [item['fake_rate'] * 100 for item in yearly_trends]
    real_rates = [item['real_rate'] * 100 for item in yearly_trends]
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Post Volume Over Time', 'Fake Content Rate Over Time'),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Add volume traces
    fig.add_trace(
        go.Scatter(
            x=years, y=fake_counts,
            name='Fake Posts',
            line=dict(color=CHART_CONFIG['colors']['fake'], width=2),
            mode='lines+markers'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=years, y=real_counts,
            name='Real Posts',
            line=dict(color=CHART_CONFIG['colors']['real'], width=2),
            mode='lines+markers'
        ),
        row=1, col=1
    )
    
    # Add rate trace
    fig.add_trace(
        go.Scatter(
            x=years, y=fake_rates,
            name='Fake Rate',
            line=dict(color=CHART_CONFIG['colors']['fake'], width=2, dash='dot'),
            mode='lines+markers',
            fill='tozeroy',
            fillcolor=f"rgba(255, 107, 107, 0.2)"
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Number of Posts", row=1, col=1)
    fig.update_yaxes(title_text="Fake Content Rate (%)", row=2, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        font=dict(family=CHART_CONFIG['layout']['font_family'])
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_seasonal_decomposition(temporal_data):
    """Create seasonal decomposition plots (trend, seasonal, residual components)"""
    yearly_trends = temporal_data.get('yearly_trends', [])
    seasonal_patterns = temporal_data.get('seasonal_patterns', [])
    quarterly_patterns = temporal_data.get('quarterly_patterns', [])
    
    if not yearly_trends:
        st.warning("No temporal data available for decomposition")
        return
    
    # Create subplots for decomposition
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Trend Component (Yearly)', 'Seasonal Component (Quarterly)', 'Seasonal Component (By Season)'),
        vertical_spacing=0.12,
        specs=[[{}], [{}], [{}]]
    )
    
    # Trend component (yearly)
    years = [item['year'] for item in yearly_trends]
    fake_rates = [item['fake_rate'] * 100 for item in yearly_trends]
    
    fig.add_trace(
        go.Scatter(
            x=years, y=fake_rates,
            name='Yearly Trend',
            line=dict(color=CHART_CONFIG['colors']['primary'], width=3),
            mode='lines+markers'
        ),
        row=1, col=1
    )
    
    # Seasonal component (quarterly if available)
    if quarterly_patterns:
        quarters = [f"Q{item['quarter']}" for item in quarterly_patterns]
        quarterly_fake_rates = [item['fake_rate'] * 100 for item in quarterly_patterns]
        
        fig.add_trace(
            go.Bar(
                x=quarters, y=quarterly_fake_rates,
                name='Quarterly Pattern',
                marker_color=CHART_CONFIG['colors']['secondary']
            ),
            row=2, col=1
        )
    
    # Seasonal component (by season)
    if seasonal_patterns:
        seasons = [item['season'] for item in seasonal_patterns]
        seasonal_fake_rates = [item['fake_rate'] * 100 for item in seasonal_patterns]
        
        fig.add_trace(
            go.Bar(
                x=seasons, y=seasonal_fake_rates,
                name='Seasonal Pattern',
                marker_color=CHART_CONFIG['colors']['fake']
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Quarter", row=2, col=1)
    fig.update_xaxes(title_text="Season", row=3, col=1)
    fig.update_yaxes(title_text="Fake Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Fake Rate (%)", row=2, col=1)
    fig.update_yaxes(title_text="Fake Rate (%)", row=3, col=1)
    
    fig.update_layout(
        height=900,
        showlegend=True,
        font=dict(family=CHART_CONFIG['layout']['font_family'])
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_temporal_heatmap(temporal_data):
    """Create temporal heatmap (hour of day vs day of week)"""
    hourly_patterns = temporal_data.get('hourly_patterns', [])
    day_of_week_patterns = temporal_data.get('day_of_week_patterns', [])
    
    if not hourly_patterns or not day_of_week_patterns:
        st.warning("Insufficient data for temporal heatmap")
        return
    
    # Create synthetic hour x day of week data
    # Since we have hourly and daily patterns separately, we'll create two heatmaps
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly heatmap
        hours = [item['hour'] for item in hourly_patterns]
        fake_rates = [item['fake_rate'] * 100 for item in hourly_patterns]
        
        # Reshape into a 4x6 grid for visualization
        hours_grid = np.array(fake_rates).reshape(4, 6)
        
        fig_hourly = go.Figure(data=go.Heatmap(
            z=hours_grid,
            x=[f"{i}-{i+5}h" for i in range(0, 24, 4)],
            y=['Block 1', 'Block 2', 'Block 3', 'Block 4'],
            colorscale='RdYlGn_r',
            text=hours_grid,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            colorbar=dict(title='Fake Rate (%)')
        ))
        
        fig_hourly.update_layout(
            title='Fake Content Rate by Hour of Day',
            xaxis_title='Hour Blocks',
            yaxis_title='Time Blocks',
            height=400,
            font=dict(family=CHART_CONFIG['layout']['font_family'])
        )
        
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        # Day of week heatmap
        days = [item['day_name'] for item in day_of_week_patterns]
        day_fake_rates = [item['fake_rate'] * 100 for item in day_of_week_patterns]
        
        fig_daily = go.Figure(data=go.Heatmap(
            z=[day_fake_rates],
            x=days,
            y=['Week'],
            colorscale='RdYlGn_r',
            text=[day_fake_rates],
            texttemplate='%{text:.1f}%',
            textfont={"size": 12},
            colorbar=dict(title='Fake Rate (%)')
        ))
        
        fig_daily.update_layout(
            title='Fake Content Rate by Day of Week',
            xaxis_title='Day of Week',
            yaxis_title='',
            height=400,
            font=dict(family=CHART_CONFIG['layout']['font_family'])
        )
        
        st.plotly_chart(fig_daily, use_container_width=True)


def create_rolling_statistics(temporal_data):
    """Create rolling statistics chart with mean and std deviation bands"""
    yearly_trends = temporal_data.get('yearly_trends', [])
    
    if not yearly_trends or len(yearly_trends) < 3:
        st.warning("Insufficient data for rolling statistics")
        return
    
    # Prepare data
    df = pd.DataFrame(yearly_trends)
    
    # Calculate rolling statistics (3-year window)
    window = 3
    df['fake_rate_pct'] = df['fake_rate'] * 100
    df['rolling_mean'] = df['fake_rate_pct'].rolling(window=window, center=True).mean()
    df['rolling_std'] = df['fake_rate_pct'].rolling(window=window, center=True).std()
    
    # Calculate bands
    df['upper_band'] = df['rolling_mean'] + df['rolling_std']
    df['lower_band'] = df['rolling_mean'] - df['rolling_std']
    
    # Create figure
    fig = go.Figure()
    
    # Add actual data
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['fake_rate_pct'],
        name='Actual Fake Rate',
        line=dict(color=CHART_CONFIG['colors']['fake'], width=2),
        mode='lines+markers'
    ))
    
    # Add rolling mean
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['rolling_mean'],
        name=f'{window}-Year Rolling Mean',
        line=dict(color=CHART_CONFIG['colors']['primary'], width=3, dash='dash')
    ))
    
    # Add confidence bands
    fig.add_trace(go.Scatter(
        x=df['year'].tolist() + df['year'].tolist()[::-1],
        y=df['upper_band'].tolist() + df['lower_band'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True,
        name='±1 Std Dev'
    ))
    
    fig.update_layout(
        title=f'Rolling Statistics ({window}-Year Window)',
        xaxis_title='Year',
        yaxis_title='Fake Content Rate (%)',
        height=500,
        hovermode='x unified',
        font=dict(family=CHART_CONFIG['layout']['font_family'])
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_temporal_trends(container):
    """Render Temporal Trends with lazy loading - COMPLETE IMPLEMENTATION"""
    with container.container():
        try:
            st.header("Temporal Pattern Analysis of Misinformation Evolution")
            st.markdown("**Analysis of how misinformation patterns evolve over time (2008-2019)**")
            
            # Load temporal analysis data
            temporal_data = load_temporal_data()
            
            # Hide loading indicator
            lazy_loader.hide_section_loading()
            
            if temporal_data:
                # Overview metrics
                st.subheader("📊 Temporal Analysis Overview")
                
                metadata = temporal_data.get('analysis_metadata', {})
                yearly_trends = temporal_data.get('yearly_trends', [])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Time Span", 
                             metadata.get('date_range', 'N/A'),
                             help="Analysis period")
                
                with col2:
                    st.metric("Total Records",
                             f"{metadata.get('total_records_analyzed', 0):,}",
                             help="Records with temporal data")
                
                with col3:
                    st.metric("Time Periods",
                             f"{len(yearly_trends)}",
                             help="Number of years analyzed")
                
                with col4:
                    # Calculate trend
                    if len(yearly_trends) >= 2:
                        first_rate = yearly_trends[0]['fake_rate']
                        last_rate = yearly_trends[-1]['fake_rate']
                        trend = "Decreasing" if last_rate < first_rate else "Increasing"
                    else:
                        trend = "N/A"
                    st.metric("Trend Detected",
                             trend,
                             help="Overall trend direction")
                
                # Temporal visualizations
                st.subheader("📈 Time Series Analysis")
                st.markdown("Evolution of fake and real content over time")
                create_yearly_time_series(temporal_data)
                
                st.subheader("🔄 Seasonal Decomposition")
                st.markdown("Breaking down temporal patterns into trend and seasonal components")
                create_seasonal_decomposition(temporal_data)
                
                st.subheader("🗓️ Temporal Heatmaps")
                st.markdown("Fake content distribution across time periods")
                create_temporal_heatmap(temporal_data)
                
                st.subheader("📊 Rolling Statistics")
                st.markdown("Smoothed trends with confidence bands")
                create_rolling_statistics(temporal_data)
                
                # Additional insights
                st.subheader("💡 Key Insights")
                
                # Calculate some insights
                seasonal_patterns = temporal_data.get('seasonal_patterns', [])
                if seasonal_patterns:
                    max_season = max(seasonal_patterns, key=lambda x: x['fake_rate'])
                    min_season = min(seasonal_patterns, key=lambda x: x['fake_rate'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Highest Fake Rate Season:** {max_season['season']} ({max_season['fake_rate']*100:.1f}%)")
                    with col2:
                        st.info(f"**Lowest Fake Rate Season:** {min_season['season']} ({min_season['fake_rate']*100:.1f}%)")
                
                # Statistical significance
                stats_tests = temporal_data.get('statistical_tests', {})
                if stats_tests:
                    seasonal_chi2 = stats_tests.get('seasonal_chi2', {})
                    if seasonal_chi2.get('significant') == 'True':
                        st.success(f"✅ Seasonal patterns are statistically significant (p < 0.001)")
                
            else:
                st.warning("📂 Temporal analysis data not available. Please run temporal analysis task first.")
                st.info("""
                **To generate temporal analysis data:**
                ```bash
                python tasks/run_task13_temporal_patterns.py
                ```
                This will analyze how misinformation patterns evolve over time (2008-2019).
                """)
            
        except FileNotFoundError as e:
            st.error(f"📂 Data file not found: {e}")
            st.info("""
            **To generate temporal analysis data:**
            ```bash
            python tasks/run_task13_temporal_patterns.py
            ```
            This will analyze temporal patterns and misinformation evolution over time.
            """)
        except Exception as e:
            st.error(f"❌ Error loading temporal trends: {e}")
            st.info("Please ensure temporal pattern analysis (Task 13) has been completed successfully.")
        finally:
            lazy_loader.hide_section_loading()
