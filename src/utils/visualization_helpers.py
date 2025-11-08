"""
Visualization Helper Functions for Dashboard

This module provides reusable visualization functions for creating consistent,
interactive charts across all dashboard pages using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats

# Visualization Configuration Constants
CHART_CONFIG = {
    'colors': {
        'fake': '#FF6B6B',
        'real': '#4ECDC4',
        'neutral': '#95A5A6',
        'primary': '#3498DB',
        'secondary': '#9B59B6'
    },
    'layout': {
        'font_family': 'Source Sans Pro',
        'title_font_size': 18,
        'axis_font_size': 12,
        'legend_font_size': 11
    },
    'interactivity': {
        'hovermode': 'closest',
        'dragmode': 'zoom'
    }
}


def get_color(category: str) -> str:
    """Get color for a category."""
    return CHART_CONFIG['colors'].get(category.lower(), CHART_CONFIG['colors']['neutral'])


def get_base_layout(title: str, **kwargs) -> Dict[str, Any]:
    """Get base layout configuration for charts."""
    layout = {
        'title': {
            'text': title,
            'font': {
                'size': CHART_CONFIG['layout']['title_font_size'],
                'color': 'black'
            }
        },
        'font': {
            'family': CHART_CONFIG['layout']['font_family'],
            'size': CHART_CONFIG['layout']['axis_font_size'],
            'color': 'black'
        },
        'xaxis': {
            'title': {'font': {'color': 'black'}},
            'tickfont': {'color': 'black'},
            'color': 'black'
        },
        'yaxis': {
            'title': {'font': {'color': 'black'}},
            'tickfont': {'color': 'black'},
            'color': 'black'
        },
        'legend': {
            'font': {'color': 'black'}
        },
        'hovermode': CHART_CONFIG['interactivity']['hovermode'],
        'dragmode': CHART_CONFIG['interactivity']['dragmode'],
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white'
    }
    layout.update(kwargs)
    return layout


def create_comparison_bar_chart(
    fake_data: pd.Series,
    real_data: pd.Series,
    title: str,
    labels: Dict[str, str],
    show_percentages: bool = True
) -> go.Figure:
    """
    Create a grouped bar chart comparing fake vs real content.
    
    Args:
        fake_data: Series with fake content values
        real_data: Series with real content values
        title: Chart title
        labels: Dict with 'x' and 'y' axis labels
        show_percentages: Whether to show percentage labels on bars
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Add fake data bars
    fig.add_trace(go.Bar(
        name='Fake',
        x=fake_data.index,
        y=fake_data.values,
        marker_color=CHART_CONFIG['colors']['fake'],
        text=fake_data.values if show_percentages else None,
        texttemplate='%{text:.1f}%' if show_percentages else None,
        textposition='outside'
    ))
    
    # Add real data bars
    fig.add_trace(go.Bar(
        name='Real',
        x=real_data.index,
        y=real_data.values,
        marker_color=CHART_CONFIG['colors']['real'],
        text=real_data.values if show_percentages else None,
        texttemplate='%{text:.1f}%' if show_percentages else None,
        textposition='outside'
    ))
    
    base_layout = get_base_layout(title)
    base_layout.update({
        'xaxis_title': labels.get('x', ''),
        'yaxis_title': labels.get('y', ''),
        'barmode': 'group',
        'legend': dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font={'color': 'black'}
        )
    })
    fig.update_layout(**base_layout)
    
    return fig


def create_distribution_plot(
    data: pd.DataFrame,
    value_column: str,
    category_column: str,
    title: str,
    labels: Dict[str, str],
    plot_type: str = 'histogram'
) -> go.Figure:
    """
    Create distribution plots (histogram or density) for fake vs real comparison.
    
    Args:
        data: DataFrame with data
        value_column: Column name for values
        category_column: Column name for categories (fake/real)
        title: Chart title
        labels: Dict with axis labels
        plot_type: 'histogram' or 'density'
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    categories = data[category_column].unique()
    
    for category in categories:
        category_data = data[data[category_column] == category][value_column]
        color = get_color(str(category))
        
        if plot_type == 'histogram':
            fig.add_trace(go.Histogram(
                x=category_data,
                name=str(category).title(),
                marker_color=color,
                opacity=0.6,
                nbinsx=30
            ))
        elif plot_type == 'density':
            fig.add_trace(go.Violin(
                x=category_data,
                name=str(category).title(),
                line_color=color,
                fillcolor=color,
                opacity=0.6
            ))
    
    fig.update_layout(
        **get_base_layout(title),
        xaxis_title=labels.get('x', ''),
        yaxis_title=labels.get('y', ''),
        barmode='overlay' if plot_type == 'histogram' else None
    )
    
    return fig


def create_scatter_plot(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str,
    title: str,
    labels: Dict[str, str],
    size_column: Optional[str] = None
) -> go.Figure:
    """
    Create scatter plot with color coding.
    
    Args:
        data: DataFrame with data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        color_column: Column name for color coding
        title: Chart title
        labels: Dict with axis labels
        size_column: Optional column name for point sizes
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    categories = data[color_column].unique()
    
    for category in categories:
        category_data = data[data[color_column] == category]
        color = get_color(str(category))
        
        size = category_data[size_column] if size_column else 8
        
        fig.add_trace(go.Scatter(
            x=category_data[x_column],
            y=category_data[y_column],
            mode='markers',
            name=str(category).title(),
            marker=dict(
                color=color,
                size=size,
                opacity=0.6,
                line=dict(width=0.5, color='white')
            )
        ))
    
    fig.update_layout(
        **get_base_layout(title),
        xaxis_title=labels.get('x', ''),
        yaxis_title=labels.get('y', '')
    )
    
    return fig


def create_heatmap(
    data: pd.DataFrame,
    title: str,
    annotations: bool = True,
    colorscale: str = 'RdBu_r'
) -> go.Figure:
    """
    Create heatmap with optional annotations.
    
    Args:
        data: DataFrame with data (rows and columns will be used as axes)
        title: Chart title
        annotations: Whether to show value annotations
        colorscale: Plotly colorscale name
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale=colorscale,
        text=data.values if annotations else None,
        texttemplate='%{text:.2f}' if annotations else None,
        textfont={"size": 10},
        colorbar=dict(title='Value')
    ))
    
    fig.update_layout(
        **get_base_layout(title),
        xaxis_title='',
        yaxis_title=''
    )
    
    return fig


def create_box_plot(
    data: pd.DataFrame,
    value_column: str,
    category_column: str,
    title: str,
    labels: Dict[str, str]
) -> go.Figure:
    """
    Create box plots for statistical distributions.
    
    Args:
        data: DataFrame with data
        value_column: Column name for values
        category_column: Column name for categories
        title: Chart title
        labels: Dict with axis labels
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    categories = data[category_column].unique()
    
    for category in categories:
        category_data = data[data[category_column] == category][value_column]
        color = get_color(str(category))
        
        fig.add_trace(go.Box(
            y=category_data,
            name=str(category).title(),
            marker_color=color,
            boxmean='sd'
        ))
    
    fig.update_layout(
        **get_base_layout(title),
        xaxis_title=labels.get('x', ''),
        yaxis_title=labels.get('y', '')
    )
    
    return fig


def add_statistical_annotations(
    fig: go.Figure,
    p_value: float,
    effect_size: float,
    x_pos: float = 0.5,
    y_pos: float = 0.95
) -> go.Figure:
    """
    Add statistical annotations to a figure.
    
    Args:
        fig: Plotly Figure object
        p_value: P-value from statistical test
        effect_size: Effect size (e.g., Cohen's d)
        x_pos: X position for annotation (0-1, relative to plot)
        y_pos: Y position for annotation (0-1, relative to plot)
    
    Returns:
        Updated Plotly Figure object
    """
    # Determine significance stars
    if p_value < 0.001:
        sig_stars = '***'
    elif p_value < 0.01:
        sig_stars = '**'
    elif p_value < 0.05:
        sig_stars = '*'
    else:
        sig_stars = 'ns'
    
    annotation_text = f"p = {p_value:.4f} {sig_stars}<br>Effect size (d) = {effect_size:.3f}"
    
    fig.add_annotation(
        text=annotation_text,
        xref='paper',
        yref='paper',
        x=x_pos,
        y=y_pos,
        showarrow=False,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='black',
        borderwidth=1,
        font=dict(size=10)
    )
    
    return fig


def create_time_series(
    data: pd.DataFrame,
    date_column: str,
    value_column: str,
    category_column: str,
    title: str,
    labels: Dict[str, str],
    show_confidence: bool = False
) -> go.Figure:
    """
    Create time series line chart with separate lines for categories.
    
    Args:
        data: DataFrame with time series data
        date_column: Column name for dates
        value_column: Column name for values
        category_column: Column name for categories (fake/real)
        title: Chart title
        labels: Dict with axis labels
        show_confidence: Whether to show confidence intervals
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    categories = data[category_column].unique()
    
    for category in categories:
        category_data = data[data[category_column] == category].sort_values(date_column)
        color = get_color(str(category))
        
        fig.add_trace(go.Scatter(
            x=category_data[date_column],
            y=category_data[value_column],
            mode='lines+markers',
            name=str(category).title(),
            line=dict(color=color, width=2),
            marker=dict(size=4)
        ))
        
        # Add confidence interval if requested
        if show_confidence and 'std' in category_data.columns:
            upper = category_data[value_column] + category_data['std']
            lower = category_data[value_column] - category_data['std']
            
            fig.add_trace(go.Scatter(
                x=category_data[date_column].tolist() + category_data[date_column].tolist()[::-1],
                y=upper.tolist() + lower.tolist()[::-1],
                fill='toself',
                fillcolor=color,
                opacity=0.2,
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name=f'{category} CI'
            ))
    
    fig.update_layout(
        **get_base_layout(title),
        xaxis_title=labels.get('x', ''),
        yaxis_title=labels.get('y', ''),
        hovermode='x unified'
    )
    
    return fig


def create_violin_plot(
    data: pd.DataFrame,
    value_column: str,
    category_column: str,
    title: str,
    labels: Dict[str, str]
) -> go.Figure:
    """
    Create violin plots for distribution visualization.
    
    Args:
        data: DataFrame with data
        value_column: Column name for values
        category_column: Column name for categories
        title: Chart title
        labels: Dict with axis labels
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    categories = data[category_column].unique()
    
    for category in categories:
        category_data = data[data[category_column] == category][value_column]
        color = get_color(str(category))
        
        fig.add_trace(go.Violin(
            y=category_data,
            name=str(category).title(),
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            line_color=color,
            opacity=0.6
        ))
    
    fig.update_layout(
        **get_base_layout(title),
        xaxis_title=labels.get('x', ''),
        yaxis_title=labels.get('y', '')
    )
    
    return fig


def create_pie_chart(
    data: pd.Series,
    title: str,
    colors: Optional[List[str]] = None
) -> go.Figure:
    """
    Create pie chart for categorical data.
    
    Args:
        data: Series with category counts
        title: Chart title
        colors: Optional list of colors for categories
    
    Returns:
        Plotly Figure object
    """
    if colors is None:
        colors = [CHART_CONFIG['colors']['primary'], 
                  CHART_CONFIG['colors']['secondary'],
                  CHART_CONFIG['colors']['neutral']]
    
    fig = go.Figure(data=[go.Pie(
        labels=data.index,
        values=data.values,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='auto'
    )])
    
    fig.update_layout(**get_base_layout(title))
    
    return fig


def create_radar_chart(
    data: pd.DataFrame,
    categories: List[str],
    title: str
) -> go.Figure:
    """
    Create radar chart for multivariate comparison.
    
    Args:
        data: DataFrame with data (rows are groups, columns are dimensions)
        categories: List of category names for radar axes
        title: Chart title
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    for idx, row in data.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row.values,
            theta=categories,
            fill='toself',
            name=str(idx)
        ))
    
    fig.update_layout(
        **get_base_layout(title),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(data.max())]
            )
        ),
        showlegend=True
    )
    
    return fig


def calculate_statistics(fake_data: np.ndarray, real_data: np.ndarray) -> Tuple[float, float]:
    """
    Calculate p-value and effect size for fake vs real comparison.
    
    Args:
        fake_data: Array of fake content values
        real_data: Array of real content values
    
    Returns:
        Tuple of (p_value, effect_size)
    """
    # T-test
    t_stat, p_value = stats.ttest_ind(fake_data, real_data, nan_policy='omit')
    
    # Cohen's d effect size
    mean_diff = np.nanmean(fake_data) - np.nanmean(real_data)
    pooled_std = np.sqrt((np.nanstd(fake_data)**2 + np.nanstd(real_data)**2) / 2)
    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
    
    return p_value, effect_size
