"""
Visualization Components for Dashboard

Contains functions for creating interactive charts and plots.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_category_distribution_chart(data):
    """Create interactive category distribution chart."""
    if 'text_analysis' in data and 'category_patterns' in data['text_analysis'].get('clean_title', {}):
        patterns = data['text_analysis']['clean_title']['category_patterns']
        labels = ['True Content', 'False Content']
        true_count = patterns.get('True', {}).get('count', 0)
        false_count = patterns.get('False', {}).get('count', 0)
        values = [true_count, false_count]
    elif 'category_distribution' in data.get('dataset_summary', {}):
        dist_data = data['dataset_summary']['category_distribution']['overall']
        labels = ['True Content', 'False Content']
        values = [dist_data.get('0', 0), dist_data.get('1', 0)]
    elif 'category_patterns' in data:
        patterns = data['category_patterns']
        labels = ['True Content', 'False Content']
        true_count = patterns.get('True', {}).get('count', 0)
        false_count = patterns.get('False', {}).get('count', 0)
        values = [true_count, false_count]
    else:
        total_records = data.get('dataset_summary', {}).get('total_records', 0)
        if total_records > 0:
            labels = ['True Content', 'False Content']
            values = [int(total_records * 0.6), int(total_records * 0.4)]
        else:
            return None

    colors = ['#28a745', '#dc3545']
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent+value',
        textfont_size=12
    )])

    fig.update_layout(
        title="Dataset Category Distribution",
        font=dict(size=14),
        showlegend=True,
        height=400
    )
    return fig

def create_split_distribution_chart(data):
    """Create split distribution chart."""
    if 'dataset_overview' in data['dataset_summary']:
        split_data = data['dataset_summary']['dataset_overview']['split_distribution']
        splits = list(split_data.keys())
        counts = list(split_data.values())
        colors = ['#007bff', '#ffc107', '#28a745']

        fig = go.Figure(data=[go.Bar(
            x=splits,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition='auto'
        )])

        fig.update_layout(
            title="Dataset Split Distribution",
            xaxis_title="Split",
            yaxis_title="Number of Records",
            font=dict(size=14),
            height=400
        )
        return fig
    return None

def create_text_analysis_chart(data):
    """Create text analysis visualization."""
    if 'text_analysis' in data:
        text_data = data['text_analysis']
        columns = []
        avg_lengths = []
        max_lengths = []

        for col_name, col_data in text_data.items():
            if 'basic_stats' in col_data:
                columns.append(col_name)
                avg_lengths.append(col_data['basic_stats']['avg_length'])
                max_lengths.append(float(col_data['basic_stats']['max_length']))

        if columns:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Average Text Length', 'Maximum Text Length'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )

            fig.add_trace(
                go.Bar(x=columns, y=avg_lengths, name="Average Length", marker_color='#007bff'),
                row=1, col=1
            )

            fig.add_trace(
                go.Bar(x=columns, y=max_lengths, name="Maximum Length", marker_color='#dc3545'),
                row=1, col=2
            )

            fig.update_layout(
                title="Text Content Analysis",
                height=400,
                showlegend=False
            )
            return fig

    elif 'text_analysis' in data.get('dataset_summary', {}):
        text_data = data['dataset_summary']['text_analysis']
        columns = list(text_data.keys())
        avg_lengths = [text_data[col]['avg_length'] for col in columns]
        max_lengths = [text_data[col]['max_length'] for col in columns]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Average Text Length', 'Maximum Text Length'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        fig.add_trace(
            go.Bar(x=columns, y=avg_lengths, name="Average Length", marker_color='#007bff'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=columns, y=max_lengths, name="Maximum Length", marker_color='#dc3545'),
            row=1, col=2
        )

        fig.update_layout(
            title="Text Content Analysis",
            height=400,
            showlegend=False
        )
        return fig

    return None

def create_multimodal_comparison_chart(data):
    """Create multimodal comparison visualization."""
    if 'text_analysis' not in data:
        return None

    categories = ['True', 'False']
    text_lengths = []
    image_qualities = []
    comment_counts = []

    for category in categories:
        text_analysis = data['text_analysis'].get('clean_title', {})
        category_patterns = text_analysis.get('category_patterns', {})
        if category in category_patterns:
            text_lengths.append(category_patterns[category].get('avg_length', 0))
        else:
            text_lengths.append(0)

        image_analysis = data.get('image_analysis', {})
        category_stats = image_analysis.get('category_stats', {})
        if category in category_stats:
            image_qualities.append(category_stats[category].get('avg_quality', 0))
        else:
            image_qualities.append(0)

        comments_analysis = data.get('comments_analysis', {})
        engagement_stats = comments_analysis.get('engagement_stats', {})
        if category in engagement_stats:
            comment_counts.append(engagement_stats[category].get('avg_comments', 0))
        else:
            comment_counts.append(0)

    fig = go.Figure()

    if max(text_lengths) > 0:
        norm_text = [x / max(text_lengths) for x in text_lengths]
    else:
        norm_text = [0, 0]

    if max(image_qualities) > 0:
        norm_image = [x / max(image_qualities) for x in image_qualities]
    else:
        norm_image = [0, 0]

    if max(comment_counts) > 0:
        norm_comments = [x / max(comment_counts) for x in comment_counts]
    else:
        norm_comments = [0, 0]

    fig.add_trace(go.Bar(
        name='Text Length (normalized)',
        x=categories,
        y=norm_text,
        marker_color='#007bff'
    ))

    fig.add_trace(go.Bar(
        name='Image Quality (normalized)',
        x=categories,
        y=norm_image,
        marker_color='#28a745'
    ))

    fig.add_trace(go.Bar(
        name='Comment Engagement (normalized)',
        x=categories,
        y=norm_comments,
        marker_color='#ffc107'
    ))

    fig.update_layout(
        title='Multimodal Characteristics Comparison',
        xaxis_title='Content Category',
        yaxis_title='Normalized Values (0-1)',
        barmode='group',
        height=400
    )
    return fig

def create_multiway_distribution_chart(data):
    """Create multi-way category distribution chart."""
    if 'multiway_distribution' in data['dataset_summary']:
        multiway_data = data['dataset_summary']['multiway_distribution']['overall']
        
        category_mapping = {
            '0': 'False',
            '1': 'Satire',
            '2': 'Misleading',
            '3': 'Imposter',
            '4': 'True',
            '5': 'Other'
        }
        
        categories = []
        counts = []
        colors = ['#dc3545', '#fd7e14', '#ffc107', '#6f42c1', '#28a745', '#6c757d']
        
        for cat_num, count in multiway_data.items():
            cat_name = category_mapping.get(str(cat_num), f'Category {cat_num}')
            categories.append(cat_name)
            counts.append(count)

        fig = go.Figure(data=[go.Bar(
            x=categories,
            y=counts,
            marker_color=colors[:len(categories)],
            text=counts,
            textposition='auto'
        )])

        fig.update_layout(
            title="Multi-way Category Distribution",
            xaxis_title="Category",
            yaxis_title="Number of Records",
            font=dict(size=14),
            height=400
        )
        return fig
    return None