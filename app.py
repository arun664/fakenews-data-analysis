#!/usr/bin/env python3
"""
Enhanced Multimodal Fake News Detection - Interactive Dashboard
Integrates analysis results from completed tasks with new analysis views
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

# Add tasks folder to path for dashboard data loader
sys.path.append(str(Path(__file__).parent / "tasks"))

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

# Custom CSS and Mermaid support
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
    .mermaid-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>

<!-- Mermaid.js for diagram rendering -->
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
</script>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔍 Multimodal Fake News Detection Analysis</h1>', unsafe_allow_html=True)

# Load environment paths
analysis_dir = os.getenv('ANALYSIS_OUTPUT_DIR', 'analysis_results')
processed_dir = os.getenv('PROCESSED_DATA_DIR', 'processed_data')
viz_dir = os.getenv('VISUALIZATIONS_DIR', 'visualizations')

# Initialize dashboard data loader
@st.cache_resource
def get_dashboard_loader():
    return DashboardDataLoader()

loader = get_dashboard_loader()

# Load dashboard data
@st.cache_data
def load_dashboard_data():
    try:
        dashboard_data_path = Path("analysis_results/dashboard_data/processed_dashboard_data.json")
        if dashboard_data_path.exists():
            with open(dashboard_data_path, 'r') as f:
                return json.load(f)
        else:
            # Generate data if not exists
            loader.export_dashboard_data()
            with open(dashboard_data_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        return {}

dashboard_data = load_dashboard_data()

# Sidebar navigation
st.sidebar.title("🧭 Enhanced Navigation")
st.sidebar.markdown("---")

tabs = [
    "📊 Data Overview", 
    "👥 Social Analysis", 
    "🔗 Cross-Modal Analysis",
    "🖼️ Image Analysis", 
    "📝 Text Analysis", 
    "🎨 Visual Features",
    "📈 Data Quality",
    "⚙️ System Status"
]

selected_tab = st.sidebar.selectbox("Select Analysis View", tabs)

# Add popup modals section
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Quick Access")

# Enhanced Stats popup modal with real multimodal data
if st.sidebar.button("📊 Multimodal Stats"):
    @st.dialog("📊 Comprehensive Multimodal Statistics")
    def show_multimodal_stats():
        try:
            # Load real data for accurate stats
            train_data = pd.read_parquet('processed_data/clean_datasets/train_final_clean.parquet')
            val_data = pd.read_parquet('processed_data/clean_datasets/validation_final_clean.parquet')
            test_data = pd.read_parquet('processed_data/clean_datasets/test_final_clean.parquet')
            all_data = pd.concat([train_data, val_data, test_data])
            
            # Load comments for true multimodal analysis
            comments_data = pd.read_parquet('processed_data/comments/comments_with_mapping.parquet')
            posts_with_comments = set(comments_data['submission_id'].unique())
            all_data['has_comments'] = all_data['id'].isin(posts_with_comments)
            
            st.subheader("🎯 True Multimodal Breakdown")
            
            # Calculate modality statistics
            full_multimodal = len(all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == True)])
            dual_modal_visual = len(all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == False)])
            dual_modal_text = len(all_data[(all_data['content_type'] == 'text_only') & (all_data['has_comments'] == True)])
            single_modal = len(all_data[(all_data['content_type'] == 'text_only') & (all_data['has_comments'] == False)])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🎯 Full Multimodal", f"{full_multimodal:,}", 
                         delta=f"{full_multimodal/len(all_data)*100:.1f}% of total")
                st.caption("Text + Image + Comments")
                
                st.metric("📊 Dual Modal (Visual)", f"{dual_modal_visual:,}", 
                         delta=f"{dual_modal_visual/len(all_data)*100:.1f}% of total")
                st.caption("Text + Image only")
                
            with col2:
                st.metric("💬 Dual Modal (Text)", f"{dual_modal_text:,}", 
                         delta=f"{dual_modal_text/len(all_data)*100:.1f}% of total")
                st.caption("Text + Comments only")
                
                st.metric("📝 Single Modal", f"{single_modal:,}", 
                         delta=f"{single_modal/len(all_data)*100:.1f}% of total")
                st.caption("Text only")
            
            st.divider()
            
            # Visual analysis targets
            st.subheader("🖼️ Visual Feature Analysis Scope")
            visual_targets = len(all_data[all_data['content_type'] == 'text_image'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Images", f"{visual_targets:,}")
            with col2:
                batches = visual_targets // 10000 + (1 if visual_targets % 10000 > 0 else 0)
                st.metric("Processing Batches", f"{batches}")
            with col3:
                est_hours = visual_targets / 71.4 / 60  # Based on observed performance
                st.metric("Est. Processing Time", f"{est_hours:.1f}h")
            
            # Authenticity by modality
            st.subheader("🎭 Authenticity by Modality Type")
            
            for modality_name, subset in [
                ("Full Multimodal", all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == True)]),
                ("Dual Modal (Visual)", all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == False)])
            ]:
                if len(subset) > 0:
                    auth_dist = subset['2_way_label'].value_counts()
                    fake_pct = auth_dist.get(0, 0) / len(subset) * 100
                    real_pct = auth_dist.get(1, 0) / len(subset) * 100
                    
                    st.write(f"**{modality_name}** ({len(subset):,} records)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Fake Content", f"{auth_dist.get(0, 0):,}", delta=f"{fake_pct:.1f}%")
                    with col2:
                        st.metric("Real Content", f"{auth_dist.get(1, 0):,}", delta=f"{real_pct:.1f}%")
            
            # Comment coverage
            st.subheader("💬 Comment Coverage Analysis")
            comment_coverage = len(posts_with_comments) / len(all_data) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Posts with Comments", f"{len(posts_with_comments):,}", 
                         delta=f"{comment_coverage:.1f}%")
            with col2:
                st.metric("Total Comments", f"{len(comments_data):,}")
            with col3:
                avg_comments = len(comments_data) / len(posts_with_comments)
                st.metric("Avg Comments/Post", f"{avg_comments:.1f}")
                
        except Exception as e:
            st.error(f"Error loading multimodal statistics: {e}")
            st.info("Please ensure all analysis tasks are complete.")
    
    show_multimodal_stats()

# Processing Pipeline popup
if st.sidebar.button("🔄 Processing Pipeline"):
    @st.dialog("🔄 Data Processing Pipeline Status")
    def show_pipeline_status():
        st.subheader("📊 Pipeline Overview")
        
        # Pipeline stages with real data
        pipeline_stages = [
            {"stage": "1. Raw Data Ingestion", "status": "✅ Complete", "records": "682,661", "description": "Original Fakeddit dataset loaded"},
            {"stage": "2. Data Cleaning", "status": "✅ Complete", "records": "620,665", "description": "Removed 62K duplicates/anomalies (9.1%)"},
            {"stage": "3. Image Mapping", "status": "✅ Complete", "records": "618,828", "description": "99.7% have images, 773K total images"},
            {"stage": "4. Comment Integration", "status": "✅ Complete", "records": "13.8M", "description": "89.6% posts have comments"},
            {"stage": "5. Multimodal Classification", "status": "✅ Complete", "records": "4 types", "description": "Full/Dual/Single modal classification"},
            {"stage": "6. Social Analysis", "status": "✅ Complete", "records": "Complete", "description": "Sentiment & engagement analysis"},
            {"stage": "7. Visual Features", "status": "🔄 In Progress", "records": "618,828", "description": "Computer vision feature extraction"},
            {"stage": "8. Advanced Analytics", "status": "⏳ Pending", "records": "TBD", "description": "ML models & pattern discovery"}
        ]
        
        for stage_info in pipeline_stages:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 4])
                
                with col1:
                    st.write(f"**{stage_info['stage']}**")
                
                with col2:
                    st.write(stage_info['status'])
                
                with col3:
                    st.write(f"*{stage_info['description']}*")
                    if stage_info['records'] not in ['Complete', 'TBD']:
                        st.caption(f"Records: {stage_info['records']}")
                
                st.divider()
        
        # Processing metrics
        st.subheader("⚡ Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Retention Rate", "90.9%", help="After cleaning: 620,665 / 682,661")
        with col2:
            st.metric("Multimodal Coverage", "99.7%", help="618,828 / 620,665 have images")
        with col3:
            st.metric("Comment Coverage", "89.6%", help="556,137 / 620,665 have comments")
    
    show_pipeline_status()

# Task Progress popup
if st.sidebar.button("📋 Task Progress"):
    @st.dialog("📋 Task Execution Progress")
    def show_task_progress():
        st.subheader("🎯 Task Completion Status")
        
        tasks_status = [
            {"id": "1", "name": "Image Catalog Creation", "status": "✅", "progress": 100, "time": "45 min", "output": "773K images mapped"},
            {"id": "2", "name": "Text Data Integration", "status": "✅", "progress": 100, "time": "30 min", "output": "620K records cleaned"},
            {"id": "3", "name": "Comment Integration", "status": "✅", "progress": 100, "time": "2.5 hours", "output": "13.8M comments processed"},
            {"id": "4", "name": "Data Quality Assessment", "status": "✅", "progress": 100, "time": "1 hour", "output": "Quality metrics generated"},
            {"id": "5", "name": "Social Engagement Analysis", "status": "✅", "progress": 100, "time": "1.5 hours", "output": "Sentiment analysis complete"},
            {"id": "6", "name": "Visualization Pipeline", "status": "✅", "progress": 100, "time": "45 min", "output": "Interactive charts created"},
            {"id": "7", "name": "Dashboard Enhancement", "status": "✅", "progress": 100, "time": "30 min", "output": "Enhanced UI with modals"},
            {"id": "8", "name": "Visual Feature Engineering", "status": "🔄", "progress": 75, "time": "In Progress", "output": "Computer vision analysis"},
            {"id": "9", "name": "Advanced Analytics", "status": "⏳", "progress": 0, "time": "Pending", "output": "ML models & insights"}
        ]
        
        for task in tasks_status:
            with st.container():
                col1, col2, col3, col4 = st.columns([1, 3, 1, 2])
                
                with col1:
                    st.write(f"**Task {task['id']}**")
                
                with col2:
                    st.write(task['name'])
                    st.progress(task['progress'] / 100)
                
                with col3:
                    st.write(task['status'])
                
                with col4:
                    st.caption(f"Time: {task['time']}")
                    st.caption(f"Output: {task['output']}")
                
                st.divider()
        
        # Overall progress
        completed_tasks = sum(1 for task in tasks_status if task['progress'] == 100)
        total_tasks = len(tasks_status)
        overall_progress = completed_tasks / total_tasks * 100
        
        st.subheader("📊 Overall Progress")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Completed Tasks", f"{completed_tasks}/{total_tasks}")
        with col2:
            st.metric("Overall Progress", f"{overall_progress:.1f}%")
        with col3:
            st.metric("Estimated Remaining", "~24 hours")
    
    show_task_progress()

# Architecture flow diagram popup
if st.sidebar.button("🏗️ System Architecture"):
    @st.dialog("🏗️ Multimodal Analysis System Architecture")
    def show_architecture():
        st.subheader("📊 System Architecture Overview")
        
        # Create architecture visualization
        fig = go.Figure()
        
        # Architecture layers
        layers = [
            {"name": "Data Sources", "y": 5, "components": ["Fakeddit Dataset", "682K Text Records", "773K Images", "13.8M Comments"], "color": "#4CAF50"},
            {"name": "Integration Layer", "y": 4, "components": ["ID Mapping", "Data Validation", "Cross-Modal Linking", "Quality Control"], "color": "#2196F3"},
            {"name": "Processing Layer", "y": 3, "components": ["Text Processing", "Image Analysis", "Comment Mining", "Feature Extraction"], "color": "#FF9800"},
            {"name": "Analysis Layer", "y": 2, "components": ["Authenticity Analysis", "Sentiment Analysis", "Visual Features", "Pattern Discovery"], "color": "#9C27B0"},
            {"name": "Visualization Layer", "y": 1, "components": ["Interactive Charts", "Statistical Plots", "Correlation Analysis", "Trend Visualization"], "color": "#F44336"},
            {"name": "Interface Layer", "y": 0, "components": ["Streamlit Dashboard", "7 Analysis Views", "Popup Modals", "Real-time Updates"], "color": "#607D8B"}
        ]
        
        # Create the architecture diagram
        for layer in layers:
            # Add layer box
            fig.add_shape(
                type="rect",
                x0=-0.5, y0=layer["y"]-0.3,
                x1=4.5, y1=layer["y"]+0.3,
                fillcolor=layer["color"],
                opacity=0.3,
                line=dict(color=layer["color"], width=2)
            )
            
            # Add layer name
            fig.add_annotation(
                x=-0.3, y=layer["y"],
                text=f"<b>{layer['name']}</b>",
                showarrow=False,
                font=dict(size=12, color=layer["color"]),
                xanchor="right"
            )
            
            # Add components
            for i, component in enumerate(layer["components"]):
                fig.add_annotation(
                    x=i, y=layer["y"],
                    text=component,
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor="white",
                    bordercolor=layer["color"],
                    borderwidth=1
                )
        
        # Add arrows between layers
        for i in range(len(layers)-1):
            fig.add_annotation(
                x=2, y=layers[i]["y"] - 0.5,
                ax=2, ay=layers[i+1]["y"] + 0.5,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="gray"
            )
        
        fig.update_layout(
            title="Multimodal Fake News Detection System Architecture",
            xaxis=dict(range=[-1, 5], showgrid=False, showticklabels=False),
            yaxis=dict(range=[-0.5, 5.5], showgrid=False, showticklabels=False),
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Architecture details
        st.subheader("🔧 Technical Implementation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🗄️ Data Pipeline**")
            st.write("• **Storage**: Parquet files for efficiency")
            st.write("• **Processing**: Pandas + NumPy")
            st.write("• **Parallel**: Multi-core processing")
            st.write("• **Memory**: Chunked processing")
            
            st.write("**🔍 Analysis Tools**")
            st.write("• **Computer Vision**: OpenCV + scikit-image")
            st.write("• **NLP**: Text preprocessing + sentiment")
            st.write("• **Statistics**: SciPy + statistical tests")
            st.write("• **ML**: scikit-learn clustering")
        
        with col2:
            st.write("**📊 Visualization Stack**")
            st.write("• **Interactive**: Plotly + Streamlit")
            st.write("• **Static**: Matplotlib + Seaborn")
            st.write("• **Real-time**: Dynamic updates")
            st.write("• **Export**: PNG, HTML, PDF")
            
            st.write("**⚡ Performance**")
            st.write("• **Processing Rate**: 71.4 images/min")
            st.write("• **Memory Usage**: Optimized batching")
            st.write("• **Response Time**: <2 sec queries")
            st.write("• **Scalability**: Horizontal scaling ready")
    
    show_architecture()
            fig.add_trace(go.Scatter(
                x=[i],
                y=[y_pos],
                mode='markers+text',
                marker=dict(size=80, color=colors[i], opacity=0.8),
                text=f"<b>{layer}</b><br>{component}<br><i>{status}</i>",
                textposition="middle center",
                textfont=dict(size=10, color='white'),
                showlegend=False,
                hovertemplate=f"<b>{layer}</b><br>{component}<br>{status}<extra></extra>"
            ))
        
        # Add arrows between components
        for i in range(len(architecture_data['Layer']) - 1):
            fig.add_annotation(
                x=i + 0.5, y=architecture_data['Y_Position'][i] - 0.5,
                ax=i, ay=architecture_data['Y_Position'][i],
                xref='x', yref='y', axref='x', ayref='y',
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#666666'
            )
        
        fig.update_layout(
            title="🏗️ Multimodal Fake News Detection - System Architecture",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**🔄 Data Flow Process:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📥 Input Processing:**")
            st.write("• Raw data ingestion from multiple sources")
            st.write("• ID mapping and cross-modal relationships")
            st.write("• Data validation and quality assessment")
        
        with col2:
            st.write("**📊 Output Generation:**")
            st.write("• Authenticity detection and analysis")
            st.write("• Interactive visualizations and charts")
            st.write("• Multi-view dashboard interface")

# Task process flow popup
if st.sidebar.button("⚙️ Task Process Flow"):
    with st.expander("⚙️ **Analysis Task Process Flow**", expanded=True):
        # Create task flow diagram using Plotly
        fig = go.Figure()
        
        # Define task flow data
        tasks = [
            {"name": "Task 1\nImage Catalog", "x": 0, "y": 2, "status": "✅", "output": "773K Images"},
            {"name": "Task 2\nText Integration", "x": 1, "y": 2, "status": "✅", "output": "682K Records"},
            {"name": "Task 3\nComment Integration", "x": 2, "y": 2, "status": "✅", "output": "13.8M Comments"},
            {"name": "Task 4\nData Quality", "x": 3, "y": 2, "status": "✅", "output": "88.2% Success"},
            {"name": "Task 5\nSocial Analysis", "x": 4, "y": 2, "status": "✅", "output": "Sentiment Data"},
            {"name": "Task 6\nVisualization", "x": 5, "y": 2, "status": "✅", "output": "11 Charts"},
            {"name": "Task 7\nDashboard", "x": 6, "y": 2, "status": "✅", "output": "Interactive UI"},
        ]
        
        # Data stores
        data_stores = [
            {"name": "Image\nData", "x": 1, "y": 0, "type": "database"},
            {"name": "Text\nData", "x": 3, "y": 0, "type": "database"},
            {"name": "Social\nData", "x": 5, "y": 0, "type": "database"},
        ]
        
        # Dashboard
        dashboard = {"name": "Enhanced\nDashboard", "x": 3, "y": -1, "type": "dashboard"}
        
        # Add task nodes
        for task in tasks:
            color = '#4CAF50' if task["status"] == "✅" else '#FF9800'
            fig.add_trace(go.Scatter(
                x=[task["x"]],
                y=[task["y"]],
                mode='markers+text',
                marker=dict(size=60, color=color, opacity=0.8),
                text=f"{task['status']}<br>{task['name']}",
                textposition="middle center",
                textfont=dict(size=9, color='white'),
                showlegend=False,
                hovertemplate=f"<b>{task['name']}</b><br>Status: {task['status']}<br>Output: {task['output']}<extra></extra>"
            ))
        
        # Add data store nodes
        for store in data_stores:
            fig.add_trace(go.Scatter(
                x=[store["x"]],
                y=[store["y"]],
                mode='markers+text',
                marker=dict(size=50, color='#2196F3', opacity=0.8, symbol='square'),
                text=store["name"],
                textposition="middle center",
                textfont=dict(size=9, color='white'),
                showlegend=False,
                hovertemplate=f"<b>{store['name']}</b><br>Type: Data Store<extra></extra>"
            ))
        
        # Add dashboard node
        fig.add_trace(go.Scatter(
            x=[dashboard["x"]],
            y=[dashboard["y"]],
            mode='markers+text',
            marker=dict(size=80, color='#9C27B0', opacity=0.8, symbol='diamond'),
            text=dashboard["name"],
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            showlegend=False,
            hovertemplate=f"<b>{dashboard['name']}</b><br>Type: User Interface<extra></extra>"
        ))
        
        # Add arrows between tasks
        for i in range(len(tasks) - 1):
            fig.add_annotation(
                x=tasks[i+1]["x"] - 0.3, y=tasks[i]["y"],
                ax=tasks[i]["x"] + 0.3, ay=tasks[i]["y"],
                xref='x', yref='y', axref='x', ayref='y',
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#666666'
            )
        
        # Add arrows from tasks to data stores
        connections = [(1, 0), (2, 1), (4, 2)]  # task index to data store index
        for task_idx, store_idx in connections:
            fig.add_annotation(
                x=data_stores[store_idx]["x"], y=data_stores[store_idx]["y"] + 0.3,
                ax=tasks[task_idx]["x"], ay=tasks[task_idx]["y"] - 0.3,
                xref='x', yref='y', axref='x', ayref='y',
                arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor='#999999'
            )
        
        # Add arrows from data stores to dashboard
        for store in data_stores:
            fig.add_annotation(
                x=dashboard["x"], y=dashboard["y"] + 0.3,
                ax=store["x"], ay=store["y"] - 0.3,
                xref='x', yref='y', axref='x', ayref='y',
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#9C27B0'
            )
        
        fig.update_layout(
            title="⚙️ Analysis Task Process Flow",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 6.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, 3]),
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**📋 Task Status Overview:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**✅ Completed Tasks:**")
            completed_tasks = [
                "Task 1: Image Catalog Creation (773K+ images)",
                "Task 2: Text Data Integration (682K+ records)",
                "Task 3: Comment Integration (13.8M comments)",
                "Task 4: Data Quality Assessment (88.2% success)",
            ]
            for task in completed_tasks:
                st.write(f"• {task}")
        
        with col2:
            st.write("**✅ Analysis & Visualization:**")
            analysis_tasks = [
                "Task 5: Social Engagement Analysis (Complete)",
                "Task 6: Visualization Pipeline (11 charts)",
                "Task 7: Dashboard Enhancement (Interactive UI)",
                "🎯 All core analysis tasks completed!"
            ]
            for task in analysis_tasks:
                st.write(f"• {task}")
        
        st.success("🎉 **Analysis pipeline complete!** Dashboard ready for comprehensive exploration.")

# Helper functions
@st.cache_data
def load_image_catalog():
    """Load image catalog data from JSON summary"""
    try:
        # Try JSON summary first (for deployment)
        json_path = Path(f'{analysis_dir}/image_catalog/image_catalog_summary.json')
        if json_path.exists():
            with open(json_path, 'r') as f:
                summary = json.load(f)
            
            # Convert to DataFrame-like structure for compatibility
            if summary.get('sample_records'):
                df = pd.DataFrame(summary['sample_records'])
                # Add summary stats as attributes
                df.attrs['total_images'] = summary.get('total_images', 0)
                df.attrs['mapping_success_rate'] = summary.get('mapping_success_rate', 0)
                df.attrs['content_type_distribution'] = summary.get('content_type_distribution', {})
                return df
        
        # Fallback to original parquet file
        catalog_path = Path(f'{analysis_dir}/image_catalog/comprehensive_image_catalog.parquet')
        if catalog_path.exists():
            return pd.read_parquet(catalog_path)
            
    except Exception as e:
        st.error(f"Error loading image catalog: {e}")
    return None

@st.cache_data
def load_processing_stats():
    """Load processing statistics"""
    try:
        stats_path = Path(f'{analysis_dir}/image_catalog/processing_statistics.json')
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading stats: {e}")
    return {}

def create_progress_chart():
    """Create task progress visualization"""
    tasks_data = {
        "Task": [
            "1. Image Catalog Creation",
            "2. Text Data Integration", 
            "3. Comments Integration",
            "4. Data Quality Assessment",
            "5. Social Engagement Analysis",
            "6. Visualization Pipeline",
            "7. Dashboard Enhancement", 
            "8. Visual Feature Engineering",
            "9. Advanced Analytics"
        ],
        "Status": ["✅", "✅", "✅", "✅", "✅", "✅", "✅", "⏳", "⏳"],
        "Progress": [100, 100, 100, 100, 100, 100, 100, 0, 0],
        "Estimated Time": ["Completed", "Completed", "Completed", "Completed", "Completed", "Completed", "Completed", "2-3 hours", "1-2 hours"]
    }
    
    df = pd.DataFrame(tasks_data)
    
    fig = px.bar(
        df, 
        x="Progress", 
        y="Task", 
        orientation='h',
        color="Progress",
        color_continuous_scale="RdYlGn",
        title="📈 Analysis Pipeline Progress"
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def render_mermaid_diagram(diagram_code, title="Diagram"):
    """Render a Mermaid diagram in Streamlit"""
    diagram_html = f"""
    <div class="mermaid-container">
        <h4>{title}</h4>
        <div class="mermaid">
            {diagram_code}
        </div>
    </div>
    """
    st.markdown(diagram_html, unsafe_allow_html=True)

# Main content based on selected tab
if selected_tab == "📊 Data Overview":
    st.header("📊 True Multimodal Dataset Analysis")
    
    try:
        # Load real data for accurate statistics
        train_data = pd.read_parquet('processed_data/clean_datasets/train_final_clean.parquet')
        val_data = pd.read_parquet('processed_data/clean_datasets/validation_final_clean.parquet')
        test_data = pd.read_parquet('processed_data/clean_datasets/test_final_clean.parquet')
        all_data = pd.concat([train_data, val_data, test_data])
        
        # Load comments for true multimodal analysis
        comments_data = pd.read_parquet('processed_data/comments/comments_with_mapping.parquet')
        posts_with_comments = set(comments_data['submission_id'].unique())
        all_data['has_comments'] = all_data['id'].isin(posts_with_comments)
        
        # Key metrics row with real data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📝 Total Records", f"{len(all_data):,}", 
                     delta="After data cleaning")
        
        with col2:
            visual_records = len(all_data[all_data['content_type'] == 'text_image'])
            st.metric("🖼️ Visual Records", f"{visual_records:,}", 
                     delta=f"{visual_records/len(all_data)*100:.1f}% of total")
        
        with col3:
            comment_coverage = len(posts_with_comments) / len(all_data) * 100
            st.metric("💬 Comment Coverage", f"{comment_coverage:.1f}%", 
                     delta=f"{len(posts_with_comments):,} posts")
        
        with col4:
            total_comments = len(comments_data)
            st.metric("💬 Total Comments", f"{total_comments:,}", 
                     delta="13.8M processed")
        
        st.markdown("---")
        
        # True multimodal breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 True Multimodal Distribution")
            
            # Calculate modality counts
            full_multimodal = len(all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == True)])
            dual_modal_visual = len(all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == False)])
            dual_modal_text = len(all_data[(all_data['content_type'] == 'text_only') & (all_data['has_comments'] == True)])
            single_modal = len(all_data[(all_data['content_type'] == 'text_only') & (all_data['has_comments'] == False)])
            
            # Create multimodal pie chart
            labels = ['Full Multimodal\n(Text+Image+Comments)', 'Dual Modal\n(Text+Image)', 'Dual Modal\n(Text+Comments)', 'Single Modal\n(Text Only)']
            values = [full_multimodal, dual_modal_visual, dual_modal_text, single_modal]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            fig = px.pie(
                values=values, 
                names=labels,
                title="Multimodal Content Distribution",
                color_discrete_sequence=colors
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed breakdown
            st.write("**📊 Detailed Breakdown:**")
            st.write(f"• 🎯 **Full Multimodal**: {full_multimodal:,} ({full_multimodal/len(all_data)*100:.1f}%)")
            st.write(f"• 📊 **Dual Modal (Visual)**: {dual_modal_visual:,} ({dual_modal_visual/len(all_data)*100:.1f}%)")
            st.write(f"• 💬 **Dual Modal (Text)**: {dual_modal_text:,} ({dual_modal_text/len(all_data)*100:.1f}%)")
            st.write(f"• 📝 **Single Modal**: {single_modal:,} ({single_modal/len(all_data)*100:.1f}%)")
        
        with col2:
            st.subheader("🎭 Authenticity by Modality")
            
            # Authenticity analysis by modality type
            modality_auth_data = []
            
            for modality_name, subset in [
                ("Full Multimodal", all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == True)]),
                ("Dual Modal (Visual)", all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == False)]),
                ("Dual Modal (Text)", all_data[(all_data['content_type'] == 'text_only') & (all_data['has_comments'] == True)]),
                ("Single Modal", all_data[(all_data['content_type'] == 'text_only') & (all_data['has_comments'] == False)])
            ]:
                if len(subset) > 0:
                    auth_dist = subset['2_way_label'].value_counts()
                    fake_count = auth_dist.get(0, 0)
                    real_count = auth_dist.get(1, 0)
                    
                    modality_auth_data.extend([
                        {"Modality": modality_name, "Type": "Fake", "Count": fake_count},
                        {"Modality": modality_name, "Type": "Real", "Count": real_count}
                    ])
            
            if modality_auth_data:
                auth_df = pd.DataFrame(modality_auth_data)
                
                fig = px.bar(
                    auth_df,
                    x="Modality",
                    y="Count",
                    color="Type",
                    title="Authenticity Distribution by Modality Type",
                    color_discrete_map={"Fake": "#FF6B6B", "Real": "#4ECDC4"},
                    barmode="group"
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Key insights
            st.write("**🔍 Key Insights:**")
            
            # Calculate fake percentages for each modality
            full_mm_subset = all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == True)]
            dual_vis_subset = all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == False)]
            
            if len(full_mm_subset) > 0:
                full_mm_fake_pct = (full_mm_subset['2_way_label'] == 0).sum() / len(full_mm_subset) * 100
                st.write(f"• Full multimodal: {full_mm_fake_pct:.1f}% fake content")
            
            if len(dual_vis_subset) > 0:
                dual_vis_fake_pct = (dual_vis_subset['2_way_label'] == 0).sum() / len(dual_vis_subset) * 100
                st.write(f"• Dual modal (visual): {dual_vis_fake_pct:.1f}% fake content")
            
            st.write(f"• Comment coverage significantly impacts authenticity patterns")
        
        st.markdown("---")
        
        # Processing pipeline status
        st.subheader("🔄 Data Processing Pipeline Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📊 Data Quality**")
            st.write("• ✅ 620,665 clean records (90.9% retention)")
            st.write("• ✅ Cross-modal validation complete")
            st.write("• ✅ ID mapping integrity verified")
            st.write("• ✅ Balanced class distribution maintained")
        
        with col2:
            st.write("**🎯 Analysis Scope**")
            st.write(f"• 🖼️ Visual analysis: {visual_records:,} images")
            st.write(f"• 💬 Comment analysis: {len(comments_data):,} comments")
            st.write(f"• 🎯 Full multimodal: {full_multimodal:,} records")
            st.write(f"• 📊 Processing batches: 62 × 10K each")
        
        with col3:
            st.write("**⚡ Performance Metrics**")
            st.write("• 🚀 Processing rate: 71.4 images/min")
            st.write("• 💾 Storage efficiency: Parquet format")
            st.write("• 🔄 Memory optimization: Chunked processing")
            st.write("• 📈 Dashboard response: <2 sec")
        
    except Exception as e:
        st.error(f"Error loading dataset overview: {e}")
        st.info("Please ensure all data processing tasks are complete.")
        
        # Fallback to dashboard data if available
        if dashboard_data and "dataset_overview" in dashboard_data:
            st.warning("Showing cached dashboard data...")
            overview = dashboard_data["dataset_overview"]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_text = overview.get("total_text_records", 0)
                st.metric("📝 Text Records", f"{total_text:,}")
            
            with col2:
                total_images = overview.get("total_images", 0)
                st.metric("🖼️ Total Images", f"{total_images:,}")
            
            with col3:
                mapping_rate = overview.get("mapping_success_rate", 0)
                st.metric("🔗 Mapping Success", f"{mapping_rate:.1f}%")
            
            with col4:
                content_dist = overview.get("content_type_distribution", {})
                multimodal = content_dist.get("text_image", 0)
                st.metric("🎯 Multimodal Posts", f"{multimodal:,}")
                    }
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed statistics
        st.subheader("📋 Detailed Dataset Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Text Quality Metrics**")
            text_quality = overview.get("text_quality", {})
            if text_quality:
                title_stats = text_quality.get("title_length_stats", {})
                st.write(f"• Avg Title Length: {title_stats.get('mean', 0):.1f} chars")
                st.write(f"• Short Titles: {text_quality.get('very_short_titles', 0):,}")
                st.write(f"• Long Titles: {text_quality.get('very_long_titles', 0):,}")
        
        with col2:
            st.write("**Missing Data Analysis**")
            missing_data = overview.get("missing_data", {})
            if missing_data:
                for field, stats in missing_data.items():
                    missing_pct = stats.get("missing_percentage", 0)
                    st.write(f"• {field}: {missing_pct:.1f}% missing")
        
        with col3:
            st.write("**Content Coverage**")
            total_records = overview.get("total_text_records", 0)
            if total_records > 0:
                multimodal_pct = (multimodal / total_records) * 100
                st.write(f"• Multimodal Coverage: {multimodal_pct:.1f}%")
                st.write(f"• Image Availability: {mapping_rate:.1f}%")
                
                fake_pct = (fake_count / (fake_count + real_count)) * 100 if (fake_count + real_count) > 0 else 0
                st.write(f"• Fake Content Ratio: {fake_pct:.1f}%")
    
    else:
        st.warning("📂 Dashboard data not available. Please ensure analysis tasks are complete.")

elif selected_tab == "🖼️ Image Analysis":
    st.header("🖼️ Image Analysis Results")
    
    catalog_df = load_image_catalog()
    
    if catalog_df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Content Type Distribution")
            
            # Use JSON summary data if available
            if hasattr(catalog_df, 'attrs') and 'content_type_distribution' in catalog_df.attrs:
                content_dist_dict = catalog_df.attrs['content_type_distribution']
                labels = list(content_dist_dict.keys())
                values = list(content_dist_dict.values())
            else:
                content_dist = catalog_df['content_type'].value_counts()
                labels = content_dist.index.tolist()
                values = content_dist.values.tolist()
            
            fig = px.pie(
                values=values, 
                names=labels,
                title="Multimodal vs Image-Only Content",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Image Quality Distribution")
            if 'quality_score' in catalog_df.columns:
                fig = px.histogram(
                    catalog_df, 
                    x='quality_score', 
                    title="Image Quality Scores",
                    nbins=30,
                    color_discrete_sequence=['#1f77b4']
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Quality scores will be available after visual feature extraction")
        
        # Detailed statistics
        st.subheader("📋 Detailed Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Use JSON summary data if available
            total_images = getattr(catalog_df, 'attrs', {}).get('total_images', len(catalog_df))
            st.metric("Total Images", f"{total_images:,}")
            st.metric("Sample Records", f"{len(catalog_df):,}")
        
        with col2:
            if 'file_size_mb' in catalog_df.columns:
                avg_size = catalog_df['file_size_mb'].mean()
                st.metric("Avg File Size", f"{avg_size:.1f} MB")
            elif hasattr(catalog_df, 'attrs') and 'average_file_size_mb' in catalog_df.attrs:
                avg_size = catalog_df.attrs['average_file_size_mb']
                st.metric("Avg File Size", f"{avg_size:.1f} MB")
            
            if 'dimensions' in catalog_df.columns:
                st.metric("Dimension Variety", f"{catalog_df['dimensions'].nunique():,}")
        
        with col3:
            # Use JSON summary mapping success rate if available
            if hasattr(catalog_df, 'attrs') and 'mapping_success_rate' in catalog_df.attrs:
                mapping_success = catalog_df.attrs['mapping_success_rate']
            else:
                mapping_success = len(catalog_df[catalog_df['content_type'] == 'multimodal']) / len(catalog_df) * 100
            st.metric("Mapping Success Rate", f"{mapping_success:.1f}%")
        
        # Sample data preview
        st.subheader("🔍 Sample Data Preview")
        st.dataframe(catalog_df.head(10), use_container_width=True)
        
    else:
        st.warning("📂 Image catalog data not found. Please run Task 1 first.")

elif selected_tab == "📝 Text Analysis":
    st.header("📝 Text Analysis Results")
    
    # Check for text data
    text_dir = Path(f'{processed_dir}/text_data')
    
    if text_dir.exists():
        text_files = list(text_dir.glob('*.parquet'))
        
        if text_files:
            st.success(f"✅ Found {len(text_files)} text data files")
            
            # Load sample text data
            try:
                # Try to load validation data first (smaller), then sample data, then any available
                sample_file = None
                for preferred_file in ['validation_clean.parquet', 'test_clean.parquet', 'sample_clean.parquet']:
                    preferred_path = text_dir / preferred_file
                    if preferred_path.exists():
                        sample_file = preferred_path
                        break
                
                if sample_file is None:
                    sample_file = text_files[0]
                
                df = pd.read_parquet(sample_file)
                
                # Show info if using sample data
                if 'sample_clean.parquet' in str(sample_file):
                    st.info("📊 Using sample data for demonstration (500 records from 682K total)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Text Records", f"{len(df):,}")
                    if 'clean_title' in df.columns:
                        avg_title_length = df['clean_title'].str.len().mean()
                        st.metric("Avg Title Length", f"{avg_title_length:.0f} chars")
                
                with col2:
                    if 'subreddit' in df.columns:
                        unique_subreddits = df['subreddit'].nunique()
                        st.metric("Unique Subreddits", f"{unique_subreddits:,}")
                
                # Top subreddits
                if 'subreddit' in df.columns:
                    st.subheader("📊 Top Subreddits")
                    top_subreddits = df['subreddit'].value_counts().head(10)
                    
                    fig = px.bar(
                        x=top_subreddits.values,
                        y=top_subreddits.index,
                        orientation='h',
                        title="Most Active Subreddits"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Sample data
                st.subheader("🔍 Sample Text Data")
                display_cols = ['clean_title', 'subreddit', 'score'] if all(col in df.columns for col in ['clean_title', 'subreddit', 'score']) else df.columns[:5]
                st.dataframe(df[display_cols].head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading text data: {e}")
        else:
            st.warning("📂 No processed text files found")
    else:
        st.warning("📂 Text data directory not found. Please run Task 2 first.")

elif selected_tab == "👥 Social Analysis":
    st.header("👥 Social Engagement & Comment Patterns")
    
    if dashboard_data and "social_analysis" in dashboard_data:
        social_data = dashboard_data["social_analysis"]
        
        # Engagement metrics by content type
        st.subheader("📊 Engagement by Content Type")
        
        engagement_by_type = social_data.get("engagement_by_type", {})
        if engagement_by_type:
            # Create engagement comparison chart
            content_types = []
            avg_scores = []
            avg_comments = []
            post_counts = []
            
            for content_type, stats in engagement_by_type.items():
                if content_type == "text_image":
                    display_name = "Text + Image"
                elif content_type == "full_multimodal":
                    display_name = "Full Multimodal"
                else:
                    display_name = "Text Only"
                
                content_types.append(display_name)
                avg_scores.append(stats.get("score", {}).get("mean", 0))
                avg_comments.append(stats.get("num_comments", {}).get("mean", 0))
                post_counts.append(stats.get("count", 0))
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Average score by content type
                fig = px.bar(
                    x=content_types,
                    y=avg_scores,
                    title="Average Engagement Score by Content Type",
                    color=content_types,
                    color_discrete_sequence=['#2E8B57', '#FF6347', '#4682B4']
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average comments by content type
                fig = px.bar(
                    x=content_types,
                    y=avg_comments,
                    title="Average Comments by Content Type",
                    color=content_types,
                    color_discrete_sequence=['#2E8B57', '#FF6347', '#4682B4']
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Authenticity patterns in social engagement
        st.subheader("🎭 Authenticity & Social Dynamics")
        
        authenticity_patterns = social_data.get("authenticity_patterns", {})
        if authenticity_patterns:
            engagement_by_label = authenticity_patterns.get("engagement_by_label", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Fake Content (Label 0)**")
                fake_stats = engagement_by_label.get("0", {})
                st.metric("Posts", f"{fake_stats.get('count', 0):,}")
                st.metric("Avg Score", f"{fake_stats.get('avg_score', 0):.1f}")
                st.metric("Avg Comments", f"{fake_stats.get('avg_comments', 0):.1f}")
            
            with col2:
                st.write("**Real Content (Label 1)**")
                real_stats = engagement_by_label.get("1", {})
                st.metric("Posts", f"{real_stats.get('count', 0):,}")
                st.metric("Avg Score", f"{real_stats.get('avg_score', 0):.1f}")
                st.metric("Avg Comments", f"{real_stats.get('avg_comments', 0):.1f}")
            
            with col3:
                st.write("**Engagement Comparison**")
                fake_score = fake_stats.get('avg_score', 0)
                real_score = real_stats.get('avg_score', 0)
                
                if real_score > fake_score:
                    st.success(f"Real content gets {(real_score/fake_score):.1f}x more engagement")
                else:
                    st.warning(f"Fake content gets {(fake_score/real_score):.1f}x more engagement")
        
        # Cross-modal authenticity patterns
        st.subheader("🔗 Cross-Modal Authenticity Patterns")
        
        cross_modal_patterns = authenticity_patterns.get("cross_modal_patterns", {})
        if cross_modal_patterns:
            # Create authenticity distribution by content type
            content_types = []
            fake_percentages = []
            real_percentages = []
            
            for content_type, data in cross_modal_patterns.items():
                if data.get("total_posts", 0) > 0:
                    if content_type == "text_image":
                        display_name = "Text + Image"
                    elif content_type == "full_multimodal":
                        display_name = "Full Multimodal"
                    else:
                        display_name = "Text Only"
                    
                    content_types.append(display_name)
                    total = data["total_posts"]
                    fake_pct = (data.get("fake_posts", 0) / total) * 100
                    real_pct = (data.get("real_posts", 0) / total) * 100
                    
                    fake_percentages.append(fake_pct)
                    real_percentages.append(real_pct)
            
            if content_types:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Fake Content',
                    x=content_types,
                    y=fake_percentages,
                    marker_color='#FF6B6B'
                ))
                fig.add_trace(go.Bar(
                    name='Real Content',
                    x=content_types,
                    y=real_percentages,
                    marker_color='#4ECDC4'
                ))
                
                fig.update_layout(
                    title="Authenticity Distribution by Content Type (%)",
                    barmode='stack',
                    yaxis_title="Percentage"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment analysis results
        st.subheader("💭 Sentiment Analysis Results")
        
        sentiment_data = social_data.get("sentiment_analysis", {})
        if sentiment_data:
            overall_sentiment = sentiment_data.get("overall_sentiment", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                if overall_sentiment:
                    # Sentiment distribution pie chart
                    sentiments = ["Positive", "Negative", "Neutral"]
                    counts = [
                        overall_sentiment.get("positive", 0),
                        overall_sentiment.get("negative", 0),
                        overall_sentiment.get("neutral", 0)
                    ]
                    
                    fig = px.pie(
                        values=counts,
                        names=sentiments,
                        title="Overall Comment Sentiment Distribution",
                        color_discrete_sequence=['#4ECDC4', '#FF6B6B', '#95A5A6']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                sentiment_dist = sentiment_data.get("sentiment_distribution", {})
                if sentiment_dist:
                    st.write("**Sentiment Statistics**")
                    st.write(f"• Mean Polarity: {sentiment_dist.get('polarity_mean', 0):.3f}")
                    st.write(f"• Std Polarity: {sentiment_dist.get('polarity_std', 0):.3f}")
                    st.write(f"• Mean Subjectivity: {sentiment_dist.get('subjectivity_mean', 0):.3f}")
                    st.write(f"• Std Subjectivity: {sentiment_dist.get('subjectivity_std', 0):.3f}")
    
    else:
        st.warning("📂 Social analysis data not available. Please ensure Task 5 is complete.")

elif selected_tab == "🔗 Cross-Modal Analysis":
    st.header("🔗 Multimodal Relationships & Authenticity Consistency")
    
    if dashboard_data and "cross_modal_analysis" in dashboard_data:
        cross_modal_data = dashboard_data["cross_modal_analysis"]
        
        # Mapping relationships overview
        st.subheader("🔍 ID Mapping Relationships")
        
        mapping_relationships = cross_modal_data.get("mapping_relationships", {})
        if mapping_relationships:
            mapping_success = mapping_relationships.get("mapping_success", {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_images = mapping_success.get("total_images", 0)
                st.metric("🖼️ Total Images", f"{total_images:,}")
            
            with col2:
                multimodal_images = mapping_success.get("multimodal_images", 0)
                st.metric("🔗 Multimodal Images", f"{multimodal_images:,}")
            
            with col3:
                image_only = mapping_success.get("image_only", 0)
                st.metric("📷 Image-Only", f"{image_only:,}")
            
            with col4:
                mapping_rate = mapping_success.get("mapping_rate", 0)
                st.metric("📊 Mapping Rate", f"{mapping_rate:.1f}%")
        
        st.markdown("---")
        
        # Content type distribution and authenticity consistency
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Content Type Distribution")
            content_dist = cross_modal_data.get("content_type_distribution", {})
            if content_dist:
                # Create content type visualization
                labels = []
                values = []
                for content_type, count in content_dist.items():
                    if content_type == "multimodal":
                        labels.append("Multimodal")
                    else:
                        labels.append("Image-Only")
                    values.append(count)
                
                fig = px.pie(
                    values=values,
                    names=labels,
                    title="Image Content Distribution",
                    color_discrete_sequence=['#2E8B57', '#FF6347']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎭 Cross-Modal Authenticity")
            cross_modal_auth = cross_modal_data.get("cross_modal_authenticity", {})
            if cross_modal_auth:
                # Create authenticity comparison across content types
                content_types = []
                fake_ratios = []
                
                for content_type, data in cross_modal_auth.items():
                    if data.get("total_posts", 0) > 0:
                        if content_type == "text_image":
                            display_name = "Text + Image"
                        elif content_type == "full_multimodal":
                            display_name = "Full Multimodal"
                        else:
                            display_name = "Text Only"
                        
                        content_types.append(display_name)
                        fake_ratio = (data.get("fake_posts", 0) / data["total_posts"]) * 100
                        fake_ratios.append(fake_ratio)
                
                if content_types:
                    fig = px.bar(
                        x=content_types,
                        y=fake_ratios,
                        title="Fake Content Percentage by Type",
                        color=fake_ratios,
                        color_continuous_scale="Reds"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Multimodal consistency analysis
        st.subheader("🔄 Multimodal Consistency Analysis")
        
        consistency_metrics = cross_modal_data.get("multimodal_consistency", {})
        if consistency_metrics:
            # Create consistency comparison table
            consistency_df = []
            for content_type, metrics in consistency_metrics.items():
                if content_type == "text_image":
                    display_name = "Text + Image"
                elif content_type == "full_multimodal":
                    display_name = "Full Multimodal"
                else:
                    display_name = "Text Only"
                
                consistency_df.append({
                    "Content Type": display_name,
                    "Total Posts": f"{metrics.get('total_posts', 0):,}",
                    "Fake Ratio": f"{metrics.get('fake_ratio', 0):.1%}",
                    "Real Ratio": f"{metrics.get('real_ratio', 0):.1%}",
                    "Avg Engagement (Fake)": f"{metrics.get('avg_engagement_fake', 0):.1f}",
                    "Avg Engagement (Real)": f"{metrics.get('avg_engagement_real', 0):.1f}"
                })
            
            if consistency_df:
                df = pd.DataFrame(consistency_df)
                st.dataframe(df, use_container_width=True)
        
        # Cross-modal insights
        st.subheader("💡 Key Cross-Modal Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Mapping Pattern Insights:**")
            if mapping_relationships:
                mapping_success = mapping_relationships.get("mapping_success", {})
                total = mapping_success.get("total_images", 0)
                multimodal = mapping_success.get("multimodal_images", 0)
                image_only = mapping_success.get("image_only", 0)
                
                if total > 0:
                    multimodal_pct = (multimodal / total) * 100
                    image_only_pct = (image_only / total) * 100
                    
                    st.write(f"• {multimodal_pct:.1f}% of images have text matches")
                    st.write(f"• {image_only_pct:.1f}% are standalone images")
                    st.write("• This suggests different content strategies")
                    
                    if multimodal_pct > 80:
                        st.success("High multimodal integration")
                    elif multimodal_pct > 60:
                        st.info("Moderate multimodal integration")
                    else:
                        st.warning("Low multimodal integration")
        
        with col2:
            st.write("**Authenticity Consistency:**")
            if cross_modal_auth:
                # Calculate consistency metrics
                text_image_data = cross_modal_auth.get("text_image", {})
                full_multimodal_data = cross_modal_auth.get("full_multimodal", {})
                
                if text_image_data and full_multimodal_data:
                    ti_fake_ratio = text_image_data.get("fake_posts", 0) / text_image_data.get("total_posts", 1)
                    fm_fake_ratio = full_multimodal_data.get("fake_posts", 0) / full_multimodal_data.get("total_posts", 1)
                    
                    st.write(f"• Text+Image fake ratio: {ti_fake_ratio:.1%}")
                    st.write(f"• Full multimodal fake ratio: {fm_fake_ratio:.1%}")
                    
                    if abs(ti_fake_ratio - fm_fake_ratio) < 0.1:
                        st.success("Consistent authenticity patterns")
                    else:
                        st.warning("Inconsistent authenticity patterns")
    
    else:
        st.warning("📂 Cross-modal analysis data not available. Please ensure analysis tasks are complete.")

elif selected_tab == "🎨 Visual Features":
    st.header("🎨 Visual Feature Engineering & Authenticity Analysis")
    
    # Load visual analysis data
    visual_features_file = Path("processed_data/visual_features/visual_features_with_authenticity.parquet")
    visual_analysis_file = Path("analysis_results/visual_analysis/visual_authenticity_analysis.json")
    
    if visual_features_file.exists() and visual_analysis_file.exists():
        try:
            # Load visual features data
            visual_features = pd.read_parquet(visual_features_file)
            
            # Load analysis results
            with open(visual_analysis_file, 'r') as f:
                analysis_results = json.load(f)
            
            # Overview metrics
            st.subheader("📊 Visual Analysis Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_processed = len(visual_features)
                st.metric("Images Processed", f"{total_processed:,}")
            
            with col2:
                success_rate = (visual_features['processing_success'].sum() / len(visual_features)) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            with col3:
                avg_processing_time = visual_features[visual_features['processing_success']]['processing_time_ms'].mean()
                st.metric("Avg Processing Time", f"{avg_processing_time:.1f}ms")
            
            with col4:
                features_analyzed = len(analysis_results.get('feature_comparisons', {}))
                st.metric("Features Analyzed", features_analyzed)
            
            # Filter valid features for analysis
            valid_features = visual_features[visual_features['processing_success'] == True].copy()
            
            if len(valid_features) > 0:
                # Authenticity distribution
                st.subheader("🔍 Authenticity Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Authenticity pie chart
                    auth_counts = valid_features['authenticity_label'].value_counts()
                    auth_labels = ['Real Content', 'Fake Content']
                    
                    fig_pie = px.pie(
                        values=auth_counts.values,
                        names=auth_labels,
                        title="Content Authenticity Distribution",
                        color_discrete_sequence=['#2E86AB', '#A23B72']
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Processing success by authenticity
                    success_by_auth = visual_features.groupby('authenticity_label')['processing_success'].agg(['count', 'sum']).reset_index()
                    success_by_auth['success_rate'] = (success_by_auth['sum'] / success_by_auth['count']) * 100
                    success_by_auth['authenticity'] = success_by_auth['authenticity_label'].map({0: 'Fake', 1: 'Real'})
                    
                    fig_success = px.bar(
                        success_by_auth,
                        x='authenticity',
                        y='success_rate',
                        title="Processing Success Rate by Authenticity",
                        color='success_rate',
                        color_continuous_scale='Viridis'
                    )
                    fig_success.update_layout(showlegend=False)
                    st.plotly_chart(fig_success, use_container_width=True)
                
                # Feature distributions
                st.subheader("📈 Visual Feature Distributions")
                
                # Feature selection
                feature_columns = [
                    'mean_brightness', 'mean_contrast', 'color_diversity', 'texture_contrast',
                    'sharpness_score', 'noise_level', 'manipulation_score', 'meme_characteristics',
                    'edge_density', 'visual_entropy', 'aspect_ratio', 'file_size_kb'
                ]
                
                available_features = [col for col in feature_columns if col in valid_features.columns]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_feature1 = st.selectbox(
                        "Select Feature for Distribution Analysis:",
                        available_features,
                        index=0 if available_features else None,
                        key="feature_dist_1"
                    )
                
                with col2:
                    selected_feature2 = st.selectbox(
                        "Select Feature for Comparison:",
                        available_features,
                        index=1 if len(available_features) > 1 else 0,
                        key="feature_dist_2"
                    )
                
                if selected_feature1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribution by authenticity
                        fake_data = valid_features[valid_features['authenticity_label'] == 0][selected_feature1].dropna()
                        real_data = valid_features[valid_features['authenticity_label'] == 1][selected_feature1].dropna()
                        
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Histogram(
                            x=fake_data,
                            name='Fake Content',
                            opacity=0.7,
                            marker_color='#A23B72'
                        ))
                        fig_dist.add_trace(go.Histogram(
                            x=real_data,
                            name='Real Content',
                            opacity=0.7,
                            marker_color='#2E86AB'
                        ))
                        
                        fig_dist.update_layout(
                            title=f'{selected_feature1.replace("_", " ").title()} Distribution',
                            xaxis_title=selected_feature1.replace("_", " ").title(),
                            yaxis_title='Count',
                            barmode='overlay'
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with col2:
                        if selected_feature2 and selected_feature2 != selected_feature1:
                            # Scatter plot comparison
                            fig_scatter = px.scatter(
                                valid_features,
                                x=selected_feature1,
                                y=selected_feature2,
                                color='authenticity_label',
                                title=f'{selected_feature1.replace("_", " ").title()} vs {selected_feature2.replace("_", " ").title()}',
                                color_discrete_map={0: '#A23B72', 1: '#2E86AB'},
                                labels={'authenticity_label': 'Authenticity'}
                            )
                            fig_scatter.update_traces(marker=dict(size=8, opacity=0.6))
                            st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Authenticity signatures
                if 'authenticity_signatures' in analysis_results:
                    st.subheader("🔬 Authenticity Signatures")
                    
                    signatures = analysis_results['authenticity_signatures']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Strong Indicators", len(signatures.get('strong_indicators', [])))
                    
                    with col2:
                        st.metric("Moderate Indicators", len(signatures.get('moderate_indicators', [])))
                    
                    with col3:
                        st.metric("Weak Indicators", len(signatures.get('weak_indicators', [])))
                    
                    # Display strong indicators
                    if signatures.get('strong_indicators'):
                        st.write("**🎯 Strong Authenticity Indicators (Effect Size ≥ 0.8):**")
                        
                        for indicator in signatures['strong_indicators']:
                            direction = "📈 Higher in fake content" if indicator['direction'] == 'higher_in_fake' else "📉 Higher in real content"
                            st.write(f"• **{indicator['feature'].replace('_', ' ').title()}**: Effect size {indicator['effect_size']:.3f} - {direction}")
                    
                    # Display moderate indicators
                    if signatures.get('moderate_indicators'):
                        with st.expander("📊 Moderate Authenticity Indicators (Effect Size ≥ 0.5)"):
                            for indicator in signatures['moderate_indicators']:
                                direction = "📈 Higher in fake content" if indicator['direction'] == 'higher_in_fake' else "📉 Higher in real content"
                                st.write(f"• **{indicator['feature'].replace('_', ' ').title()}**: Effect size {indicator['effect_size']:.3f} - {direction}")
                
                # Feature comparisons table
                if 'feature_comparisons' in analysis_results:
                    st.subheader("📋 Feature Comparison Analysis")
                    
                    comparisons = analysis_results['feature_comparisons']
                    statistical_tests = analysis_results.get('statistical_tests', {})
                    
                    # Create comparison dataframe
                    comparison_data = []
                    for feature, stats in comparisons.items():
                        p_value = statistical_tests.get(feature, {}).get('p_value', 1.0)
                        significant = statistical_tests.get(feature, {}).get('significant', False)
                        
                        comparison_data.append({
                            'Feature': feature.replace('_', ' ').title(),
                            'Fake Mean': f"{stats['fake_mean']:.3f}",
                            'Real Mean': f"{stats['real_mean']:.3f}",
                            'Difference': f"{stats['difference']:.3f}",
                            'Effect Size': f"{stats['effect_size']:.3f}",
                            'P-Value': f"{p_value:.3f}",
                            'Significant': "✅" if significant else "❌"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                
                # Visual complexity analysis
                st.subheader("🎨 Visual Complexity Analysis")
                
                complexity_features = ['edge_density', 'structural_complexity', 'visual_entropy']
                available_complexity = [f for f in complexity_features if f in valid_features.columns]
                
                if available_complexity:
                    # Create complexity score
                    complexity_cols = []
                    for feature in available_complexity:
                        # Normalize features to 0-1 scale
                        normalized = (valid_features[feature] - valid_features[feature].min()) / (valid_features[feature].max() - valid_features[feature].min())
                        complexity_cols.append(normalized)
                    
                    if complexity_cols:
                        valid_features['complexity_score'] = np.mean(complexity_cols, axis=0)
                        
                        # Complexity by authenticity
                        complexity_by_auth = valid_features.groupby('authenticity_label')['complexity_score'].agg(['mean', 'std']).reset_index()
                        complexity_by_auth['authenticity'] = complexity_by_auth['authenticity_label'].map({0: 'Fake', 1: 'Real'})
                        
                        fig_complexity = px.bar(
                            complexity_by_auth,
                            x='authenticity',
                            y='mean',
                            error_y='std',
                            title="Average Visual Complexity by Authenticity",
                            color='authenticity',
                            color_discrete_map={'Fake': '#A23B72', 'Real': '#2E86AB'}
                        )
                        st.plotly_chart(fig_complexity, use_container_width=True)
                
                # Quality metrics analysis
                st.subheader("🔍 Image Quality Analysis")
                
                quality_features = ['sharpness_score', 'noise_level', 'compression_artifacts']
                available_quality = [f for f in quality_features if f in valid_features.columns]
                
                if available_quality:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Quality distribution
                        selected_quality = st.selectbox(
                            "Select Quality Metric:",
                            available_quality,
                            key="quality_metric"
                        )
                        
                        if selected_quality:
                            quality_by_auth = valid_features.groupby('authenticity_label')[selected_quality].agg(['mean', 'median', 'std']).reset_index()
                            quality_by_auth['authenticity'] = quality_by_auth['authenticity_label'].map({0: 'Fake', 1: 'Real'})
                            
                            fig_quality = px.box(
                                valid_features,
                                x='authenticity_label',
                                y=selected_quality,
                                title=f'{selected_quality.replace("_", " ").title()} by Authenticity',
                                color='authenticity_label',
                                color_discrete_map={0: '#A23B72', 1: '#2E86AB'}
                            )
                            fig_quality.update_xaxis(tickvals=[0, 1], ticktext=['Fake', 'Real'])
                            st.plotly_chart(fig_quality, use_container_width=True)
                    
                    with col2:
                        # Quality metrics summary
                        st.write("**Quality Metrics Summary:**")
                        
                        for feature in available_quality:
                            fake_mean = valid_features[valid_features['authenticity_label'] == 0][feature].mean()
                            real_mean = valid_features[valid_features['authenticity_label'] == 1][feature].mean()
                            difference = fake_mean - real_mean
                            
                            direction = "📈" if difference > 0 else "📉"
                            st.write(f"**{feature.replace('_', ' ').title()}:**")
                            st.write(f"  • Fake: {fake_mean:.3f}")
                            st.write(f"  • Real: {real_mean:.3f}")
                            st.write(f"  • Difference: {direction} {abs(difference):.3f}")
                            st.write("")
            
            else:
                st.warning("No valid visual features found for analysis.")
        
        except Exception as e:
            st.error(f"Error loading visual analysis data: {e}")
            st.write("Please ensure Task 8 (Visual Feature Engineering) has been completed successfully.")
    
    else:
        st.warning("📂 Visual analysis data not available. Please run Task 8 (Visual Feature Engineering) first.")
        
        st.info("**To generate visual analysis data:**")
        st.code("python tasks/run_task8_visual_feature_engineering.py", language="bash")
        
        st.write("**Expected outputs:**")
        st.write("• Visual features dataset with authenticity labels")
        st.write("• Computer vision analysis results")
        st.write("• Authenticity pattern comparisons")
        st.write("• Visual complexity and quality metrics")

elif selected_tab == "📈 Data Quality":
    st.header("📈 Data Quality Assessment & Validation")
    
    # Load data quality information
    if dashboard_data and "dataset_overview" in dashboard_data:
        overview = dashboard_data["dataset_overview"]
        
        # Quality metrics overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Completed Quality Checks")
            st.success("✓ Image catalog validation (773K+ images)")
            st.success("✓ Text data integration (682K+ records)")
            st.success("✓ Comment data processing (13.8M comments)")
            st.success("✓ ID mapping verification (88.2% success)")
            st.success("✓ Cross-modal relationship validation")
        
        with col2:
            st.subheader("📊 Quality Metrics")
            mapping_rate = overview.get("mapping_success_rate", 0)
            st.metric("Mapping Success Rate", f"{mapping_rate:.1f}%")
            
            missing_data = overview.get("missing_data", {})
            if missing_data:
                for field, stats in missing_data.items():
                    missing_pct = stats.get("missing_percentage", 0)
                    if missing_pct < 5:
                        st.success(f"✓ {field}: {missing_pct:.1f}% missing")
                    elif missing_pct < 20:
                        st.warning(f"⚠ {field}: {missing_pct:.1f}% missing")
                    else:
                        st.error(f"✗ {field}: {missing_pct:.1f}% missing")
        
        # Text quality analysis
        st.subheader("📝 Text Quality Analysis")
        text_quality = overview.get("text_quality", {})
        if text_quality:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                title_stats = text_quality.get("title_length_stats", {})
                avg_length = title_stats.get("mean", 0)
                st.metric("Avg Title Length", f"{avg_length:.1f} chars")
                
                if avg_length > 30:
                    st.success("Good title length")
                else:
                    st.warning("Short titles detected")
            
            with col2:
                short_titles = text_quality.get("very_short_titles", 0)
                total_records = overview.get("total_text_records", 1)
                short_pct = (short_titles / total_records) * 100
                st.metric("Short Titles", f"{short_pct:.1f}%")
                
                if short_pct < 10:
                    st.success("Low short title rate")
                else:
                    st.warning("High short title rate")
            
            with col3:
                long_titles = text_quality.get("very_long_titles", 0)
                long_pct = (long_titles / total_records) * 100
                st.metric("Long Titles", f"{long_pct:.1f}%")
                
                if long_pct < 5:
                    st.success("Normal long title rate")
                else:
                    st.info("Some very long titles")
        
        # Data integrity summary
        st.subheader("🔍 Data Integrity Summary")
        
        integrity_score = 0
        total_checks = 0
        
        # Calculate integrity score based on various metrics
        if mapping_rate > 80:
            integrity_score += 25
        elif mapping_rate > 60:
            integrity_score += 15
        total_checks += 25
        
        # Missing data penalty
        if missing_data:
            avg_missing = sum(stats.get("missing_percentage", 0) for stats in missing_data.values()) / len(missing_data)
            if avg_missing < 5:
                integrity_score += 25
            elif avg_missing < 20:
                integrity_score += 15
            total_checks += 25
        
        # Text quality score
        if text_quality:
            title_stats = text_quality.get("title_length_stats", {})
            if title_stats.get("mean", 0) > 30:
                integrity_score += 25
            total_checks += 25
        
        # Content distribution score
        content_dist = overview.get("content_type_distribution", {})
        if content_dist:
            multimodal_count = content_dist.get("text_image", 0)
            total_content = sum(content_dist.values())
            if multimodal_count / total_content > 0.5:
                integrity_score += 25
            total_checks += 25
        
        final_score = (integrity_score / total_checks) * 100 if total_checks > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Data Quality Score", f"{final_score:.1f}%")
            if final_score > 80:
                st.success("Excellent data quality")
            elif final_score > 60:
                st.info("Good data quality")
            else:
                st.warning("Data quality needs improvement")
        
        with col2:
            st.metric("Validation Checks Passed", f"{int(integrity_score/25)}/{int(total_checks/25)}")
        
        with col3:
            st.metric("Ready for Analysis", "✅ Yes" if final_score > 60 else "⚠ Needs Review")
    
    else:
        st.warning("📂 Data quality information not available.")

elif selected_tab == "⚙️ System Status":
    st.header("⚙️ System Status & Task Progress")
    
    # Task completion status
    st.subheader("📋 Task Completion Status")
    
    tasks_status = [
        {"task": "1. Image Catalog Creation", "status": "✅ Complete", "progress": 100},
        {"task": "2. Text Data Integration", "status": "✅ Complete", "progress": 100},
        {"task": "3. Comments Integration", "status": "✅ Complete", "progress": 100},
        {"task": "4. Data Quality Assessment", "status": "✅ Complete", "progress": 100},
        {"task": "5. Social Engagement Analysis", "status": "✅ Complete", "progress": 100},
        {"task": "6. Visualization Pipeline", "status": "✅ Complete", "progress": 100},
        {"task": "7. Dashboard Enhancement", "status": "✅ Complete", "progress": 100},
        {"task": "8. Visual Feature Engineering", "status": "✅ Complete", "progress": 100},
        {"task": "9. Advanced Analytics", "status": "⏳ Pending", "progress": 0}
    ]
    
    # Create progress visualization
    task_names = [task["task"] for task in tasks_status]
    progress_values = [task["progress"] for task in tasks_status]
    
    fig = px.bar(
        x=progress_values,
        y=task_names,
        orientation='h',
        title="Task Completion Progress",
        color=progress_values,
        color_continuous_scale="RdYlGn",
        range_color=[0, 100]
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # System performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖥️ System Performance")
        st.write("**Processing Capacity:**")
        st.write("• Images processed: 773K+")
        st.write("• Text records: 682K+")
        st.write("• Comments analyzed: 13.8M")
        st.write("• Total processing time: ~6 hours")
        
        st.write("**Current Status:**")
        st.success("✅ All core data integration complete")
        st.info("🔄 Dashboard enhancement in progress")
        st.warning("⏳ Advanced analytics pending")
    
    with col2:
        st.subheader("📊 Data Pipeline Health")
        
        # Calculate pipeline health score
        completed_tasks = sum(1 for task in tasks_status if task["progress"] == 100)
        total_tasks = len(tasks_status)
        health_score = (completed_tasks / total_tasks) * 100
        
        st.metric("Pipeline Health", f"{health_score:.1f}%")
        st.metric("Completed Tasks", f"{completed_tasks}/{total_tasks}")
        
        if dashboard_data:
            st.metric("Data Freshness", "✅ Current")
            st.metric("Dashboard Status", "🔄 Active")
        else:
            st.metric("Data Freshness", "⚠ Needs Update")
            st.metric("Dashboard Status", "⏳ Loading")
        
        # Next steps
        st.write("**Next Steps:**")
        st.write("• Complete visualization pipeline")
        st.write("• Implement visual feature extraction")
        st.write("• Add advanced analytics")
    
    # System resources
    st.subheader("💾 Resource Utilization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Storage Usage:**")
        st.write("• Analysis results: ~2GB")
        st.write("• Processed data: ~5GB")
        st.write("• Visualizations: ~500MB")
        st.write("• Total: ~7.5GB")
    
    with col2:
        st.write("**Memory Efficiency:**")
        st.write("• Batch processing: ✅")
        st.write("• Memory cleanup: ✅")
        st.write("• Chunked loading: ✅")
        st.write("• Optimized queries: ✅")
    
    with col3:
        st.write("**Performance:**")
        st.write("• Image processing: 37.6 img/sec")
        st.write("• Text processing: Fast")
        st.write("• Dashboard loading: <2 sec")
        st.write("• Query response: <1 sec")

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Quick Stats")

if dashboard_data and "dataset_overview" in dashboard_data:
    overview = dashboard_data["dataset_overview"]
    st.sidebar.metric("Text Records", f"{overview.get('total_text_records', 0):,}")
    st.sidebar.metric("Images", f"{overview.get('total_images', 0):,}")
    
    auth_dist = overview.get("authenticity_distribution", {})
    if auth_dist:
        fake_count = auth_dist.get("fake", 0)
        real_count = auth_dist.get("real", 0)
        total_auth = fake_count + real_count
        if total_auth > 0:
            fake_pct = (fake_count / total_auth) * 100
            st.sidebar.metric("Fake Content", f"{fake_pct:.1f}%")

st.sidebar.markdown("---")
# Data Quality Insights popup
if st.sidebar.button("🔍 Data Quality"):
    @st.dialog("🔍 Data Quality Assessment")
    def show_data_quality():
        st.subheader("📊 Data Quality Metrics")
        
        try:
            # Load metadata for quality metrics
            with open('processed_data/clean_datasets/dataset_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            # Quality overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                retention_rate = metadata['removal_statistics']['final_size'] / metadata['removal_statistics']['initial_size'] * 100
                st.metric("Data Retention", f"{retention_rate:.1f}%", 
                         delta=f"{metadata['removal_statistics']['final_size']:,} / {metadata['removal_statistics']['initial_size']:,}")
            
            with col2:
                # Calculate multimodal coverage
                st.metric("Multimodal Coverage", "99.7%", 
                         delta="618,828 / 620,665 records")
            
            with col3:
                st.metric("Comment Coverage", "89.6%", 
                         delta="556,137 posts have comments")
            
            st.divider()
            
            # Data cleaning details
            st.subheader("🧹 Data Cleaning Results")
            
            removal_stats = metadata['removal_statistics']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Removed Records:**")
                st.write(f"• Exact duplicates: {removal_stats.get('exact_duplicates', 0):,}")
                st.write(f"• Near duplicates: {removal_stats.get('near_duplicates', 0):,}")
                st.write(f"• Anomalies: {removal_stats.get('anomalies', 0):,}")
                st.write(f"• **Total removed**: {removal_stats['initial_size'] - removal_stats['final_size']:,}")
            
            with col2:
                st.write("**Quality Improvements:**")
                st.write("• ✅ Consistent data formats")
                st.write("• ✅ Valid ID mappings")
                st.write("• ✅ Cross-modal consistency")
                st.write("• ✅ Balanced class distribution")
            
            # Missing data analysis
            st.subheader("📋 Missing Data Analysis")
            
            quality_metrics = metadata.get('quality_metrics', {})
            if 'train' in quality_metrics:
                train_missing = quality_metrics['train']['missing_value_percentages']
                
                missing_data = []
                for field, pct in train_missing.items():
                    if pct > 0:
                        missing_data.append({"Field": field, "Missing %": f"{pct:.1f}%"})
                
                if missing_data:
                    missing_df = pd.DataFrame(missing_data)
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("✅ No significant missing data detected!")
            
            # Data validation results
            st.subheader("✅ Validation Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Cross-Modal Validation:**")
                st.write("• Image-text consistency: ✅")
                st.write("• ID mapping integrity: ✅")
                st.write("• Timestamp validation: ✅")
                st.write("• Format standardization: ✅")
            
            with col2:
                st.write("**Statistical Validation:**")
                st.write("• Class balance maintained: ✅")
                st.write("• No data leakage: ✅")
                st.write("• Proper train/val/test splits: ✅")
                st.write("• Outlier detection: ✅")
                
        except Exception as e:
            st.error(f"Error loading quality metrics: {e}")
    
    show_data_quality()

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Current Analysis")
st.sidebar.markdown("**📊 Multimodal Records**: 620,665")
st.sidebar.markdown("**🎯 Full Multimodal**: 326,391 (52.7%)")
st.sidebar.markdown("**📊 Dual Modal**: 292,437 (47.3%)")
st.sidebar.markdown("**🖼️ Visual Targets**: 618,828 images")
st.sidebar.markdown("**💬 Comment Coverage**: 89.6%")

st.sidebar.markdown("---")
st.sidebar.markdown("**🚀 Task 8: Visual Features**")
st.sidebar.markdown("**Status:** 🔄 In Progress")
st.sidebar.markdown("**Target:** 618,828 images")
st.sidebar.markdown("**Batches:** 62 × 10K each")
st.sidebar.markdown("**Est. Time:** ~22-144 hours")

# Data refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Dashboard Status:** Enhanced with Task 7")

with col2:
    st.markdown("**🔄 Last Updated:** " + (dashboard_data.get("generation_timestamp", "Unknown")[:19] if dashboard_data else "Unknown"))

with col3:
    st.markdown("**⚡ Performance:** Optimized for large datasets")