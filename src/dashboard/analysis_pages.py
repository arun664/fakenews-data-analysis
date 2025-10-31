"""
Analysis Result Pages for Dashboard

Contains functions for displaying different analysis results.
"""

import streamlit as st
import plotly.graph_objects as go
import json
from .styles import create_metric_card, create_insight_box, create_warning_box, create_success_box

def show_task1_results(eda_data):
    """Display Task 1: Dataset Loading results."""
    st.header("Task 1: Dataset Loading and Preprocessing")
    
    try:
        with open('analysis_results/task1_dataset_loading_report.json', 'r') as f:
            task1_data = json.load(f)
    except:
        st.error("Task 1 report not found.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            total_records = int(task1_data['dataset_statistics']['total_records'])
        except (KeyError, ValueError, TypeError):
            total_records = 0
        st.markdown(create_metric_card("Total Records", f"{total_records:,}"), unsafe_allow_html=True)
    
    with col2:
        try:
            columns_count = int(task1_data['data_quality']['columns_total'])
        except (KeyError, ValueError, TypeError):
            columns_count = 0
        st.markdown(create_metric_card("Features", f"{columns_count}"), unsafe_allow_html=True)
    
    with col3:
        try:
            multimodal_values = task1_data['multimodal_features']['multimodal_samples'].values()
            multimodal_count = sum(int(v) for v in multimodal_values if str(v).isdigit())
        except (KeyError, ValueError, TypeError):
            multimodal_count = 0
        st.markdown(create_metric_card("Multimodal Samples", f"{multimodal_count:,}"), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card("Status", "Complete"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Dataset Split Distribution")
    splits_data = task1_data['dataset_statistics']['splits']
    
    col1, col2 = st.columns(2)
    
    with col1:
        splits = list(splits_data.keys())
        counts = [splits_data[split]['records'] for split in splits]
        
        fig = go.Figure(data=[go.Bar(
            x=splits, 
            y=counts, 
            marker_color=['#007bff', '#ffc107', '#28a745']
        )])
        
        fig.update_layout(
            title="Records per Split",
            xaxis_title="Split",
            yaxis_title="Number of Records"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Category Distribution by Split:**")
        for split, data in task1_data['category_distribution'].items():
            if data:
                true_count = data.get(0, data.get('0', 0))
                false_count = data.get(1, data.get('1', 0))
                st.write(f"**{split.capitalize()}:** {true_count} true, {false_count} false")
    
    st.subheader("Key Insights")
    for insight in task1_data['key_insights']:
        st.markdown(create_insight_box(f"• {insight}"), unsafe_allow_html=True)
    
    st.subheader("Preprocessing Applied")
    for process in task1_data['preprocessing_applied']:
        st.markdown(f"- {process}")

def show_task2_results(eda_data):
    """Display Task 2: Leakage Analysis results."""
    st.header("Task 2: Data Leakage Detection and Mitigation")
    
    try:
        with open('analysis_results/task2_leakage_analysis_report.json', 'r') as f:
            task2_data = json.load(f)
    except:
        st.error("Task 2 report not found.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            initial_score = float(task2_data['leakage_detection']['initial_leakage_score'])
        except (KeyError, ValueError, TypeError):
            initial_score = 0.0
        st.markdown(create_metric_card("Initial Leakage Score", f"{initial_score:.3f}"), unsafe_allow_html=True)
    
    with col2:
        try:
            final_score = float(task2_data['mitigation_results']['final_leakage_score'])
        except (KeyError, ValueError, TypeError):
            final_score = 0.0
        st.markdown(create_metric_card("Final Leakage Score", f"{final_score:.3f}"), unsafe_allow_html=True)
    
    with col3:
        try:
            exact_removed = int(task2_data['mitigation_results']['actions_taken']['exact_duplicates_removed'])
            near_removed = int(task2_data['mitigation_results']['actions_taken']['near_duplicates_removed'])
            duplicates_removed = exact_removed + near_removed
        except (KeyError, ValueError, TypeError):
            duplicates_removed = 0
        st.markdown(create_metric_card("Duplicates Removed", f"{duplicates_removed}"), unsafe_allow_html=True)
    
    with col4:
        retention_rate = task2_data['mitigation_results']['data_retention']['retention_rate']
        st.markdown(create_metric_card("Data Retained", f"{retention_rate:.1f}%"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Leakage Detection Results")
        issues = task2_data['leakage_detection']['issues_found']
        
        warning_content = f"""
        <h4>Issues Identified:</h4>
        <ul>
            <li>Exact duplicates: {issues['exact_duplicates']}</li>
            <li>Near duplicates: {issues['near_duplicates']}</li>
            <li>Temporal issues: {issues['temporal_issues']}</li>
            <li>Metadata leakage: {issues['metadata_leakage']} splits</li>
        </ul>
        """
        st.markdown(create_warning_box("Issues Identified", warning_content), unsafe_allow_html=True)
    
    with col2:
        st.subheader("Mitigation Results")
        actions = task2_data['mitigation_results']['actions_taken']
        
        success_content = f"""
        <h4>Actions Taken:</h4>
        <ul>
            <li>Removed {actions['exact_duplicates_removed']} exact duplicates</li>
            <li>Removed {actions['near_duplicates_removed']} near duplicates</li>
            <li>Applied temporal splitting: {'' if actions['temporal_splitting_applied'] else 'Error: '}</li>
            <li>Applied engagement capping: {'' if actions['engagement_capping_applied'] else 'Error: '}</li>
        </ul>
        """
        st.markdown(create_success_box("Actions Taken", success_content), unsafe_allow_html=True)
    
    st.subheader("Final Dataset Quality")
    final_data = task2_data['final_dataset']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Train Records", f"{final_data['train_records']:,}")
    
    with col2:
        st.metric("Validation Records", f"{final_data['validation_records']:,}")
    
    with col3:
        st.metric("Test Records", f"{final_data['test_records']:,}")
    
    st.subheader("Key Insights")
    for insight in task2_data['key_insights']:
        st.markdown(create_insight_box(f"• {insight}"), unsafe_allow_html=True)

def show_text_analysis_results(eda_data):
    """Display text analysis results."""
    st.header("Text Analysis Results")
    
    if 'text_analysis' not in eda_data:
        st.info("Text analysis data not available. Please run the text analysis first.")
        return
    
    text_data = eda_data['text_analysis']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'clean_title' in text_data and 'basic_stats' in text_data['clean_title']:
            avg_length = text_data['clean_title']['basic_stats']['avg_length']
            st.metric("Average Text Length", f"{avg_length:.1f} chars")
    
    with col2:
        if 'clean_title' in text_data and 'basic_stats' in text_data['clean_title']:
            max_length = text_data['clean_title']['basic_stats']['max_length']
            st.metric("Maximum Text Length", f"{max_length} chars")
    
    with col3:
        if 'clean_title' in text_data and 'basic_stats' in text_data['clean_title']:
            avg_words = text_data['clean_title']['basic_stats']['avg_words']
            st.metric("Average Word Count", f"{avg_words:.1f} words")
    
    st.markdown("---")
    
    if 'category_patterns' in text_data.get('clean_title', {}):
        st.subheader("Category-Specific Text Patterns")
        patterns = text_data['clean_title']['category_patterns']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**True Content Characteristics:**")
            if 'True' in patterns:
                true_data = patterns['True']
                st.write(f"- Average Length: {true_data['avg_length']:.1f} characters")
                st.write(f"- Exclamation Density: {true_data['exclamation_density']:.3f}")
                st.write(f"- Question Density: {true_data['question_density']:.3f}")
                st.write(f"- Caps Ratio: {true_data['caps_ratio']:.3f}")
        
        with col2:
            st.write("**False Content Characteristics:**")
            if 'False' in patterns:
                false_data = patterns['False']
                st.write(f"- Average Length: {false_data['avg_length']:.1f} characters")
                st.write(f"- Exclamation Density: {false_data['exclamation_density']:.3f}")
                st.write(f"- Question Density: {false_data['question_density']:.3f}")
                st.write(f"- Caps Ratio: {false_data['caps_ratio']:.3f}")
        
        if 'True' in patterns and 'False' in patterns:
            st.subheader("Text Pattern Comparison")
            
            true_len = patterns['True']['avg_length']
            false_len = patterns['False']['avg_length']
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='True Content',
                x=['Average Length'],
                y=[true_len],
                marker_color='#28a745'
            ))
            fig.add_trace(go.Bar(
                name='False Content',
                x=['Average Length'],
                y=[false_len],
                marker_color='#dc3545'
            ))
            
            fig.update_layout(
                title='Average Text Length by Category',
                yaxis_title='Characters',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            diff_pct = ((false_len - true_len) / true_len) * 100
            insight_text = f"**Key Insight:** False content has {abs(diff_pct):.1f}% {'longer' if diff_pct > 0 else 'shorter'} headlines than true content, suggesting {'sensationalization' if diff_pct > 0 else 'brevity'} as a distinguishing characteristic."
            st.markdown(create_insight_box(insight_text), unsafe_allow_html=True)