"""
Dashboard Pages
Individual page components for the dashboard
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any
from .components import MetricCard, StatusIndicator, ProgressBar, ChartFactory, DataTable, Layout
from .data_loader import DataLoader, DataProcessor
from .config import DashboardConfig

class AssociationRulesPage:
    """Association rule mining analysis page"""
    
    @staticmethod
    def render():
        """Render association rules page"""
        
        Layout.create_header("Cross-Modal Association Rule Mining", "Discover patterns between visual, textual, and authenticity features")
        
        # Load association mining data
        try:
            association_data = DataLoader.load_json("analysis_results/dashboard_data/association_mining_dashboard_data.json")
            
            if association_data:
                # Overview metrics
                st.subheader("📊 Association Mining Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    MetricCard.render(
                        "Total Rules", 
                        f"{association_data['association_mining_overview']['total_rules']:,}",
                        icon="🔗"
                    )
                
                with col2:
                    MetricCard.render(
                        "Authenticity Rules",
                        f"{association_data['association_mining_overview']['authenticity_rules']:,}",
                        icon="🎯"
                    )
                
                with col3:
                    MetricCard.render(
                        "Fake Predictors",
                        f"{association_data['association_mining_overview']['fake_content_rules']:,}",
                        icon="❌"
                    )
                
                with col4:
                    MetricCard.render(
                        "Real Predictors",
                        f"{association_data['association_mining_overview']['authentic_content_rules']:,}",
                        icon="✅"
                    )
                
                # Mining parameters
                st.subheader("⚙️ Mining Parameters")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Min Support", f"{association_data['association_mining_overview']['min_support']:.3f}")
                with col2:
                    st.metric("Min Confidence", f"{association_data['association_mining_overview']['min_confidence']:.1f}")
                with col3:
                    st.metric("Min Lift", f"{association_data['association_mining_overview']['min_lift']:.1f}")
                
                # Top fake content indicators
                st.subheader("🚨 Top Fake Content Indicators")
                if association_data.get('top_fake_indicators'):
                    fake_df = pd.DataFrame(association_data['top_fake_indicators'])
                    fake_df['features_str'] = fake_df['features'].apply(lambda x: ', '.join(x))
                    
                    fig_fake = px.bar(
                        fake_df.head(10), 
                        x='confidence', 
                        y='features_str',
                        title="Top 10 Features Predicting Fake Content",
                        labels={'confidence': 'Confidence', 'features_str': 'Feature Combination'},
                        orientation='h'
                    )
                    fig_fake.update_layout(height=400)
                    st.plotly_chart(fig_fake, use_container_width=True)
                
                # Top authentic content indicators  
                st.subheader("✅ Top Authentic Content Indicators")
                if association_data.get('top_authentic_indicators'):
                    auth_df = pd.DataFrame(association_data['top_authentic_indicators'])
                    auth_df['features_str'] = auth_df['features'].apply(lambda x: ', '.join(x))
                    
                    fig_auth = px.bar(
                        auth_df.head(10), 
                        x='confidence', 
                        y='features_str',
                        title="Top 10 Features Predicting Authentic Content",
                        labels={'confidence': 'Confidence', 'features_str': 'Feature Combination'},
                        orientation='h',
                        color_discrete_sequence=['green']
                    )
                    fig_auth.update_layout(height=400)
                    st.plotly_chart(fig_auth, use_container_width=True)
                
                # Rule quality metrics
                st.subheader("📈 Rule Quality Metrics")
                if association_data.get('rule_metrics'):
                    metrics = association_data['rule_metrics']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Support Statistics**")
                        st.write(f"Mean: {metrics['support_stats']['mean']:.4f}")
                        st.write(f"Std: {metrics['support_stats']['std']:.4f}")
                        st.write(f"Range: {metrics['support_stats']['min']:.4f} - {metrics['support_stats']['max']:.4f}")
                    
                    with col2:
                        st.write("**Confidence Statistics**")
                        st.write(f"Mean: {metrics['confidence_stats']['mean']:.4f}")
                        st.write(f"Std: {metrics['confidence_stats']['std']:.4f}")
                        st.write(f"Range: {metrics['confidence_stats']['min']:.4f} - {metrics['confidence_stats']['max']:.4f}")
                
                # Interactive visualizations
                st.subheader("🎨 Interactive Visualizations")
                
                viz_tabs = st.tabs(["Rule Metrics", "Top Rules", "Authenticity Patterns"])
                
                with viz_tabs[0]:
                    if Path("visualizations/association_patterns/rule_metrics.html").exists():
                        with open("visualizations/association_patterns/rule_metrics.html", 'r') as f:
                            st.components.v1.html(f.read(), height=600)
                    else:
                        st.info("Rule metrics visualization not available")
                
                with viz_tabs[1]:
                    if Path("visualizations/association_patterns/top_rules.html").exists():
                        with open("visualizations/association_patterns/top_rules.html", 'r') as f:
                            st.components.v1.html(f.read(), height=800)
                    else:
                        st.info("Top rules visualization not available")
                
                with viz_tabs[2]:
                    if Path("visualizations/association_patterns/authenticity_patterns.html").exists():
                        with open("visualizations/association_patterns/authenticity_patterns.html", 'r') as f:
                            st.components.v1.html(f.read(), height=600)
                    else:
                        st.info("Authenticity patterns visualization not available")
                
            else:
                st.warning("Association mining data not available. Please run Task 11 first.")
                
        except Exception as e:
            st.error(f"Error loading association mining data: {e}")

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

class ComparativeAnalysisPage:
    """Cross-Modal Authenticity Comparative Analysis Page"""
    
    @staticmethod
    def render():
        """Render comparative analysis page"""
        
        Layout.create_header("Cross-Modal Authenticity Comparative Analysis", "Compare authenticity patterns across content types")
        
        # Load comparative analysis data
        try:
            with open('analysis_results/dashboard_data/comparative_analysis_dashboard.json', 'r') as f:
                data = json.load(f)
            
            comparative_data = data['comparative_analysis']
            
            st.markdown("### 📊 Content Type Distribution")
            
            # Display authenticity by content type
            if 'authenticity_by_content_type' in comparative_data:
                auth_data = comparative_data['authenticity_by_content_type']
                
                # Create DataFrame for display
                display_data = []
                for content_type, stats in auth_data.items():
                    display_data.append({
                        'Content Type': content_type.replace('_', ' ').title(),
                        'Total Records': f"{stats['total_count']:,}",
                        'Fake Count': f"{stats['fake_count']:,}",
                        'Real Count': f"{stats['real_count']:,}",
                        'Fake Rate': f"{stats['fake_rate']:.1%}",
                        'Real Rate': f"{stats['real_rate']:.1%}"
                    })
                
                df_display = pd.DataFrame(display_data)
                st.dataframe(df_display, use_container_width=True)
            
            st.markdown("### 📈 Statistical Significance Tests")
            
            # Display statistical test results
            if 'statistical_tests' in comparative_data:
                stat_tests = comparative_data['statistical_tests']
                
                # Chi-square test results
                if 'chi_square_test' in stat_tests:
                    chi2_test = stat_tests['chi_square_test']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        MetricCard.render("Chi-square Statistic", f"{chi2_test['chi2_statistic']:.2f}", icon="📊")
                    with col2:
                        MetricCard.render("P-value", f"{chi2_test['p_value']:.2e}", icon="🔬")
                    with col3:
                        significance = "Significant" if chi2_test['significant'] else "Not Significant"
                        MetricCard.render("Result", significance, icon="✅" if chi2_test['significant'] else "❌")
                
                # Pairwise comparisons
                if 'pairwise_comparisons' in stat_tests:
                    st.markdown("#### Pairwise Comparisons")
                    
                    pairwise_data = []
                    for comparison, stats in stat_tests['pairwise_comparisons'].items():
                        pairwise_data.append({
                            'Comparison': comparison.replace('_vs_', ' vs ').replace('_', ' ').title(),
                            'P-value': f"{stats['p_value']:.2e}",
                            'Cohen\'s d': f"{stats['cohens_d']:.3f}",
                            'Effect Size': stats['effect_size'].title(),
                            'Significant': "Yes" if stats['significant'] else "No"
                        })
                    
                    pairwise_df = pd.DataFrame(pairwise_data)
                    st.dataframe(pairwise_df, use_container_width=True)
            
            # Display interactive charts
            st.markdown("### 📊 Interactive Visualizations")
            
            # Load and display interactive dashboard if available
            interactive_file = 'visualizations/comparative_charts/interactive_comparative_dashboard.html'
            if Path(interactive_file).exists():
                with open(interactive_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=800)
            
            # Display static charts
            chart_files = [
                ('Authenticity by Content Type', 'visualizations/comparative_charts/authenticity_by_content_type.png'),
                ('Statistical Significance', 'visualizations/comparative_charts/statistical_significance_heatmap.png')
            ]
            
            for chart_title, chart_path in chart_files:
                if Path(chart_path).exists():
                    st.markdown(f"#### {chart_title}")
                    st.image(chart_path, use_column_width=True)
            
        except Exception as e:
            st.error(f"Error loading comparative analysis data: {e}")
            st.info("Please ensure the comparative analysis has been completed.")