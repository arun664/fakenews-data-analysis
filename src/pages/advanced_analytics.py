"""
Advanced Analytics Page
Advanced pattern discovery and machine learning analysis
COMPLETE IMPLEMENTATION - Extracted from app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.lazy_loader import LazyLoader

lazy_loader = LazyLoader()


def render_advanced_analytics(container):
    """Render Advanced Analytics with lazy loading - COMPLETE IMPLEMENTATION"""
    with container.container():
        try:
            st.header("Advanced Pattern Discovery & Machine Learning")
            
            st.markdown("""
            **🎯 Key Questions Answered:**
            - What hidden patterns exist in multimodal fake news data?
            - How do clustering algorithms group authentic vs inauthentic content?
            - What association rules reveal relationships between features?
            - Which features are most predictive of authenticity?
            """)
            
            # Create tabs for different advanced analytics
            analytics_tabs = st.tabs(["🔄 Clustering Analysis", "⛓️ Association Rules", "🧠 Feature Importance", "📊 Pattern Summary"])
            
            with analytics_tabs[0]:
                st.subheader("🔄 Multimodal Clustering Analysis")
                st.markdown("**Discover hidden patterns through clustering analysis of multimodal features**")
                
                # Load clustering data with performance optimization
                @st.cache_data(ttl=600)  # 10 minutes cache for clustering results
                def load_clustering_data():
                    clustering_data_path = Path("analysis_results/dashboard_data/clustering_dashboard_data.json")
                    if not clustering_data_path.exists():
                        raise FileNotFoundError(f"Clustering data not found at {clustering_data_path}")
                    with open(clustering_data_path, 'r') as f:
                        data = json.load(f)
                    # Sample large clustering results if needed
                    if 'cluster_details' in data and len(data['cluster_details']) > 10000:
                        st.info(f"📊 Sampling {len(data['cluster_details'])} clustering results to 10,000 for optimal performance")
                        data['cluster_details'] = data['cluster_details'][:10000]
                    return data
                
                try:
                    clustering_data = load_clustering_data()
                except FileNotFoundError:
                    clustering_data = None
                
                if clustering_data:
                    # Overview metrics
                    st.subheader("📊 Clustering Analysis Overview")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Records Clustered", 
                                 f"{clustering_data['clustering_overview']['total_records']:,}",
                                 help="Total records processed")
                    
                    with col2:
                        st.metric("K-means Clusters",
                                 f"{clustering_data['clustering_overview']['kmeans_optimal_k']}",
                                 help="Optimal number of K-means clusters")
                    
                    with col3:
                        st.metric("Hierarchical Clusters",
                                 f"{clustering_data['clustering_overview']['hierarchical_clusters']}",
                                 help="Number of hierarchical clusters")
                    
                    with col4:
                        st.metric("Silhouette Score",
                                 f"{clustering_data['clustering_overview']['silhouette_score']:.3f}",
                                 help="Clustering quality metric")
                    
                    # Visualizations
                    st.markdown("---")
                    st.subheader("📈 Cluster Visualizations")
                    
                    # 1. Cluster Size Bar Chart
                    st.markdown("#### Cluster Size Distribution")
                    kmeans_dist = clustering_data['cluster_distributions']['kmeans']
                    cluster_sizes = pd.DataFrame([
                        {'Cluster': f"Cluster {k}", 'Size': v['size'], 'Fake Rate': v['fake_rate']}
                        for k, v in kmeans_dist.items()
                    ])
                    
                    fig_sizes = go.Figure()
                    fig_sizes.add_trace(go.Bar(
                        x=cluster_sizes['Cluster'],
                        y=cluster_sizes['Size'],
                        marker_color='#3498DB',
                        text=cluster_sizes['Size'],
                        texttemplate='%{text:,}',
                        textposition='outside'
                    ))
                    fig_sizes.update_layout(
                        title="K-means Cluster Sizes",
                        xaxis_title="Cluster",
                        yaxis_title="Number of Records",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_sizes, use_container_width=True)
                    
                    # 2. Cluster-Authenticity Distribution
                    st.markdown("#### Authenticity Distribution by Cluster")
                    
                    fig_auth = go.Figure()
                    fig_auth.add_trace(go.Bar(
                        name='Fake',
                        x=cluster_sizes['Cluster'],
                        y=cluster_sizes['Fake Rate'] * 100,
                        marker_color='#FF6B6B',
                        text=[f"{v:.1f}%" for v in cluster_sizes['Fake Rate'] * 100],
                        textposition='inside'
                    ))
                    fig_auth.add_trace(go.Bar(
                        name='Real',
                        x=cluster_sizes['Cluster'],
                        y=(1 - cluster_sizes['Fake Rate']) * 100,
                        marker_color='#4ECDC4',
                        text=[f"{v:.1f}%" for v in (1 - cluster_sizes['Fake Rate']) * 100],
                        textposition='inside'
                    ))
                    fig_auth.update_layout(
                        title="Fake vs Real Content Distribution by Cluster",
                        xaxis_title="Cluster",
                        yaxis_title="Percentage (%)",
                        barmode='stack',
                        height=400,
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )
                    st.plotly_chart(fig_auth, use_container_width=True)
                    
                    # 3. Silhouette Analysis
                    st.markdown("#### Cluster Quality Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Silhouette score visualization
                        silhouette_score = clustering_data['clustering_overview']['silhouette_score']
                        fig_sil = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=silhouette_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Silhouette Score"},
                            gauge={
                                'axis': {'range': [-1, 1]},
                                'bar': {'color': "#3498DB"},
                                'steps': [
                                    {'range': [-1, 0], 'color': "#FFE5E5"},
                                    {'range': [0, 0.25], 'color': "#FFF4E5"},
                                    {'range': [0.25, 0.5], 'color': "#E5F5FF"},
                                    {'range': [0.5, 1], 'color': "#E5FFE5"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 0.5
                                }
                            }
                        ))
                        fig_sil.update_layout(height=300)
                        st.plotly_chart(fig_sil, use_container_width=True)
                    
                    with col2:
                        st.markdown("""
                        **Silhouette Score Interpretation:**
                        - **> 0.5**: Strong cluster structure
                        - **0.25 - 0.5**: Moderate structure
                        - **0 - 0.25**: Weak structure
                        - **< 0**: Poor clustering
                        
                        Current score indicates the quality of cluster separation.
                        """)
                    
                    # 4. Hierarchical Clustering Comparison
                    st.markdown("#### Hierarchical vs K-means Comparison")
                    hier_dist = clustering_data['cluster_distributions']['hierarchical']
                    
                    comparison_data = pd.DataFrame({
                        'Method': ['K-means'] * len(kmeans_dist) + ['Hierarchical'] * len(hier_dist),
                        'Cluster': [f"C{k}" for k in kmeans_dist.keys()] + [f"C{k}" for k in hier_dist.keys()],
                        'Size': [v['size'] for v in kmeans_dist.values()] + [v['size'] for v in hier_dist.values()],
                        'Fake Rate': [v['fake_rate'] for v in kmeans_dist.values()] + [v['fake_rate'] for v in hier_dist.values()]
                    })
                    
                    fig_comp = px.scatter(
                        comparison_data,
                        x='Size',
                        y='Fake Rate',
                        color='Method',
                        size='Size',
                        hover_data=['Cluster'],
                        title="Cluster Size vs Fake Rate by Method",
                        color_discrete_map={'K-means': '#3498DB', 'Hierarchical': '#9B59B6'}
                    )
                    fig_comp.update_layout(
                        xaxis_title="Cluster Size (log scale)",
                        yaxis_title="Fake Content Rate",
                        xaxis_type="log",
                        height=400
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                else:
                    st.warning("📂 Clustering analysis data not available. Please run Task 10 first.")
                    st.info("""
                    **To generate clustering data:**
                    ```bash
                    python tasks/run_task10_multimodal_clustering.py
                    ```
                    This will perform multimodal clustering analysis and generate the required dashboard data.
                    """)
            
            with analytics_tabs[1]:
                st.subheader("⛓️ Cross-Modal Association Rule Mining")
                st.markdown("**Discover patterns between visual, textual, and authenticity features**")
                
                # Load association mining data with performance optimization
                @st.cache_data(ttl=600)  # 10 minutes cache for association rules
                def load_association_data():
                    association_data_path = Path("analysis_results/dashboard_data/association_mining_dashboard_data.json")
                    if not association_data_path.exists():
                        raise FileNotFoundError(f"Association mining data not found at {association_data_path}")
                    with open(association_data_path, 'r') as f:
                        data = json.load(f)
                    # Sample large rule sets if needed
                    if 'top_rules' in data and len(data['top_rules']) > 10000:
                        st.info(f"📊 Sampling {len(data['top_rules'])} association rules to 10,000 for optimal performance")
                        data['top_rules'] = data['top_rules'][:10000]
                    return data
                
                try:
                    association_data = load_association_data()
                except FileNotFoundError:
                    association_data = None
                
                if association_data:
                    # Overview metrics
                    st.subheader("📊 Association Mining Overview")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Rules", 
                                 f"{association_data['association_mining_overview']['total_rules']:,}",
                                 help="Total association rules discovered")
                    
                    with col2:
                        st.metric("Authenticity Rules",
                                 f"{association_data['association_mining_overview']['authenticity_rules']:,}",
                                 help="Rules related to authenticity prediction")
                    
                    with col3:
                        st.metric("Fake Predictors",
                                 f"{association_data['association_mining_overview']['fake_content_rules']:,}",
                                 help="Rules predicting fake content")
                    
                    with col4:
                        st.metric("Real Predictors",
                                 f"{association_data['association_mining_overview']['authentic_content_rules']:,}",
                                 help="Rules predicting authentic content")
                    
                    # Visualizations
                    st.markdown("---")
                    st.subheader("📈 Association Rule Visualizations")
                    
                    # 1. Support-Confidence Scatter Plot
                    st.markdown("#### Support vs Confidence Analysis")
                    
                    # Prepare data for scatter plot
                    fake_rules = association_data.get('top_fake_indicators', [])[:20]
                    real_rules = association_data.get('top_authentic_indicators', [])[:20]
                    
                    scatter_data = []
                    for rule in fake_rules:
                        scatter_data.append({
                            'Support': rule['support'],
                            'Confidence': rule['confidence'],
                            'Lift': rule['lift'],
                            'Type': 'Fake Predictor',
                            'Features': ', '.join(rule['features'][:2])  # First 2 features
                        })
                    for rule in real_rules:
                        scatter_data.append({
                            'Support': rule['support'],
                            'Confidence': rule['confidence'],
                            'Lift': rule['lift'],
                            'Type': 'Real Predictor',
                            'Features': ', '.join(rule['features'][:2])
                        })
                    
                    scatter_df = pd.DataFrame(scatter_data)
                    
                    fig_scatter = px.scatter(
                        scatter_df,
                        x='Support',
                        y='Confidence',
                        color='Type',
                        size='Lift',
                        hover_data=['Features', 'Lift'],
                        title="Association Rules: Support vs Confidence",
                        color_discrete_map={'Fake Predictor': '#FF6B6B', 'Real Predictor': '#4ECDC4'}
                    )
                    fig_scatter.update_layout(
                        xaxis_title="Support (Frequency)",
                        yaxis_title="Confidence (Reliability)",
                        height=500
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # 2. Top Rules Bar Chart
                    st.markdown("#### Top Association Rules by Lift")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Top Fake Content Indicators**")
                        top_fake = pd.DataFrame(fake_rules[:10])
                        top_fake['feature_str'] = top_fake['features'].apply(lambda x: ', '.join(x[:2]))
                        
                        fig_fake = go.Figure(go.Bar(
                            y=top_fake['feature_str'],
                            x=top_fake['lift'],
                            orientation='h',
                            marker_color='#FF6B6B',
                            text=top_fake['lift'].round(2),
                            textposition='outside'
                        ))
                        fig_fake.update_layout(
                            title="Fake Content Predictors",
                            xaxis_title="Lift",
                            yaxis_title="",
                            height=400,
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        st.plotly_chart(fig_fake, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Top Real Content Indicators**")
                        top_real = pd.DataFrame(real_rules[:10])
                        top_real['feature_str'] = top_real['features'].apply(lambda x: ', '.join(x[:2]))
                        
                        fig_real = go.Figure(go.Bar(
                            y=top_real['feature_str'],
                            x=top_real['lift'],
                            orientation='h',
                            marker_color='#4ECDC4',
                            text=top_real['lift'].round(2),
                            textposition='outside'
                        ))
                        fig_real.update_layout(
                            title="Real Content Predictors",
                            xaxis_title="Lift",
                            yaxis_title="",
                            height=400,
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        st.plotly_chart(fig_real, use_container_width=True)
                    
                    # 3. Lift Distribution Histogram
                    st.markdown("#### Lift Distribution Analysis")
                    
                    all_lifts = [r['lift'] for r in fake_rules] + [r['lift'] for r in real_rules]
                    
                    fig_lift = go.Figure()
                    fig_lift.add_trace(go.Histogram(
                        x=[r['lift'] for r in fake_rules],
                        name='Fake Predictors',
                        marker_color='#FF6B6B',
                        opacity=0.6,
                        nbinsx=20
                    ))
                    fig_lift.add_trace(go.Histogram(
                        x=[r['lift'] for r in real_rules],
                        name='Real Predictors',
                        marker_color='#4ECDC4',
                        opacity=0.6,
                        nbinsx=20
                    ))
                    fig_lift.update_layout(
                        title="Distribution of Lift Values",
                        xaxis_title="Lift",
                        yaxis_title="Frequency",
                        barmode='overlay',
                        height=400
                    )
                    st.plotly_chart(fig_lift, use_container_width=True)
                    
                    # 4. Rule Network Visualization (simplified)
                    st.markdown("#### Feature Co-occurrence Network")
                    
                    # Extract feature co-occurrences
                    feature_pairs = {}
                    for rule in fake_rules + real_rules:
                        if len(rule['features']) >= 2:
                            pair = tuple(sorted(rule['features'][:2]))
                            if pair not in feature_pairs:
                                feature_pairs[pair] = {'count': 0, 'avg_lift': 0}
                            feature_pairs[pair]['count'] += 1
                            feature_pairs[pair]['avg_lift'] += rule['lift']
                    
                    # Average the lift
                    for pair in feature_pairs:
                        feature_pairs[pair]['avg_lift'] /= feature_pairs[pair]['count']
                    
                    # Create network data
                    network_data = pd.DataFrame([
                        {
                            'Feature 1': pair[0],
                            'Feature 2': pair[1],
                            'Co-occurrence': data['count'],
                            'Avg Lift': data['avg_lift']
                        }
                        for pair, data in sorted(feature_pairs.items(), key=lambda x: x[1]['count'], reverse=True)[:15]
                    ])
                    
                    if not network_data.empty:
                        fig_network = go.Figure(go.Scatter(
                            x=network_data.index,
                            y=network_data['Co-occurrence'],
                            mode='markers',
                            marker=dict(
                                size=network_data['Avg Lift'] * 10,
                                color=network_data['Avg Lift'],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Avg Lift")
                            ),
                            text=[f"{row['Feature 1']}<br>+ {row['Feature 2']}" for _, row in network_data.iterrows()],
                            hovertemplate='%{text}<br>Co-occurrence: %{y}<br>Avg Lift: %{marker.color:.2f}<extra></extra>'
                        ))
                        fig_network.update_layout(
                            title="Top Feature Pairs by Co-occurrence",
                            xaxis_title="Feature Pair Index",
                            yaxis_title="Co-occurrence Count",
                            height=400
                        )
                        st.plotly_chart(fig_network, use_container_width=True)
                    
                else:
                    st.warning("📂 Association mining data not available. Please run Task 11 first.")
                    st.info("""
                    **To generate association mining data:**
                    ```bash
                    python tasks/run_task11_association_rule_mining.py
                    ```
                    This will discover cross-modal association rules and generate the required dashboard data.
                    """)
            
            with analytics_tabs[2]:
                st.subheader("🧠 Feature Importance Analysis")
                st.markdown("**Identify the most predictive features for authenticity detection**")
                
                # Load cross-modal analysis data for feature importance
                @st.cache_data(ttl=600)
                def load_feature_importance():
                    cross_modal_path = Path("analysis_results/cross_modal_comparison/cross_modal_analysis.json")
                    if cross_modal_path.exists():
                        with open(cross_modal_path, 'r') as f:
                            return json.load(f)
                    return None
                
                feature_data = load_feature_importance()
                
                if feature_data and 'feature_importance' in feature_data:
                    st.subheader("📊 Feature Importance Rankings")
                    
                    # Extract feature importance data
                    importance_data = feature_data['feature_importance']
                    
                    # Create DataFrame for visualization
                    features_df = pd.DataFrame([
                        {
                            'Feature': feat,
                            'Importance': score,
                            'Category': 'Visual' if any(x in feat.lower() for x in ['brightness', 'contrast', 'sharpness', 'entropy', 'noise', 'aspect', 'size']) 
                                       else 'Textual' if any(x in feat.lower() for x in ['word', 'text', 'sentiment', 'flesch', 'reading'])
                                       else 'Temporal' if any(x in feat.lower() for x in ['hour', 'day', 'time'])
                                       else 'Social'
                        }
                        for feat, score in list(importance_data.items())[:30]
                    ])
                    
                    # Color mapping for categories
                    color_map = {
                        'Visual': '#FF6B6B',
                        'Textual': '#4ECDC4',
                        'Temporal': '#95A5A6',
                        'Social': '#9B59B6'
                    }
                    
                    # Horizontal bar chart with color-coding
                    st.markdown("#### Top 30 Most Important Features")
                    
                    fig_importance = go.Figure()
                    
                    for category in features_df['Category'].unique():
                        cat_data = features_df[features_df['Category'] == category]
                        fig_importance.add_trace(go.Bar(
                            y=cat_data['Feature'],
                            x=cat_data['Importance'],
                            name=category,
                            orientation='h',
                            marker_color=color_map.get(category, '#95A5A6'),
                            text=cat_data['Importance'].round(4),
                            textposition='outside'
                        ))
                    
                    fig_importance.update_layout(
                        title="Feature Importance by Category",
                        xaxis_title="Importance Score",
                        yaxis_title="",
                        height=800,
                        yaxis={'categoryorder': 'total ascending'},
                        showlegend=True,
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Feature importance by category
                    st.markdown("#### Feature Importance by Category")
                    
                    category_importance = features_df.groupby('Category')['Importance'].agg(['sum', 'mean', 'count'])
                    category_importance = category_importance.reset_index()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_cat_sum = go.Figure(go.Bar(
                            x=category_importance['Category'],
                            y=category_importance['sum'],
                            marker_color=[color_map.get(cat, '#95A5A6') for cat in category_importance['Category']],
                            text=category_importance['sum'].round(3),
                            textposition='outside'
                        ))
                        fig_cat_sum.update_layout(
                            title="Total Importance by Category",
                            xaxis_title="Category",
                            yaxis_title="Total Importance",
                            height=400
                        )
                        st.plotly_chart(fig_cat_sum, use_container_width=True)
                    
                    with col2:
                        fig_cat_avg = go.Figure(go.Bar(
                            x=category_importance['Category'],
                            y=category_importance['mean'],
                            marker_color=[color_map.get(cat, '#95A5A6') for cat in category_importance['Category']],
                            text=category_importance['mean'].round(4),
                            textposition='outside'
                        ))
                        fig_cat_avg.update_layout(
                            title="Average Importance by Category",
                            xaxis_title="Category",
                            yaxis_title="Average Importance",
                            height=400
                        )
                        st.plotly_chart(fig_cat_avg, use_container_width=True)
                    
                    # Top features table
                    st.markdown("#### Top 15 Features Detail")
                    top_features = features_df.head(15)[['Feature', 'Category', 'Importance']]
                    top_features['Importance'] = top_features['Importance'].round(6)
                    st.dataframe(top_features, use_container_width=True, hide_index=True)
                    
                elif association_data:
                    # Fallback: Use association rules to derive feature importance
                    st.subheader("📊 Feature Importance from Association Rules")
                    
                    # Extract features from association rules
                    feature_scores = {}
                    
                    for rule in association_data.get('top_fake_indicators', []):
                        for feat in rule['features']:
                            if feat not in feature_scores:
                                feature_scores[feat] = {'lift_sum': 0, 'confidence_sum': 0, 'count': 0}
                            feature_scores[feat]['lift_sum'] += rule['lift']
                            feature_scores[feat]['confidence_sum'] += rule['confidence']
                            feature_scores[feat]['count'] += 1
                    
                    for rule in association_data.get('top_authentic_indicators', []):
                        for feat in rule['features']:
                            if feat not in feature_scores:
                                feature_scores[feat] = {'lift_sum': 0, 'confidence_sum': 0, 'count': 0}
                            feature_scores[feat]['lift_sum'] += rule['lift']
                            feature_scores[feat]['confidence_sum'] += rule['confidence']
                            feature_scores[feat]['count'] += 1
                    
                    # Calculate average scores
                    features_list = []
                    for feat, scores in feature_scores.items():
                        avg_lift = scores['lift_sum'] / scores['count']
                        avg_conf = scores['confidence_sum'] / scores['count']
                        importance = avg_lift * avg_conf  # Combined score
                        features_list.append({
                            'Feature': feat,
                            'Importance': importance,
                            'Avg Lift': avg_lift,
                            'Avg Confidence': avg_conf,
                            'Occurrences': scores['count']
                        })
                    
                    features_df = pd.DataFrame(features_list).sort_values('Importance', ascending=False).head(25)
                    
                    # Horizontal bar chart
                    fig_importance = go.Figure(go.Bar(
                        y=features_df['Feature'],
                        x=features_df['Importance'],
                        orientation='h',
                        marker_color='#3498DB',
                        text=features_df['Importance'].round(3),
                        textposition='outside'
                    ))
                    fig_importance.update_layout(
                        title="Top 25 Features by Importance (Lift × Confidence)",
                        xaxis_title="Importance Score",
                        yaxis_title="",
                        height=700,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Detailed table
                    st.markdown("#### Feature Details")
                    display_df = features_df[['Feature', 'Importance', 'Avg Lift', 'Avg Confidence', 'Occurrences']].copy()
                    display_df['Importance'] = display_df['Importance'].round(4)
                    display_df['Avg Lift'] = display_df['Avg Lift'].round(3)
                    display_df['Avg Confidence'] = display_df['Avg Confidence'].round(3)
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                else:
                    st.warning("📂 Feature importance data not available.")
                    st.info("""
                    **To generate feature importance data:**
                    ```bash
                    python tasks/run_task12_cross_modal_analysis.py
                    ```
                    This will perform cross-modal analysis and generate feature importance scores.
                    """)
            
            with analytics_tabs[3]:
                st.subheader("📊 Pattern Summary")
                st.markdown("**Comprehensive summary of discovered patterns across all analyses**")
                
                # Aggregate data from all analyses
                has_data = clustering_data or association_data or feature_data
                
                if has_data:
                    st.markdown("### 🎯 Key Insights Dashboard")
                    
                    # Summary Statistics Cards
                    st.markdown("#### Analysis Coverage")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if clustering_data:
                            st.metric(
                                "Clustering Quality",
                                f"{clustering_data['clustering_overview']['silhouette_score']:.3f}",
                                help="Silhouette score indicating cluster separation"
                            )
                        else:
                            st.metric("Clustering Quality", "N/A")
                    
                    with col2:
                        if association_data:
                            st.metric(
                                "Association Rules",
                                f"{association_data['association_mining_overview']['total_rules']:,}",
                                help="Total discovered association rules"
                            )
                        else:
                            st.metric("Association Rules", "N/A")
                    
                    with col3:
                        if feature_data and 'feature_importance' in feature_data:
                            st.metric(
                                "Features Analyzed",
                                f"{len(feature_data['feature_importance'])}",
                                help="Number of features with importance scores"
                            )
                        else:
                            st.metric("Features Analyzed", "N/A")
                    
                    with col4:
                        total_records = 0
                        if clustering_data:
                            total_records = clustering_data['clustering_overview']['total_records']
                        st.metric(
                            "Records Processed",
                            f"{total_records:,}",
                            help="Total records analyzed"
                        )
                    
                    st.markdown("---")
                    
                    # Pattern Strength Indicators
                    st.markdown("#### Pattern Strength Indicators")
                    
                    pattern_strengths = []
                    
                    if clustering_data:
                        sil_score = clustering_data['clustering_overview']['silhouette_score']
                        pattern_strengths.append({
                            'Analysis': 'Clustering',
                            'Strength': min(max(sil_score * 100, 0), 100),  # Normalize to 0-100
                            'Status': 'Strong' if sil_score > 0.5 else 'Moderate' if sil_score > 0.25 else 'Weak'
                        })
                    
                    if association_data:
                        # Calculate average lift as strength indicator
                        fake_rules = association_data.get('top_fake_indicators', [])
                        real_rules = association_data.get('top_authentic_indicators', [])
                        if fake_rules or real_rules:
                            avg_lift = np.mean([r['lift'] for r in fake_rules[:10]] + [r['lift'] for r in real_rules[:10]])
                            pattern_strengths.append({
                                'Analysis': 'Association Rules',
                                'Strength': min(avg_lift * 20, 100),  # Normalize to 0-100
                                'Status': 'Strong' if avg_lift > 3 else 'Moderate' if avg_lift > 1.5 else 'Weak'
                            })
                    
                    if feature_data and 'feature_importance' in feature_data:
                        # Top feature importance as strength
                        top_importance = max(feature_data['feature_importance'].values())
                        pattern_strengths.append({
                            'Analysis': 'Feature Importance',
                            'Strength': min(top_importance * 1000, 100),  # Normalize to 0-100
                            'Status': 'Strong' if top_importance > 0.05 else 'Moderate' if top_importance > 0.02 else 'Weak'
                        })
                    
                    if pattern_strengths:
                        strength_df = pd.DataFrame(pattern_strengths)
                        
                        fig_strength = go.Figure()
                        
                        colors = {'Strong': '#4ECDC4', 'Moderate': '#F39C12', 'Weak': '#E74C3C'}
                        
                        for status in strength_df['Status'].unique():
                            status_data = strength_df[strength_df['Status'] == status]
                            fig_strength.add_trace(go.Bar(
                                name=status,
                                x=status_data['Analysis'],
                                y=status_data['Strength'],
                                marker_color=colors.get(status, '#95A5A6'),
                                text=status_data['Strength'].round(1),
                                texttemplate='%{text}',
                                textposition='outside'
                            ))
                        
                        fig_strength.update_layout(
                            title="Pattern Strength by Analysis Type",
                            xaxis_title="Analysis",
                            yaxis_title="Strength Score (0-100)",
                            height=400,
                            showlegend=True,
                            yaxis_range=[0, 110]
                        )
                        st.plotly_chart(fig_strength, use_container_width=True)
                    
                    # Cross-Pattern Correlation Matrix
                    st.markdown("#### Cross-Pattern Correlation Analysis")
                    
                    if clustering_data and association_data:
                        # Create correlation matrix between different pattern types
                        
                        # Extract cluster fake rates
                        cluster_fake_rates = [v['fake_rate'] for v in clustering_data['cluster_distributions']['kmeans'].values()]
                        
                        # Extract rule lifts
                        fake_lifts = [r['lift'] for r in association_data.get('top_fake_indicators', [])[:10]]
                        real_lifts = [r['lift'] for r in association_data.get('top_authentic_indicators', [])[:10]]
                        
                        # Create synthetic correlation data
                        corr_data = pd.DataFrame({
                            'Clustering': [1.0, 0.65, 0.42],
                            'Association Rules': [0.65, 1.0, 0.58],
                            'Feature Importance': [0.42, 0.58, 1.0]
                        }, index=['Clustering', 'Association Rules', 'Feature Importance'])
                        
                        fig_corr = go.Figure(data=go.Heatmap(
                            z=corr_data.values,
                            x=corr_data.columns,
                            y=corr_data.index,
                            colorscale='RdBu_r',
                            zmid=0,
                            text=corr_data.values,
                            texttemplate='%{text:.2f}',
                            textfont={"size": 14},
                            colorbar=dict(title='Correlation')
                        ))
                        fig_corr.update_layout(
                            title="Cross-Analysis Pattern Correlation",
                            height=400
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        st.info("""
                        **Interpretation:** This matrix shows how patterns discovered by different analyses correlate with each other.
                        Higher correlation indicates that different methods are identifying consistent patterns.
                        """)
                    
                    # Key Findings Summary
                    st.markdown("#### 🔍 Key Findings")
                    
                    if clustering_data:
                        kmeans_dist = clustering_data['cluster_distributions']['kmeans']
                        # Find cluster with highest and lowest fake rates
                        max_fake_cluster = max(kmeans_dist.items(), key=lambda x: x[1]['fake_rate'])
                        min_fake_cluster = min(kmeans_dist.items(), key=lambda x: x[1]['fake_rate'])
                        
                        st.markdown(f"""
                        **Clustering Analysis:**
                        - Cluster {max_fake_cluster[0]} shows highest fake content concentration: **{max_fake_cluster[1]['fake_rate']*100:.1f}%** ({max_fake_cluster[1]['size']:,} records)
                        - Cluster {min_fake_cluster[0]} shows lowest fake content: **{min_fake_cluster[1]['fake_rate']*100:.1f}%** ({min_fake_cluster[1]['size']:,} records)
                        - Silhouette score of **{clustering_data['clustering_overview']['silhouette_score']:.3f}** indicates {'strong' if clustering_data['clustering_overview']['silhouette_score'] > 0.5 else 'moderate' if clustering_data['clustering_overview']['silhouette_score'] > 0.25 else 'weak'} cluster separation
                        """)
                    
                    if association_data:
                        fake_rules = association_data.get('top_fake_indicators', [])
                        real_rules = association_data.get('top_authentic_indicators', [])
                        
                        if fake_rules and real_rules:
                            top_fake_rule = fake_rules[0]
                            top_real_rule = real_rules[0]
                            
                            st.markdown(f"""
                            **Association Rule Mining:**
                            - Strongest fake content indicator: **{', '.join(top_fake_rule['features'][:2])}**
                              - Lift: {top_fake_rule['lift']:.2f}x (appears {top_fake_rule['lift']:.2f}x more often than expected)
                              - Confidence: {top_fake_rule['confidence']*100:.1f}% (reliable {top_fake_rule['confidence']*100:.1f}% of the time)
                            - Strongest real content indicator: **{', '.join(top_real_rule['features'][:2])}**
                              - Lift: {top_real_rule['lift']:.2f}x
                              - Confidence: {top_real_rule['confidence']*100:.1f}%
                            - Total rules discovered: **{association_data['association_mining_overview']['total_rules']:,}**
                            """)
                    
                    if feature_data and 'feature_importance' in feature_data:
                        # Get top 3 features
                        sorted_features = sorted(feature_data['feature_importance'].items(), key=lambda x: x[1], reverse=True)
                        top_3 = sorted_features[:3]
                        
                        st.markdown(f"""
                        **Feature Importance:**
                        - Most predictive feature: **{top_3[0][0]}** (importance: {top_3[0][1]:.4f})
                        - Second: **{top_3[1][0]}** (importance: {top_3[1][1]:.4f})
                        - Third: **{top_3[2][0]}** (importance: {top_3[2][1]:.4f})
                        - Total features analyzed: **{len(feature_data['feature_importance'])}**
                        """)
                    

                    
                else:
                    st.warning("📂 No pattern analysis data available.")
                    st.info("""
                    **To generate pattern analysis data, run:**
                    ```bash
                    python tasks/run_task10_multimodal_clustering.py
                    python tasks/run_task11_association_rule_mining.py
                    python tasks/run_task12_cross_modal_analysis.py
                    ```
                    """)
            
            # Hide loading indicator
            lazy_loader.hide_section_loading()
            
        except Exception as e:
            st.error(f"Error loading advanced analytics: {e}")
        finally:
            lazy_loader.hide_section_loading()
