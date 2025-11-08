"""
Visual Patterns Page
Displays visual feature analysis and patterns - COMPLETE IMPLEMENTATION
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.lazy_loader import LazyLoader
from src.utils.visualization_helpers import (
    create_box_plot,
    create_heatmap,
    calculate_statistics,
    add_statistical_annotations,
    CHART_CONFIG
)

lazy_loader = LazyLoader()


def render_visual_patterns(container):
    """Render Visual Patterns with lazy loading - COMPLETE IMPLEMENTATION"""
    with container.container():
        try:
            st.header("Visual Characteristics: Fake vs Real Images")
            
            st.markdown("""
            **Key Questions Answered:**
            - Do fake and real images have different visual characteristics?
            - What visual features best distinguish authentic from inauthentic content?
            - How do image quality metrics correlate with authenticity?
            """)
            
            # Lazy load visual features data with caching and performance optimization
            @st.cache_data(ttl=600)  # Cache for 10 minutes
            def load_visual_features():
                data = pd.read_parquet('processed_data/visual_features/visual_features_with_authenticity.parquet')
                original_size = len(data)
                # Performance optimization: Sample large datasets
                if original_size > 50000:
                    data = data.sample(n=50000, random_state=42)
                return data, original_size
            
            visual_features, original_size = load_visual_features()
            
            # Hide loading indicator after all data is loaded
            lazy_loader.hide_section_loading()
            
            # Show sampling notification if data was sampled
            if original_size > 50000:
                st.info(f"📊 Performance Optimization: Displaying analysis of 50,000 sampled images (from {original_size:,} total) for optimal dashboard performance")
            
            if len(visual_features) > 0:
                # Create authenticity label mapping
                visual_features['authenticity'] = visual_features['authenticity_label'].map({0: 'fake', 1: 'real'})
                # Pre-filter data once for performance
                fake_images = visual_features[visual_features['authenticity_label'] == 0].copy()
                real_images = visual_features[visual_features['authenticity_label'] == 1].copy()
                
                # Overall visual analysis metrics
                st.subheader("📊 Visual Analysis Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🖼️ Total Images Analyzed", f"{len(visual_features):,}")
                
                with col2:
                    st.metric("🔴 Fake Images", f"{len(fake_images):,}", 
                             delta=f"{len(fake_images)/len(visual_features)*100:.1f}%")
                
                with col3:
                    st.metric("🟢 Real Images", f"{len(real_images):,}", 
                             delta=f"{len(real_images)/len(visual_features)*100:.1f}%")
                
                with col4:
                    processing_success = (visual_features['processing_success'] == True).mean() * 100
                    st.metric("✅ Processing Success", f"{processing_success:.1f}%")
                
                # 1. Feature Distribution Grid (2x2: brightness, sharpness, entropy, noise level)
                st.subheader("📊 Feature Distribution Grid")
                st.markdown("Comparing distributions of key visual features between fake and real images")
                
                feature_columns = {
                    'mean_brightness': 'Brightness',
                    'sharpness_score': 'Sharpness',
                    'visual_entropy': 'Visual Entropy',
                    'noise_level': 'Noise Level'
                }
                
                # Create 2x2 subplot grid
                fig_grid = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=list(feature_columns.values()),
                    vertical_spacing=0.12,
                    horizontal_spacing=0.1
                )
                
                positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
                
                for (feature_col, feature_name), (row, col) in zip(feature_columns.items(), positions):
                    if feature_col in visual_features.columns:
                        fake_values = fake_images[feature_col].dropna()
                        real_values = real_images[feature_col].dropna()
                        
                        # Add fake histogram
                        fig_grid.add_trace(
                            go.Histogram(
                                x=fake_values,
                                name='Fake',
                                marker_color=CHART_CONFIG['colors']['fake'],
                                opacity=0.6,
                                nbinsx=30,
                                legendgroup='fake',
                                showlegend=(row == 1 and col == 1)
                            ),
                            row=row, col=col
                        )
                        
                        # Add real histogram
                        fig_grid.add_trace(
                            go.Histogram(
                                x=real_values,
                                name='Real',
                                marker_color=CHART_CONFIG['colors']['real'],
                                opacity=0.6,
                                nbinsx=30,
                                legendgroup='real',
                                showlegend=(row == 1 and col == 1)
                            ),
                            row=row, col=col
                        )
                        
                        # Update axes labels
                        fig_grid.update_xaxes(title_text=feature_name, row=row, col=col)
                        fig_grid.update_yaxes(title_text='Count', row=row, col=col)
                
                fig_grid.update_layout(
                    height=600,
                    barmode='overlay',
                    title_text="Visual Feature Distributions: Fake vs Real",
                    showlegend=True,
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    )
                )
                
                st.plotly_chart(fig_grid, use_container_width=True)
                
                # 2. Visual Feature Correlation Matrix
                st.subheader("🔗 Visual Feature Correlation Matrix")
                st.markdown("Correlation between visual features with statistical annotations")
                
                # Select numeric visual features for correlation
                correlation_features = [
                    'mean_brightness', 'sharpness_score', 'visual_entropy', 'noise_level',
                    'contrast_score', 'color_diversity', 'edge_density'
                ]
                
                available_features = [f for f in correlation_features if f in visual_features.columns]
                
                if len(available_features) >= 4:
                    correlation_data = visual_features[available_features].corr()
                    
                    # Create correlation heatmap
                    fig_corr = create_heatmap(
                        correlation_data,
                        title="Visual Feature Correlation Matrix",
                        annotations=True,
                        colorscale='RdBu_r'
                    )
                    
                    fig_corr.update_layout(height=500)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Show key correlations
                    st.markdown("**🔍 Key Correlations:**")
                    
                    # Find strongest correlations (excluding diagonal)
                    corr_pairs = []
                    for i in range(len(correlation_data.columns)):
                        for j in range(i+1, len(correlation_data.columns)):
                            corr_pairs.append({
                                'feature1': correlation_data.columns[i],
                                'feature2': correlation_data.columns[j],
                                'correlation': correlation_data.iloc[i, j]
                            })
                    
                    corr_pairs_df = pd.DataFrame(corr_pairs)
                    corr_pairs_df = corr_pairs_df.reindex(corr_pairs_df['correlation'].abs().sort_values(ascending=False).index)
                    
                    for idx, row in corr_pairs_df.head(3).iterrows():
                        st.write(f"• **{row['feature1']}** ↔ **{row['feature2']}**: {row['correlation']:.3f}")
                else:
                    st.warning("Insufficient features available for correlation analysis")
                
                # 3. Manipulation Score Box Plots
                st.subheader("📦 Manipulation Score Analysis")
                st.markdown("Statistical comparison of manipulation scores between fake and real images")
                
                if 'manipulation_score' in visual_features.columns:
                    # Create box plot
                    fig_box = create_box_plot(
                        visual_features,
                        value_column='manipulation_score',
                        category_column='authenticity',
                        title='Manipulation Score Distribution: Fake vs Real',
                        labels={'x': 'Authenticity', 'y': 'Manipulation Score'}
                    )
                    
                    # Calculate statistics
                    fake_manip = fake_images['manipulation_score'].dropna()
                    real_manip = real_images['manipulation_score'].dropna()
                    
                    if len(fake_manip) > 0 and len(real_manip) > 0:
                        p_value, effect_size = calculate_statistics(
                            fake_manip.values,
                            real_manip.values
                        )
                        
                        # Add statistical annotations
                        fig_box = add_statistical_annotations(fig_box, p_value, effect_size)
                    
                    fig_box.update_layout(height=500)
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Fake Mean", f"{fake_manip.mean():.3f}")
                    with col2:
                        st.metric("Real Mean", f"{real_manip.mean():.3f}")
                    with col3:
                        diff_pct = ((fake_manip.mean() - real_manip.mean()) / real_manip.mean() * 100) if real_manip.mean() != 0 else 0
                        st.metric("Difference", f"{diff_pct:+.1f}%")
                else:
                    st.warning("Manipulation score data not available")
                
                # 4. Feature Scatter Matrix for Top 4 Discriminative Features
                st.subheader("🎯 Feature Scatter Matrix")
                st.markdown("Relationships between top discriminative features")
                
                # Identify top discriminative features by calculating effect sizes
                discriminative_features = []
                
                for feature_col, feature_name in feature_columns.items():
                    if feature_col in visual_features.columns:
                        fake_vals = fake_images[feature_col].dropna()
                        real_vals = real_images[feature_col].dropna()
                        
                        if len(fake_vals) > 100 and len(real_vals) > 100:
                            _, effect_size = calculate_statistics(fake_vals.values, real_vals.values)
                            discriminative_features.append({
                                'feature': feature_col,
                                'name': feature_name,
                                'effect_size': abs(effect_size)
                            })
                
                # Sort by effect size and take top 4
                discriminative_features = sorted(discriminative_features, key=lambda x: x['effect_size'], reverse=True)[:4]
                
                if len(discriminative_features) >= 4:
                    top_features = [f['feature'] for f in discriminative_features]
                    top_feature_names = [f['name'] for f in discriminative_features]
                    
                    # Create scatter matrix
                    fig_scatter = make_subplots(
                        rows=4, cols=4,
                        subplot_titles=[f"{top_feature_names[i//4]} vs {top_feature_names[i%4]}" 
                                       if i//4 != i%4 else top_feature_names[i//4]
                                       for i in range(16)],
                        vertical_spacing=0.05,
                        horizontal_spacing=0.05
                    )
                    
                    for i, feat1 in enumerate(top_features):
                        for j, feat2 in enumerate(top_features):
                            row, col = i + 1, j + 1
                            
                            if i == j:
                                # Diagonal: histograms
                                fake_vals = fake_images[feat1].dropna()
                                real_vals = real_images[feat1].dropna()
                                
                                fig_scatter.add_trace(
                                    go.Histogram(
                                        x=fake_vals,
                                        name='Fake',
                                        marker_color=CHART_CONFIG['colors']['fake'],
                                        opacity=0.6,
                                        nbinsx=20,
                                        showlegend=(i == 0)
                                    ),
                                    row=row, col=col
                                )
                                
                                fig_scatter.add_trace(
                                    go.Histogram(
                                        x=real_vals,
                                        name='Real',
                                        marker_color=CHART_CONFIG['colors']['real'],
                                        opacity=0.6,
                                        nbinsx=20,
                                        showlegend=(i == 0)
                                    ),
                                    row=row, col=col
                                )
                            else:
                                # Off-diagonal: scatter plots
                                # Sample for performance
                                sample_size = min(5000, len(visual_features))
                                sample_data = visual_features.sample(n=sample_size, random_state=42)
                                
                                for auth_label, color in [(0, CHART_CONFIG['colors']['fake']), 
                                                          (1, CHART_CONFIG['colors']['real'])]:
                                    auth_data = sample_data[sample_data['authenticity_label'] == auth_label]
                                    
                                    fig_scatter.add_trace(
                                        go.Scatter(
                                            x=auth_data[feat2],
                                            y=auth_data[feat1],
                                            mode='markers',
                                            marker=dict(
                                                color=color,
                                                size=3,
                                                opacity=0.4
                                            ),
                                            showlegend=False
                                        ),
                                        row=row, col=col
                                    )
                    
                    fig_scatter.update_layout(
                        height=1000,
                        title_text="Feature Scatter Matrix: Top 4 Discriminative Features",
                        showlegend=True,
                        barmode='overlay'
                    )
                    
                    # Hide axis labels for cleaner look
                    fig_scatter.update_xaxes(showticklabels=False)
                    fig_scatter.update_yaxes(showticklabels=False)
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Show feature importance
                    st.markdown("**🏆 Top Discriminative Features (by effect size):**")
                    for idx, feat in enumerate(discriminative_features, 1):
                        st.write(f"{idx}. **{feat['name']}**: Effect size = {feat['effect_size']:.3f}")
                else:
                    st.warning("Insufficient features for scatter matrix analysis")
                
                # Key visual insights with detailed analysis
                st.subheader("🎯 Key Visual Insights")
                
                # Calculate feature statistics for insights
                feature_stats = {}
                for feature_col, feature_name in feature_columns.items():
                    if feature_col in visual_features.columns:
                        fake_values = fake_images[feature_col].dropna()
                        real_values = real_images[feature_col].dropna()
                        
                        if len(fake_values) > 100 and len(real_values) > 100:
                            feature_stats[feature_name] = {
                                'fake_mean': fake_values.mean(),
                                'real_mean': real_values.mean(),
                                'difference_pct': ((fake_values.mean() - real_values.mean()) / real_values.mean() * 100) if real_values.mean() != 0 else 0
                            }
                
                # Create two columns for fake vs real characteristics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🔴 Fake Image Characteristics")
                    fake_characteristics = []
                    
                    if feature_stats:
                        for feature_name, stats in feature_stats.items():
                            diff_pct = stats['difference_pct']
                            if diff_pct > 2:  # Fake is higher
                                if feature_name == 'Brightness':
                                    fake_characteristics.append(f"• **Brighter images** ({abs(diff_pct):.1f}% higher) - may indicate artificial enhancement")
                                elif feature_name == 'Sharpness':
                                    fake_characteristics.append(f"• **Over-sharpened** ({abs(diff_pct):.1f}% higher) - suggests digital manipulation")
                                elif feature_name == 'Visual Entropy':
                                    fake_characteristics.append(f"• **More complex** ({abs(diff_pct):.1f}% higher) - potentially added artifacts")
                                elif feature_name == 'Noise Level':
                                    fake_characteristics.append(f"• **Higher noise** ({abs(diff_pct):.1f}% higher) - compression or editing artifacts")
                    
                    # Manipulation score
                    if 'manipulation_score' in visual_features.columns:
                        fake_manipulation = fake_images['manipulation_score'].mean()
                        real_manipulation = real_images['manipulation_score'].mean()
                        manipulation_diff = ((fake_manipulation - real_manipulation) / real_manipulation) * 100 if real_manipulation != 0 else 0
                        
                        if manipulation_diff > 5:
                            fake_characteristics.append(f"• **Higher manipulation scores** ({abs(manipulation_diff):.1f}% higher) - detected editing patterns")
                    
                    if fake_characteristics:
                        for char in fake_characteristics:
                            st.markdown(char)
                    else:
                        st.info("No significant distinguishing characteristics detected")
                
                with col2:
                    st.markdown("### 🟢 Real Image Characteristics")
                    real_characteristics = []
                    
                    if feature_stats:
                        for feature_name, stats in feature_stats.items():
                            diff_pct = stats['difference_pct']
                            if diff_pct < -2:  # Real is higher (fake is lower)
                                if feature_name == 'Brightness':
                                    real_characteristics.append(f"• **Natural brightness** ({abs(diff_pct):.1f}% higher) - unmodified lighting")
                                elif feature_name == 'Sharpness':
                                    real_characteristics.append(f"• **Natural sharpness** ({abs(diff_pct):.1f}% higher) - authentic image quality")
                                elif feature_name == 'Visual Entropy':
                                    real_characteristics.append(f"• **Simpler composition** ({abs(diff_pct):.1f}% higher) - natural scenes")
                                elif feature_name == 'Noise Level':
                                    real_characteristics.append(f"• **Lower noise** ({abs(diff_pct):.1f}% higher) - cleaner captures")
                    
                    # Manipulation score
                    if 'manipulation_score' in visual_features.columns:
                        fake_manipulation = fake_images['manipulation_score'].mean()
                        real_manipulation = real_images['manipulation_score'].mean()
                        manipulation_diff = ((fake_manipulation - real_manipulation) / real_manipulation) * 100 if real_manipulation != 0 else 0
                        
                        if manipulation_diff < -5:
                            real_characteristics.append(f"• **Lower manipulation scores** ({abs(manipulation_diff):.1f}% lower) - minimal editing detected")
                    
                    if real_characteristics:
                        for char in real_characteristics:
                            st.markdown(char)
                    else:
                        st.info("No significant distinguishing characteristics detected")
                
                # Summary insight box
                st.markdown("---")
                st.subheader("💡 Visual Analysis Summary")
                
                if feature_stats:
                    # Calculate overall pattern
                    significant_diffs = sum(1 for stats in feature_stats.values() if abs(stats['difference_pct']) > 5)
                    total_features = len(feature_stats)
                    
                    if significant_diffs > 0:
                        st.success(f"""
                        **Key Finding:** {significant_diffs} out of {total_features} visual features show significant differences (>5%) between fake and real images.
                        
                        **Implication:** Visual analysis can contribute to authenticity detection, particularly when combined with other modalities.
                        The detected patterns suggest that fake images often undergo digital manipulation that leaves measurable traces in visual features.
                        """)
                    else:
                        st.info("""
                        **Key Finding:** Visual features show minimal differences between fake and real images in this dataset.
                        
                        **Implication:** Sophisticated fake content may be visually indistinguishable from authentic content, 
                        highlighting the importance of multimodal analysis (combining visual, textual, and social signals).
                        """)
                else:
                    st.warning("Insufficient data for comprehensive visual analysis.")
            
        except FileNotFoundError as e:
            st.error(f"📂 Data file not found: {e}")
            st.info("""
            **To generate visual features data:**
            ```bash
            python tasks/run_task8_visual_feature_engineering.py
            ```
            This will extract visual features from images and generate authenticity analysis data.
            """)
        except Exception as e:
            st.error(f"❌ Error loading visual features data: {e}")
            st.info("Please ensure visual feature analysis is complete and data files are accessible.")
        finally:
            lazy_loader.hide_section_loading()
