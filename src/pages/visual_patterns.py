"""
Visual Patterns Page
Displays visual feature analysis and patterns - COMPLETE IMPLEMENTATION
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.lazy_loader import LazyLoader

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
                
                # Visual feature comparison: Fake vs Real
                st.subheader("🎨 Visual Feature Comparison: Fake vs Real")
                
                # Key visual features to analyze (reduced for performance)
                visual_features_to_analyze = [
                    ('mean_brightness', 'Brightness'),
                    ('sharpness_score', 'Sharpness'),
                    ('visual_entropy', 'Visual Complexity'),
                    ('noise_level', 'Noise Level')
                ]
                
                # Pre-calculate statistics for performance
                feature_stats = {}
                for feature_col, feature_name in visual_features_to_analyze:
                    if feature_col in visual_features.columns:
                        fake_values = fake_images[feature_col].dropna()
                        real_values = real_images[feature_col].dropna()
                        
                        if len(fake_values) > 100 and len(real_values) > 100:
                            feature_stats[feature_name] = {
                                'fake_mean': fake_values.mean(),
                                'real_mean': real_values.mean(),
                                'fake_std': fake_values.std(),
                                'real_std': real_values.std(),
                                'difference_pct': ((fake_values.mean() - real_values.mean()) / real_values.mean() * 100) if real_values.mean() != 0 else 0
                            }
                
                # Create a single comprehensive comparison chart
                if feature_stats:
                    features = list(feature_stats.keys())
                    fake_means = [feature_stats[f]['fake_mean'] for f in features]
                    real_means = [feature_stats[f]['real_mean'] for f in features]
                    
                    # Normalize values for comparison (0-1 scale)
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    
                    # Combine and scale
                    all_values = np.array([fake_means, real_means]).T
                    scaled_values = scaler.fit_transform(all_values)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Fake Images',
                        x=features,
                        y=scaled_values[:, 0],
                        marker_color='#FF6B6B',
                        text=[f"{val:.3f}" for val in fake_means],
                        textposition='outside'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Real Images',
                        x=features,
                        y=scaled_values[:, 1],
                        marker_color='#4ECDC4',
                        text=[f"{val:.3f}" for val in real_means],
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title="Visual Features: Fake vs Real Images (Normalized)",
                        yaxis_title="Normalized Score (0-1)",
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show key differences
                    st.write("**📊 Key Differences:**")
                    for feature_name, stats in feature_stats.items():
                        diff_pct = stats['difference_pct']
                        if abs(diff_pct) > 5:  # Only show meaningful differences
                            direction = "higher" if diff_pct > 0 else "lower"
                            st.write(f"• **{feature_name}**: Fake images are {abs(diff_pct):.1f}% {direction}")
                
                # Simplified quality analysis
                st.subheader("🔍 Image Quality Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Quality metrics summary
                    st.write("**📊 Quality Metrics Summary**")
                    
                    quality_metrics = {}
                    if 'sharpness_score' in visual_features.columns:
                        fake_sharpness = fake_images['sharpness_score'].mean()
                        real_sharpness = real_images['sharpness_score'].mean()
                        quality_metrics['Sharpness'] = {
                            'fake': fake_sharpness,
                            'real': real_sharpness,
                            'diff': ((fake_sharpness - real_sharpness) / real_sharpness * 100) if real_sharpness != 0 else 0
                        }
                    
                    if 'noise_level' in visual_features.columns:
                        fake_noise = fake_images['noise_level'].mean()
                        real_noise = real_images['noise_level'].mean()
                        quality_metrics['Noise Level'] = {
                            'fake': fake_noise,
                            'real': real_noise,
                            'diff': ((fake_noise - real_noise) / real_noise * 100) if real_noise != 0 else 0
                        }
                    
                    for metric, values in quality_metrics.items():
                        st.metric(
                            f"{metric} Difference",
                            f"{values['diff']:+.1f}%",
                            delta=f"Fake: {values['fake']:.3f} vs Real: {values['real']:.3f}"
                        )
                
                with col2:
                    # Processing success rates
                    st.write("**⚙️ Processing Statistics**")
                    
                    fake_success_rate = (fake_images['processing_success'] == True).mean() * 100
                    real_success_rate = (real_images['processing_success'] == True).mean() * 100
                    
                    st.metric("Fake Images Success Rate", f"{fake_success_rate:.1f}%")
                    st.metric("Real Images Success Rate", f"{real_success_rate:.1f}%")
                    
                    if 'processing_time_ms' in visual_features.columns:
                        fake_processing_time = fake_images['processing_time_ms'].mean()
                        real_processing_time = real_images['processing_time_ms'].mean()
                        st.metric("Avg Processing Time Difference", 
                                 f"{((fake_processing_time - real_processing_time) / real_processing_time * 100):+.1f}%")
                
                # Key visual insights with detailed analysis
                st.subheader("🎯 Key Visual Insights: What Distinguishes Fake from Real?")
                
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
                                elif feature_name == 'Visual Complexity':
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
                                elif feature_name == 'Visual Complexity':
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
