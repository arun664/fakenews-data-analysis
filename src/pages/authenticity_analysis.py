"""
Authenticity Analysis Page
Comprehensive analysis of fake vs real content across all modalities
COMPLETE IMPLEMENTATION - Extracted from app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.lazy_loader import LazyLoader

lazy_loader = LazyLoader()


def render_authenticity_analysis(container):
    """Render Authenticity Analysis with lazy loading - COMPLETE IMPLEMENTATION"""
    with container.container():
        try:
            st.header("Fake vs Real Content: Comprehensive Analysis")
            
            st.markdown("""
            **Key Questions Answered:**
            - What distinguishes fake from real content across all modalities?
            - How do engagement patterns differ between authentic and inauthentic posts?
            - What are the statistical signatures of misinformation?
            """)
            
            # Load integrated data from lightweight JSON summary
            @st.cache_data(ttl=600)  # 10 minutes cache for integrated dataset
            def load_integrated_data():
                import json
                summary_path = Path('analysis_results/dashboard_data/authenticity_analysis_summary.json')
                
                if not summary_path.exists():
                    raise FileNotFoundError(f"Authenticity analysis summary not found at {summary_path}")
                
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                # Convert sample data to DataFrame
                data = pd.DataFrame(summary['sample_data'])
                
                # Map label column for compatibility
                if '2_way_label' in data.columns:
                    data['authenticity_label'] = data['2_way_label']
                elif 'authenticity_label' not in data.columns:
                    raise ValueError("No authenticity label column found in summary data")
                
                original_size = summary.get('total_records', len(data))
                return data, original_size
            
            integrated_data, original_size = load_integrated_data()
            
            # Show sampling notification if data was sampled
            if original_size > 100000:
                st.info(f"📊 Performance Optimization: Analyzing 100,000 sampled records (from {original_size:,} total) for optimal dashboard performance")
            
            # Hide loading indicator after data is loaded
            lazy_loader.hide_section_loading()
            
            # Overall authenticity distribution
            st.subheader("📊 Authenticity Distribution Across Dataset")
            
            fake_count = len(integrated_data[integrated_data['2_way_label'] == 0])
            real_count = len(integrated_data[integrated_data['2_way_label'] == 1])
            total_count = len(integrated_data)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🔴 Fake Content", f"{fake_count:,}", 
                         delta=f"{fake_count/total_count*100:.1f}% of dataset")
            
            with col2:
                st.metric("🟢 Real Content", f"{real_count:,}", 
                         delta=f"{real_count/total_count*100:.1f}% of dataset")
            
            with col3:
                ratio = fake_count / real_count if real_count > 0 else 0
                st.metric("📈 Fake:Real Ratio", f"{ratio:.2f}:1")
            
            # Key Insight Box
            st.info(f"""
            **🔍 Key Insight:** The dataset shows a significant class imbalance with {fake_count/total_count*100:.1f}% fake content, 
            reflecting the prevalence of misinformation in social media. This {ratio:.1f}:1 ratio provides robust 
            statistical power for detecting authenticity patterns.
            """)
            
            # Authenticity by content modality
            st.subheader("🎭 Authenticity Patterns by Content Type")
            
            # Calculate authenticity by modality using content_type_social field
            # Full multimodal (text + image + comments)
            full_multimodal = integrated_data[integrated_data['content_type_social'] == 'full_multimodal']
            
            # Bimodal (text + image)
            bimodal = integrated_data[integrated_data['content_type_social'] == 'text_image']
            
            # Text only
            text_only = integrated_data[integrated_data['content_type_social'] == 'text_only']
            
            modalities = {
                "Full Multimodal\n(Text+Image+Comments)": full_multimodal,
                "Bimodal\n(Text+Image)": bimodal,
                "Text Only": text_only
            }
            
            modality_stats = []
            for mod_name, mod_data in modalities.items():
                if len(mod_data) > 0:
                    fake_pct = (mod_data['2_way_label'] == 0).mean() * 100
                    real_pct = (mod_data['2_way_label'] == 1).mean() * 100
                    modality_stats.append({
                        'Modality': mod_name,
                        'Total Posts': len(mod_data),
                        'Fake %': fake_pct,
                        'Real %': real_pct,
                        'Fake Count': len(mod_data[mod_data['2_way_label'] == 0]),
                        'Real Count': len(mod_data[mod_data['2_way_label'] == 1])
                    })
            
            if modality_stats:
                # Create stacked bar chart
                fig = go.Figure()
                
                modality_names = [stat['Modality'] for stat in modality_stats]
                fake_percentages = [stat['Fake %'] for stat in modality_stats]
                real_percentages = [stat['Real %'] for stat in modality_stats]
                
                fig.add_trace(go.Bar(
                    name='🔴 Fake Content',
                    x=modality_names,
                    y=fake_percentages,
                    marker_color='#FF6B6B',
                    text=[f"{pct:.1f}%" for pct in fake_percentages],
                    textposition='inside'
                ))
                
                fig.add_trace(go.Bar(
                    name='🟢 Real Content',
                    x=modality_names,
                    y=real_percentages,
                    marker_color='#4ECDC4',
                    text=[f"{pct:.1f}%" for pct in real_percentages],
                    textposition='inside'
                ))
                
                fig.update_layout(
                    title="Authenticity Distribution by Content Modality",
                    barmode='stack',
                    yaxis_title="Percentage (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed statistics table
                st.subheader("📋 Detailed Authenticity Statistics")
                stats_df = pd.DataFrame(modality_stats)
                st.dataframe(stats_df, use_container_width=True)
            
            # Social engagement patterns by authenticity
            st.subheader("👥 Social Engagement: Fake vs Real")
            
            if 'score' in integrated_data.columns:
                # Create unified comment count field
                integrated_data['unified_comments'] = integrated_data.apply(
                    lambda row: row['comment_count'] if (row['content_type_social'] == 'full_multimodal' and pd.notna(row['comment_count'])) 
                               else row['num_comments'] if pd.notna(row['num_comments']) else 0, axis=1
                )
                
                fake_data = integrated_data[integrated_data['2_way_label'] == 0]
                real_data = integrated_data[integrated_data['2_way_label'] == 1]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Engagement score comparison
                    fake_score_mean = fake_data['score'].mean()
                    real_score_mean = real_data['score'].mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Fake Content', 'Real Content'],
                        y=[fake_score_mean, real_score_mean],
                        marker_color=['#FF6B6B', '#4ECDC4'],
                        text=[f"{fake_score_mean:.1f}", f"{real_score_mean:.1f}"],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title="Average Engagement Score",
                        yaxis_title="Score",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Comment count comparison
                    fake_comments_mean = fake_data['unified_comments'].mean()
                    real_comments_mean = real_data['unified_comments'].mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Fake Content', 'Real Content'],
                        y=[fake_comments_mean, real_comments_mean],
                        marker_color=['#FF6B6B', '#4ECDC4'],
                        text=[f"{fake_comments_mean:.1f}", f"{real_comments_mean:.1f}"],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title="Average Comment Count",
                        yaxis_title="Comments",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistical significance testing
                # T-test for engagement scores
                score_t_stat, score_p_value = stats.ttest_ind(
                    fake_data['score'].dropna(), 
                    real_data['score'].dropna()
                )
                
                # T-test for comment counts
                comment_t_stat, comment_p_value = stats.ttest_ind(
                    fake_data['unified_comments'].dropna(), 
                    real_data['unified_comments'].dropna()
                )
                
                st.subheader("📈 Statistical Significance")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Engagement Score Analysis:**")
                    st.markdown(f"• T-statistic: {score_t_stat:.3f}")
                    st.markdown(f"• P-value: {score_p_value:.6f}")
                    if score_p_value < 0.05:
                        st.success("✅ Statistically significant difference")
                    else:
                        st.warning("⚠️ No significant difference")
                
                with col2:
                    st.markdown("**Comment Count Analysis:**")
                    st.markdown(f"• T-statistic: {comment_t_stat:.3f}")
                    st.markdown(f"• P-value: {comment_p_value:.6f}")
                    if comment_p_value < 0.05:
                        st.success("✅ Statistically significant difference")
                    else:
                        st.warning("⚠️ No significant difference")
            
            # Enhanced authenticity insights with comprehensive analysis
            st.subheader("🎯 Comprehensive Authenticity Analysis: Fake vs Real")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔴 Fake Content Profile")
                fake_profile = []
                
                # Content distribution
                fake_profile.append(f"**📊 Volume:** {fake_count:,} posts ({fake_count/total_count*100:.1f}% of dataset)")
                
                # Modality risk
                if modality_stats:
                    highest_fake_modality = max(modality_stats, key=lambda x: x['Fake %'])
                    fake_profile.append(f"**🎭 Highest Risk Modality:** {highest_fake_modality['Modality']} ({highest_fake_modality['Fake %']:.1f}% fake)")
                
                # Engagement pattern
                if 'score' in integrated_data.columns:
                    if fake_score_mean > real_score_mean:
                        engagement_diff = ((fake_score_mean - real_score_mean) / real_score_mean) * 100
                        fake_profile.append(f"**👥 Engagement:** Higher scores (+{engagement_diff:.1f}%) - potentially sensational")
                    else:
                        engagement_diff = ((real_score_mean - fake_score_mean) / fake_score_mean) * 100
                        fake_profile.append(f"**👥 Engagement:** Lower scores (-{engagement_diff:.1f}%) - less viral spread")
                    
                    fake_profile.append(f"**📈 Avg Score:** {fake_score_mean:.1f} points")
                
                # Comment pattern
                if 'score' in integrated_data.columns:
                    if fake_comments_mean > real_comments_mean:
                        comment_diff = ((fake_comments_mean - real_comments_mean) / real_comments_mean) * 100
                        fake_profile.append(f"**💬 Comments:** More discussion (+{comment_diff:.1f}%) - controversial topics")
                    else:
                        comment_diff = ((real_comments_mean - fake_comments_mean) / fake_comments_mean) * 100
                        fake_profile.append(f"**💬 Comments:** Less discussion (-{comment_diff:.1f}%)")
                    
                    fake_profile.append(f"**💬 Avg Comments:** {fake_comments_mean:.1f} per post")
                
                for item in fake_profile:
                    st.markdown(f"• {item}")
            
            with col2:
                st.markdown("### 🟢 Real Content Profile")
                real_profile = []
                
                # Content distribution
                real_profile.append(f"**📊 Volume:** {real_count:,} posts ({real_count/total_count*100:.1f}% of dataset)")
                
                # Modality safety
                if modality_stats:
                    lowest_fake_modality = min(modality_stats, key=lambda x: x['Fake %'])
                    real_profile.append(f"**🎭 Safest Modality:** {lowest_fake_modality['Modality']} ({100-lowest_fake_modality['Fake %']:.1f}% real)")
                
                # Engagement pattern
                if 'score' in integrated_data.columns:
                    if real_score_mean > fake_score_mean:
                        engagement_diff = ((real_score_mean - fake_score_mean) / fake_score_mean) * 100
                        real_profile.append(f"**👥 Engagement:** Higher scores (+{engagement_diff:.1f}%) - genuine interest")
                    else:
                        engagement_diff = ((fake_score_mean - real_score_mean) / real_score_mean) * 100
                        real_profile.append(f"**👥 Engagement:** Lower scores (-{engagement_diff:.1f}%) - more subdued")
                    
                    real_profile.append(f"**📈 Avg Score:** {real_score_mean:.1f} points")
                
                # Comment pattern
                if 'score' in integrated_data.columns:
                    if real_comments_mean > fake_comments_mean:
                        comment_diff = ((real_comments_mean - fake_comments_mean) / fake_comments_mean) * 100
                        real_profile.append(f"**💬 Comments:** More discussion (+{comment_diff:.1f}%) - meaningful dialogue")
                    else:
                        comment_diff = ((fake_comments_mean - real_comments_mean) / real_comments_mean) * 100
                        real_profile.append(f"**💬 Comments:** Less discussion (-{comment_diff:.1f}%)")
                    
                    real_profile.append(f"**💬 Avg Comments:** {real_comments_mean:.1f} per post")
                
                for item in real_profile:
                    st.markdown(f"• {item}")
            
            # Comprehensive summary
            st.markdown("---")
            st.subheader("💡 Authenticity Detection Summary")
            
            # Statistical confidence
            if 'score' in integrated_data.columns:
                if score_p_value < 0.05 or comment_p_value < 0.05:
                    st.success(f"""
                    **Statistical Significance:** Detected significant differences between fake and real content (p < 0.05).
                    
                    **Key Findings:**
                    - Fake content represents {fake_count/total_count*100:.1f}% of the dataset ({ratio:.2f}:1 ratio)
                    - Social engagement patterns differ significantly between authentic and inauthentic content
                    - Multimodal analysis reveals distinct patterns across content types
                    
                    **Detection Strategy:** Combining social engagement metrics, content modality analysis, and statistical patterns 
                    provides a robust framework for authenticity detection. The {ratio:.2f}:1 class imbalance reflects real-world 
                    prevalence of misinformation on social media platforms.
                    """)
                else:
                    st.info("""
                    **Statistical Significance:** Minimal statistical differences detected between fake and real content.
                    
                    **Implication:** Sophisticated misinformation may closely mimic authentic content in social engagement patterns.
                    This highlights the importance of multimodal analysis combining visual, textual, and social signals for 
                    reliable authenticity detection.
                    """)
            
        except FileNotFoundError as e:
            st.error(f"📂 Data file not found: {e}")
            st.info("""
            **To generate the integrated dataset:**
            ```bash
            python tasks/run_task15_final_integration.py
            ```
            This will create the complete multimodal dataset with all features integrated.
            """)
        except Exception as e:
            st.error(f"❌ Error loading authenticity analysis data: {e}")
            st.info("Please ensure the final integration (Task 15) is complete and the integrated dataset is accessible.")
        finally:
            lazy_loader.hide_section_loading()
