"""
Social Patterns Page
Social engagement patterns analysis - fake vs real
COMPLETE IMPLEMENTATION - Extracted from app.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.lazy_loader import LazyLoader

lazy_loader = LazyLoader()


def render_social_patterns(container):
    """Render Social Patterns with lazy loading - COMPLETE IMPLEMENTATION"""
    with container.container():
        try:
            st.header("Social Engagement Patterns: Fake vs Real")
            
            st.markdown("""
            **🎯 Key Questions Answered:**
            - How do social engagement patterns differ between fake and real content?
            - What social signals indicate misinformation?
            - How does community interaction vary with content authenticity?
            """)
            
            # Load from JSON summary (FULL dataset with pre-computed visualizations)
            @st.cache_data(ttl=600)  # 10 minutes cache for social engagement data
            def load_social_data():
                import json
                
                summary_path = Path('analysis_results/dashboard_data/social_engagement_summary.json')
                
                if not summary_path.exists():
                    raise FileNotFoundError(f"Social engagement summary not found at {summary_path}")
                
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                return summary
            
            summary = load_social_data()
            
            total_records = summary.get('total_records', 0)
            fake_count = summary.get('fake_count', 0)
            real_count = summary.get('real_count', 0)
            
            # Hide loading indicator
            lazy_loader.hide_section_loading()
            
            if total_records > 0:
                # Overall social engagement metrics
                st.subheader("📊 Social Engagement Overview")
                
                engagement_stats = summary.get('engagement_stats', {})
                fake_stats = engagement_stats.get('fake', {})
                real_stats = engagement_stats.get('real', {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📝 Total Posts", f"{total_records:,}")
                
                with col2:
                    fake_engagement = fake_stats.get('score', {}).get('mean', 0)
                    st.metric("🔴 Fake Avg Score", f"{fake_engagement:.1f}")
                
                with col3:
                    real_engagement = real_stats.get('score', {}).get('mean', 0)
                    st.metric("🟢 Real Avg Score", f"{real_engagement:.1f}")
                
                with col4:
                    if real_engagement > 0:
                        engagement_ratio = fake_engagement / real_engagement
                        st.metric("📊 Fake:Real Ratio", f"{engagement_ratio:.2f}:1")
                
                # Engagement distribution comparison (using pre-computed histograms)
                st.subheader("📈 Engagement Score Distribution (Full Dataset)")
                
                histograms = summary.get('histograms', {})
                if 'score' in histograms:
                    fig = go.Figure()
                    
                    # Add fake histogram
                    if 'fake' in histograms['score']:
                        fake_hist = histograms['score']['fake']
                        bin_centers = [(fake_hist['bin_edges'][i] + fake_hist['bin_edges'][i+1]) / 2 
                                      for i in range(len(fake_hist['bin_edges']) - 1)]
                        
                        fig.add_trace(go.Bar(
                            x=bin_centers,
                            y=fake_hist['counts'],
                            name='🔴 Fake Content',
                            opacity=0.7,
                            marker_color='#FF6B6B'
                        ))
                    
                    # Add real histogram
                    if 'real' in histograms['score']:
                        real_hist = histograms['score']['real']
                        bin_centers = [(real_hist['bin_edges'][i] + real_hist['bin_edges'][i+1]) / 2 
                                      for i in range(len(real_hist['bin_edges']) - 1)]
                        
                        fig.add_trace(go.Bar(
                            x=bin_centers,
                            y=real_hist['counts'],
                            name='🟢 Real Content',
                            opacity=0.7,
                            marker_color='#4ECDC4'
                        ))
                    
                    fig.update_layout(
                        title=f"Engagement Score Distribution: Fake vs Real ({total_records:,} posts)",
                        xaxis_title="Engagement Score",
                        yaxis_title="Count",
                        barmode='overlay',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Comment patterns (using pre-computed data)
                st.subheader("💬 Comment Engagement Patterns (Full Dataset)")
                
                if 'num_comments' in engagement_stats.get('fake', {}):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fake_comments = fake_stats.get('num_comments', {}).get('mean', 0)
                        real_comments = real_stats.get('num_comments', {}).get('mean', 0)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=['Fake Content', 'Real Content'],
                            y=[fake_comments, real_comments],
                            marker_color=['#FF6B6B', '#4ECDC4'],
                            text=[f"{fake_comments:.1f}", f"{real_comments:.1f}"],
                            textposition='outside'
                        ))
                        
                        fig.update_layout(
                            title=f"Average Comments per Post ({total_records:,} posts)",
                            yaxis_title="Number of Comments",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Use pre-computed box plot data
                        boxplot_data = summary.get('boxplot_data', {})
                        if 'num_comments' in boxplot_data:
                            fig = go.Figure()
                            
                            for label_name, color in [('fake', '#FF6B6B'), ('real', '#4ECDC4')]:
                                if label_name in boxplot_data['num_comments']:
                                    box_stats = boxplot_data['num_comments'][label_name]
                                    
                                    fig.add_trace(go.Box(
                                        name=f'{"🔴" if label_name == "fake" else "🟢"} {label_name.capitalize()}',
                                        q1=[box_stats['q1']],
                                        median=[box_stats['median']],
                                        q3=[box_stats['q3']],
                                        lowerfence=[box_stats['min']],
                                        upperfence=[box_stats['max']],
                                        y=box_stats.get('outliers', [])[:100],  # Limit outliers for display
                                        marker_color=color,
                                        boxmean='sd'
                                    ))
                            
                            fig.update_layout(
                                title=f"Comment Count Distribution ({total_records:,} posts)",
                                yaxis_title="Number of Comments",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                # Statistical significance testing (using full dataset statistics)
                st.subheader("📈 Statistical Analysis (Full Dataset)")
                
                if 'score' in engagement_stats.get('fake', {}):
                    # Get statistics from pre-computed data
                    fake_score_stats = fake_stats.get('score', {})
                    real_score_stats = real_stats.get('score', {})
                    
                    fake_mean = fake_score_stats.get('mean', 0)
                    real_mean = real_score_stats.get('mean', 0)
                    fake_std = fake_score_stats.get('std', 1)
                    real_std = real_score_stats.get('std', 1)
                    
                    if fake_count > 1 and real_count > 1:
                        # Calculate t-statistic from summary statistics
                        try:
                            # Welch's t-test formula using summary statistics
                            se_fake = fake_std / np.sqrt(fake_count)
                            se_real = real_std / np.sqrt(real_count)
                            se_diff = np.sqrt(se_fake**2 + se_real**2)
                            t_stat = (fake_mean - real_mean) / se_diff if se_diff > 0 else 0
                            
                            # Approximate p-value (two-tailed)
                            from scipy import stats as sp_stats
                            df = fake_count + real_count - 2
                            p_value = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df))
                        except Exception as e:
                            st.error(f"Error calculating statistics: {e}")
                            t_stat, p_value = 0, 1
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Engagement Score Significance:**")
                            st.write(f"• T-statistic: {t_stat:.3f}")
                            st.write(f"• P-value: {p_value:.6f}")
                            if p_value < 0.05:
                                st.success("✅ Statistically significant difference (p < 0.05)")
                            elif p_value < 0.10:
                                st.info("📊 Marginally significant (p < 0.10)")
                            else:
                                st.warning("⚠️ No significant difference (p ≥ 0.05)")
                        
                        with col2:
                            # Calculate Cohen's d from summary statistics
                            try:
                                # Pooled standard deviation
                                pooled_std = np.sqrt(((fake_count - 1) * fake_std**2 + (real_count - 1) * real_std**2) / (fake_count + real_count - 2))
                                
                                # Cohen's d
                                if pooled_std > 0:
                                    cohens_d = (real_mean - fake_mean) / pooled_std
                                else:
                                    cohens_d = 0
                                    
                            except Exception as e:
                                st.error(f"Error calculating effect size: {e}")
                                cohens_d = 0
                            
                            st.write("**Effect Size Analysis:**")
                            st.write(f"• Cohen's d: {cohens_d:.3f}")
                            
                            # Interpret effect size
                            abs_d = abs(cohens_d)
                            if abs_d > 0.8:
                                effect_size = "Large effect"
                                effect_emoji = "🔴"
                            elif abs_d > 0.5:
                                effect_size = "Medium effect"
                                effect_emoji = "🟡"
                            elif abs_d > 0.2:
                                effect_size = "Small effect"
                                effect_emoji = "🟢"
                            else:
                                effect_size = "Negligible effect"
                                effect_emoji = "⚪"
                            
                            st.write(f"• {effect_emoji} Interpretation: {effect_size}")
                            
                            # Direction interpretation
                            if cohens_d > 0:
                                st.write("• Direction: Real content has higher engagement")
                            elif cohens_d < 0:
                                st.write("• Direction: Fake content has higher engagement")
                            else:
                                st.write("• Direction: No difference")
                    else:
                        st.warning("⚠️ Insufficient data for statistical analysis (need at least 2 records per group)")
                
                # Enhanced social insights with fake vs real comparison
                st.subheader("🎯 Social Engagement Insights: What Distinguishes Fake from Real?")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🔴 Fake Content Social Patterns")
                    fake_patterns = []
                    
                    if 'score' in engagement_stats.get('fake', {}):
                        if fake_engagement > real_engagement:
                            engagement_diff = ((fake_engagement - real_engagement) / real_engagement) * 100
                            fake_patterns.append(f"• **Higher engagement scores** (+{engagement_diff:.1f}%) - may indicate viral spread or bot activity")
                        else:
                            engagement_diff = ((real_engagement - fake_engagement) / fake_engagement) * 100
                            fake_patterns.append(f"• **Lower engagement scores** (-{engagement_diff:.1f}%) - less community interest")
                    
                    if 'num_comments' in engagement_stats.get('fake', {}):
                        if fake_comments > real_comments:
                            comment_diff = ((fake_comments - real_comments) / real_comments) * 100
                            fake_patterns.append(f"• **More comments** (+{comment_diff:.1f}%) - potentially controversial or polarizing")
                        else:
                            comment_diff = ((real_comments - fake_comments) / fake_comments) * 100
                            fake_patterns.append(f"• **Fewer comments** (-{comment_diff:.1f}%) - less discussion generated")
                    
                    fake_patterns.append(f"• **Average score:** {fake_engagement:.1f} points")
                    fake_patterns.append(f"• **Average comments:** {fake_comments:.1f} per post")
                    
                    for pattern in fake_patterns:
                        st.markdown(pattern)
                
                with col2:
                    st.markdown("### 🟢 Real Content Social Patterns")
                    real_patterns = []
                    
                    if 'score' in engagement_stats.get('real', {}):
                        if real_engagement > fake_engagement:
                            engagement_diff = ((real_engagement - fake_engagement) / fake_engagement) * 100
                            real_patterns.append(f"• **Higher engagement scores** (+{engagement_diff:.1f}%) - genuine community interest")
                        else:
                            engagement_diff = ((fake_engagement - real_engagement) / real_engagement) * 100
                            real_patterns.append(f"• **Lower engagement scores** (-{engagement_diff:.1f}%) - more subdued response")
                    
                    if 'num_comments' in engagement_stats.get('real', {}):
                        if real_comments > fake_comments:
                            comment_diff = ((real_comments - fake_comments) / fake_comments) * 100
                            real_patterns.append(f"• **More comments** (+{comment_diff:.1f}%) - stimulates meaningful discussion")
                        else:
                            comment_diff = ((fake_comments - real_comments) / real_comments) * 100
                            real_patterns.append(f"• **Fewer comments** (-{comment_diff:.1f}%) - less controversial")
                    
                    real_patterns.append(f"• **Average score:** {real_engagement:.1f} points")
                    real_patterns.append(f"• **Average comments:** {real_comments:.1f} per post")
                    
                    for pattern in real_patterns:
                        st.markdown(pattern)
                
                # Summary insight
                st.markdown("---")
                st.subheader("💡 Social Engagement Summary")
                
                if 'score' in engagement_stats.get('fake', {}) and fake_count > 1 and real_count > 1:
                    # Calculate engagement difference percentage
                    if real_engagement > 0 and fake_engagement > 0:
                        if real_engagement > fake_engagement:
                            engagement_diff_pct = ((real_engagement - fake_engagement) / fake_engagement) * 100
                        else:
                            engagement_diff_pct = ((fake_engagement - real_engagement) / real_engagement) * 100
                    else:
                        engagement_diff_pct = 0
                    
                    # Determine which type gets more engagement
                    if abs(cohens_d) > 0.5:
                        if real_engagement > fake_engagement:
                            st.success(f"""
                            **Key Finding:** Real content receives significantly higher social engagement ({engagement_diff_pct:.1f}% more) with a {effect_size.lower()}.
                            
                            **Implication:** Authentic content tends to generate more genuine community interaction and sustained engagement.
                            This pattern can be used as a social signal for authenticity detection.
                            """)
                        else:
                            st.warning(f"""
                            **Key Finding:** Fake content receives significantly higher social engagement ({engagement_diff_pct:.1f}% more) with a {effect_size.lower()}.
                            
                            **Implication:** Misinformation may be designed to be more sensational or controversial, driving higher engagement.
                            High engagement alone is not a reliable indicator of authenticity and may actually signal potential misinformation.
                            """)
                    else:
                        st.info("""
                        **Key Finding:** Social engagement patterns show minimal differences between fake and real content.
                        
                        **Implication:** Social metrics alone may not be sufficient for authenticity detection in this dataset.
                        Combining social signals with textual and visual analysis provides more reliable results.
                        """)
            
        except FileNotFoundError as e:
            st.error(f"📂 Data file not found: {e}")
            st.info("""
            **To generate social engagement data:**
            ```bash
            python tasks/run_task5_social_engagement_analysis.py
            ```
            This will analyze social engagement patterns and generate integrated engagement metrics.
            """)
        except Exception as e:
            st.error(f"❌ Error loading social engagement data: {e}")
            st.info("Please ensure social engagement analysis (Task 5) is complete and data files are accessible.")
        finally:
            lazy_loader.hide_section_loading()
