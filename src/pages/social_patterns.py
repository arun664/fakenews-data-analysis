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
            
            # Load social engagement data with performance optimization
            @st.cache_data(ttl=600)  # 10 minutes cache for social engagement data
            def load_social_data():
                data = pd.read_parquet('processed_data/social_engagement/integrated_engagement_data.parquet')
                original_size = len(data)
                # Performance optimization: Sample very large datasets
                if original_size > 100000:
                    data = data.sample(n=100000, random_state=42)
                return data, original_size
            
            social_data, original_size = load_social_data()
            
            # Show sampling notification if data was sampled
            if original_size > 100000:
                st.info(f"📊 Performance Optimization: Analyzing {len(social_data):,} sampled records (from {original_size:,} total) for optimal dashboard performance")
            
            # Hide loading indicator
            lazy_loader.hide_section_loading()
            
            if len(social_data) > 0:
                # Overall social engagement metrics
                st.subheader("📊 Social Engagement Overview")
                
                fake_posts = social_data[social_data['2_way_label'] == 0]
                real_posts = social_data[social_data['2_way_label'] == 1]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📝 Total Posts", f"{len(social_data):,}")
                
                with col2:
                    fake_engagement = fake_posts['score'].mean() if 'score' in fake_posts.columns else 0
                    st.metric("🔴 Fake Avg Score", f"{fake_engagement:.1f}")
                
                with col3:
                    real_engagement = real_posts['score'].mean() if 'score' in real_posts.columns else 0
                    st.metric("🟢 Real Avg Score", f"{real_engagement:.1f}")
                
                with col4:
                    if real_engagement > 0:
                        engagement_ratio = fake_engagement / real_engagement
                        st.metric("📊 Fake:Real Ratio", f"{engagement_ratio:.2f}:1")
                
                # Engagement distribution comparison
                st.subheader("📈 Engagement Score Distribution")
                
                if 'score' in social_data.columns:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=fake_posts['score'].dropna(),
                        name='🔴 Fake Content',
                        opacity=0.7,
                        marker_color='#FF6B6B',
                        nbinsx=50
                    ))
                    
                    fig.add_trace(go.Histogram(
                        x=real_posts['score'].dropna(),
                        name='🟢 Real Content',
                        opacity=0.7,
                        marker_color='#4ECDC4',
                        nbinsx=50
                    ))
                    
                    fig.update_layout(
                        title="Engagement Score Distribution: Fake vs Real",
                        xaxis_title="Engagement Score",
                        yaxis_title="Count",
                        barmode='overlay',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Comment patterns
                st.subheader("💬 Comment Engagement Patterns")
                
                if 'num_comments' in social_data.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fake_comments = fake_posts['num_comments'].mean()
                        real_comments = real_posts['num_comments'].mean()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=['Fake Content', 'Real Content'],
                            y=[fake_comments, real_comments],
                            marker_color=['#FF6B6B', '#4ECDC4'],
                            text=[f"{fake_comments:.1f}", f"{real_comments:.1f}"],
                            textposition='outside'
                        ))
                        
                        fig.update_layout(
                            title="Average Comments per Post",
                            yaxis_title="Number of Comments",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = go.Figure()
                        
                        fig.add_trace(go.Box(
                            y=fake_posts['num_comments'].dropna(),
                            name='🔴 Fake',
                            marker_color='#FF6B6B',
                            boxpoints='outliers'
                        ))
                        
                        fig.add_trace(go.Box(
                            y=real_posts['num_comments'].dropna(),
                            name='🟢 Real',
                            marker_color='#4ECDC4',
                            boxpoints='outliers'
                        ))
                        
                        fig.update_layout(
                            title="Comment Count Distribution",
                            yaxis_title="Number of Comments",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Statistical significance testing
                st.subheader("📈 Statistical Analysis")
                
                if 'score' in social_data.columns:
                    fake_scores = fake_posts['score'].dropna()
                    real_scores = real_posts['score'].dropna()
                    
                    if len(fake_scores) > 0 and len(real_scores) > 0:
                        t_stat, p_value = stats.ttest_ind(fake_scores, real_scores)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Engagement Score Significance:**")
                            st.write(f"• T-statistic: {t_stat:.3f}")
                            st.write(f"• P-value: {p_value:.6f}")
                            if p_value < 0.05:
                                st.success("✅ Statistically significant difference")
                            else:
                                st.warning("⚠️ No significant difference")
                        
                        with col2:
                            pooled_std = np.sqrt(((len(fake_scores)-1)*fake_scores.var() + 
                                                (len(real_scores)-1)*real_scores.var()) / 
                                               (len(fake_scores) + len(real_scores) - 2))
                            cohens_d = (real_scores.mean() - fake_scores.mean()) / pooled_std if pooled_std > 0 else 0
                            
                            st.write("**Effect Size Analysis:**")
                            st.write(f"• Cohen's d: {cohens_d:.3f}")
                            
                            if abs(cohens_d) > 0.8:
                                effect_size = "Large effect"
                            elif abs(cohens_d) > 0.5:
                                effect_size = "Medium effect"
                            elif abs(cohens_d) > 0.2:
                                effect_size = "Small effect"
                            else:
                                effect_size = "Negligible effect"
                            
                            st.write(f"• Interpretation: {effect_size}")
                
                # Enhanced social insights with fake vs real comparison
                st.subheader("🎯 Social Engagement Insights: What Distinguishes Fake from Real?")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🔴 Fake Content Social Patterns")
                    fake_patterns = []
                    
                    if 'score' in social_data.columns:
                        if fake_engagement > real_engagement:
                            engagement_diff = ((fake_engagement - real_engagement) / real_engagement) * 100
                            fake_patterns.append(f"• **Higher engagement scores** (+{engagement_diff:.1f}%) - may indicate viral spread or bot activity")
                        else:
                            engagement_diff = ((real_engagement - fake_engagement) / fake_engagement) * 100
                            fake_patterns.append(f"• **Lower engagement scores** (-{engagement_diff:.1f}%) - less community interest")
                    
                    if 'num_comments' in social_data.columns:
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
                    
                    if 'score' in social_data.columns:
                        if real_engagement > fake_engagement:
                            engagement_diff = ((real_engagement - fake_engagement) / fake_engagement) * 100
                            real_patterns.append(f"• **Higher engagement scores** (+{engagement_diff:.1f}%) - genuine community interest")
                        else:
                            engagement_diff = ((fake_engagement - real_engagement) / real_engagement) * 100
                            real_patterns.append(f"• **Lower engagement scores** (-{engagement_diff:.1f}%) - more subdued response")
                    
                    if 'num_comments' in social_data.columns:
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
                
                if 'score' in social_data.columns:
                    # Determine which type gets more engagement
                    if abs(cohens_d) > 0.5:
                        if real_engagement > fake_engagement:
                            st.success(f"""
                            **Key Finding:** Real content receives significantly higher social engagement ({engagement_diff:.1f}% more) with a {effect_size.lower()}.
                            
                            **Implication:** Authentic content tends to generate more genuine community interaction and sustained engagement.
                            This pattern can be used as a social signal for authenticity detection.
                            """)
                        else:
                            st.warning(f"""
                            **Key Finding:** Fake content receives significantly higher social engagement ({engagement_diff:.1f}% more) with a {effect_size.lower()}.
                            
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
