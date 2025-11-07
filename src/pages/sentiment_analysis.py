"""
Sentiment Analysis Page
Displays sentiment analysis results for fake vs real content
"""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.lazy_loader import LazyLoader

lazy_loader = LazyLoader()


def render_sentiment_analysis(container):
    """Render Sentiment Analysis with lazy loading"""
    with container.container():
        try:
            st.header("Comprehensive Sentiment Analysis: Fake vs Real Content")
            
            st.markdown("""
            **Key Questions Answered:**
            - Do fake and real content have different emotional tones in titles?
            - How does comment sentiment differ between authentic and inauthentic posts?
            - What emotional patterns distinguish misinformation from legitimate content?
            - Are there psychological manipulation tactics visible in sentiment data?
            """)
            
            # Lazy load sentiment data with performance optimization
            @st.cache_data(ttl=600)  # 10 minutes cache for sentiment analysis (static results)
            def load_sentiment_data():
                sentiment_results_path = Path("analysis_results/sentiment_analysis/comprehensive_sentiment_analysis.json")
                if not sentiment_results_path.exists():
                    raise FileNotFoundError(f"Sentiment analysis results not found at {sentiment_results_path}")
                with open(sentiment_results_path, 'r') as f:
                    data = json.load(f)
                # Sample large sentiment datasets if needed
                if 'detailed_results' in data and len(data['detailed_results']) > 50000:
                    st.info(f"📊 Sampling {len(data['detailed_results'])} sentiment results to 50,000 for optimal performance")
                    data['detailed_results'] = data['detailed_results'][:50000]
                return data
            
            try:
                sentiment_data = load_sentiment_data()
            except FileNotFoundError:
                sentiment_data = None
            
            # Hide loading indicator after data is loaded
            lazy_loader.hide_section_loading()
            
            if sentiment_data:
                # Analysis overview
                st.subheader("📊 Sentiment Analysis Overview")
                metadata = sentiment_data.get('analysis_metadata', {})
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Posts Analyzed", f"{metadata.get('total_posts_analyzed', 0):,}")
                with col2:
                    st.metric("Fake Posts", f"{metadata.get('fake_posts', 0):,}")
                with col3:
                    st.metric("Real Posts", f"{metadata.get('real_posts', 0):,}")
                with col4:
                    st.metric("Posts with Comment Sentiment", f"{metadata.get('posts_with_comment_sentiment', 0):,}")
                
                # Sentiment comparison: Fake vs Real
                st.subheader("😊 Sentiment Patterns: Fake vs Real Content")
                
                if sentiment_data.get('title_sentiment_comparison'):
                    title_sentiment = sentiment_data['title_sentiment_comparison']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🔴 Fake Content Sentiment")
                        fake_sent = title_sentiment.get('fake', {})
                        st.markdown(f"**Title Sentiment:**")
                        st.markdown(f"• Positive: {fake_sent.get('positive', 0):.1%}")
                        st.markdown(f"• Negative: {fake_sent.get('negative', 0):.1%}")
                        st.markdown(f"• Neutral: {fake_sent.get('neutral', 0):.1%}")
                        st.markdown(f"• Avg Compound: {fake_sent.get('compound_mean', 0):.3f}")
                        
                        if fake_sent.get('negative', 0) > fake_sent.get('positive', 0):
                            st.info("**Pattern:** Fake content tends toward negative sentiment - may use fear or outrage")
                        elif fake_sent.get('positive', 0) > fake_sent.get('negative', 0):
                            st.info("**Pattern:** Fake content tends toward positive sentiment - may use sensationalism")
                        else:
                            st.info("**Pattern:** Fake content shows balanced sentiment distribution")
                    
                    with col2:
                        st.markdown("### 🟢 Real Content Sentiment")
                        real_sent = title_sentiment.get('real', {})
                        st.markdown(f"**Title Sentiment:**")
                        st.markdown(f"• Positive: {real_sent.get('positive', 0):.1%}")
                        st.markdown(f"• Negative: {real_sent.get('negative', 0):.1%}")
                        st.markdown(f"• Neutral: {real_sent.get('neutral', 0):.1%}")
                        st.markdown(f"• Avg Compound: {real_sent.get('compound_mean', 0):.3f}")
                        
                        if real_sent.get('neutral', 0) > 0.5:
                            st.info("**Pattern:** Real content tends toward neutral sentiment - more objective reporting")
                        elif real_sent.get('negative', 0) > real_sent.get('positive', 0):
                            st.info("**Pattern:** Real content tends toward negative sentiment - serious news topics")
                        else:
                            st.info("**Pattern:** Real content shows varied sentiment distribution")
                
                # Comment sentiment analysis
                if sentiment_data.get('comment_sentiment_comparison'):
                    st.subheader("💬 Comment Sentiment Analysis")
                    
                    comment_sentiment = sentiment_data['comment_sentiment_comparison']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🔴 Comments on Fake Posts")
                        fake_comments = comment_sentiment.get('fake', {})
                        st.markdown(f"• Avg Sentiment: {fake_comments.get('avg_sentiment', 0):.3f}")
                        st.markdown(f"• Sentiment Variance: {fake_comments.get('sentiment_variance', 0):.3f}")
                        st.markdown(f"• Polarization: {fake_comments.get('polarization', 0):.3f}")
                    
                    with col2:
                        st.markdown("### 🟢 Comments on Real Posts")
                        real_comments = comment_sentiment.get('real', {})
                        st.markdown(f"• Avg Sentiment: {real_comments.get('avg_sentiment', 0):.3f}")
                        st.markdown(f"• Sentiment Variance: {real_comments.get('sentiment_variance', 0):.3f}")
                        st.markdown(f"• Polarization: {real_comments.get('polarization', 0):.3f}")
                
                # Summary insights
                st.markdown("---")
                st.subheader("💡 Sentiment Analysis Summary")
                
                if sentiment_data.get('title_sentiment_comparison'):
                    fake_sent = title_sentiment.get('fake', {})
                    real_sent = title_sentiment.get('real', {})
                    
                    fake_compound = fake_sent.get('compound_mean', 0)
                    real_compound = real_sent.get('compound_mean', 0)
                    
                    sentiment_diff = abs(fake_compound - real_compound)
                    
                    if sentiment_diff > 0.1:
                        st.success(f"""
                        **Key Finding:** Significant sentiment differences detected between fake and real content.
                        
                        **Fake Content:** Compound sentiment score of {fake_compound:.3f}
                        **Real Content:** Compound sentiment score of {real_compound:.3f}
                        **Difference:** {sentiment_diff:.3f}
                        
                        **Implication:** Sentiment analysis can serve as a valuable signal for authenticity detection.
                        Fake content often employs emotional manipulation through extreme positive or negative sentiment,
                        while authentic content tends toward more neutral, objective language.
                        """)
                    else:
                        st.info("""
                        **Key Finding:** Minimal sentiment differences between fake and real content.
                        
                        **Implication:** Sophisticated misinformation may mimic the sentiment patterns of authentic content.
                        Sentiment analysis should be combined with other features (visual, social, linguistic) for
                        reliable authenticity detection.
                        """)
            else:
                st.warning("📂 Sentiment analysis data not available. Please run the sentiment analysis first.")
                
        except FileNotFoundError as e:
            st.error(f"📂 Data file not found: {e}")
            st.info("""
            **To generate sentiment analysis data:**
            ```bash
            python tasks/run_sentiment_analysis.py
            ```
            This will analyze sentiment patterns in titles and comments for fake vs real content.
            """)
        except Exception as e:
            st.error(f"❌ Error loading sentiment analysis: {e}")
            st.info("Please check that sentiment analysis has been completed successfully.")
        finally:
            lazy_loader.hide_section_loading()
