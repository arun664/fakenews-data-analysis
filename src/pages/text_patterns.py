"""
Text Patterns Page
Linguistic pattern mining and text analysis
COMPLETE IMPLEMENTATION - Extracted from app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.lazy_loader import LazyLoader

lazy_loader = LazyLoader()


def render_text_patterns(container):
    """Render Text Patterns with lazy loading - COMPLETE IMPLEMENTATION"""
    with container.container():
        try:
            st.header("Linguistic Pattern Mining & Text Analysis")
            st.markdown("**Advanced NLP analysis of text patterns and authenticity features**")
            
            # Load linguistic analysis data with performance optimization
            @st.cache_data(ttl=600)  # 10 minutes cache for linguistic analysis (static results)
            def load_linguistic_data():
                linguistic_data_path = Path("analysis_results/dashboard_data/linguistic_analysis_dashboard.json")
                if not linguistic_data_path.exists():
                    raise FileNotFoundError(f"Linguistic analysis data not found at {linguistic_data_path}")
                with open(linguistic_data_path, 'r') as f:
                    data = json.load(f)
                # Sample large topic modeling results if needed
                if 'topic_modeling' in data and 'topics' in data['topic_modeling']:
                    if len(data['topic_modeling']['topics']) > 100:
                        st.info(f"📊 Displaying top 100 topics (from {len(data['topic_modeling']['topics'])} total) for optimal performance")
                        data['topic_modeling']['topics'] = data['topic_modeling']['topics'][:100]
                return data
            
            try:
                linguistic_data = load_linguistic_data()
            except FileNotFoundError:
                linguistic_data = None
            
            # Hide loading indicator
            lazy_loader.hide_section_loading()
            
            if linguistic_data:
                # Overview metrics
                st.subheader("📊 Linguistic Analysis Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Records Analyzed", f"{linguistic_data.get('total_records', 0):,}",
                             help="Total text records processed")
                
                with col2:
                    st.metric("Features Extracted", f"{linguistic_data.get('feature_count', 26)}",
                             help="Linguistic features per record")
                
                with col3:
                    st.metric("Topics Discovered", f"{linguistic_data.get('topic_count', 10)}",
                             help="LDA topic modeling results")
                
                with col4:
                    st.metric("Authenticity Patterns", f"{linguistic_data.get('significant_patterns', 'N/A')}",
                             help="Statistically significant differences")
                
                # Feature categories
                st.subheader("🔤 Linguistic Feature Categories")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**📏 Text Metrics**")
                    st.write("• Text length & word count")
                    st.write("• Average word length")
                    st.write("• Sentence count & structure")
                    st.write("• Character-level statistics")
                
                with col2:
                    st.write("**📖 Readability**")
                    st.write("• Flesch Reading Ease")
                    st.write("• Flesch-Kincaid Grade")
                    st.write("• Complexity measures")
                    st.write("• Vocabulary diversity")
                
                with col3:
                    st.write("**😊 Sentiment Analysis**")
                    st.write("• Compound sentiment score")
                    st.write("• Positive/Negative/Neutral")
                    st.write("• Emotional intensity")
                    st.write("• Subjectivity measures")
                
                # Authenticity analysis
                if linguistic_data.get('authenticity_analysis'):
                    st.subheader("🎯 Authenticity Pattern Analysis")
                    
                    auth_analysis = linguistic_data['authenticity_analysis']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🚨 Fake Content Characteristics**")
                        fake_patterns = auth_analysis.get('fake_patterns', [])
                        for pattern in fake_patterns[:5]:
                            st.write(f"• {pattern}")
                    
                    with col2:
                        st.write("**✅ Authentic Content Characteristics**")
                        real_patterns = auth_analysis.get('real_patterns', [])
                        for pattern in real_patterns[:5]:
                            st.write(f"• {pattern}")
                
                # Topic modeling results
                if linguistic_data.get('topic_modeling'):
                    st.subheader("🏷️ Topic Modeling Results")
                    
                    topics = linguistic_data['topic_modeling'].get('topics', [])
                    
                    if topics:
                        topic_df = pd.DataFrame(topics)
                        
                        if 'fake_prevalence' in topic_df.columns and 'real_prevalence' in topic_df.columns:
                            fig_topics = px.scatter(
                                topic_df,
                                x='fake_prevalence',
                                y='real_prevalence', 
                                size='total_docs',
                                hover_name='topic_id',
                                hover_data=['top_words'],
                                title="Topic Distribution: Fake vs Real Content",
                                labels={
                                    'fake_prevalence': 'Prevalence in Fake Content (%)',
                                    'real_prevalence': 'Prevalence in Real Content (%)'
                                }
                            )
                            
                            fig_topics.add_shape(
                                type="line",
                                x0=0, y0=0, x1=100, y1=100,
                                line=dict(color="red", width=2, dash="dash")
                            )
                            
                            st.plotly_chart(fig_topics, use_container_width=True)
                        
                        # Topic details table
                        with st.expander("📋 Detailed Topic Analysis"):
                            display_topics = []
                            for topic in topics[:10]:
                                display_topics.append({
                                    'Topic ID': topic.get('topic_id', 'N/A'),
                                    'Top Words': ', '.join(topic.get('top_words', [])[:5]),
                                    'Fake %': f"{topic.get('fake_prevalence', 0):.1f}%",
                                    'Real %': f"{topic.get('real_prevalence', 0):.1f}%",
                                    'Total Docs': topic.get('total_docs', 0)
                                })
                            
                            st.dataframe(pd.DataFrame(display_topics), use_container_width=True)
                
                # Statistical analysis
                if linguistic_data.get('statistical_analysis'):
                    st.subheader("📊 Statistical Analysis Results")
                    
                    stats_data = linguistic_data['statistical_analysis']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Significant Features", stats_data.get('significant_features', 'N/A'))
                    
                    with col2:
                        st.metric("Average Effect Size", f"{stats_data.get('avg_effect_size', 0):.3f}")
                    
                    with col3:
                        st.metric("P-value Threshold", f"{stats_data.get('p_threshold', 0.05):.3f}")
                
                # Interactive visualizations
                st.subheader("🎨 Interactive Visualizations")
                
                viz_tabs = st.tabs(["Feature Distributions", "Sentiment Analysis", "Topic Models", "Authenticity Heatmap"])
                
                with viz_tabs[0]:
                    viz_path = Path("visualizations/linguistic_patterns/feature_distributions.png")
                    if viz_path.exists():
                        st.image(str(viz_path), caption="Linguistic Feature Distributions by Authenticity")
                    else:
                        st.info("📊 Feature distribution visualization not available")
                
                with viz_tabs[1]:
                    viz_path = Path("visualizations/linguistic_patterns/wordclouds_authenticity.png")
                    if viz_path.exists():
                        st.image(str(viz_path), caption="Word Clouds by Content Authenticity")
                    else:
                        st.info("📊 Sentiment analysis visualization not available")
                
                with viz_tabs[2]:
                    viz_path = Path("visualizations/linguistic_patterns/topic_modeling.png")
                    if viz_path.exists():
                        st.image(str(viz_path), caption="Topic Modeling Results")
                    else:
                        st.info("📊 Topic modeling visualization not available")
                
                with viz_tabs[3]:
                    viz_path = Path("visualizations/linguistic_patterns/authenticity_heatmap.png")
                    if viz_path.exists():
                        st.image(str(viz_path), caption="Authenticity Pattern Heatmap")
                    else:
                        st.info("📊 Authenticity heatmap not available")
                
                # Comprehensive linguistic summary
                st.markdown("---")
                st.subheader("💡 Linguistic Analysis Summary: Fake vs Real")
                
                if linguistic_data.get('authenticity_analysis'):
                    auth_analysis = linguistic_data['authenticity_analysis']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🔴 Fake Content Linguistic Profile")
                        fake_patterns = auth_analysis.get('fake_patterns', [])
                        if fake_patterns:
                            st.markdown("**Key Characteristics:**")
                            for pattern in fake_patterns[:7]:
                                st.markdown(f"• {pattern}")
                        else:
                            st.info("No distinct fake content patterns identified")
                    
                    with col2:
                        st.markdown("### 🟢 Real Content Linguistic Profile")
                        real_patterns = auth_analysis.get('real_patterns', [])
                        if real_patterns:
                            st.markdown("**Key Characteristics:**")
                            for pattern in real_patterns[:7]:
                                st.markdown(f"• {pattern}")
                        else:
                            st.info("No distinct real content patterns identified")
                    
                    # Overall summary
                    st.markdown("---")
                    if linguistic_data.get('statistical_analysis'):
                        stats = linguistic_data['statistical_analysis']
                        sig_features = stats.get('significant_features', 0)
                        
                        if sig_features > 0:
                            st.success(f"""
                            **Key Finding:** {sig_features} linguistic features show statistically significant differences between fake and real content.
                            
                            **Implication:** Textual analysis reveals distinct linguistic patterns in misinformation. Fake content often exhibits:
                            - Different readability levels
                            - Distinct sentiment patterns
                            - Unique vocabulary choices
                            - Varying complexity and structure
                            
                            These patterns can be leveraged for automated authenticity detection when combined with visual and social signals.
                            """)
                        else:
                            st.info("""
                            **Key Finding:** Minimal linguistic differences detected between fake and real content.
                            
                            **Implication:** Sophisticated misinformation may use similar linguistic patterns to authentic content,
                            making text-only detection challenging. Multimodal analysis combining text, images, and social signals
                            provides more reliable authenticity assessment.
                            """)
                else:
                    st.info("Detailed authenticity analysis not available. Run linguistic pattern mining task for comprehensive results.")
            else:
                st.warning("📂 Linguistic analysis data not available. Please run Task 9 first.")
                st.info("""
                **To generate linguistic analysis data:**
                ```bash
                python tasks/run_task9_linguistic_pattern_mining.py
                ```
                Task 9 performs comprehensive NLP analysis including sentiment analysis, topic modeling, and authenticity pattern discovery.
                """)
                
        except FileNotFoundError as e:
            st.error(f"📂 Data file not found: {e}")
            st.info("""
            **To generate linguistic analysis data:**
            ```bash
            python tasks/run_task9_linguistic_pattern_mining.py
            ```
            This will perform comprehensive NLP analysis including sentiment, topic modeling, and authenticity patterns.
            """)
        except Exception as e:
            st.error(f"❌ Error loading linguistic analysis data: {e}")
            st.info("Please ensure linguistic pattern mining (Task 9) has been completed successfully.")
        finally:
            lazy_loader.hide_section_loading()
