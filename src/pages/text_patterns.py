"""
Text Patterns Page
Linguistic pattern mining and text analysis
COMPLETE IMPLEMENTATION - Extracted from app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from wordcloud import WordCloud
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.lazy_loader import LazyLoader
from src.utils.visualization_helpers import CHART_CONFIG

lazy_loader = LazyLoader()


def load_linguistic_features_data():
    """Load linguistic features summary with pre-computed statistics"""
    try:
        import json
        
        summary_path = Path("analysis_results/dashboard_data/linguistic_features_summary.json")
        
        if not summary_path.exists():
            st.warning(f"Linguistic features summary not found at {summary_path}")
            return None
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Return the full summary (not sample_data)
        return summary
            
    except Exception as e:
        st.error(f"Error loading linguistic features: {e}")
        return None


def render_feature_distributions_tab(linguistic_data):
    """Tab 1: Feature Distributions - Readability, vocabulary, text length, sentence complexity"""
    st.markdown("### 📏 Linguistic Feature Distributions (Full Dataset)")
    
    # Load summary with pre-computed data
    summary = load_linguistic_features_data()
    
    if summary is not None:
        features_by_auth = summary.get('features_by_authenticity', {})
        histograms = summary.get('histograms', {})
        violin_data = summary.get('violin_data', {})
        
        total_records = summary.get('total_records', 0)
        fake_count = summary.get('fake_count', 0)
        real_count = summary.get('real_count', 0)
        
        if fake_count == 0 or real_count == 0:
            st.warning(f"⚠️ Insufficient data: {fake_count} fake, {real_count} real records. Need both types for comparison.")
            return
        
        # 1. Readability Score Violin Plots (using pre-computed data)
        st.markdown("#### 📖 Readability Scores")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'flesch_reading_ease' in violin_data:
                fig_fre = go.Figure()
                for label_name, color in [('fake', CHART_CONFIG['colors']['fake']), ('real', CHART_CONFIG['colors']['real'])]:
                    if label_name in violin_data['flesch_reading_ease']:
                        vdata = violin_data['flesch_reading_ease'][label_name]
                        fig_fre.add_trace(go.Violin(
                            y=vdata['percentiles'],
                            name=label_name.capitalize(),
                            box_visible=True,
                            meanline_visible=True,
                            fillcolor=color,
                            opacity=0.6,
                            line_color=color
                        ))
                
                fig_fre.update_layout(
                    title=f"Flesch Reading Ease Score Distribution ({total_records:,} records)",
                    yaxis_title="Reading Ease Score",
                    showlegend=True,
                    height=400
                )
                st.plotly_chart(fig_fre, use_container_width=True)
        
        with col2:
            if 'flesch_kincaid_grade' in violin_data:
                fig_fkg = go.Figure()
                for label_name, color in [('fake', CHART_CONFIG['colors']['fake']), ('real', CHART_CONFIG['colors']['real'])]:
                    if label_name in violin_data['flesch_kincaid_grade']:
                        vdata = violin_data['flesch_kincaid_grade'][label_name]
                        fig_fkg.add_trace(go.Violin(
                            y=vdata['percentiles'],
                            name=label_name.capitalize(),
                            box_visible=True,
                            meanline_visible=True,
                            fillcolor=color,
                            opacity=0.6,
                            line_color=color
                        ))
                
                fig_fkg.update_layout(
                    title=f"Flesch-Kincaid Grade Level Distribution ({total_records:,} records)",
                    yaxis_title="Grade Level",
                    showlegend=True,
                    height=400
                )
                st.plotly_chart(fig_fkg, use_container_width=True)
        
        # 2. Vocabulary Diversity Bar Charts (using pre-computed statistics)
        st.markdown("#### 📚 Vocabulary Diversity")
        if 'unique_word_ratio' in features_by_auth.get('fake', {}) and 'unique_word_ratio' in features_by_auth.get('real', {}):
            fake_stats = features_by_auth['fake']['unique_word_ratio']
            real_stats = features_by_auth['real']['unique_word_ratio']
            
            fig_vocab = go.Figure()
            fig_vocab.add_trace(go.Bar(
                x=['Fake', 'Real'],
                y=[fake_stats['mean'], real_stats['mean']],
                error_y=dict(type='data', array=[fake_stats['std'], real_stats['std']]),
                marker_color=[CHART_CONFIG['colors']['fake'], CHART_CONFIG['colors']['real']],
                text=[f"{fake_stats['mean']:.4f}", f"{real_stats['mean']:.4f}"],
                textposition='outside'
            ))
            
            fig_vocab.update_layout(
                title=f"Unique Word Ratio by Authenticity ({total_records:,} records)",
                xaxis_title="Content Type",
                yaxis_title="Unique Word Ratio (Mean ± SD)",
                height=400
            )
            st.plotly_chart(fig_vocab, use_container_width=True)
        
        # 3. Text Length Histograms (using pre-computed data)
        st.markdown("#### 📝 Text Length Distributions")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'text_length' in histograms:
                fig_len = go.Figure()
                for label_name, color in [('fake', CHART_CONFIG['colors']['fake']), ('real', CHART_CONFIG['colors']['real'])]:
                    if label_name in histograms['text_length']:
                        hist = histograms['text_length'][label_name]
                        bin_centers = [(hist['bin_edges'][i] + hist['bin_edges'][i+1]) / 2 
                                      for i in range(len(hist['bin_edges']) - 1)]
                        fig_len.add_trace(go.Bar(
                            x=bin_centers,
                            y=hist['counts'],
                            name=label_name.capitalize(),
                            opacity=0.6,
                            marker_color=color
                        ))
                
                fig_len.update_layout(
                    title=f"Text Length Distribution ({total_records:,} records)",
                    xaxis_title="Text Length (characters)",
                    yaxis_title="Frequency",
                    barmode='overlay',
                    height=400
                )
                st.plotly_chart(fig_len, use_container_width=True)
        
        with col2:
            if 'word_count' in histograms:
                fig_words = go.Figure()
                for label_name, color in [('fake', CHART_CONFIG['colors']['fake']), ('real', CHART_CONFIG['colors']['real'])]:
                    if label_name in histograms['word_count']:
                        hist = histograms['word_count'][label_name]
                        bin_centers = [(hist['bin_edges'][i] + hist['bin_edges'][i+1]) / 2 
                                      for i in range(len(hist['bin_edges']) - 1)]
                        fig_words.add_trace(go.Bar(
                            x=bin_centers,
                            y=hist['counts'],
                            name=label_name.capitalize(),
                            opacity=0.6,
                            marker_color=color
                        ))
                
                fig_words.update_layout(
                    title=f"Word Count Distribution ({total_records:,} records)",
                    xaxis_title="Word Count",
                    yaxis_title="Frequency",
                    barmode='overlay',
                    height=400
                )
                st.plotly_chart(fig_words, use_container_width=True)
        
        # 4. Sentence Complexity (using pre-computed statistics)
        st.markdown("#### 📊 Sentence Complexity")
        if 'avg_sentence_length' in features_by_auth.get('fake', {}) and 'avg_sentence_length' in features_by_auth.get('real', {}):
            fake_stats = features_by_auth['fake']['avg_sentence_length']
            real_stats = features_by_auth['real']['avg_sentence_length']
            
            fig_sent = go.Figure()
            fig_sent.add_trace(go.Bar(
                x=['Fake', 'Real'],
                y=[fake_stats['mean'], real_stats['mean']],
                error_y=dict(type='data', array=[fake_stats['std'], real_stats['std']]),
                marker_color=[CHART_CONFIG['colors']['fake'], CHART_CONFIG['colors']['real']],
                text=[f"{fake_stats['mean']:.2f}", f"{real_stats['mean']:.2f}"],
                textposition='outside'
            ))
            
            fig_sent.update_layout(
                title=f"Average Sentence Length by Authenticity ({total_records:,} records)",
                yaxis_title="Average Sentence Length (words)",
                height=400
            )
            st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.info("📊 Detailed feature data not available. Showing summary statistics from linguistic analysis.")
        
        # Show summary from linguistic_data
        if linguistic_data and 'top_discriminative_features' in linguistic_data:
            features = linguistic_data['top_discriminative_features']
            df_features = pd.DataFrame(features)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_features['feature_name'],
                y=df_features['effect_size'].abs(),
                marker_color=[CHART_CONFIG['colors']['fake'] if x < 0 else CHART_CONFIG['colors']['real'] 
                             for x in df_features['effect_size']],
                text=df_features['effect_size'].round(3),
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Top Discriminative Features (Effect Size)",
                xaxis_title="Feature",
                yaxis_title="Absolute Effect Size",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)


def render_sentiment_analysis_tab(linguistic_data):
    """Tab 2: Sentiment Analysis - Word clouds, sentiment distribution, emotion intensity, subjectivity"""
    st.markdown("### 😊 Sentiment & Emotion Analysis (Full Dataset)")
    
    summary = load_linguistic_features_data()
    
    if summary is not None:
        features_by_auth = summary.get('features_by_authenticity', {})
        total_records = summary.get('total_records', 0)
        
        # 1. Word Clouds for Fake vs Real
        st.markdown("#### ☁️ Word Clouds by Authenticity")
        st.info("💡 Word clouds require text data. This visualization shows sentiment patterns instead.")
        
        # 2. Sentiment Distribution Pie Charts (using pre-computed statistics)
        st.markdown("#### 🎭 Sentiment Distribution")
        
        if all(col in features_by_auth.get('fake', {}) for col in ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']):
            col1, col2 = st.columns(2)
            
            with col1:
                fake_stats = features_by_auth.get('fake', {})
                sentiment_fake = {
                    'Positive': fake_stats['sentiment_positive']['mean'],
                    'Negative': fake_stats['sentiment_negative']['mean'],
                    'Neutral': fake_stats['sentiment_neutral']['mean']
                }
                
                fig_pie_fake = go.Figure(data=[go.Pie(
                    labels=list(sentiment_fake.keys()),
                    values=list(sentiment_fake.values()),
                    marker=dict(colors=['#4ECDC4', '#FF6B6B', '#95A5A6']),
                    hole=0.3,
                    textinfo='label+percent',
                    textposition='auto'
                )])
                fig_pie_fake.update_layout(title=f"Fake Content Sentiment ({summary.get('fake_count', 0):,} records)", height=400)
                st.plotly_chart(fig_pie_fake, use_container_width=True)
            
            with col2:
                real_stats = features_by_auth.get('real', {})
                sentiment_real = {
                    'Positive': real_stats['sentiment_positive']['mean'],
                    'Negative': real_stats['sentiment_negative']['mean'],
                    'Neutral': real_stats['sentiment_neutral']['mean']
                }
                
                fig_pie_real = go.Figure(data=[go.Pie(
                    labels=list(sentiment_real.keys()),
                    values=list(sentiment_real.values()),
                    marker=dict(colors=['#4ECDC4', '#FF6B6B', '#95A5A6']),
                    hole=0.3,
                    textinfo='label+percent',
                    textposition='auto'
                )])
                fig_pie_real.update_layout(title=f"Real Content Sentiment ({summary.get('real_count', 0):,} records)", height=400)
                st.plotly_chart(fig_pie_real, use_container_width=True)
        
        # 3. Emotion Intensity Radar Charts (using pre-computed statistics)
        st.markdown("#### 🎯 Emotion Intensity Comparison")
        
        # Check for required columns
        required_cols = ['sentiment_positive', 'sentiment_negative']
        
        if all(col in features_by_auth.get('fake', {}) for col in required_cols):
            fake_stats = features_by_auth.get('fake', {})
            real_stats = features_by_auth.get('real', {})
            
            categories = []
            fake_values = []
            real_values = []
            
            # Add sentiment scores
            categories.append('Positive')
            fake_values.append(fake_stats['sentiment_positive']['mean'])
            real_values.append(real_stats['sentiment_positive']['mean'])
            
            categories.append('Negative')
            fake_values.append(fake_stats['sentiment_negative']['mean'])
            real_values.append(real_stats['sentiment_negative']['mean'])
            
            # Add optional metrics if available
            if 'polarity' in fake_stats and 'polarity' in real_stats:
                categories.append('Polarity')
                fake_values.append(abs(fake_stats['polarity']['mean']))
                real_values.append(abs(real_stats['polarity']['mean']))
            
            if 'subjectivity' in fake_stats and 'subjectivity' in real_stats:
                categories.append('Subjectivity')
                fake_values.append(fake_stats['subjectivity']['mean'])
                real_values.append(real_stats['subjectivity']['mean'])
            
            if 'exclamation_count' in fake_stats and 'exclamation_count' in real_stats:
                categories.append('Exclamation')
                # Normalize exclamation count to 0-1 scale
                max_excl = max(fake_stats['exclamation_count']['max'], real_stats['exclamation_count']['max'])
                if max_excl > 0:
                    fake_values.append(fake_stats['exclamation_count']['mean'] / max_excl)
                    real_values.append(real_stats['exclamation_count']['mean'] / max_excl)
                else:
                    fake_values.append(0)
                    real_values.append(0)
            
            # Create radar chart
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=fake_values,
                theta=categories,
                fill='toself',
                name='Fake',
                line_color=CHART_CONFIG['colors']['fake'],
                fillcolor=CHART_CONFIG['colors']['fake'],
                opacity=0.5
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=real_values,
                theta=categories,
                fill='toself',
                name='Real',
                line_color=CHART_CONFIG['colors']['real'],
                fillcolor=CHART_CONFIG['colors']['real'],
                opacity=0.5
            ))
            
            max_val = max(max(fake_values), max(real_values))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max_val * 1.1])),
                showlegend=True,
                title=f"Emotion Intensity Radar Chart ({total_records:,} records)",
                height=500
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.warning("Sentiment data not available for radar chart")
        
        # 4. Subjectivity Scatter Plot (using pre-computed 2D histogram)
        st.markdown("#### 📈 Polarity vs Subjectivity Density")
        
        # Load linguistic data to get scatter_data
        try:
            linguistic_data_path = Path("analysis_results/dashboard_data/linguistic_features_summary.json")
            if linguistic_data_path.exists():
                with open(linguistic_data_path, 'r') as f:
                    ling_summary = json.load(f)
                
                scatter_data = ling_summary.get('scatter_data', {})
                
                if 'polarity_vs_subjectivity_fake' in scatter_data and 'polarity_vs_subjectivity_real' in scatter_data:
                    # Create density heatmap from pre-computed 2D histogram
                    fig_scatter = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=['Fake Content', 'Real Content']
                    )
                    
                    for idx, (label_name, color) in enumerate([('fake', CHART_CONFIG['colors']['fake']), 
                                                                ('real', CHART_CONFIG['colors']['real'])], 1):
                        scatter_key = f'polarity_vs_subjectivity_{label_name}'
                        if scatter_key in scatter_data:
                            data_2d = scatter_data[scatter_key]
                            
                            fig_scatter.add_trace(
                                go.Heatmap(
                                    x=data_2d['x_edges'],
                                    y=data_2d['y_edges'],
                                    z=data_2d['counts'],
                                    colorscale='Viridis',
                                    showscale=(idx == 2)
                                ),
                                row=1, col=idx
                            )
                    
                    fig_scatter.update_xaxes(title_text="Subjectivity", row=1, col=1)
                    fig_scatter.update_xaxes(title_text="Subjectivity", row=1, col=2)
                    fig_scatter.update_yaxes(title_text="Polarity", row=1, col=1)
                    fig_scatter.update_yaxes(title_text="Polarity", row=1, col=2)
                    
                    fig_scatter.update_layout(
                        title="Sentiment Polarity vs Subjectivity Density (Full Dataset)",
                        height=400
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    st.info("💡 Darker regions show higher density of content. Upper regions show positive sentiment, lower regions show negative sentiment.")
                else:
                    st.info("📊 Scatter density data not available.")
            else:
                st.info("📊 Linguistic features summary not found.")
        except Exception as e:
            st.warning(f"Could not load scatter data: {e}")
    else:
        st.info("📊 Detailed sentiment data not available.")


def render_topic_models_tab(linguistic_data):
    """Tab 3: Topic Models - Topic prevalence, topic-authenticity heatmap, topic words, distribution over time"""
    st.markdown("### 🏷️ Topic Modeling Analysis")
    
    # Check if topic modeling data exists
    has_topics = linguistic_data and 'topic_modeling' in linguistic_data and linguistic_data['topic_modeling'].get('topics')
    
    if has_topics:
        topics = linguistic_data['topic_modeling'].get('topics', [])
        
        if topics:
            topic_df = pd.DataFrame(topics)
            
            # 1. Topic Prevalence Bar Chart
            st.markdown("#### 📊 Topic Prevalence")
            
            if 'total_docs' in topic_df.columns:
                topic_df_sorted = topic_df.sort_values('total_docs', ascending=False).head(15)
                
                fig_prev = go.Figure()
                fig_prev.add_trace(go.Bar(
                    x=topic_df_sorted['topic_id'],
                    y=topic_df_sorted['total_docs'],
                    marker_color=CHART_CONFIG['colors']['neutral'],
                    text=topic_df_sorted['total_docs'],
                    textposition='outside',
                    hovertext=topic_df_sorted['top_words'].apply(lambda x: ', '.join(x[:5]) if isinstance(x, list) else str(x))
                ))
                
                fig_prev.update_layout(
                    title="Top 15 Topics by Document Count",
                    xaxis_title="Topic ID",
                    yaxis_title="Number of Documents",
                    height=400
                )
                st.plotly_chart(fig_prev, use_container_width=True)
            
            # 2. Topic-Authenticity Heatmap
            st.markdown("#### 🔥 Topic-Authenticity Heatmap")
            
            if 'fake_prevalence' in topic_df.columns and 'real_prevalence' in topic_df.columns:
                # Create heatmap data
                top_topics = topic_df.sort_values('total_docs', ascending=False).head(20)
                
                heatmap_data = []
                for _, row in top_topics.iterrows():
                    heatmap_data.append({
                        'Topic': f"Topic {row['topic_id']}",
                        'Fake': row['fake_prevalence'],
                        'Real': row['real_prevalence']
                    })
                
                heatmap_df = pd.DataFrame(heatmap_data)
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=[heatmap_df['Fake'].values, heatmap_df['Real'].values],
                    x=heatmap_df['Topic'].values,
                    y=['Fake Content', 'Real Content'],
                    colorscale='RdYlGn',
                    text=[[f"{v:.1f}%" for v in heatmap_df['Fake'].values],
                          [f"{v:.1f}%" for v in heatmap_df['Real'].values]],
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="Prevalence %")
                ))
                
                fig_heatmap.update_layout(
                    title="Topic Prevalence by Content Authenticity (Top 20 Topics)",
                    xaxis_title="Topic",
                    yaxis_title="Content Type",
                    height=300
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # 3. Topic Word Importance Charts
            st.markdown("#### 💬 Top Topic Keywords")
            
            # Show top 5 topics with their keywords
            top_5_topics = topic_df.sort_values('total_docs', ascending=False).head(5)
            
            for idx, row in top_5_topics.iterrows():
                with st.expander(f"Topic {row['topic_id']} - {row.get('total_docs', 0)} documents"):
                    if 'top_words' in row and isinstance(row['top_words'], list):
                        words = row['top_words'][:10]
                        
                        # Create horizontal bar chart for word importance
                        fig_words = go.Figure()
                        fig_words.add_trace(go.Bar(
                            y=words,
                            x=list(range(len(words), 0, -1)),  # Decreasing importance
                            orientation='h',
                            marker_color=CHART_CONFIG['colors']['neutral']
                        ))
                        
                        fig_words.update_layout(
                            title=f"Top Keywords for Topic {row['topic_id']}",
                            xaxis_title="Relative Importance",
                            yaxis_title="Keyword",
                            height=300,
                            showlegend=False
                        )
                        st.plotly_chart(fig_words, use_container_width=True)
                        
                        # Show prevalence stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Documents", f"{row.get('total_docs', 0):,}")
                        with col2:
                            st.metric("Fake Prevalence", f"{row.get('fake_prevalence', 0):.1f}%")
                        with col3:
                            st.metric("Real Prevalence", f"{row.get('real_prevalence', 0):.1f}%")
            
            # 4. Topic Distribution Scatter (Fake vs Real)
            st.markdown("#### 🎯 Topic Distribution: Fake vs Real")
            
            if 'fake_prevalence' in topic_df.columns and 'real_prevalence' in topic_df.columns:
                fig_scatter = px.scatter(
                    topic_df,
                    x='fake_prevalence',
                    y='real_prevalence',
                    size='total_docs',
                    hover_name='topic_id',
                    hover_data={'top_words': True, 'total_docs': True},
                    title="Topic Distribution Across Content Types",
                    labels={
                        'fake_prevalence': 'Prevalence in Fake Content (%)',
                        'real_prevalence': 'Prevalence in Real Content (%)'
                    }
                )
                
                # Add diagonal line
                fig_scatter.add_shape(
                    type="line",
                    x0=0, y0=0, x1=100, y1=100,
                    line=dict(color="red", width=2, dash="dash")
                )
                
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                st.info("📌 Topics above the diagonal line are more prevalent in real content, while topics below are more common in fake content.")
        else:
            st.info("📊 No topic modeling data available.")
    else:
        st.info("📊 Topic modeling analysis not available in CORE version.")
        
        st.markdown("**Note:** The CORE version of linguistic pattern mining focuses on essential features for performance. Topic modeling requires the full version which includes LDA analysis.")
        
        st.markdown("**Alternative Analysis Available:**")
        st.markdown("- View discriminative features in the **Authenticity Heatmap** tab")
        st.markdown("- Explore sentiment patterns in the **Sentiment Analysis** tab")
        st.markdown("- Check feature distributions in the **Feature Distributions** tab")
        
        # Show top discriminative features as an alternative
        if linguistic_data and 'top_discriminative_features' in linguistic_data:
            st.markdown("#### 🎯 Top Discriminative Linguistic Features")
            st.markdown("These features show the strongest differences between fake and real content:")
            
            features = linguistic_data['top_discriminative_features']
            
            # Create visualization
            feature_names = [f['feature_name'].replace('_', ' ').title() for f in features]
            effect_sizes = [abs(f['effect_size']) for f in features]
            directions = ['Fake' if f['effect_size'] > 0 else 'Real' for f in features]
            colors = [CHART_CONFIG['colors']['fake'] if d == 'Fake' else CHART_CONFIG['colors']['real'] for d in directions]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=feature_names,
                x=effect_sizes,
                orientation='h',
                marker_color=colors,
                text=[f"{es:.3f}" for es in effect_sizes],
                textposition='outside',
                hovertext=[f"Higher in {d} content" for d in directions],
                hoverinfo='text+x'
            ))
            
            fig.update_layout(
                title="Linguistic Features by Effect Size",
                xaxis_title="Effect Size (Cohen's d)",
                yaxis_title="Feature",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show feature details
            with st.expander("📋 Feature Details"):
                for feature in features:
                    direction_text = "higher" if feature['effect_size'] > 0 else "lower"
                    content_type = "fake" if feature['effect_size'] > 0 else "real"
                    
                    st.markdown(f"""
                    **{feature['feature_name'].replace('_', ' ').title()}**
                    - Effect Size: {feature['effect_size']:.4f}
                    - Direction: {direction_text} in {content_type} content
                    - Fake Mean: {feature['fake_mean']:.4f}
                    - Real Mean: {feature['real_mean']:.4f}
                    - P-value: {feature['p_value']:.2e}
                    """)
                    st.markdown("---")


def render_authenticity_heatmap_tab(linguistic_data):
    """Tab 4: Authenticity Heatmap - Feature-authenticity correlation with statistical significance and clustering"""
    st.markdown("### 🎯 Feature-Authenticity Correlation Analysis")
    
    # Load authenticity patterns data
    try:
        patterns_path = Path("analysis_results/linguistic_analysis/authenticity_patterns.json")
        if patterns_path.exists():
            with open(patterns_path, 'r') as f:
                patterns_data = json.load(f)
            
            # 1. Feature-Authenticity Correlation Heatmap with Statistical Significance
            st.markdown("#### 🔥 Feature Correlation Heatmap")
            
            if 'overall_comparisons' in patterns_data:
                comparisons = patterns_data['overall_comparisons']
                
                # Prepare data for heatmap
                features = []
                effect_sizes = []
                p_values = []
                fake_means = []
                real_means = []
                
                for feature, stats in comparisons.items():
                    features.append(feature.replace('_', ' ').title())
                    effect_sizes.append(stats.get('effect_size', 0))
                    p_values.append(stats.get('p_value', 1))
                    fake_means.append(stats.get('fake_mean', 0))
                    real_means.append(stats.get('real_mean', 0))
                
                # Create DataFrame
                corr_df = pd.DataFrame({
                    'Feature': features,
                    'Effect Size': effect_sizes,
                    'P-Value': p_values,
                    'Fake Mean': fake_means,
                    'Real Mean': real_means
                })
                
                # Sort by absolute effect size
                corr_df['Abs Effect Size'] = corr_df['Effect Size'].abs()
                corr_df = corr_df.sort_values('Abs Effect Size', ascending=False)
                
                # Create heatmap with effect sizes
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=[corr_df['Effect Size'].values],
                    x=corr_df['Feature'].values,
                    y=['Effect Size'],
                    colorscale='RdBu',
                    zmid=0,
                    text=[[f"{v:.3f}{'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}" 
                           for v, p in zip(corr_df['Effect Size'].values, corr_df['P-Value'].values)]],
                    texttemplate='%{text}',
                    textfont={"size": 9},
                    colorbar=dict(title="Effect Size")
                ))
                
                fig_heatmap.update_layout(
                    title="Feature-Authenticity Effect Sizes with Significance Markers<br><sub>*** p<0.001, ** p<0.01, * p<0.05</sub>",
                    xaxis_title="Linguistic Feature",
                    height=300,
                    xaxis={'tickangle': -45}
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                st.info("🔴 Negative effect sizes (red) indicate features higher in fake content. 🔵 Positive effect sizes (blue) indicate features higher in real content.")
                
                # 2. Top Discriminative Features Bar Chart
                st.markdown("#### 📊 Top Discriminative Features")
                
                top_features = corr_df.head(15)
                
                fig_bar = go.Figure()
                
                colors = [CHART_CONFIG['colors']['fake'] if x < 0 else CHART_CONFIG['colors']['real'] 
                         for x in top_features['Effect Size']]
                
                fig_bar.add_trace(go.Bar(
                    y=top_features['Feature'],
                    x=top_features['Effect Size'],
                    orientation='h',
                    marker_color=colors,
                    text=top_features['Effect Size'].round(3),
                    textposition='outside'
                ))
                
                fig_bar.update_layout(
                    title="Top 15 Discriminative Features (by Effect Size)",
                    xaxis_title="Effect Size (Cohen's d)",
                    yaxis_title="Feature",
                    height=600
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # 3. Hierarchical Clustering Dendrogram
                st.markdown("#### 🌳 Feature Clustering Dendrogram")
                
                # Prepare data for clustering
                feature_matrix = np.column_stack([
                    corr_df['Effect Size'].values,
                    corr_df['Fake Mean'].values,
                    corr_df['Real Mean'].values
                ])
                
                # Perform hierarchical clustering
                linkage_matrix = linkage(feature_matrix, method='ward')
                
                # Create dendrogram
                fig_dendro = go.Figure()
                
                # Calculate dendrogram
                from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
                dendro_data = scipy_dendrogram(linkage_matrix, labels=corr_df['Feature'].values, no_plot=True)
                
                # Plot dendrogram
                icoord = np.array(dendro_data['icoord'])
                dcoord = np.array(dendro_data['dcoord'])
                
                for i in range(len(icoord)):
                    fig_dendro.add_trace(go.Scatter(
                        x=icoord[i],
                        y=dcoord[i],
                        mode='lines',
                        line=dict(color=CHART_CONFIG['colors']['neutral'], width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                fig_dendro.update_layout(
                    title="Hierarchical Clustering of Linguistic Features",
                    xaxis_title="Feature Index",
                    yaxis_title="Distance",
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig_dendro, use_container_width=True)
                
                st.info("📌 The dendrogram shows how linguistic features cluster together based on their statistical properties. Features that cluster together have similar patterns across fake and real content.")
                
                # 4. Statistical Summary Table
                st.markdown("#### 📋 Detailed Statistical Summary")
                
                summary_df = corr_df[['Feature', 'Effect Size', 'P-Value', 'Fake Mean', 'Real Mean']].head(10)
                summary_df['Significance'] = summary_df['P-Value'].apply(
                    lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                )
                summary_df['Effect Size'] = summary_df['Effect Size'].round(4)
                summary_df['Fake Mean'] = summary_df['Fake Mean'].round(4)
                summary_df['Real Mean'] = summary_df['Real Mean'].round(4)
                
                st.dataframe(summary_df, use_container_width=True)
                
        else:
            st.info("📊 Authenticity patterns data not available.")
            
    except Exception as e:
        st.error(f"Error loading authenticity patterns: {e}")


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
            
            # Load data
            features_df = load_linguistic_features_data()
            
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
                    render_feature_distributions_tab(linguistic_data)
                
                with viz_tabs[1]:
                    render_sentiment_analysis_tab(linguistic_data)
                
                with viz_tabs[2]:
                    render_topic_models_tab(linguistic_data)
                
                with viz_tabs[3]:
                    render_authenticity_heatmap_tab(linguistic_data)
                

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
