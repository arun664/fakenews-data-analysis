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
    """Load detailed linguistic features for visualizations"""
    try:
        features_path = Path("processed_data/linguistic_features/linguistic_features.parquet")
        if features_path.exists():
            # Load with sampling for performance
            df = pd.read_parquet(features_path)
            if len(df) > 50000:
                # Stratified sampling
                df = df.groupby('authenticity_label').apply(
                    lambda x: x.sample(n=min(len(x), 25000), random_state=42)
                ).reset_index(drop=True)
            return df
        return None
    except Exception as e:
        st.error(f"Error loading linguistic features: {e}")
        return None


def render_feature_distributions_tab(linguistic_data):
    """Tab 1: Feature Distributions - Readability, vocabulary, text length, sentence complexity"""
    st.markdown("### 📏 Linguistic Feature Distributions")
    
    # Load detailed features
    features_df = load_linguistic_features_data()
    
    if features_df is not None and 'authenticity_label' in features_df.columns:
        # Map labels to readable names
        features_df['Authenticity'] = features_df['authenticity_label'].map({0: 'Fake', 1: 'Real'})
        
        # 1. Readability Score Violin Plots
        st.markdown("#### 📖 Readability Scores")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'flesch_reading_ease' in features_df.columns:
                fig_fre = go.Figure()
                for label, color in [('Fake', CHART_CONFIG['colors']['fake']), ('Real', CHART_CONFIG['colors']['real'])]:
                    data = features_df[features_df['Authenticity'] == label]['flesch_reading_ease']
                    fig_fre.add_trace(go.Violin(
                        y=data,
                        name=label,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=color,
                        opacity=0.6,
                        line_color=color
                    ))
                
                fig_fre.update_layout(
                    title="Flesch Reading Ease Score Distribution",
                    yaxis_title="Reading Ease Score",
                    showlegend=True,
                    height=400
                )
                st.plotly_chart(fig_fre, use_container_width=True)
        
        with col2:
            if 'flesch_kincaid_grade' in features_df.columns:
                fig_fkg = go.Figure()
                for label, color in [('Fake', CHART_CONFIG['colors']['fake']), ('Real', CHART_CONFIG['colors']['real'])]:
                    data = features_df[features_df['Authenticity'] == label]['flesch_kincaid_grade']
                    fig_fkg.add_trace(go.Violin(
                        y=data,
                        name=label,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=color,
                        opacity=0.6,
                        line_color=color
                    ))
                
                fig_fkg.update_layout(
                    title="Flesch-Kincaid Grade Level Distribution",
                    yaxis_title="Grade Level",
                    showlegend=True,
                    height=400
                )
                st.plotly_chart(fig_fkg, use_container_width=True)
        
        # 2. Vocabulary Diversity Bar Charts
        st.markdown("#### 📚 Vocabulary Diversity")
        if 'unique_word_ratio' in features_df.columns:
            vocab_stats = features_df.groupby('Authenticity')['unique_word_ratio'].agg(['mean', 'std']).reset_index()
            
            fig_vocab = go.Figure()
            fig_vocab.add_trace(go.Bar(
                x=vocab_stats['Authenticity'],
                y=vocab_stats['mean'],
                error_y=dict(type='data', array=vocab_stats['std']),
                marker_color=[CHART_CONFIG['colors']['fake'], CHART_CONFIG['colors']['real']],
                text=vocab_stats['mean'].round(4),
                textposition='outside'
            ))
            
            fig_vocab.update_layout(
                title="Unique Word Ratio by Authenticity",
                xaxis_title="Content Type",
                yaxis_title="Unique Word Ratio (Mean ± SD)",
                height=400
            )
            st.plotly_chart(fig_vocab, use_container_width=True)
        
        # 3. Text Length Histograms
        st.markdown("#### 📝 Text Length Distributions")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'text_length' in features_df.columns:
                fig_len = go.Figure()
                for label, color in [('Fake', CHART_CONFIG['colors']['fake']), ('Real', CHART_CONFIG['colors']['real'])]:
                    data = features_df[features_df['Authenticity'] == label]['text_length']
                    fig_len.add_trace(go.Histogram(
                        x=data,
                        name=label,
                        opacity=0.6,
                        marker_color=color,
                        nbinsx=50
                    ))
                
                fig_len.update_layout(
                    title="Text Length Distribution",
                    xaxis_title="Text Length (characters)",
                    yaxis_title="Frequency",
                    barmode='overlay',
                    height=400
                )
                st.plotly_chart(fig_len, use_container_width=True)
        
        with col2:
            if 'word_count' in features_df.columns:
                fig_words = go.Figure()
                for label, color in [('Fake', CHART_CONFIG['colors']['fake']), ('Real', CHART_CONFIG['colors']['real'])]:
                    data = features_df[features_df['Authenticity'] == label]['word_count']
                    fig_words.add_trace(go.Histogram(
                        x=data,
                        name=label,
                        opacity=0.6,
                        marker_color=color,
                        nbinsx=50
                    ))
                
                fig_words.update_layout(
                    title="Word Count Distribution",
                    xaxis_title="Word Count",
                    yaxis_title="Frequency",
                    barmode='overlay',
                    height=400
                )
                st.plotly_chart(fig_words, use_container_width=True)
        
        # 4. Sentence Complexity Box Plots
        st.markdown("#### 📊 Sentence Complexity")
        if 'avg_sentence_length' in features_df.columns:
            fig_sent = go.Figure()
            for label, color in [('Fake', CHART_CONFIG['colors']['fake']), ('Real', CHART_CONFIG['colors']['real'])]:
                data = features_df[features_df['Authenticity'] == label]['avg_sentence_length']
                fig_sent.add_trace(go.Box(
                    y=data,
                    name=label,
                    marker_color=color,
                    boxmean='sd'
                ))
            
            fig_sent.update_layout(
                title="Average Sentence Length by Authenticity",
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
    st.markdown("### 😊 Sentiment & Emotion Analysis")
    
    features_df = load_linguistic_features_data()
    
    if features_df is not None and 'authenticity_label' in features_df.columns:
        features_df['Authenticity'] = features_df['authenticity_label'].map({0: 'Fake', 1: 'Real'})
        
        # 1. Word Clouds for Fake vs Real
        st.markdown("#### ☁️ Word Clouds by Authenticity")
        st.info("💡 Word clouds require text data. This visualization shows sentiment patterns instead.")
        
        # 2. Sentiment Distribution Pie Charts
        st.markdown("#### 🎭 Sentiment Distribution")
        
        if all(col in features_df.columns for col in ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']):
            col1, col2 = st.columns(2)
            
            with col1:
                fake_df = features_df[features_df['Authenticity'] == 'Fake']
                sentiment_fake = {
                    'Positive': fake_df['sentiment_positive'].mean(),
                    'Negative': fake_df['sentiment_negative'].mean(),
                    'Neutral': fake_df['sentiment_neutral'].mean()
                }
                
                fig_pie_fake = go.Figure(data=[go.Pie(
                    labels=list(sentiment_fake.keys()),
                    values=list(sentiment_fake.values()),
                    marker=dict(colors=['#4ECDC4', '#FF6B6B', '#95A5A6']),
                    hole=0.3
                )])
                fig_pie_fake.update_layout(title="Fake Content Sentiment", height=400)
                st.plotly_chart(fig_pie_fake, use_container_width=True)
            
            with col2:
                real_df = features_df[features_df['Authenticity'] == 'Real']
                sentiment_real = {
                    'Positive': real_df['sentiment_positive'].mean(),
                    'Negative': real_df['sentiment_negative'].mean(),
                    'Neutral': real_df['sentiment_neutral'].mean()
                }
                
                fig_pie_real = go.Figure(data=[go.Pie(
                    labels=list(sentiment_real.keys()),
                    values=list(sentiment_real.values()),
                    marker=dict(colors=['#4ECDC4', '#FF6B6B', '#95A5A6']),
                    hole=0.3
                )])
                fig_pie_real.update_layout(title="Real Content Sentiment", height=400)
                st.plotly_chart(fig_pie_real, use_container_width=True)
        
        # 3. Emotion Intensity Radar Charts
        st.markdown("#### 🎯 Emotion Intensity Comparison")
        
        if all(col in features_df.columns for col in ['sentiment_positive', 'sentiment_negative', 'polarity', 'subjectivity']):
            fake_df = features_df[features_df['Authenticity'] == 'Fake']
            real_df = features_df[features_df['Authenticity'] == 'Real']
            
            categories = ['Positive', 'Negative', 'Polarity', 'Subjectivity', 'Exclamation']
            
            fake_values = [
                fake_df['sentiment_positive'].mean(),
                fake_df['sentiment_negative'].mean(),
                fake_df['polarity'].mean() if 'polarity' in fake_df.columns else 0,
                fake_df['subjectivity'].mean() if 'subjectivity' in fake_df.columns else 0,
                fake_df['exclamation_count'].mean() if 'exclamation_count' in fake_df.columns else 0
            ]
            
            real_values = [
                real_df['sentiment_positive'].mean(),
                real_df['sentiment_negative'].mean(),
                real_df['polarity'].mean() if 'polarity' in real_df.columns else 0,
                real_df['subjectivity'].mean() if 'subjectivity' in real_df.columns else 0,
                real_df['exclamation_count'].mean() if 'exclamation_count' in real_df.columns else 0
            ]
            
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
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(max(fake_values), max(real_values)) * 1.1])),
                showlegend=True,
                title="Emotion Intensity Radar Chart",
                height=500
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # 4. Subjectivity Scatter Plot
        st.markdown("#### 📈 Polarity vs Subjectivity")
        
        if 'polarity' in features_df.columns and 'subjectivity' in features_df.columns:
            # Sample for performance
            sample_df = features_df.sample(n=min(5000, len(features_df)), random_state=42)
            
            fig_scatter = px.scatter(
                sample_df,
                x='subjectivity',
                y='polarity',
                color='Authenticity',
                color_discrete_map={'Fake': CHART_CONFIG['colors']['fake'], 'Real': CHART_CONFIG['colors']['real']},
                opacity=0.5,
                title="Sentiment Polarity vs Subjectivity",
                labels={'subjectivity': 'Subjectivity', 'polarity': 'Polarity'}
            )
            
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
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
