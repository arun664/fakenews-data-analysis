"""
Cross-Modal Insights Page
Analysis of multimodal relationships and authenticity consistency
COMPLETE IMPLEMENTATION - Extracted from app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.lazy_loader import LazyLoader
from src.utils.visualization_helpers import (
    create_heatmap,
    create_comparison_bar_chart,
    CHART_CONFIG
)

lazy_loader = LazyLoader()


def create_sankey_diagram(cross_modal_data):
    """Create Sankey diagram showing content flow across modalities."""
    try:
        cross_modal_auth = cross_modal_data.get("cross_modal_authenticity", {})
        
        # Define nodes
        labels = [
            "Text+Image",      # 0
            "Full Multimodal", # 1
            "Text Only",       # 2
            "Fake",            # 3
            "Real",            # 4
            "Low Engagement",  # 5
            "High Engagement"  # 6
        ]
        
        # Build flows
        sources = []
        targets = []
        values = []
        colors = []
        
        # Content type to authenticity flows
        for idx, (content_type, data) in enumerate(cross_modal_auth.items()):
            if content_type == "text_image":
                source_idx = 0
            elif content_type == "full_multimodal":
                source_idx = 1
            else:  # text_only
                source_idx = 2
            
            fake_posts = data.get("fake_posts", 0)
            real_posts = data.get("real_posts", 0)
            
            if fake_posts > 0:
                sources.append(source_idx)
                targets.append(3)  # Fake
                values.append(fake_posts)
                colors.append('rgba(255, 107, 107, 0.4)')
            
            if real_posts > 0:
                sources.append(source_idx)
                targets.append(4)  # Real
                values.append(real_posts)
                colors.append('rgba(78, 205, 196, 0.4)')
        
        # Authenticity to engagement flows
        for content_type, data in cross_modal_auth.items():
            avg_eng_fake = data.get("avg_engagement_fake", 0)
            avg_eng_real = data.get("avg_engagement_real", 0)
            fake_posts = data.get("fake_posts", 0)
            real_posts = data.get("real_posts", 0)
            
            # Fake to engagement
            if fake_posts > 0 and not np.isnan(avg_eng_fake):
                if avg_eng_fake < 100:
                    sources.append(3)
                    targets.append(5)
                    values.append(fake_posts * 0.5)
                    colors.append('rgba(255, 107, 107, 0.3)')
                else:
                    sources.append(3)
                    targets.append(6)
                    values.append(fake_posts * 0.5)
                    colors.append('rgba(255, 107, 107, 0.3)')
            
            # Real to engagement
            if real_posts > 0 and not np.isnan(avg_eng_real):
                if avg_eng_real < 100:
                    sources.append(4)
                    targets.append(5)
                    values.append(real_posts * 0.5)
                    colors.append('rgba(78, 205, 196, 0.3)')
                else:
                    sources.append(4)
                    targets.append(6)
                    values.append(real_posts * 0.5)
                    colors.append('rgba(78, 205, 196, 0.3)')
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=['#3498DB', '#9B59B6', '#95A5A6', '#FF6B6B', '#4ECDC4', '#E8E8E8', '#2ECC71']
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=colors
            )
        )])
        
        fig.update_layout(
            title="Content Flow: Modality → Authenticity → Engagement",
            font=dict(size=12, family='Source Sans Pro'),
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating Sankey diagram: {e}")
        return None


def create_modality_correlation_heatmap(cross_modal_data):
    """Create correlation matrix heatmap for modality features."""
    try:
        cross_modal_auth = cross_modal_data.get("cross_modal_authenticity", {})
        
        # Build feature matrix
        features_data = []
        for content_type, data in cross_modal_auth.items():
            if content_type == "text_image":
                display_name = "Text+Image"
            elif content_type == "full_multimodal":
                display_name = "Full Multimodal"
            else:
                display_name = "Text Only"
            
            fake_ratio = data.get("fake_posts", 0) / max(data.get("total_posts", 1), 1)
            avg_eng_fake = data.get("avg_engagement_fake", 0)
            avg_eng_real = data.get("avg_engagement_real", 0)
            avg_comments_fake = data.get("avg_comments_fake", 0)
            avg_comments_real = data.get("avg_comments_real", 0)
            
            # Handle NaN values
            if np.isnan(avg_eng_fake):
                avg_eng_fake = 0
            if np.isnan(avg_eng_real):
                avg_eng_real = 0
            if np.isnan(avg_comments_fake):
                avg_comments_fake = 0
            if np.isnan(avg_comments_real):
                avg_comments_real = 0
            
            features_data.append({
                'Content Type': display_name,
                'Fake Ratio': fake_ratio,
                'Avg Engagement (Fake)': avg_eng_fake,
                'Avg Engagement (Real)': avg_eng_real,
                'Avg Comments (Fake)': avg_comments_fake,
                'Avg Comments (Real)': avg_comments_real
            })
        
        df = pd.DataFrame(features_data)
        df = df.set_index('Content Type')
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title='Correlation')
        ))
        
        fig.update_layout(
            title="Modality Feature Correlation Matrix",
            font=dict(size=11, family='Source Sans Pro'),
            height=500,
            xaxis=dict(tickangle=-45)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {e}")
        return None


def create_engagement_by_content_type(cross_modal_data):
    """Create grouped bar chart for engagement by content type."""
    try:
        cross_modal_auth = cross_modal_data.get("cross_modal_authenticity", {})
        
        # Prepare data
        content_types = []
        fake_engagement = []
        real_engagement = []
        
        for content_type, data in cross_modal_auth.items():
            if content_type == "text_image":
                display_name = "Text+Image"
            elif content_type == "full_multimodal":
                display_name = "Full Multimodal"
            else:
                display_name = "Text Only"
            
            content_types.append(display_name)
            
            avg_eng_fake = data.get("avg_engagement_fake", 0)
            avg_eng_real = data.get("avg_engagement_real", 0)
            
            # Handle NaN values
            fake_engagement.append(0 if np.isnan(avg_eng_fake) else avg_eng_fake)
            real_engagement.append(0 if np.isnan(avg_eng_real) else avg_eng_real)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Fake Content',
            x=content_types,
            y=fake_engagement,
            marker_color=CHART_CONFIG['colors']['fake'],
            text=[f"{val:.1f}" for val in fake_engagement],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Real Content',
            x=content_types,
            y=real_engagement,
            marker_color=CHART_CONFIG['colors']['real'],
            text=[f"{val:.1f}" for val in real_engagement],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Average Engagement Score by Content Type and Authenticity",
            xaxis_title="Content Type",
            yaxis_title="Average Engagement Score",
            barmode='group',
            font=dict(size=12, family='Source Sans Pro'),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating engagement chart: {e}")
        return None


def create_authenticity_consistency_heatmap(cross_modal_data):
    """Create heatmap showing authenticity consistency across modalities."""
    try:
        multimodal_consistency = cross_modal_data.get("multimodal_consistency", {})
        
        # Prepare data for heatmap
        content_types = []
        fake_ratios = []
        real_ratios = []
        total_posts = []
        
        for content_type, metrics in multimodal_consistency.items():
            if content_type == "text_image":
                display_name = "Text+Image"
            elif content_type == "full_multimodal":
                display_name = "Full Multimodal"
            else:
                display_name = "Text Only"
            
            content_types.append(display_name)
            fake_ratios.append(metrics.get('fake_ratio', 0))
            real_ratios.append(metrics.get('real_ratio', 0))
            total_posts.append(metrics.get('total_posts', 0))
        
        # Create DataFrame for heatmap
        heatmap_data = pd.DataFrame({
            'Fake Ratio': fake_ratios,
            'Real Ratio': real_ratios,
            'Post Volume (normalized)': [p / max(total_posts) for p in total_posts]
        }, index=content_types)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values.T,
            x=heatmap_data.index,
            y=heatmap_data.columns,
            colorscale='RdYlGn',
            text=heatmap_data.values.T,
            texttemplate='%{text:.2f}',
            textfont={"size": 12},
            colorbar=dict(title='Value')
        ))
        
        fig.update_layout(
            title="Authenticity Consistency Across Content Modalities",
            font=dict(size=12, family='Source Sans Pro'),
            height=400,
            xaxis=dict(side='bottom'),
            yaxis=dict(side='left')
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating authenticity consistency heatmap: {e}")
        return None


def render_cross_modal_insights(container):
    """Render Cross-Modal Insights with lazy loading - COMPLETE IMPLEMENTATION"""
    with container.container():
        try:
            st.header("Multimodal Relationships & Authenticity Consistency")
            
            # Load dashboard data with performance optimization
            @st.cache_data(ttl=600)  # 10 minutes cache for cross-modal analysis (static results)
            def load_dashboard_data():
                dashboard_data_path = Path("analysis_results/dashboard_data/processed_dashboard_data.json")
                if not dashboard_data_path.exists():
                    raise FileNotFoundError(f"Dashboard data not found at {dashboard_data_path}")
                with open(dashboard_data_path, 'r') as f:
                    return json.load(f)
            
            try:
                dashboard_data = load_dashboard_data()
            except FileNotFoundError:
                dashboard_data = None
            
            # Hide loading indicator
            lazy_loader.hide_section_loading()
            
            if dashboard_data and "cross_modal_analysis" in dashboard_data:
                cross_modal_data = dashboard_data["cross_modal_analysis"]
                
                # Mapping relationships overview
                st.subheader("🔍 ID Mapping Relationships")
                
                mapping_relationships = cross_modal_data.get("mapping_relationships", {})
                if mapping_relationships:
                    mapping_success = mapping_relationships.get("mapping_success", {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_images = mapping_success.get("total_images", 0)
                        st.metric("🖼️ Total Images", f"{total_images:,}")
                    
                    with col2:
                        multimodal_images = mapping_success.get("multimodal_images", 0)
                        st.metric("🔗 Multimodal Images", f"{multimodal_images:,}")
                    
                    with col3:
                        image_only = mapping_success.get("image_only", 0)
                        st.metric("📷 Image-Only", f"{image_only:,}")
                    
                    with col4:
                        mapping_rate = mapping_success.get("mapping_rate", 0)
                        st.metric("📊 Mapping Rate", f"{mapping_rate:.1f}%")
                
                st.markdown("---")
                
                # NEW VISUALIZATIONS - Task 1.7
                st.subheader("📊 Enhanced Cross-Modal Visualizations")
                
                # 1. Sankey Diagram - Content Flow Across Modalities
                st.markdown("#### 1️⃣ Content Flow Across Modalities")
                sankey_fig = create_sankey_diagram(cross_modal_data)
                if sankey_fig:
                    st.plotly_chart(sankey_fig, use_container_width=True)
                    st.caption("This Sankey diagram shows how content flows from source types through authenticity labels to engagement levels.")
                
                st.markdown("---")
                
                # 2. Modality Correlation Matrix Heatmap
                st.markdown("#### 2️⃣ Modality Feature Correlation Matrix")
                correlation_fig = create_modality_correlation_heatmap(cross_modal_data)
                if correlation_fig:
                    st.plotly_chart(correlation_fig, use_container_width=True)
                    st.caption("Correlation matrix showing relationships between different modality features and authenticity.")
                
                st.markdown("---")
                
                # 3. Engagement by Content Type Grouped Bar Chart
                st.markdown("#### 3️⃣ Engagement Patterns by Content Type")
                engagement_fig = create_engagement_by_content_type(cross_modal_data)
                if engagement_fig:
                    st.plotly_chart(engagement_fig, use_container_width=True)
                    st.caption("Comparison of engagement metrics across different content types and authenticity labels.")
                
                st.markdown("---")
                
                # 4. Authenticity Consistency Heatmap
                st.markdown("#### 4️⃣ Authenticity Consistency Across Modalities")
                consistency_fig = create_authenticity_consistency_heatmap(cross_modal_data)
                if consistency_fig:
                    st.plotly_chart(consistency_fig, use_container_width=True)
                    st.caption("Heatmap showing the consistency of authenticity patterns across different content modalities.")
                
                st.markdown("---")
                
                # Content type distribution and authenticity consistency
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Content Type Distribution")
                    content_dist = cross_modal_data.get("content_type_distribution", {})
                    if content_dist:
                        labels = []
                        values = []
                        for content_type, count in content_dist.items():
                            if content_type == "multimodal":
                                labels.append("Multimodal")
                            else:
                                labels.append("Image-Only")
                            values.append(count)
                        
                        fig = px.pie(
                            values=values,
                            names=labels,
                            title="Image Content Distribution",
                            color_discrete_sequence=['#2E8B57', '#FF6347']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("🎭 Cross-Modal Authenticity")
                    cross_modal_auth = cross_modal_data.get("cross_modal_authenticity", {})
                    if cross_modal_auth:
                        content_types = []
                        fake_ratios = []
                        
                        for content_type, data in cross_modal_auth.items():
                            if data.get("total_posts", 0) > 0:
                                if content_type == "text_image":
                                    display_name = "Text + Image"
                                elif content_type == "full_multimodal":
                                    display_name = "Full Multimodal"
                                else:
                                    display_name = "Text Only"
                                
                                content_types.append(display_name)
                                fake_ratio = (data.get("fake_posts", 0) / data["total_posts"]) * 100
                                fake_ratios.append(fake_ratio)
                        
                        if content_types:
                            fig = px.bar(
                                x=content_types,
                                y=fake_ratios,
                                title="Fake Content Percentage by Type",
                                color=fake_ratios,
                                color_continuous_scale="Reds"
                            )
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                
                # Multimodal consistency analysis
                st.subheader("🔄 Multimodal Consistency Analysis")
                
                consistency_metrics = cross_modal_data.get("multimodal_consistency", {})
                if consistency_metrics:
                    consistency_df = []
                    for content_type, metrics in consistency_metrics.items():
                        if content_type == "text_image":
                            display_name = "Text + Image"
                        elif content_type == "full_multimodal":
                            display_name = "Full Multimodal"
                        else:
                            display_name = "Text Only"
                        
                        consistency_df.append({
                            "Content Type": display_name,
                            "Total Posts": f"{metrics.get('total_posts', 0):,}",
                            "Fake Ratio": f"{metrics.get('fake_ratio', 0):.1%}",
                            "Real Ratio": f"{metrics.get('real_ratio', 0):.1%}",
                            "Avg Engagement (Fake)": f"{metrics.get('avg_engagement_fake', 0):.1f}",
                            "Avg Engagement (Real)": f"{metrics.get('avg_engagement_real', 0):.1f}"
                        })
                    
                    if consistency_df:
                        df = pd.DataFrame(consistency_df)
                        st.dataframe(df, use_container_width=True)
                
                # Enhanced cross-modal insights
                st.subheader("💡 Cross-Modal Analysis: Fake vs Real Patterns")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🔴 Fake Content Cross-Modal Profile")
                    fake_profile = []
                    
                    if cross_modal_auth:
                        # Find modality with highest fake content
                        highest_fake_modality = None
                        highest_fake_pct = 0
                        
                        for content_type, data in cross_modal_auth.items():
                            if data.get("total_posts", 0) > 0:
                                fake_pct = (data.get("fake_posts", 0) / data["total_posts"]) * 100
                                if fake_pct > highest_fake_pct:
                                    highest_fake_pct = fake_pct
                                    if content_type == "text_image":
                                        highest_fake_modality = "Text + Image"
                                    elif content_type == "full_multimodal":
                                        highest_fake_modality = "Full Multimodal"
                                    else:
                                        highest_fake_modality = "Text Only"
                        
                        if highest_fake_modality:
                            fake_profile.append(f"• **Highest Risk Modality:** {highest_fake_modality} ({highest_fake_pct:.1f}% fake)")
                        
                        # Multimodal integration pattern
                        if mapping_relationships:
                            mapping_success = mapping_relationships.get("mapping_success", {})
                            multimodal_pct = (mapping_success.get("multimodal_images", 0) / mapping_success.get("total_images", 1)) * 100
                            fake_profile.append(f"• **Multimodal Integration:** {multimodal_pct:.1f}% of images have text")
                        
                        # Engagement patterns
                        if consistency_metrics:
                            for content_type, metrics in consistency_metrics.items():
                                if content_type == "text_image":
                                    fake_eng = metrics.get('avg_engagement_fake', 0)
                                    fake_profile.append(f"• **Text+Image Engagement:** {fake_eng:.1f} avg score")
                    
                    for item in fake_profile:
                        st.markdown(item)
                
                with col2:
                    st.markdown("### 🟢 Real Content Cross-Modal Profile")
                    real_profile = []
                    
                    if cross_modal_auth:
                        # Find modality with lowest fake content (highest real)
                        lowest_fake_modality = None
                        lowest_fake_pct = 100
                        
                        for content_type, data in cross_modal_auth.items():
                            if data.get("total_posts", 0) > 0:
                                fake_pct = (data.get("fake_posts", 0) / data["total_posts"]) * 100
                                if fake_pct < lowest_fake_pct:
                                    lowest_fake_pct = fake_pct
                                    if content_type == "text_image":
                                        lowest_fake_modality = "Text + Image"
                                    elif content_type == "full_multimodal":
                                        lowest_fake_modality = "Full Multimodal"
                                    else:
                                        lowest_fake_modality = "Text Only"
                        
                        if lowest_fake_modality:
                            real_profile.append(f"• **Safest Modality:** {lowest_fake_modality} ({100-lowest_fake_pct:.1f}% real)")
                        
                        # Multimodal integration pattern
                        if mapping_relationships:
                            mapping_success = mapping_relationships.get("mapping_success", {})
                            multimodal_pct = (mapping_success.get("multimodal_images", 0) / mapping_success.get("total_images", 1)) * 100
                            real_profile.append(f"• **Content Richness:** {multimodal_pct:.1f}% multimodal posts")
                        
                        # Engagement patterns
                        if consistency_metrics:
                            for content_type, metrics in consistency_metrics.items():
                                if content_type == "text_image":
                                    real_eng = metrics.get('avg_engagement_real', 0)
                                    real_profile.append(f"• **Text+Image Engagement:** {real_eng:.1f} avg score")
                    
                    for item in real_profile:
                        st.markdown(item)
                
                # Comprehensive summary
                st.markdown("---")
                st.subheader("💡 Cross-Modal Integration Summary")
                
                if mapping_relationships and cross_modal_auth:
                    mapping_success = mapping_relationships.get("mapping_success", {})
                    multimodal_pct = (mapping_success.get("multimodal_images", 0) / mapping_success.get("total_images", 1)) * 100
                    
                    # Calculate authenticity consistency
                    text_image_data = cross_modal_auth.get("text_image", {})
                    full_multimodal_data = cross_modal_auth.get("full_multimodal", {})
                    
                    if text_image_data and full_multimodal_data:
                        ti_fake_ratio = text_image_data.get("fake_posts", 0) / text_image_data.get("total_posts", 1)
                        fm_fake_ratio = full_multimodal_data.get("fake_posts", 0) / full_multimodal_data.get("total_posts", 1)
                        consistency_diff = abs(ti_fake_ratio - fm_fake_ratio)
                        
                        if consistency_diff < 0.05:
                            st.success(f"""
                            **Key Finding:** High cross-modal consistency detected ({multimodal_pct:.1f}% multimodal integration).
                            
                            **Authenticity Patterns:**
                            - Text+Image posts: {ti_fake_ratio:.1%} fake content
                            - Full Multimodal posts: {fm_fake_ratio:.1%} fake content
                            - Consistency difference: {consistency_diff:.1%}
                            
                            **Implication:** Authenticity patterns remain consistent across modalities, suggesting that 
                            multimodal features can be reliably combined for detection. The high integration rate indicates 
                            that most content naturally combines multiple modalities, making multimodal analysis essential.
                            """)
                        else:
                            st.warning(f"""
                            **Key Finding:** Moderate cross-modal consistency ({multimodal_pct:.1f}% multimodal integration).
                            
                            **Authenticity Patterns:**
                            - Text+Image posts: {ti_fake_ratio:.1%} fake content
                            - Full Multimodal posts: {fm_fake_ratio:.1%} fake content
                            - Consistency difference: {consistency_diff:.1%}
                            
                            **Implication:** Authenticity patterns vary across modalities, suggesting that different content 
                            types may require specialized detection approaches. Multimodal analysis should account for these 
                            variations when combining features from different modalities.
                            """)
            else:
                st.warning("📂 Cross-modal analysis data not available. Please ensure analysis tasks are complete.")
                st.info("""
                **To generate cross-modal analysis data:**
                ```bash
                python tasks/run_task12_cross_modal_analysis.py
                ```
                This will analyze multimodal relationships and authenticity consistency patterns.
                """)
            
        except FileNotFoundError as e:
            st.error(f"📂 Data file not found: {e}")
            st.info("""
            **To generate cross-modal analysis data:**
            ```bash
            python tasks/run_task12_cross_modal_analysis.py
            ```
            This will analyze multimodal relationships and generate dashboard data.
            """)
        except Exception as e:
            st.error(f"❌ Error loading cross-modal insights: {e}")
            st.info("Please ensure cross-modal analysis (Task 12) has been completed successfully.")
        finally:
            lazy_loader.hide_section_loading()
