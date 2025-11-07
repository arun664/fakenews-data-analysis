"""
Cross-Modal Insights Page
Analysis of multimodal relationships and authenticity consistency
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
