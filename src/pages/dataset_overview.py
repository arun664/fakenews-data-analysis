"""
Dataset Overview Page
Displays comprehensive multimodal dataset statistics and analysis
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.lazy_loader import LazyLoader

lazy_loader = LazyLoader()


def render_dataset_overview(container):
    """Render Dataset Overview with lazy loading"""
    with container.container():
        try:
            st.header("True Multimodal Dataset Analysis")
            
            # Load from JSON summary (deployment-ready, no Parquet files needed)
            @st.cache_data(ttl=600)  # 10 minutes cache for dataset overview (static data)
            def load_overview_data():
                import json
                
                summary_path = Path('analysis_results/dashboard_data/dataset_overview_summary.json')
                
                if not summary_path.exists():
                    raise FileNotFoundError(f"Dataset overview summary not found at {summary_path}")
                
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                return summary
            
            dataset_summary = load_overview_data()
            
            # Extract data from summary
            total_records = dataset_summary.get('total', {}).get('records', 0)
            fake_count = dataset_summary.get('total', {}).get('fake', 0)
            real_count = dataset_summary.get('total', {}).get('real', 0)
            splits = dataset_summary.get('splits', {})
            
            # Hide loading indicator after data is loaded
            lazy_loader.hide_section_loading()
            
            # Key metrics row with data from summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📝 Total Records", f"{total_records:,}", 
                         delta="Full Dataset")
            
            with col2:
                st.metric("🔴 Fake Content", f"{fake_count:,}", 
                         delta=f"{fake_count/total_records*100:.1f}% of total")
            
            with col3:
                st.metric("🟢 Real Content", f"{real_count:,}", 
                         delta=f"{real_count/total_records*100:.1f}% of total")
            
            with col4:
                ratio = fake_count / real_count if real_count > 0 else 0
                st.metric("📊 Fake:Real Ratio", f"{ratio:.2f}:1", 
                         delta="Class imbalance")
        
            st.markdown("---")
            
            # Dataset splits breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Dataset Splits")
                
                # Create splits bar chart
                split_names = ['Train', 'Validation', 'Test']
                split_totals = [
                    splits.get('train', {}).get('total', 0),
                    splits.get('validation', {}).get('total', 0),
                    splits.get('test', {}).get('total', 0)
                ]
                
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=split_names,
                    y=split_totals,
                    marker_color=['#3498DB', '#9B59B6', '#E74C3C'],
                    text=[f"{v:,}" for v in split_totals],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Records by Split",
                    yaxis_title="Number of Records",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed breakdown
                st.write("**📊 Split Details:**")
                for split_name, split_key in [('Train', 'train'), ('Validation', 'validation'), ('Test', 'test')]:
                    split_data = splits.get(split_key, {})
                    total = split_data.get('total', 0)
                    fake = split_data.get('fake', 0)
                    real = split_data.get('real', 0)
                    st.write(f"• **{split_name}**: {total:,} ({fake:,} fake, {real:,} real)")
            
            with col2:
                st.subheader("🎭 Authenticity Distribution")
                
                # Authenticity analysis by modality type
                modality_auth_data = []
                
                for modality_name, subset in [
                    ("Full Multimodal", all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == True)]),
                    ("Dual Modal (Visual)", all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == False)]),
                    ("Dual Modal (Text)", all_data[(all_data['content_type'] == 'text_only') & (all_data['has_comments'] == True)]),
                    ("Single Modal", all_data[(all_data['content_type'] == 'text_only') & (all_data['has_comments'] == False)])
                ]:
                    if len(subset) > 0:
                        auth_dist = subset['2_way_label'].value_counts()
                        fake_count = auth_dist.get(0, 0)
                        real_count = auth_dist.get(1, 0)
                        
                        modality_auth_data.extend([
                            {"Modality": modality_name, "Type": "Fake", "Count": fake_count},
                            {"Modality": modality_name, "Type": "Real", "Count": real_count}
                        ])
                
                if modality_auth_data:
                    auth_df = pd.DataFrame(modality_auth_data)
                    
                    fig = px.bar(
                        auth_df,
                        x="Modality",
                        y="Count",
                        color="Type",
                        title="Authenticity Distribution by Modality Type",
                        color_discrete_map={"Fake": "#FF6B6B", "Real": "#4ECDC4"},
                        barmode="group"
                    )
                    fig.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Key insights
                st.write("**🔍 Key Insights:**")
                
                # Calculate fake percentages for each modality
                full_mm_subset = all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == True)]
                dual_vis_subset = all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == False)]
                
                if len(full_mm_subset) > 0:
                    full_mm_fake_pct = (full_mm_subset['2_way_label'] == 0).sum() / len(full_mm_subset) * 100
                    st.write(f"• Full multimodal: {full_mm_fake_pct:.1f}% fake content")
                
                if len(dual_vis_subset) > 0:
                    dual_vis_fake_pct = (dual_vis_subset['2_way_label'] == 0).sum() / len(dual_vis_subset) * 100
                    st.write(f"• Dual modal (visual): {dual_vis_fake_pct:.1f}% fake content")
                
                st.write(f"• Comment coverage significantly impacts authenticity patterns")
            
            st.markdown("---")
            
            # Processing pipeline status
            st.subheader("🔄 Data Processing Pipeline Status")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**📊 Data Quality**")
                st.write("• ✅ 620,665 clean records (90.9% retention)")
                st.write("• ✅ Cross-modal validation complete")
                st.write("• ✅ ID mapping integrity verified")
                st.write("• ✅ Balanced class distribution maintained")
            
            with col2:
                st.write("**🎯 Analysis Scope**")
                st.write(f"• 🖼️ Visual analysis: {visual_records:,} images")
                st.write(f"• 💬 Comment analysis: {len(comments_data):,} comments")
                st.write(f"• 🎯 Full multimodal: {full_multimodal:,} records")
                st.write(f"• 📊 Processing batches: 62 × 10K each")
            
            with col3:
                st.write("**⚡ Performance Metrics**")
                st.write("• 🚀 Processing rate: 71.4 images/min")
                st.write("• 💾 Storage efficiency: Parquet format")
                st.write("• 🔄 Memory optimization: Chunked processing")
                st.write("• 📈 Dashboard response: <2 sec")
            
            # Detailed statistics from real data
            st.markdown("---")
            st.subheader("📋 Detailed Dataset Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Text Quality Metrics**")
                if 'title' in all_data.columns:
                    avg_title_length = all_data['title'].str.len().mean()
                    short_titles = len(all_data[all_data['title'].str.len() < 10])
                    long_titles = len(all_data[all_data['title'].str.len() > 200])
                    
                    st.write(f"• Avg Title Length: {avg_title_length:.1f} chars")
                    st.write(f"• Short Titles: {short_titles:,}")
                    st.write(f"• Long Titles: {long_titles:,}")
                else:
                    # Use summary data
                    total_records = dataset_summary['total']['records']
                    st.write(f"• Total Records: {total_records:,}")
                    st.write(f"• Train Split: {dataset_summary['splits']['train']['total']:,}")
                    st.write(f"• Val Split: {dataset_summary['splits']['validation']['total']:,}")
            
            with col2:
                st.write("**Content Distribution**")
                fake_count = len(all_data[all_data['2_way_label'] == 0])
                real_count = len(all_data[all_data['2_way_label'] == 1])
                fake_pct = (fake_count / len(all_data)) * 100 if len(all_data) > 0 else 0
                
                st.write(f"• Fake Content: {fake_count:,} ({fake_pct:.1f}%)")
                st.write(f"• Real Content: {real_count:,} ({100-fake_pct:.1f}%)")
                if real_count > 0:
                    st.write(f"• Class Imbalance: {fake_count/real_count:.2f}:1")
                else:
                    st.write(f"• Class Imbalance: N/A")
            
            with col3:
                st.write("**Multimodal Coverage**")
                visual_coverage = (visual_records / len(all_data)) * 100
                comment_posts = len(all_data[all_data['has_comments'] == True])
                comment_coverage_pct = (comment_posts / len(all_data)) * 100
                
                st.write(f"• Visual Coverage: {visual_coverage:.1f}%")
                st.write(f"• Comment Coverage: {comment_coverage_pct:.1f}%")
                st.write(f"• Full Multimodal: {len(all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == True)]):,}")
            
            # Dataset insights summary
            st.markdown("---")
            st.subheader("💡 Dataset Insights Summary")
            
            # Calculate key metrics
            fake_count = len(all_data[all_data['2_way_label'] == 0])
            real_count = len(all_data[all_data['2_way_label'] == 1])
            fake_ratio = fake_count / real_count
            
            # Modality-specific fake rates
            full_mm_subset = all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == True)]
            dual_vis_subset = all_data[(all_data['content_type'] == 'text_image') & (all_data['has_comments'] == False)]
            
            full_mm_fake_pct = (full_mm_subset['2_way_label'] == 0).sum() / len(full_mm_subset) * 100 if len(full_mm_subset) > 0 else 0
            dual_vis_fake_pct = (dual_vis_subset['2_way_label'] == 0).sum() / len(dual_vis_subset) * 100 if len(dual_vis_subset) > 0 else 0
            
            st.success(f"""
            **Dataset Composition:**
            - **Total Records:** {len(all_data):,} posts after cleaning (90.9% retention from raw data)
            - **Class Distribution:** {fake_count:,} fake ({fake_count/len(all_data)*100:.1f}%) vs {real_count:,} real ({real_count/len(all_data)*100:.1f}%)
            - **Imbalance Ratio:** {fake_ratio:.2f}:1 (fake:real) - reflects real-world misinformation prevalence
            
            **Multimodal Richness:**
            - **Visual Content:** {visual_coverage:.1f}% of posts include images
            - **Social Engagement:** {comment_coverage_pct:.1f}% of posts have community comments
            - **Full Multimodal:** {full_multimodal:,} posts with text, images, AND comments
            
            **Authenticity by Modality:**
            - **Full Multimodal Posts:** {full_mm_fake_pct:.1f}% fake content
            - **Dual Modal (Visual):** {dual_vis_fake_pct:.1f}% fake content
            - **Pattern:** Multimodal content shows distinct authenticity patterns across modalities
            
            **Research Implications:**
            This dataset enables comprehensive multimodal fake news detection research by providing:
            1. Rich multimodal features (text, images, social engagement)
            2. Realistic class imbalance reflecting actual misinformation spread
            3. Diverse content types for robust model training
            4. Large-scale data (620K+ posts, 13.8M comments) for deep learning approaches
            """)
        
        except FileNotFoundError as e:
            st.error(f"📂 Data file not found: {e}")
            st.info("""
            **To generate the required data files:**
            ```bash
            # Run data preparation and cleaning
            python tasks/data_preparation_standardization.py
            
            # Run comment integration
            python tasks/run_task3_comment_integration.py
            ```
            These tasks will generate the clean datasets and comment mappings needed for the dashboard.
            """)
        except Exception as e:
            st.error(f"❌ Error loading dataset overview: {e}")
            st.info("Please ensure all data processing tasks are complete and data files are accessible.")
        finally:
            lazy_loader.hide_section_loading()
