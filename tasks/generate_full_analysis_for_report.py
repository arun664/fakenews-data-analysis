"""
Generate Full Analysis Data for Report

This script generates comprehensive analysis using the FULL dataset (no sampling)
for creating detailed reports with complete visualizations and statistics.

Output: High-resolution charts and complete statistical analysis saved to reports/
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Output directory for report assets
REPORT_DIR = Path('reports/analysis_images')
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(value):
    """Safely convert value to float"""
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except:
        return 0.0


def safe_int(value):
    """Safely convert value to int"""
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except:
        return 0


def generate_visual_features_analysis():
    """Generate complete visual features analysis with full dataset"""
    logger.info("Generating visual features analysis (FULL DATA)...")
    
    try:
        file_path = Path('processed_data/visual_features/visual_features_with_authenticity.parquet')
        if not file_path.exists():
            logger.warning(f"Visual features file not found: {file_path}")
            return
        
        data = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(data):,} visual feature records")
        
        label_col = 'authenticity_label' if 'authenticity_label' in data.columns else '2_way_label'
        
        fake_data = data[data[label_col] == 0]
        real_data = data[data[label_col] == 1]
        
        # 1. Feature Distribution Comparison
        logger.info("Creating feature distribution charts...")
        feature_columns = {
            'mean_brightness': 'Brightness',
            'sharpness_score': 'Sharpness',
            'visual_entropy': 'Visual Entropy',
            'noise_level': 'Noise Level'
        }
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(feature_columns.values()),
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for (feature_col, feature_name), (row, col) in zip(feature_columns.items(), positions):
            if feature_col in data.columns:
                fig.add_trace(
                    go.Histogram(x=fake_data[feature_col], name='Fake', marker_color='#FF6B6B',
                                opacity=0.6, nbinsx=50, legendgroup='fake', showlegend=(row==1 and col==1)),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Histogram(x=real_data[feature_col], name='Real', marker_color='#4ECDC4',
                                opacity=0.6, nbinsx=50, legendgroup='real', showlegend=(row==1 and col==1)),
                    row=row, col=col
                )
        
        fig.update_layout(height=800, title_text="Visual Feature Distributions: Fake vs Real (Full Dataset)",
                         barmode='overlay', font=dict(color='black'))
        fig.write_html(REPORT_DIR / 'visual_feature_distributions.html')
        fig.write_image(REPORT_DIR / 'visual_feature_distributions.png', width=1200, height=800)
        logger.info("✓ Saved visual_feature_distributions")
        
        # 2. Statistical Summary
        stats_summary = {
            'total_records': len(data),
            'fake_count': len(fake_data),
            'real_count': len(real_data),
            'features': {}
        }
        
        for feature_col, feature_name in feature_columns.items():
            if feature_col in data.columns:
                fake_vals = fake_data[feature_col].dropna()
                real_vals = real_data[feature_col].dropna()
                
                t_stat, p_value = stats.ttest_ind(fake_vals, real_vals)
                cohens_d = (fake_vals.mean() - real_vals.mean()) / np.sqrt((fake_vals.std()**2 + real_vals.std()**2) / 2)
                
                stats_summary['features'][feature_name] = {
                    'fake_mean': float(fake_vals.mean()),
                    'fake_std': float(fake_vals.std()),
                    'real_mean': float(real_vals.mean()),
                    'real_std': float(real_vals.std()),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'cohens_d': float(cohens_d),
                    'significant': p_value < 0.05
                }
        
        with open(REPORT_DIR / 'visual_features_statistics.json', 'w') as f:
            json.dump(stats_summary, f, indent=2)
        
        logger.info(f"✓ Visual features analysis complete: {len(data):,} records analyzed")
        
    except Exception as e:
        logger.error(f"Error in visual features analysis: {e}")
        import traceback
        traceback.print_exc()


def generate_linguistic_features_analysis():
    """Generate complete linguistic features analysis with full dataset"""
    logger.info("Generating linguistic features analysis (FULL DATA)...")
    
    try:
        file_path = Path('processed_data/linguistic_features/linguistic_features.parquet')
        if not file_path.exists():
            logger.warning(f"Linguistic features file not found: {file_path}")
            return
        
        data = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(data):,} linguistic feature records")
        
        label_col = 'authenticity_label' if 'authenticity_label' in data.columns else '2_way_label'
        
        fake_data = data[data[label_col] == 0]
        real_data = data[data[label_col] == 1]
        
        # Key linguistic features
        key_features = [
            'flesch_reading_ease', 'avg_word_length', 'avg_sentence_length',
            'lexical_diversity', 'sentiment_polarity', 'sentiment_subjectivity'
        ]
        
        available_features = [f for f in key_features if f in data.columns]
        
        if len(available_features) >= 4:
            # Create box plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=available_features[:4],
                vertical_spacing=0.15,
                horizontal_spacing=0.12
            )
            
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            
            for feature, (row, col) in zip(available_features[:4], positions):
                fig.add_trace(
                    go.Box(y=fake_data[feature], name='Fake', marker_color='#FF6B6B',
                          legendgroup='fake', showlegend=(row==1 and col==1)),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Box(y=real_data[feature], name='Real', marker_color='#4ECDC4',
                          legendgroup='real', showlegend=(row==1 and col==1)),
                    row=row, col=col
                )
            
            fig.update_layout(height=800, title_text="Linguistic Feature Distributions: Fake vs Real (Full Dataset)",
                             font=dict(color='black'))
            fig.write_html(REPORT_DIR / 'linguistic_feature_distributions.html')
            fig.write_image(REPORT_DIR / 'linguistic_feature_distributions.png', width=1200, height=800)
            logger.info("✓ Saved linguistic_feature_distributions")
        
        # Statistical summary
        stats_summary = {
            'total_records': len(data),
            'fake_count': len(fake_data),
            'real_count': len(real_data),
            'features': {}
        }
        
        for feature in available_features:
            fake_vals = fake_data[feature].dropna()
            real_vals = real_data[feature].dropna()
            
            if len(fake_vals) > 0 and len(real_vals) > 0:
                t_stat, p_value = stats.ttest_ind(fake_vals, real_vals)
                cohens_d = (fake_vals.mean() - real_vals.mean()) / np.sqrt((fake_vals.std()**2 + real_vals.std()**2) / 2)
                
                stats_summary['features'][feature] = {
                    'fake_mean': float(fake_vals.mean()),
                    'fake_std': float(fake_vals.std()),
                    'real_mean': float(real_vals.mean()),
                    'real_std': float(real_vals.std()),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'cohens_d': float(cohens_d),
                    'significant': p_value < 0.05
                }
        
        with open(REPORT_DIR / 'linguistic_features_statistics.json', 'w') as f:
            json.dump(stats_summary, f, indent=2)
        
        logger.info(f"✓ Linguistic features analysis complete: {len(data):,} records analyzed")
        
    except Exception as e:
        logger.error(f"Error in linguistic features analysis: {e}")
        import traceback
        traceback.print_exc()


def generate_temporal_analysis():
    """Generate temporal patterns analysis"""
    logger.info("Generating temporal analysis (FULL DATA)...")
    
    try:
        # Load temporal data
        temporal_path = Path('processed_data/temporal_analysis')
        if not temporal_path.exists():
            logger.warning("Temporal analysis directory not found")
            return
        
        # Yearly trends
        yearly_file = temporal_path / 'yearly_authenticity_trends.parquet'
        if yearly_file.exists():
            yearly_data = pd.read_parquet(yearly_file)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly_data['year'], y=yearly_data['fake_percentage'],
                mode='lines+markers', name='Fake Content %',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=10)
            ))
            fig.add_trace(go.Scatter(
                x=yearly_data['year'], y=yearly_data['real_percentage'],
                mode='lines+markers', name='Real Content %',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title="Yearly Authenticity Trends (Full Dataset)",
                xaxis_title="Year",
                yaxis_title="Percentage (%)",
                height=600,
                font=dict(color='black')
            )
            fig.write_html(REPORT_DIR / 'temporal_yearly_trends.html')
            fig.write_image(REPORT_DIR / 'temporal_yearly_trends.png', width=1200, height=600)
            logger.info("✓ Saved temporal_yearly_trends")
        
        logger.info("✓ Temporal analysis complete")
        
    except Exception as e:
        logger.error(f"Error in temporal analysis: {e}")
        import traceback
        traceback.print_exc()


def generate_dataset_overview():
    """Generate dataset overview visualizations"""
    logger.info("Generating dataset overview (FULL DATA)...")
    
    try:
        train = pd.read_parquet('processed_data/clean_datasets/train_final_clean.parquet')
        val = pd.read_parquet('processed_data/clean_datasets/validation_final_clean.parquet')
        test = pd.read_parquet('processed_data/clean_datasets/test_final_clean.parquet')
        
        # Split distribution
        split_data = pd.DataFrame({
            'Split': ['Train', 'Validation', 'Test'],
            'Records': [len(train), len(val), len(test)],
            'Fake': [
                (train['2_way_label'] == 0).sum(),
                (val['2_way_label'] == 0).sum(),
                (test['2_way_label'] == 0).sum()
            ],
            'Real': [
                (train['2_way_label'] == 1).sum(),
                (val['2_way_label'] == 1).sum(),
                (test['2_way_label'] == 1).sum()
            ]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Fake', x=split_data['Split'], y=split_data['Fake'],
                            marker_color='#FF6B6B'))
        fig.add_trace(go.Bar(name='Real', x=split_data['Split'], y=split_data['Real'],
                            marker_color='#4ECDC4'))
        
        fig.update_layout(
            title="Dataset Split Distribution (Full Dataset)",
            xaxis_title="Split",
            yaxis_title="Number of Records",
            barmode='group',
            height=600,
            font=dict(color='black')
        )
        fig.write_html(REPORT_DIR / 'dataset_split_distribution.html')
        fig.write_image(REPORT_DIR / 'dataset_split_distribution.png', width=1200, height=600)
        logger.info("✓ Saved dataset_split_distribution")
        
        # Summary statistics
        summary = {
            'total_records': len(train) + len(val) + len(test),
            'train': {'total': len(train), 'fake': int((train['2_way_label'] == 0).sum()), 'real': int((train['2_way_label'] == 1).sum())},
            'validation': {'total': len(val), 'fake': int((val['2_way_label'] == 0).sum()), 'real': int((val['2_way_label'] == 1).sum())},
            'test': {'total': len(test), 'fake': int((test['2_way_label'] == 0).sum()), 'real': int((test['2_way_label'] == 1).sum())}
        }
        
        with open(REPORT_DIR / 'dataset_overview_statistics.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("✓ Dataset overview complete")
        
    except Exception as e:
        logger.error(f"Error in dataset overview: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Generate all full analysis for report"""
    logger.info("="*70)
    logger.info("GENERATING FULL ANALYSIS FOR REPORT")
    logger.info("="*70)
    logger.info("")
    logger.info("Note: This uses the COMPLETE dataset (no sampling)")
    logger.info("Output directory: reports/analysis_images/")
    logger.info("")
    
    start_time = datetime.now()
    
    # Generate all analyses
    generate_dataset_overview()
    generate_visual_features_analysis()
    generate_linguistic_features_analysis()
    generate_temporal_analysis()
    
    # Summary
    logger.info("")
    logger.info("="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70)
    
    image_files = list(REPORT_DIR.glob('*.png'))
    html_files = list(REPORT_DIR.glob('*.html'))
    json_files = list(REPORT_DIR.glob('*.json'))
    
    logger.info(f"Generated {len(image_files)} PNG images")
    logger.info(f"Generated {len(html_files)} interactive HTML charts")
    logger.info(f"Generated {len(json_files)} statistical summaries")
    logger.info("")
    logger.info("Files saved to: reports/analysis_images/")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Completed in {elapsed:.1f} seconds")
    logger.info("="*70)


if __name__ == "__main__":
    main()
