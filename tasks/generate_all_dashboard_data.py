"""
Generate Complete Dashboard Data

This script generates ALL dashboard data files needed for deployment.
It processes the complete dataset and creates lightweight JSON summaries.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def safe_float(value):
    """Safely convert value to float, handling NaN"""
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except:
        return 0.0


def safe_int(value):
    """Safely convert value to int, handling NaN"""
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except:
        return 0


def create_visual_features_summary():
    """Create compressed visualization data using FULL dataset with binning/aggregation"""
    logger.info("Creating visual features summary from FULL dataset...")
    
    try:
        file_path = Path('processed_data/visual_features/visual_features_with_authenticity.parquet')
        if not file_path.exists():
            logger.warning(f"Visual features file not found: {file_path}")
            return
        
        # Load FULL data
        data = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(data)} visual feature records (using ALL data)")
        
        # Determine label column
        if 'authenticity_label' in data.columns:
            label_col = 'authenticity_label'
        elif '2_way_label' in data.columns:
            label_col = '2_way_label'
        else:
            logger.error("No label column found in visual features")
            return
        
        # Create summary
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_records': int(len(data)),
            'fake_count': int((data[label_col] == 0).sum()),
            'real_count': int((data[label_col] == 1).sum()),
            'features_by_authenticity': {},
            'histograms': {},
            'boxplot_data': {},
            'scatter_data': {}
        }
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['authenticity_label', '2_way_label', 'record_id', 'id']]
        
        logger.info(f"Processing {len(feature_cols)} features with full data compression")
        
        # Key features for detailed visualization
        key_features = [
            'mean_brightness', 'sharpness_score', 'visual_entropy', 
            'noise_level', 'manipulation_score', 'contrast_score', 
            'color_diversity', 'edge_density'
        ]
        available_key_features = [f for f in key_features if f in feature_cols]
        
        # 1. Calculate statistics by authenticity
        for label, label_name in [(0, 'fake'), (1, 'real')]:
            subset = data[data[label_col] == label]
            summary['features_by_authenticity'][label_name] = {}
            
            for col in feature_cols:
                if col in subset.columns:
                    summary['features_by_authenticity'][label_name][col] = {
                        'mean': safe_float(subset[col].mean()),
                        'std': safe_float(subset[col].std()),
                        'min': safe_float(subset[col].min()),
                        'max': safe_float(subset[col].max()),
                        'median': safe_float(subset[col].median()),
                        'q25': safe_float(subset[col].quantile(0.25)),
                        'q75': safe_float(subset[col].quantile(0.75)),
                        'count': int(subset[col].notna().sum())
                    }
        
        # 2. Pre-compute histograms (binned data for ALL records)
        for feature in available_key_features:
            summary['histograms'][feature] = {}
            
            for label, label_name in [(0, 'fake'), (1, 'real')]:
                subset = data[data[label_col] == label][feature].dropna()
                
                if len(subset) > 0:
                    # Create 50 bins
                    hist, bin_edges = np.histogram(subset, bins=50)
                    
                    summary['histograms'][feature][label_name] = {
                        'counts': hist.tolist(),
                        'bin_edges': bin_edges.tolist()
                    }
        
        # 3. Pre-compute box plot data (quartiles + outliers)
        for feature in available_key_features:
            summary['boxplot_data'][feature] = {}
            
            for label, label_name in [(0, 'fake'), (1, 'real')]:
                subset = data[data[label_col] == label][feature].dropna()
                
                if len(subset) > 0:
                    q1 = subset.quantile(0.25)
                    q3 = subset.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Get outliers (limit to 100 for size)
                    outliers = subset[(subset < lower_bound) | (subset > upper_bound)]
                    outlier_sample = outliers.sample(min(100, len(outliers)), random_state=42).tolist() if len(outliers) > 0 else []
                    
                    summary['boxplot_data'][feature][label_name] = {
                        'min': safe_float(subset.min()),
                        'q1': safe_float(q1),
                        'median': safe_float(subset.median()),
                        'q3': safe_float(q3),
                        'max': safe_float(subset.max()),
                        'outliers': outlier_sample,
                        'outlier_count': int(len(outliers))
                    }
        
        # 4. Pre-compute scatter plot data (2D binning for correlation plots)
        # Example: brightness vs sharpness
        if 'mean_brightness' in data.columns and 'sharpness_score' in data.columns:
            for label, label_name in [(0, 'fake'), (1, 'real')]:
                subset = data[data[label_col] == label][['mean_brightness', 'sharpness_score']].dropna()
                
                if len(subset) > 0:
                    # Create 2D histogram (heatmap bins)
                    x_bins = 30
                    y_bins = 30
                    
                    hist_2d, x_edges, y_edges = np.histogram2d(
                        subset['mean_brightness'], 
                        subset['sharpness_score'],
                        bins=[x_bins, y_bins]
                    )
                    
                    summary['scatter_data'][f'brightness_vs_sharpness_{label_name}'] = {
                        'x_edges': x_edges.tolist(),
                        'y_edges': y_edges.tolist(),
                        'counts': hist_2d.tolist()
                    }
        
        # Save
        output_path = Path('analysis_results/dashboard_data/visual_features_summary.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ Visual features summary created: {size_mb:.2f} MB (FULL data, compressed via binning)")
        
    except Exception as e:
        logger.error(f"Error creating visual features summary: {e}")
        import traceback
        traceback.print_exc()


def create_linguistic_features_summary():
    """Create compressed visualization data using FULL dataset with binning/aggregation"""
    logger.info("Creating linguistic features summary from FULL dataset...")
    
    try:
        file_path = Path('processed_data/linguistic_features/linguistic_features.parquet')
        if not file_path.exists():
            logger.warning(f"Linguistic features file not found: {file_path}")
            return
        
        # Load FULL data
        data = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(data)} linguistic feature records (using ALL data)")
        
        # Determine label column
        if 'authenticity_label' in data.columns:
            label_col = 'authenticity_label'
        elif '2_way_label' in data.columns:
            label_col = '2_way_label'
        else:
            logger.error("No label column found in linguistic features")
            return
        
        # Create summary
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_records': int(len(data)),
            'fake_count': int((data[label_col] == 0).sum()),
            'real_count': int((data[label_col] == 1).sum()),
            'features_by_authenticity': {},
            'histograms': {},
            'violin_data': {},
            'scatter_data': {}
        }
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['authenticity_label', '2_way_label', 'record_id', 'id']]
        
        logger.info(f"Processing {len(feature_cols)} features with full data compression")
        
        # Key features for detailed visualization
        key_features = [
            'flesch_reading_ease', 'flesch_kincaid_grade',
            'text_length', 'word_count', 'avg_sentence_length',
            'unique_word_ratio', 'sentiment_positive', 'sentiment_negative',
            'sentiment_neutral', 'polarity', 'subjectivity'
        ]
        available_key_features = [f for f in key_features if f in feature_cols]
        
        # 1. Calculate statistics by authenticity
        for label, label_name in [(0, 'fake'), (1, 'real')]:
            subset = data[data[label_col] == label]
            summary['features_by_authenticity'][label_name] = {}
            
            for col in feature_cols:
                if col in subset.columns:
                    summary['features_by_authenticity'][label_name][col] = {
                        'mean': safe_float(subset[col].mean()),
                        'std': safe_float(subset[col].std()),
                        'min': safe_float(subset[col].min()),
                        'max': safe_float(subset[col].max()),
                        'median': safe_float(subset[col].median()),
                        'q25': safe_float(subset[col].quantile(0.25)),
                        'q75': safe_float(subset[col].quantile(0.75)),
                        'count': int(subset[col].notna().sum())
                    }
        
        # 2. Pre-compute histograms for overlapping distributions
        for feature in available_key_features:
            summary['histograms'][feature] = {}
            
            for label, label_name in [(0, 'fake'), (1, 'real')]:
                subset = data[data[label_col] == label][feature].dropna()
                
                if len(subset) > 0:
                    hist, bin_edges = np.histogram(subset, bins=50)
                    
                    summary['histograms'][feature][label_name] = {
                        'counts': hist.tolist(),
                        'bin_edges': bin_edges.tolist()
                    }
        
        # 3. Pre-compute violin plot data (kernel density estimation)
        for feature in available_key_features:
            summary['violin_data'][feature] = {}
            
            for label, label_name in [(0, 'fake'), (1, 'real')]:
                subset = data[data[label_col] == label][feature].dropna()
                
                if len(subset) > 0:
                    # Compute percentiles for violin shape
                    percentiles = np.percentile(subset, np.arange(0, 101, 2))
                    
                    summary['violin_data'][feature][label_name] = {
                        'percentiles': percentiles.tolist(),
                        'mean': safe_float(subset.mean()),
                        'median': safe_float(subset.median()),
                        'q1': safe_float(subset.quantile(0.25)),
                        'q3': safe_float(subset.quantile(0.75))
                    }
        
        # 4. Pre-compute scatter data (polarity vs subjectivity)
        if 'polarity' in data.columns and 'subjectivity' in data.columns:
            for label, label_name in [(0, 'fake'), (1, 'real')]:
                subset = data[data[label_col] == label][['polarity', 'subjectivity']].dropna()
                
                if len(subset) > 0:
                    # 2D histogram for density
                    hist_2d, x_edges, y_edges = np.histogram2d(
                        subset['subjectivity'], 
                        subset['polarity'],
                        bins=[30, 30]
                    )
                    
                    summary['scatter_data'][f'polarity_vs_subjectivity_{label_name}'] = {
                        'x_edges': x_edges.tolist(),
                        'y_edges': y_edges.tolist(),
                        'counts': hist_2d.tolist()
                    }
        
        # Save
        output_path = Path('analysis_results/dashboard_data/linguistic_features_summary.json')
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ Linguistic features summary created: {size_mb:.2f} MB (FULL data, compressed via binning)")
        
    except Exception as e:
        logger.error(f"Error creating linguistic features summary: {e}")
        import traceback
        traceback.print_exc()


def create_social_engagement_summary():
    """Create compressed visualization data using FULL dataset with binning/aggregation"""
    logger.info("Creating social engagement summary from FULL dataset...")
    
    try:
        file_path = Path('processed_data/social_engagement/integrated_engagement_data.parquet')
        if not file_path.exists():
            logger.warning(f"Social engagement file not found: {file_path}")
            return
        
        # Load FULL data
        data = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(data)} social engagement records (using ALL data)")
        
        # Determine label column
        label_col = '2_way_label' if '2_way_label' in data.columns else 'authenticity_label'
        
        # Add num_comments as alias for comment_count if it exists
        if 'comment_count' in data.columns and 'num_comments' not in data.columns:
            data['num_comments'] = data['comment_count']
        
        # Create summary
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_records': len(data),
            'fake_count': safe_int((data[label_col] == 0).sum()) if label_col in data.columns else 0,
            'real_count': safe_int((data[label_col] == 1).sum()) if label_col in data.columns else 0,
            'engagement_stats': {},
            'histograms': {},
            'boxplot_data': {}
        }
        
        # Key engagement metrics
        engagement_cols = ['score', 'num_comments']
        available_cols = [col for col in engagement_cols if col in data.columns]
        
        logger.info(f"Processing {len(available_cols)} engagement metrics with full data compression")
        
        if label_col in data.columns:
            # 1. Calculate statistics
            for label, label_name in [(0, 'fake'), (1, 'real')]:
                subset = data[data[label_col] == label]
                summary['engagement_stats'][label_name] = {}
                
                for col in available_cols:
                    if col in subset.columns:
                        summary['engagement_stats'][label_name][col] = {
                            'mean': safe_float(subset[col].mean()),
                            'std': safe_float(subset[col].std()),
                            'median': safe_float(subset[col].median()),
                            'min': safe_float(subset[col].min()),
                            'max': safe_float(subset[col].max()),
                            'q25': safe_float(subset[col].quantile(0.25)),
                            'q75': safe_float(subset[col].quantile(0.75))
                        }
            
            # 2. Pre-compute histograms
            for col in available_cols:
                summary['histograms'][col] = {}
                
                for label, label_name in [(0, 'fake'), (1, 'real')]:
                    subset = data[data[label_col] == label][col].dropna()
                    
                    if len(subset) > 0:
                        hist, bin_edges = np.histogram(subset, bins=50)
                        
                        summary['histograms'][col][label_name] = {
                            'counts': hist.tolist(),
                            'bin_edges': bin_edges.tolist()
                        }
            
            # 3. Pre-compute box plot data
            for col in available_cols:
                summary['boxplot_data'][col] = {}
                
                for label, label_name in [(0, 'fake'), (1, 'real')]:
                    subset = data[data[label_col] == label][col].dropna()
                    
                    if len(subset) > 0:
                        q1 = subset.quantile(0.25)
                        q3 = subset.quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        # Get outliers (limit to 100 for size)
                        outliers = subset[(subset < lower_bound) | (subset > upper_bound)]
                        outlier_sample = outliers.sample(min(100, len(outliers)), random_state=42).tolist() if len(outliers) > 0 else []
                        
                        summary['boxplot_data'][col][label_name] = {
                            'min': safe_float(subset.min()),
                            'q1': safe_float(q1),
                            'median': safe_float(subset.median()),
                            'q3': safe_float(q3),
                            'max': safe_float(subset.max()),
                            'outliers': outlier_sample,
                            'outlier_count': int(len(outliers))
                        }
        
        # Save
        output_path = Path('analysis_results/dashboard_data/social_engagement_summary.json')
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ Social engagement summary created: {size_mb:.2f} MB (FULL data, compressed via binning)")
        
    except Exception as e:
        logger.error(f"Error creating social engagement summary: {e}")
        import traceback
        traceback.print_exc()


def create_dataset_overview_summary():
    """Create lightweight summary of dataset overview"""
    logger.info("Creating dataset overview summary...")
    
    try:
        # Load datasets
        train_path = Path('processed_data/clean_datasets/train_final_clean.parquet')
        val_path = Path('processed_data/clean_datasets/validation_final_clean.parquet')
        test_path = Path('processed_data/clean_datasets/test_final_clean.parquet')
        
        if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
            logger.warning("One or more dataset files not found")
            return
        
        train = pd.read_parquet(train_path)
        val = pd.read_parquet(val_path)
        test = pd.read_parquet(test_path)
        
        logger.info(f"Loaded datasets: train={len(train)}, val={len(val)}, test={len(test)}")
        
        # Determine content_type column
        content_type_col = None
        for col in ['content_type', 'post_type', 'type']:
            if col in train.columns:
                content_type_col = col
                break
        
        # Calculate content type distribution
        content_type_dist = {}
        if content_type_col:
            all_data = pd.concat([train, val, test])
            content_type_dist = all_data[content_type_col].value_counts().to_dict()
        
        summary = {
            'generated_at': datetime.now().isoformat(),
            'splits': {
                'train': {
                    'total': len(train),
                    'fake': safe_int((train['2_way_label'] == 0).sum()),
                    'real': safe_int((train['2_way_label'] == 1).sum())
                },
                'validation': {
                    'total': len(val),
                    'fake': safe_int((val['2_way_label'] == 0).sum()),
                    'real': safe_int((val['2_way_label'] == 1).sum())
                },
                'test': {
                    'total': len(test),
                    'fake': safe_int((test['2_way_label'] == 0).sum()),
                    'real': safe_int((test['2_way_label'] == 1).sum())
                }
            },
            'total': {
                'records': len(train) + len(val) + len(test),
                'fake': safe_int((train['2_way_label'] == 0).sum() + (val['2_way_label'] == 0).sum() + (test['2_way_label'] == 0).sum()),
                'real': safe_int((train['2_way_label'] == 1).sum() + (val['2_way_label'] == 1).sum() + (test['2_way_label'] == 1).sum())
            },
            'content_type_distribution': content_type_dist,
            'has_content_type': content_type_col is not None
        }
        
        # Save
        output_path = Path('analysis_results/dashboard_data/dataset_overview_summary.json')
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ Dataset overview summary created: {size_mb:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error creating dataset overview summary: {e}")
        import traceback
        traceback.print_exc()


def create_authenticity_analysis_summary():
    """Create compressed visualization data using FULL dataset with binning/aggregation"""
    logger.info("Creating authenticity analysis summary from FULL dataset...")
    
    try:
        file_path = Path('processed_data/final_integrated_dataset/complete_multimodal_dataset.parquet')
        if not file_path.exists():
            logger.warning(f"Integrated dataset file not found: {file_path}")
            return
        
        # Load FULL data
        data = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(data)} integrated records (using ALL data)")
        
        # Create summary
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_records': len(data),
            'fake_count': safe_int((data['2_way_label'] == 0).sum()),
            'real_count': safe_int((data['2_way_label'] == 1).sum()),
            'metrics_by_authenticity': {},
            'histograms': {}
        }
        
        # Key metrics
        key_metrics = ['score', 'num_comments']
        available_metrics = [col for col in key_metrics if col in data.columns]
        
        logger.info(f"Processing {len(available_metrics)} metrics with full data compression")
        
        # 1. Calculate statistics
        for label, label_name in [(0, 'fake'), (1, 'real')]:
            subset = data[data['2_way_label'] == label]
            summary['metrics_by_authenticity'][label_name] = {}
            
            for col in available_metrics:
                if col in subset.columns:
                    summary['metrics_by_authenticity'][label_name][col] = {
                        'mean': safe_float(subset[col].mean()),
                        'std': safe_float(subset[col].std()),
                        'median': safe_float(subset[col].median()),
                        'min': safe_float(subset[col].min()),
                        'max': safe_float(subset[col].max()),
                        'q25': safe_float(subset[col].quantile(0.25)),
                        'q75': safe_float(subset[col].quantile(0.75))
                    }
        
        # 2. Pre-compute histograms
        for col in available_metrics:
            summary['histograms'][col] = {}
            
            for label, label_name in [(0, 'fake'), (1, 'real')]:
                subset = data[data['2_way_label'] == label][col].dropna()
                
                if len(subset) > 0:
                    hist, bin_edges = np.histogram(subset, bins=50)
                    
                    summary['histograms'][col][label_name] = {
                        'counts': hist.tolist(),
                        'bin_edges': bin_edges.tolist()
                    }
        
        # Save
        output_path = Path('analysis_results/dashboard_data/authenticity_analysis_summary.json')
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ Authenticity analysis summary created: {size_mb:.2f} MB (FULL data, compressed via binning)")
        
    except Exception as e:
        logger.error(f"Error creating authenticity analysis summary: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Generate all dashboard data"""
    logger.info("="*70)
    logger.info("GENERATING COMPLETE DASHBOARD DATA")
    logger.info("="*70)
    logger.info("")
    
    start_time = datetime.now()
    
    # Create all summaries
    create_visual_features_summary()
    create_linguistic_features_summary()
    create_social_engagement_summary()
    create_dataset_overview_summary()
    create_authenticity_analysis_summary()
    
    # Calculate total size
    logger.info("")
    logger.info("="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    dashboard_data_dir = Path('analysis_results/dashboard_data')
    total_size = 0
    file_count = 0
    
    if dashboard_data_dir.exists():
        for file in sorted(dashboard_data_dir.glob('*.json')):
            size_mb = file.stat().st_size / 1024 / 1024
            total_size += size_mb
            file_count += 1
            logger.info(f"{file.name}: {size_mb:.2f} MB")
    
    logger.info("")
    logger.info(f"Total files: {file_count}")
    logger.info(f"Total dashboard data size: {total_size:.2f} MB")
    
    if total_size < 50:
        logger.info(f"✓ SUCCESS! Data size is under 50 MB target")
    else:
        logger.warning(f"⚠ WARNING: Data size exceeds 50 MB target")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Completed in {elapsed:.1f} seconds")
    logger.info("="*70)


if __name__ == "__main__":
    main()
