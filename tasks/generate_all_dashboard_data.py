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
    """Create lightweight summary of visual features"""
    logger.info("Creating visual features summary...")
    
    try:
        file_path = Path('processed_data/visual_features/visual_features_with_authenticity.parquet')
        if not file_path.exists():
            logger.warning(f"Visual features file not found: {file_path}")
            return
        
        # Load data
        data = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(data)} visual feature records")
        
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
            'total_records': len(data),
            'fake_count': safe_int((data[label_col] == 0).sum()),
            'real_count': safe_int((data[label_col] == 1).sum()),
            'features_by_authenticity': {}
        }
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['authenticity_label', '2_way_label', 'record_id', 'id']]
        
        logger.info(f"Processing {len(feature_cols)} features...")
        
        # Calculate statistics by authenticity
        for label, label_name in [(0, 'fake'), (1, 'real')]:
            subset = data[data[label_col] == label]
            summary['features_by_authenticity'][label_name] = {}
            
            for col in list(feature_cols[:20]):  # Top 20 features
                if col in subset.columns:
                    summary['features_by_authenticity'][label_name][col] = {
                        'mean': safe_float(subset[col].mean()),
                        'std': safe_float(subset[col].std()),
                        'min': safe_float(subset[col].min()),
                        'max': safe_float(subset[col].max()),
                        'median': safe_float(subset[col].median())
                    }
        
        # Maximize sample size for deployment (target: <50MB per file)
        # Use as much data as possible while staying under size limit
        fake_count = (data[label_col] == 0).sum()
        real_count = (data[label_col] == 1).sum()
        
        # Aim for 25,000 per class = 50,000 total (excellent statistical representation, ~40MB per file)
        sample_size_per_class = 25000
        fake_sample = data[data[label_col] == 0].sample(min(sample_size_per_class, fake_count), random_state=42)
        real_sample = data[data[label_col] == 1].sample(min(sample_size_per_class, real_count), random_state=42)
        sample_data = pd.concat([fake_sample, real_sample])
        
        # Store sampling info for display on dashboard
        sampling_percentage = (len(sample_data) / len(data)) * 100
        logger.info(f"Sampled {len(sample_data)} records ({sampling_percentage:.1f}%) from {len(data)} total for visual features")
        
        summary['sampling_info'] = {
            'total_original': int(len(data)),
            'total_sampled': int(len(sample_data)),
            'sampling_percentage': float(sampling_percentage),
            'fake_original': int(fake_count),
            'fake_sampled': int(len(fake_sample)),
            'real_original': int(real_count),
            'real_sampled': int(len(real_sample))
        }
        
        # Include all important columns for visualizations
        important_cols = [
            'mean_brightness', 'sharpness_score', 'visual_entropy', 'noise_level',
            'contrast_score', 'color_diversity', 'edge_density', 'manipulation_score',
            'processing_success'
        ]
        
        # Get available columns
        key_columns = [label_col] + [col for col in important_cols if col in sample_data.columns]
        
        # Add any remaining feature columns up to 20 total
        remaining_cols = [col for col in feature_cols if col not in key_columns and col in sample_data.columns]
        key_columns.extend(remaining_cols[:max(0, 20 - len(key_columns))])
        
        summary['sample_data'] = sample_data[key_columns].fillna(0).to_dict('records')
        
        # Save
        output_path = Path('analysis_results/dashboard_data/visual_features_summary.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ Visual features summary created: {size_mb:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error creating visual features summary: {e}")
        import traceback
        traceback.print_exc()


def create_linguistic_features_summary():
    """Create lightweight summary of linguistic features"""
    logger.info("Creating linguistic features summary...")
    
    try:
        file_path = Path('processed_data/linguistic_features/linguistic_features.parquet')
        if not file_path.exists():
            logger.warning(f"Linguistic features file not found: {file_path}")
            return
        
        # Load data
        data = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(data)} linguistic feature records")
        
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
            'total_records': len(data),
            'fake_count': safe_int((data[label_col] == 0).sum()),
            'real_count': safe_int((data[label_col] == 1).sum()),
            'features_by_authenticity': {}
        }
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['authenticity_label', '2_way_label', 'record_id', 'id']]
        
        logger.info(f"Processing {len(feature_cols)} features...")
        
        # Calculate statistics
        for label, label_name in [(0, 'fake'), (1, 'real')]:
            subset = data[data[label_col] == label]
            summary['features_by_authenticity'][label_name] = {}
            
            for col in list(feature_cols):
                if col in subset.columns:
                    summary['features_by_authenticity'][label_name][col] = {
                        'mean': safe_float(subset[col].mean()),
                        'std': safe_float(subset[col].std()),
                        'min': safe_float(subset[col].min()),
                        'max': safe_float(subset[col].max()),
                        'median': safe_float(subset[col].median()),
                        'q25': safe_float(subset[col].quantile(0.25)),
                        'q75': safe_float(subset[col].quantile(0.75))
                    }
        
        # Maximize sample for deployment (target: <50MB per file)
        fake_count = (data[label_col] == 0).sum()
        real_count = (data[label_col] == 1).sum()
        
        sample_size_per_class = 25000  # 50,000 total for excellent representation (~38MB per file)
        fake_sample = data[data[label_col] == 0].sample(min(sample_size_per_class, fake_count), random_state=42)
        real_sample = data[data[label_col] == 1].sample(min(sample_size_per_class, real_count), random_state=42)
        sample_data = pd.concat([fake_sample, real_sample])
        
        sampling_percentage = (len(sample_data) / len(data)) * 100
        logger.info(f"Sampled {len(sample_data)} records ({sampling_percentage:.1f}%) from {len(data)} total for linguistic features")
        
        # Store sampling info
        summary['sampling_info'] = {
            'total_original': int(len(data)),
            'total_sampled': int(len(sample_data)),
            'sampling_percentage': float(sampling_percentage),
            'fake_original': int(fake_count),
            'fake_sampled': int(len(fake_sample)),
            'real_original': int(real_count),
            'real_sampled': int(len(real_sample))
        }
        
        # Convert sample data - include all features
        key_columns = [label_col] + list(feature_cols)
        summary['sample_data'] = sample_data[key_columns].fillna(0).to_dict('records')
        
        # Save
        output_path = Path('analysis_results/dashboard_data/linguistic_features_summary.json')
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ Linguistic features summary created: {size_mb:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error creating linguistic features summary: {e}")
        import traceback
        traceback.print_exc()


def create_social_engagement_summary():
    """Create lightweight summary of social engagement data"""
    logger.info("Creating social engagement summary...")
    
    try:
        file_path = Path('processed_data/social_engagement/integrated_engagement_data.parquet')
        if not file_path.exists():
            logger.warning(f"Social engagement file not found: {file_path}")
            return
        
        # Load data
        data = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(data)} social engagement records")
        
        # Determine label column
        label_col = '2_way_label' if '2_way_label' in data.columns else 'authenticity_label'
        
        # Create summary
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_records': len(data),
            'fake_count': safe_int((data[label_col] == 0).sum()) if label_col in data.columns else 0,
            'real_count': safe_int((data[label_col] == 1).sum()) if label_col in data.columns else 0,
            'engagement_stats': {}
        }
        
        # Key engagement metrics - map to expected column names
        engagement_cols = ['comment_count', 'engagement_score', 'share_count', 'reaction_count', 'score']
        
        # Add num_comments as alias for comment_count if it exists
        if 'comment_count' in data.columns and 'num_comments' not in data.columns:
            data['num_comments'] = data['comment_count']
        
        if label_col in data.columns:
            for label, label_name in [(0, 'fake'), (1, 'real')]:
                subset = data[data[label_col] == label]
                summary['engagement_stats'][label_name] = {}
                
                for col in engagement_cols:
                    if col in subset.columns:
                        summary['engagement_stats'][label_name][col] = {
                            'mean': safe_float(subset[col].mean()),
                            'std': safe_float(subset[col].std()),
                            'median': safe_float(subset[col].median())
                        }
            
            # Maximize sample for deployment
            fake_count = (data[label_col] == 0).sum()
            real_count = (data[label_col] == 1).sum()
            
            sample_size_per_class = 25000  # 50,000 total (~5MB per file)
            fake_sample = data[data[label_col] == 0].sample(min(sample_size_per_class, fake_count), random_state=42)
            real_sample = data[data[label_col] == 1].sample(min(sample_size_per_class, real_count), random_state=42)
            sample_data = pd.concat([fake_sample, real_sample])
            
            sampling_percentage = (len(sample_data) / len(data)) * 100
            logger.info(f"Sampled {len(sample_data)} records ({sampling_percentage:.1f}%) from {len(data)} total for social engagement")
            
            # Store sampling info
            summary['sampling_info'] = {
                'total_original': int(len(data)),
                'total_sampled': int(len(sample_data)),
                'sampling_percentage': float(sampling_percentage),
                'fake_original': int(fake_count),
                'fake_sampled': int(len(fake_sample)),
                'real_original': int(real_count),
                'real_sampled': int(len(real_sample))
            }
            
            # Include all available engagement columns
            available_cols = [col for col in engagement_cols + ['num_comments'] if col in sample_data.columns]
            key_columns = [label_col] + available_cols
            
            summary['sample_data'] = sample_data[key_columns].fillna(0).to_dict('records')
        
        # Save
        output_path = Path('analysis_results/dashboard_data/social_engagement_summary.json')
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ Social engagement summary created: {size_mb:.2f} MB")
        
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
    """Create lightweight summary for authenticity analysis"""
    logger.info("Creating authenticity analysis summary...")
    
    try:
        file_path = Path('processed_data/final_integrated_dataset/complete_multimodal_dataset.parquet')
        if not file_path.exists():
            logger.warning(f"Integrated dataset file not found: {file_path}")
            return
        
        # Load data
        data = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(data)} integrated records")
        
        # Create summary
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_records': len(data),
            'fake_count': safe_int((data['2_way_label'] == 0).sum()),
            'real_count': safe_int((data['2_way_label'] == 1).sum()),
            'metrics_by_authenticity': {}
        }
        
        # Key metrics - include all columns needed by authenticity_analysis page
        key_metrics = ['engagement_score', 'comment_count', 'num_comments', 'content_type_social', 'score']
        
        for label, label_name in [(0, 'fake'), (1, 'real')]:
            subset = data[data['2_way_label'] == label]
            summary['metrics_by_authenticity'][label_name] = {}
            
            for col in key_metrics:
                if col in subset.columns:
                    summary['metrics_by_authenticity'][label_name][col] = {
                        'mean': safe_float(subset[col].mean()) if subset[col].dtype in ['float64', 'int64'] else None,
                        'std': safe_float(subset[col].std()) if subset[col].dtype in ['float64', 'int64'] else None,
                        'median': safe_float(subset[col].median()) if subset[col].dtype in ['float64', 'int64'] else None
                    }
        
        # Maximize sample for deployment
        fake_count = (data['2_way_label'] == 0).sum()
        real_count = (data['2_way_label'] == 1).sum()
        
        sample_size_per_class = 25000  # 50,000 total (~8MB per file)
        fake_sample = data[data['2_way_label'] == 0].sample(min(sample_size_per_class, fake_count), random_state=42)
        real_sample = data[data['2_way_label'] == 1].sample(min(sample_size_per_class, real_count), random_state=42)
        sample_data = pd.concat([fake_sample, real_sample])
        
        sampling_percentage = (len(sample_data) / len(data)) * 100
        logger.info(f"Sampled {len(sample_data)} records ({sampling_percentage:.1f}%) from {len(data)} total for authenticity analysis")
        
        # Store sampling info
        summary['sampling_info'] = {
            'total_original': int(len(data)),
            'total_sampled': int(len(sample_data)),
            'sampling_percentage': float(sampling_percentage),
            'fake_original': int(fake_count),
            'fake_sampled': int(len(fake_sample)),
            'real_original': int(real_count),
            'real_sampled': int(len(real_sample))
        }
        
        # Include all available columns
        available_cols = [col for col in key_metrics if col in sample_data.columns]
        key_columns = ['2_way_label'] + available_cols
        
        summary['sample_data'] = sample_data[key_columns].fillna(0).to_dict('records')
        
        # Save
        output_path = Path('analysis_results/dashboard_data/authenticity_analysis_summary.json')
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ Authenticity analysis summary created: {size_mb:.2f} MB")
        
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
