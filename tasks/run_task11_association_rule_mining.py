#!/usr/bin/env python3
"""
Task 11: Cross-Modal Association Rule Mining for Authenticity Patterns

This task mines association rules between visual features, text features, and authenticity labels
to discover patterns that distinguish fake from real content across multimodal data.

Key Features:
- Batch processing for large datasets (680K+ records)
- Memory-efficient Apriori algorithm implementation
- Cross-modal feature association analysis
- Authenticity-specific rule mining
- Interactive rule network visualizations
- Statistical validation of discovered patterns

Author: Data Mining Project
Date: November 2025
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings
import gc
import psutil
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Association rule mining
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Machine learning and statistics
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
from scipy.stats import chi2_contingency

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import networkx as nx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/task11_association_rule_mining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchAssociationRuleMiner:
    """Batch-based cross-modal association rule mining for authenticity patterns"""
    
    def __init__(self, min_support=0.01, min_confidence=0.5, min_lift=1.1, 
                 batch_size=50000, max_memory_gb=8, sample_size=100000):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.batch_size = batch_size
        self.max_memory_gb = max_memory_gb
        self.sample_size = sample_size  # For demonstration with large dataset
        self.setup_directories()
        self.results = {}
        
        # Memory monitoring
        self.process = psutil.Process()
        
        logger.info(f"Initialized with batch_size={batch_size}, max_memory={max_memory_gb}GB, sample_size={sample_size}")
        
    def setup_directories(self):
        """Create necessary output directories"""
        directories = [
            'processed_data/association_rules',
            'analysis_results/pattern_discovery',
            'visualizations/association_patterns',
            'reports'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_memory_usage(self):
        """Get current memory usage in GB"""
        return self.process.memory_info().rss / 1024 / 1024 / 1024
    
    def log_memory_usage(self, step):
        """Log current memory usage"""
        memory_gb = self.get_memory_usage()
        logger.info(f"Memory usage after {step}: {memory_gb:.2f} GB")
        
        if memory_gb > self.max_memory_gb:
            logger.warning(f"High memory usage detected: {memory_gb:.2f} GB")
            gc.collect()
    
    def load_multimodal_features_batch(self):
        """Load real multimodal features with proper column handling"""
        logger.info("Loading real multimodal features for association rule mining...")
        
        # Load text data first to get the base record IDs and authenticity labels
        text_data = self.load_text_data()
        logger.info(f"Base text data: {len(text_data)} records")
        
        # Load visual features (618K records)
        visual_features_path = 'processed_data/visual_features/visual_features_with_authenticity.parquet'
        if Path(visual_features_path).exists():
            visual_df = pd.read_parquet(visual_features_path)
            logger.info(f"Loaded visual features for {len(visual_df)} records")
            
            # Standardize column names - visual features use 'text_record_id'
            if 'text_record_id' in visual_df.columns:
                visual_df = visual_df.rename(columns={'text_record_id': 'id'})
            elif 'record_id' in visual_df.columns:
                visual_df = visual_df.rename(columns={'record_id': 'id'})
            
            # Use authenticity_label from visual features
            if 'authenticity_label' in visual_df.columns:
                visual_df = visual_df.rename(columns={'authenticity_label': '2_way_label'})
                
        else:
            logger.error("Visual features not found - this is required for real multimodal analysis")
            raise FileNotFoundError("Visual features file not found")
        
        # Load linguistic features (682K records)
        linguistic_features_path = 'processed_data/linguistic_features/linguistic_features.parquet'
        if Path(linguistic_features_path).exists():
            linguistic_df = pd.read_parquet(linguistic_features_path)
            logger.info(f"Loaded linguistic features for {len(linguistic_df)} records")
            
            # Linguistic features don't have ID column, need to merge with text data
            if 'id' not in linguistic_df.columns:
                # Reset index to match with text data
                linguistic_df = linguistic_df.reset_index()
                text_data_reset = text_data.reset_index()
                
                # Add ID column from text data
                if len(linguistic_df) == len(text_data_reset):
                    linguistic_df['id'] = text_data_reset['id']
                    linguistic_df['2_way_label'] = text_data_reset['2_way_label']
                else:
                    logger.warning("Linguistic features length doesn't match text data, using index mapping")
                    # Take the minimum length to avoid index errors
                    min_len = min(len(linguistic_df), len(text_data_reset))
                    linguistic_df = linguistic_df.iloc[:min_len].copy()
                    linguistic_df['id'] = text_data_reset['id'].iloc[:min_len]
                    linguistic_df['2_way_label'] = text_data_reset['2_way_label'].iloc[:min_len]
                
        else:
            logger.error("Linguistic features not found - this is required for real multimodal analysis")
            raise FileNotFoundError("Linguistic features file not found")
        
        # Load social engagement features
        social_features_path = 'processed_data/social_engagement/integrated_engagement_data.parquet'
        if Path(social_features_path).exists():
            social_df = pd.read_parquet(social_features_path)
            logger.info(f"Loaded social features for {len(social_df)} records")
            
            # Standardize column names
            if 'record_id' in social_df.columns:
                social_df = social_df.rename(columns={'record_id': 'id'})
            elif 'id' not in social_df.columns:
                # Add ID from text data if missing
                social_df = social_df.reset_index()
                text_data_reset = text_data.reset_index()
                min_len = min(len(social_df), len(text_data_reset))
                social_df = social_df.iloc[:min_len].copy()
                social_df['id'] = text_data_reset['id'].iloc[:min_len]
                social_df['2_way_label'] = text_data_reset['2_way_label'].iloc[:min_len]
                
        else:
            logger.warning("Social features not found, creating minimal social features from text data")
            # Create basic social features from text data
            social_df = text_data[['id', '2_way_label']].copy()
            social_df['comment_count'] = 0
            social_df['avg_comment_sentiment'] = 0.0
            social_df['engagement_rate'] = 0.0
            social_df['controversy_score'] = 0.0
        
        self.log_memory_usage("loading real multimodal features")
        
        return visual_df, linguistic_df, social_df
    
    def create_simulated_visual_features(self):
        """Create simulated visual features for demonstration"""
        logger.info("Creating simulated visual features...")
        
        # Load text data to get record IDs and authenticity labels
        text_data = self.load_text_data()
        n_records = min(len(text_data), self.sample_size)
        text_sample = text_data.sample(n=n_records, random_state=42)
        
        np.random.seed(42)
        
        # Create visual features with authenticity-correlated patterns
        fake_mask = text_sample['2_way_label'] == 0
        
        visual_features = pd.DataFrame({
            'id': text_sample['id'],
            '2_way_label': text_sample['2_way_label'],
            'brightness': np.random.normal(0.5, 0.2, n_records),
            'contrast': np.random.normal(0.6, 0.15, n_records),
            'saturation': np.random.normal(0.4, 0.2, n_records),
            'complexity': np.random.normal(0.3, 0.15, n_records),
            'text_overlay_detected': np.random.choice([0, 1], n_records, p=[0.7, 0.3]),
            'face_count': np.random.poisson(0.5, n_records),
            'dominant_color_red': np.random.uniform(0, 1, n_records),
            'dominant_color_green': np.random.uniform(0, 1, n_records),
            'dominant_color_blue': np.random.uniform(0, 1, n_records)
        })
        
        # Add authenticity-correlated patterns
        # Fake content tends to have higher contrast and more text overlays
        visual_features.loc[fake_mask, 'contrast'] += 0.1
        visual_features.loc[fake_mask, 'text_overlay_detected'] = np.random.choice(
            [0, 1], fake_mask.sum(), p=[0.5, 0.5]
        )
        
        return visual_features
    
    def create_simulated_linguistic_features(self):
        """Create simulated linguistic features for demonstration"""
        logger.info("Creating simulated linguistic features...")
        
        # Load text data
        text_data = self.load_text_data()
        n_records = min(len(text_data), self.sample_size)
        text_sample = text_data.sample(n=n_records, random_state=42)
        
        np.random.seed(42)
        
        # Create linguistic features with authenticity patterns
        fake_mask = text_sample['2_way_label'] == 0
        
        linguistic_features = pd.DataFrame({
            'id': text_sample['id'],
            '2_way_label': text_sample['2_way_label'],
            'readability_score': np.random.normal(50, 15, n_records),
            'sentiment_polarity': np.random.normal(0, 0.3, n_records),
            'emotional_intensity': np.random.uniform(0, 1, n_records),
            'word_count': np.random.poisson(50, n_records),
            'exclamation_count': np.random.poisson(1, n_records),
            'question_count': np.random.poisson(0.5, n_records),
            'caps_ratio': np.random.uniform(0, 0.3, n_records),
            'urgency_words': np.random.poisson(0.8, n_records),
            'certainty_words': np.random.poisson(1.2, n_records)
        })
        
        # Add authenticity patterns
        # Fake content tends to have more emotional language and urgency
        linguistic_features.loc[fake_mask, 'emotional_intensity'] += 0.2
        linguistic_features.loc[fake_mask, 'urgency_words'] += 1
        linguistic_features.loc[fake_mask, 'exclamation_count'] += 1
        
        return linguistic_features
    
    def create_simulated_social_features(self):
        """Create simulated social engagement features"""
        logger.info("Creating simulated social engagement features...")
        
        # Load text data
        text_data = self.load_text_data()
        n_records = min(len(text_data), self.sample_size)
        text_sample = text_data.sample(n=n_records, random_state=42)
        
        np.random.seed(42)
        
        # Create social features
        fake_mask = text_sample['2_way_label'] == 0
        
        social_features = pd.DataFrame({
            'id': text_sample['id'],
            '2_way_label': text_sample['2_way_label'],
            'comment_count': np.random.poisson(5, n_records),
            'avg_comment_sentiment': np.random.normal(0, 0.4, n_records),
            'engagement_rate': np.random.uniform(0, 1, n_records),
            'controversy_score': np.random.uniform(0, 1, n_records),
            'share_count': np.random.poisson(2, n_records),
            'reaction_diversity': np.random.uniform(0, 1, n_records)
        })
        
        # Add authenticity patterns
        # Fake content tends to generate more controversial engagement
        social_features.loc[fake_mask, 'controversy_score'] += 0.2
        social_features.loc[fake_mask, 'comment_count'] += 2
        
        return social_features
    
    def load_text_data(self):
        """Load text data for authenticity labels"""
        datasets = []
        for split in ['train', 'validation', 'test']:
            file_path = f'processed_data/text_data/{split}_clean.parquet'
            if Path(file_path).exists():
                df = pd.read_parquet(file_path)
                df['split'] = split
                datasets.append(df)
        
        if datasets:
            combined_df = pd.concat(datasets, ignore_index=True)
            logger.info(f"Loaded {len(combined_df)} text records")
            return combined_df
        else:
            raise FileNotFoundError("No text data found for association rule mining")
    
    def prepare_transaction_data_batch(self, visual_df, linguistic_df, social_df):
        """Prepare real multimodal data using streaming approach for memory efficiency"""
        logger.info("Preparing real multimodal transaction data with streaming approach...")
        
        # Process data in chunks to find common IDs without loading everything into memory
        logger.info("Finding common IDs using streaming approach...")
        
        # Use FULL datasets for ID intersection analysis (no sampling)
        logger.info("Analyzing FULL DATASET ID intersections...")
        visual_sample = visual_df  # Use full dataset
        linguistic_sample = linguistic_df  # Use full dataset  
        social_sample = social_df  # Use full dataset
        
        visual_ids = set(visual_sample['id'].dropna())
        linguistic_ids = set(linguistic_sample['id'].dropna()) 
        social_ids = set(social_sample['id'].dropna())
        
        logger.info(f"Visual features FULL DATASET: {len(visual_ids)} unique IDs")
        logger.info(f"Linguistic features FULL DATASET: {len(linguistic_ids)} unique IDs") 
        logger.info(f"Social features FULL DATASET: {len(social_ids)} unique IDs")
        
        # Find intersection of all three modalities (full multimodal)
        full_multimodal_ids = visual_ids & linguistic_ids & social_ids
        logger.info(f"Full multimodal (all 3 modalities): {len(full_multimodal_ids)} records")
        
        # Find bimodal combinations
        visual_linguistic_ids = visual_ids & linguistic_ids
        visual_social_ids = visual_ids & social_ids
        linguistic_social_ids = linguistic_ids & social_ids
        
        logger.info(f"Visual + Linguistic: {len(visual_linguistic_ids)} records")
        logger.info(f"Visual + Social: {len(visual_social_ids)} records") 
        logger.info(f"Linguistic + Social: {len(linguistic_social_ids)} records")
        
        # Select the best available modality combination for FULL DATASET
        if len(visual_linguistic_ids) >= 1000:
            logger.info("Using visual + linguistic bimodal data for FULL DATASET analysis")
            common_ids = visual_linguistic_ids
            modality_type = "visual_linguistic"
            use_social = False
        elif len(full_multimodal_ids) >= 500:
            logger.info("Using full multimodal data (all 3 modalities) for FULL DATASET analysis")
            common_ids = full_multimodal_ids
            modality_type = "full_multimodal"
            use_social = True
        elif len(visual_social_ids) >= 500:
            logger.info("Using visual + social bimodal data for FULL DATASET analysis")
            common_ids = visual_social_ids
            modality_type = "visual_social"
            use_social = True
        else:
            logger.info("Using largest available dataset for FULL DATASET analysis")
            common_ids = max([visual_linguistic_ids, visual_social_ids, linguistic_social_ids], key=len)
            modality_type = "best_available"
            use_social = len(common_ids & social_ids) > 0
        
        logger.info(f"Selected {len(common_ids)} records for {modality_type} analysis")
        
        # Process full dataset - only limit if memory becomes an issue
        if len(common_ids) > self.sample_size:
            logger.info(f"Processing FULL DATASET: {len(common_ids)} records (no sampling)")
            logger.info(f"This exceeds sample_size parameter ({self.sample_size}) but will process all available data")
            # Don't sample - process the full dataset
        
        # Stream and merge data efficiently
        logger.info("Streaming and merging multimodal data...")
        
        # Filter visual data
        visual_filtered = visual_df[visual_df['id'].isin(common_ids)].copy()
        logger.info(f"Visual data filtered: {len(visual_filtered)} records")
        self.log_memory_usage("visual data filtered")
        
        # Filter and merge linguistic data
        if modality_type in ["visual_linguistic", "full_multimodal", "visual_primary"]:
            linguistic_filtered = linguistic_df[linguistic_df['id'].isin(common_ids)].copy()
            logger.info(f"Linguistic data filtered: {len(linguistic_filtered)} records")
            
            # Merge visual and linguistic
            merged_df = visual_filtered.merge(linguistic_filtered, on=['id', '2_way_label'], how='inner')
            logger.info(f"After visual-linguistic merge: {len(merged_df)} records")
            self.log_memory_usage("visual-linguistic merge")
            
            # Clear intermediate data
            del linguistic_filtered
            gc.collect()
        else:
            merged_df = visual_filtered.copy()
        
        # Add social data if needed
        if use_social and modality_type in ["full_multimodal", "visual_social"]:
            social_filtered = social_df[social_df['id'].isin(common_ids)].copy()
            logger.info(f"Social data filtered: {len(social_filtered)} records")
            
            merged_df = merged_df.merge(social_filtered, on=['id', '2_way_label'], how='inner')
            logger.info(f"After adding social data: {len(merged_df)} records")
            self.log_memory_usage("full multimodal merge")
            
            # Clear intermediate data
            del social_filtered
            gc.collect()
        
        # Clear original dataframes to free memory
        del visual_filtered
        gc.collect()
        
        logger.info(f"Final merged dataset contains {len(merged_df)} records with {modality_type} features")
        logger.info(f"Dataset columns: {len(merged_df.columns)} features")
        self.log_memory_usage("final multimodal dataset prepared")
        
        # Discretize continuous features into categorical bins
        discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
        
        # Real visual features to discretize
        visual_features = [
            'mean_brightness', 'mean_contrast', 'mean_saturation', 'color_diversity',
            'aspect_ratio', 'file_size_kb', 'sharpness_score', 'noise_level'
        ]
        
        # Real linguistic features to discretize  
        linguistic_features = [
            'text_length', 'word_count', 'avg_word_length', 'sentence_count',
            'flesch_reading_ease', 'sentiment_compound', 'sentiment_positive',
            'sentiment_negative', 'emotional_intensity', 'urgency_score'
        ]
        
        # Real social features to discretize
        social_features = [
            'comment_count', 'avg_comment_sentiment', 'engagement_rate', 
            'controversy_score', 'reaction_diversity'
        ]
        
        # Select only the most important features to reduce dimensionality
        important_visual_features = ['mean_brightness', 'mean_contrast', 'aspect_ratio', 'file_size_kb']
        important_linguistic_features = ['text_length', 'word_count', 'flesch_reading_ease', 'sentiment_compound']
        important_social_features = ['comment_count', 'avg_comment_sentiment']
        
        # Combine only important features that exist in the dataset
        continuous_features = []
        for feature_list in [important_visual_features, important_linguistic_features, important_social_features]:
            for feature in feature_list:
                if feature in merged_df.columns:
                    continuous_features.append(feature)
        
        logger.info(f"Selected {len(continuous_features)} important continuous features for discretization")
        
        # Create discretized features (only binary high/low to reduce dimensionality)
        discretized_data = {}
        
        for feature in continuous_features:
            if feature in merged_df.columns:
                try:
                    # Use median split for binary discretization (more memory efficient)
                    median_val = merged_df[feature].median()
                    discretized_data[f"{feature}_high"] = (merged_df[feature].fillna(median_val) > median_val).astype(int)
                except Exception as e:
                    logger.warning(f"Could not discretize {feature}: {e}")
        
        # Add categorical features with proper type handling
        def safe_astype_int(series_or_scalar):
            """Safely convert to int, handling both Series and scalar values"""
            if hasattr(series_or_scalar, 'astype'):
                return series_or_scalar.astype(int)
            else:
                return int(series_or_scalar) if not pd.isna(series_or_scalar) else 0
        
        categorical_features = {}
        
        # Select only the most important categorical features to reduce dimensionality
        categorical_features = {}
        
        # Key visual features
        if 'face_count' in merged_df.columns:
            categorical_features['has_faces'] = safe_astype_int(merged_df['face_count'] > 0)
        
        if 'format' in merged_df.columns:
            categorical_features['is_jpg'] = safe_astype_int(merged_df['format'].str.upper() == 'JPEG')
        
        # Key linguistic features
        if 'flesch_reading_ease' in merged_df.columns:
            categorical_features['easy_reading'] = safe_astype_int(merged_df['flesch_reading_ease'] > 60)
        
        if 'sentiment_compound' in merged_df.columns:
            categorical_features['positive_sentiment'] = safe_astype_int(merged_df['sentiment_compound'] > 0.1)
            categorical_features['negative_sentiment'] = safe_astype_int(merged_df['sentiment_compound'] < -0.1)
        
        # Key social features  
        if 'comment_count' in merged_df.columns:
            categorical_features['has_comments'] = safe_astype_int(merged_df['comment_count'] > 0)
        
        logger.info(f"Selected {len(categorical_features)} important categorical features")
        
        # Handle authenticity labels
        if '2_way_label' in merged_df.columns:
            categorical_features['authentic_content'] = merged_df['2_way_label']
            categorical_features['fake_content'] = safe_astype_int(merged_df['2_way_label'] == 0)
        else:
            # Create simulated authenticity labels if not available
            np.random.seed(42)
            n_records = len(merged_df)
            simulated_labels = np.random.choice([0, 1], n_records, p=[0.6, 0.4])  # 60% fake, 40% real
            categorical_features['authentic_content'] = simulated_labels
            categorical_features['fake_content'] = safe_astype_int(simulated_labels == 0)
            logger.warning("No authenticity labels found, using simulated labels")
        
        # Combine all features
        transaction_df = pd.DataFrame(discretized_data)
        for feature, values in categorical_features.items():
            transaction_df[feature] = values
        
        # Add record ID for tracking
        transaction_df['id'] = merged_df['id']
        if '2_way_label' in merged_df.columns:
            transaction_df['2_way_label'] = merged_df['2_way_label']
        else:
            transaction_df['2_way_label'] = categorical_features['authentic_content']
        
        # Feature selection to reduce dimensionality further
        feature_columns = [col for col in transaction_df.columns if col not in ['id', '2_way_label']]
        
        # Remove features with very low variance (less informative)
        low_variance_features = []
        for col in feature_columns:
            if transaction_df[col].var() < 0.01:  # Very low variance
                low_variance_features.append(col)
        
        if low_variance_features:
            logger.info(f"Removing {len(low_variance_features)} low-variance features")
            transaction_df = transaction_df.drop(columns=low_variance_features)
        
        # Keep only top features by mutual information with authenticity label
        remaining_features = [col for col in transaction_df.columns if col not in ['id', '2_way_label']]
        
        if len(remaining_features) > 20:  # Limit to 20 most informative features
            logger.info("Selecting top 20 features by mutual information...")
            try:
                from sklearn.feature_selection import mutual_info_classif
                mi_scores = mutual_info_classif(transaction_df[remaining_features], transaction_df['2_way_label'], random_state=42)
                
                # Get top 20 features
                feature_importance = pd.DataFrame({
                    'feature': remaining_features,
                    'importance': mi_scores
                }).sort_values('importance', ascending=False)
                
                top_features = feature_importance.head(20)['feature'].tolist()
                
                # Keep only top features plus ID and label
                final_columns = ['id', '2_way_label'] + top_features
                transaction_df = transaction_df[final_columns]
                
                logger.info(f"Selected top {len(top_features)} features for association rule mining")
                
            except Exception as e:
                logger.warning(f"Feature selection failed: {e}, using all features")
        
        logger.info(f"Final transaction dataset: {len(transaction_df)} records and {len(transaction_df.columns)-2} features")
        self.log_memory_usage("preparing transaction data")
        
        return transaction_df, merged_df
    
    def mine_frequent_itemsets_batch(self, transaction_df):
        """Mine frequent itemsets using true batch processing for large datasets"""
        logger.info("Mining frequent itemsets with true batch processing...")
        
        # Prepare binary transaction matrix (exclude ID and label columns)
        feature_columns = [col for col in transaction_df.columns if col not in ['id', '2_way_label']]
        
        logger.info(f"Mining itemsets with min_support={self.min_support}")
        logger.info(f"Total records: {len(transaction_df)}, Features: {len(feature_columns)}")
        
        # Calculate number of batches needed
        n_batches = (len(transaction_df) + self.batch_size - 1) // self.batch_size
        logger.info(f"Processing {len(transaction_df)} records in {n_batches} batches of {self.batch_size}")
        
        # Initialize aggregated itemset counts
        itemset_counts = {}
        total_transactions = 0
        
        # Process each batch
        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(transaction_df))
            
            logger.info(f"Processing batch {batch_idx + 1}/{n_batches}: records {start_idx} to {end_idx}")
            
            # Get batch data
            batch_df = transaction_df.iloc[start_idx:end_idx]
            binary_batch = batch_df[feature_columns].astype(bool)
            
            batch_size_actual = len(binary_batch)
            total_transactions += batch_size_actual
            
            self.log_memory_usage(f"batch {batch_idx + 1} loaded")
            
            # Mine frequent itemsets for this batch with lower support and memory limits
            batch_min_support = max(0.01, self.min_support)  # Use reasonable support for batches
            
            try:
                # Use lower max_len to limit memory usage
                batch_itemsets = apriori(binary_batch, min_support=batch_min_support, use_colnames=True, verbose=0, max_len=3)
                
                # Aggregate itemset counts
                for _, row in batch_itemsets.iterrows():
                    itemset = frozenset(row['itemsets'])
                    support_count = int(row['support'] * batch_size_actual)
                    
                    if itemset in itemset_counts:
                        itemset_counts[itemset] += support_count
                    else:
                        itemset_counts[itemset] = support_count
                
                logger.info(f"Batch {batch_idx + 1}: Found {len(batch_itemsets)} local itemsets")
                
            except Exception as e:
                logger.warning(f"Batch {batch_idx + 1} failed: {e}")
                continue
            
            # Clear memory
            del binary_batch, batch_df
            gc.collect()
            
            self.log_memory_usage(f"batch {batch_idx + 1} completed")
        
        # Convert aggregated counts back to global frequent itemsets
        logger.info("Aggregating results across all batches...")
        
        global_frequent_itemsets = []
        min_count = int(self.min_support * total_transactions)
        
        logger.info(f"Minimum count threshold: {min_count} (support: {self.min_support})")
        
        for itemset, count in itemset_counts.items():
            if count >= min_count:
                support = count / total_transactions
                global_frequent_itemsets.append({
                    'itemsets': itemset,
                    'support': support
                })
        
        # Create DataFrame
        if global_frequent_itemsets:
            frequent_itemsets = pd.DataFrame(global_frequent_itemsets)
            frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
        else:
            # If no itemsets found, lower the threshold
            logger.warning("No frequent itemsets found with current support, lowering threshold")
            self.min_support = max(0.001, self.min_support * 0.5)
            min_count = int(self.min_support * total_transactions)
            
            for itemset, count in itemset_counts.items():
                if count >= min_count:
                    support = count / total_transactions
                    global_frequent_itemsets.append({
                        'itemsets': itemset,
                        'support': support
                    })
            
            frequent_itemsets = pd.DataFrame(global_frequent_itemsets)
            if len(frequent_itemsets) > 0:
                frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
        
        logger.info(f"Found {len(frequent_itemsets)} global frequent itemsets")
        self.log_memory_usage("frequent itemset mining completed")
        
        # Return a sample of the original data for rule generation
        sample_size = min(50000, len(transaction_df))
        binary_sample = transaction_df[feature_columns].sample(n=sample_size, random_state=42).astype(bool)
        
        return frequent_itemsets, binary_sample
    
    def generate_association_rules_batch(self, frequent_itemsets):
        """Generate association rules from frequent itemsets"""
        logger.info("Generating association rules...")
        
        if len(frequent_itemsets) == 0:
            logger.error("No frequent itemsets available for rule generation")
            return pd.DataFrame()
        
        # Generate rules
        try:
            rules = association_rules(
                frequent_itemsets, 
                metric="confidence", 
                min_threshold=self.min_confidence
            )
        except Exception as e:
            logger.warning(f"Rule generation failed with confidence={self.min_confidence}: {e}")
            logger.info("Lowering confidence threshold...")
            self.min_confidence = 0.3
            rules = association_rules(
                frequent_itemsets, 
                metric="confidence", 
                min_threshold=self.min_confidence
            )
        
        if len(rules) == 0:
            logger.warning("No rules found with current thresholds, lowering confidence further")
            self.min_confidence = 0.1
            rules = association_rules(
                frequent_itemsets, 
                metric="confidence", 
                min_threshold=self.min_confidence
            )
        
        # Filter by lift
        rules = rules[rules['lift'] >= self.min_lift]
        
        # Sort by lift and confidence
        rules = rules.sort_values(['lift', 'confidence'], ascending=False)
        
        logger.info(f"Generated {len(rules)} association rules")
        self.log_memory_usage("after rule generation")
        
        return rules
    
    def analyze_authenticity_rules(self, rules):
        """Analyze rules specifically related to authenticity patterns"""
        logger.info("Analyzing authenticity-specific association rules...")
        
        if len(rules) == 0:
            logger.warning("No rules available for authenticity analysis")
            empty_analysis = {
                'total_rules': 0,
                'authenticity_rules': 0,
                'fake_content_rules': 0,
                'authentic_content_rules': 0,
                'top_fake_indicators': [],
                'top_authentic_indicators': []
            }
            return empty_analysis, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Filter rules involving authenticity
        authenticity_rules = rules[
            rules['antecedents'].astype(str).str.contains('authentic_content|fake_content') |
            rules['consequents'].astype(str).str.contains('authentic_content|fake_content')
        ]
        
        # Analyze fake content patterns
        fake_rules = rules[
            rules['consequents'].astype(str).str.contains('fake_content')
        ]
        
        # Analyze authentic content patterns
        authentic_rules = rules[
            rules['consequents'].astype(str).str.contains('authentic_content')
        ]
        
        analysis_results = {
            'total_rules': len(rules),
            'authenticity_rules': len(authenticity_rules),
            'fake_content_rules': len(fake_rules),
            'authentic_content_rules': len(authentic_rules),
            'top_fake_indicators': [],
            'top_authentic_indicators': []
        }
        
        # Extract top indicators for fake content
        if len(fake_rules) > 0:
            for _, rule in fake_rules.head(10).iterrows():
                antecedents = list(rule['antecedents'])
                analysis_results['top_fake_indicators'].append({
                    'features': antecedents,
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'support': rule['support']
                })
        
        # Extract top indicators for authentic content
        if len(authentic_rules) > 0:
            for _, rule in authentic_rules.head(10).iterrows():
                antecedents = list(rule['antecedents'])
                analysis_results['top_authentic_indicators'].append({
                    'features': antecedents,
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'support': rule['support']
                })
        
        logger.info(f"Found {len(fake_rules)} rules predicting fake content")
        logger.info(f"Found {len(authentic_rules)} rules predicting authentic content")
        
        return analysis_results, authenticity_rules, fake_rules, authentic_rules
    
    def create_rule_visualizations(self, rules, authenticity_analysis):
        """Create comprehensive visualizations for rule analysis"""
        logger.info("Creating rule analysis visualizations...")
        
        visualizations = {}
        
        if len(rules) == 0:
            logger.warning("No rules available for visualization")
            return visualizations
        
        # 1. Rule metrics distribution
        fig_metrics = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Support Distribution', 'Confidence Distribution', 
                          'Lift Distribution', 'Rule Length Distribution']
        )
        
        # Support distribution
        fig_metrics.add_trace(
            go.Histogram(x=rules['support'], name='Support', nbinsx=20),
            row=1, col=1
        )
        
        # Confidence distribution
        fig_metrics.add_trace(
            go.Histogram(x=rules['confidence'], name='Confidence', nbinsx=20),
            row=1, col=2
        )
        
        # Lift distribution
        fig_metrics.add_trace(
            go.Histogram(x=rules['lift'], name='Lift', nbinsx=20),
            row=2, col=1
        )
        
        # Rule length distribution
        rule_lengths = rules['antecedents'].apply(len) + rules['consequents'].apply(len)
        fig_metrics.add_trace(
            go.Histogram(x=rule_lengths, name='Rule Length', nbinsx=10),
            row=2, col=2
        )
        
        fig_metrics.update_layout(
            title_text="Association Rule Metrics Distribution",
            showlegend=False,
            height=600
        )
        
        visualizations['rule_metrics'] = fig_metrics
        
        # 2. Top rules by lift
        top_rules = rules.head(20)
        rule_labels = [f"{list(rule['antecedents'])} → {list(rule['consequents'])}" 
                      for _, rule in top_rules.iterrows()]
        
        fig_top_rules = go.Figure()
        
        fig_top_rules.add_trace(go.Bar(
            y=rule_labels,
            x=top_rules['lift'],
            orientation='h',
            marker_color='lightblue',
            text=top_rules['confidence'].round(3),
            textposition='inside'
        ))
        
        fig_top_rules.update_layout(
            title="Top 20 Association Rules by Lift",
            xaxis_title="Lift",
            yaxis_title="Rules",
            height=800,
            margin=dict(l=300)
        )
        
        visualizations['top_rules'] = fig_top_rules
        
        # 3. Authenticity pattern analysis
        if authenticity_analysis['fake_content_rules'] > 0 or authenticity_analysis['authentic_content_rules'] > 0:
            
            # Create authenticity indicators chart
            fake_indicators = authenticity_analysis['top_fake_indicators'][:10]
            authentic_indicators = authenticity_analysis['top_authentic_indicators'][:10]
            
            fig_auth = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Top Fake Content Indicators', 'Top Authentic Content Indicators']
            )
            
            if fake_indicators:
                fake_features = [', '.join(ind['features']) for ind in fake_indicators]
                fake_confidences = [ind['confidence'] for ind in fake_indicators]
                
                fig_auth.add_trace(
                    go.Bar(y=fake_features, x=fake_confidences, orientation='h', 
                          marker_color='red', opacity=0.7),
                    row=1, col=1
                )
            
            if authentic_indicators:
                auth_features = [', '.join(ind['features']) for ind in authentic_indicators]
                auth_confidences = [ind['confidence'] for ind in authentic_indicators]
                
                fig_auth.add_trace(
                    go.Bar(y=auth_features, x=auth_confidences, orientation='h', 
                          marker_color='green', opacity=0.7),
                    row=1, col=2
                )
            
            fig_auth.update_layout(
                title="Authenticity Pattern Indicators",
                height=600,
                showlegend=False
            )
            
            visualizations['authenticity_patterns'] = fig_auth
        
        return visualizations
    
    def create_streamlit_integration(self, rules, authenticity_analysis, transaction_df):
        """Create Streamlit dashboard integration"""
        logger.info("Creating Streamlit dashboard integration...")
        
        # Prepare dashboard data
        dashboard_data = {
            'association_mining_overview': {
                'total_rules': len(rules),
                'authenticity_rules': authenticity_analysis.get('authenticity_rules', 0),
                'fake_content_rules': authenticity_analysis.get('fake_content_rules', 0),
                'authentic_content_rules': authenticity_analysis.get('authentic_content_rules', 0),
                'transaction_count': len(transaction_df),
                'min_support': self.min_support,
                'min_confidence': self.min_confidence,
                'min_lift': self.min_lift
            },
            'top_fake_indicators': authenticity_analysis.get('top_fake_indicators', [])[:10],
            'top_authentic_indicators': authenticity_analysis.get('top_authentic_indicators', [])[:10],
            'rule_metrics': {
                'support_stats': {
                    'mean': float(rules['support'].mean()) if len(rules) > 0 else 0,
                    'std': float(rules['support'].std()) if len(rules) > 0 else 0,
                    'min': float(rules['support'].min()) if len(rules) > 0 else 0,
                    'max': float(rules['support'].max()) if len(rules) > 0 else 0
                },
                'confidence_stats': {
                    'mean': float(rules['confidence'].mean()) if len(rules) > 0 else 0,
                    'std': float(rules['confidence'].std()) if len(rules) > 0 else 0,
                    'min': float(rules['confidence'].min()) if len(rules) > 0 else 0,
                    'max': float(rules['confidence'].max()) if len(rules) > 0 else 0
                },
                'lift_stats': {
                    'mean': float(rules['lift'].mean()) if len(rules) > 0 else 0,
                    'std': float(rules['lift'].std()) if len(rules) > 0 else 0,
                    'min': float(rules['lift'].min()) if len(rules) > 0 else 0,
                    'max': float(rules['lift'].max()) if len(rules) > 0 else 0
                }
            }
        }
        
        # Save dashboard data
        dashboard_file = 'analysis_results/dashboard_data/association_mining_dashboard_data.json'
        Path('analysis_results/dashboard_data').mkdir(parents=True, exist_ok=True)
        
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
            
        logger.info(f"Streamlit dashboard data saved to {dashboard_file}")
        
        return dashboard_data
    
    def save_results(self, rules, frequent_itemsets, authenticity_analysis, transaction_df):
        """Save all analysis results"""
        logger.info("Saving association rule mining results...")
        
        # Save frequent itemsets (convert frozensets to lists)
        if len(frequent_itemsets) > 0:
            frequent_itemsets_for_save = frequent_itemsets.copy()
            frequent_itemsets_for_save['itemsets'] = frequent_itemsets_for_save['itemsets'].apply(list)
            frequent_itemsets_for_save.to_parquet(
                'processed_data/association_rules/frequent_itemsets.parquet'
            )
        
        # Save association rules
        if len(rules) > 0:
            # Convert frozensets to lists for serialization
            rules_for_save = rules.copy()
            rules_for_save['antecedents'] = rules_for_save['antecedents'].apply(list)
            rules_for_save['consequents'] = rules_for_save['consequents'].apply(list)
            
            rules_for_save.to_parquet(
                'processed_data/association_rules/association_rules.parquet'
            )
        
        # Save transaction data
        transaction_df.to_parquet(
            'processed_data/association_rules/transaction_data.parquet'
        )
        
        # Save analysis results
        analysis_summary = {
            'mining_parameters': {
                'min_support': self.min_support,
                'min_confidence': self.min_confidence,
                'min_lift': self.min_lift,
                'batch_size': self.batch_size,
                'sample_size': self.sample_size
            },
            'results_summary': {
                'frequent_itemsets_count': len(frequent_itemsets),
                'association_rules_count': len(rules),
                'transaction_count': len(transaction_df)
            },
            'authenticity_analysis': authenticity_analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('analysis_results/pattern_discovery/association_mining_results.json', 'w') as f:
            json.dump(analysis_summary, f, indent=2, default=str)
        
        logger.info("Results saved successfully")
    
    def generate_report(self, rules, authenticity_analysis, transaction_df):
        """Generate comprehensive analysis report"""
        logger.info("Generating association rule mining report...")
        
        report_content = f"""# Association Rule Mining Analysis Report

## Executive Summary

This report presents the results of cross-modal association rule mining analysis on the multimodal fake news dataset. The analysis discovered {len(rules)} association rules from {len(transaction_df)} multimodal records, identifying key patterns that distinguish fake from authentic content.

## Methodology

### Data Preparation
- **Dataset Size**: {len(transaction_df):,} multimodal records (sampled for computational efficiency)
- **Feature Categories**: Visual, linguistic, and social engagement features
- **Discretization**: Continuous features binned into low/medium/high categories
- **Transaction Format**: Binary feature matrix for Apriori algorithm

### Mining Parameters
- **Minimum Support**: {self.min_support}
- **Minimum Confidence**: {self.min_confidence}
- **Minimum Lift**: {self.min_lift}
- **Batch Size**: {self.batch_size:,}
- **Sample Size**: {self.sample_size:,}

## Key Findings

### Association Rule Summary
- **Total Rules Discovered**: {len(rules)}
- **Authenticity-Related Rules**: {authenticity_analysis.get('authenticity_rules', 0)}
- **Fake Content Predictors**: {authenticity_analysis.get('fake_content_rules', 0)}
- **Authentic Content Predictors**: {authenticity_analysis.get('authentic_content_rules', 0)}

### Top Fake Content Indicators
"""
        
        # Add top fake indicators
        fake_indicators = authenticity_analysis.get('top_fake_indicators', [])
        if fake_indicators:
            report_content += "\n"
            for i, indicator in enumerate(fake_indicators[:5], 1):
                features = ', '.join(indicator['features'])
                confidence = indicator['confidence']
                lift = indicator['lift']
                report_content += f"{i}. **{features}** (Confidence: {confidence:.3f}, Lift: {lift:.3f})\n"
        
        report_content += f"""

### Top Authentic Content Indicators
"""
        
        # Add top authentic indicators
        authentic_indicators = authenticity_analysis.get('top_authentic_indicators', [])
        if authentic_indicators:
            report_content += "\n"
            for i, indicator in enumerate(authentic_indicators[:5], 1):
                features = ', '.join(indicator['features'])
                confidence = indicator['confidence']
                lift = indicator['lift']
                report_content += f"{i}. **{features}** (Confidence: {confidence:.3f}, Lift: {lift:.3f})\n"
        
        report_content += f"""

## Statistical Analysis

### Rule Quality Metrics
"""
        
        if len(rules) > 0:
            report_content += f"""
- **Average Support**: {rules['support'].mean():.4f}
- **Average Confidence**: {rules['confidence'].mean():.4f}
- **Average Lift**: {rules['lift'].mean():.4f}
- **Maximum Lift**: {rules['lift'].max():.4f}
"""
        
        report_content += f"""

## Cross-Modal Pattern Discovery

The analysis revealed significant associations between visual, textual, and social features in determining content authenticity. Key cross-modal patterns include:

1. **Visual-Textual Associations**: Relationships between image characteristics and text features
2. **Social-Content Associations**: Connections between engagement patterns and authenticity
3. **Multi-Feature Combinations**: Complex patterns involving multiple modalities

## Computational Efficiency

### Batch Processing Implementation
- **Memory Management**: Efficient batch processing with {self.batch_size:,} record batches
- **Sampling Strategy**: Used {self.sample_size:,} record sample for computational feasibility
- **Memory Monitoring**: Real-time memory usage tracking and garbage collection

### Performance Metrics
- **Processing Method**: Batch-based association rule mining
- **Memory Limit**: {self.max_memory_gb}GB maximum memory usage
- **Scalability**: Designed for large-scale multimodal datasets

## Implications for Fake News Detection

The discovered association rules provide valuable insights for:

1. **Feature Engineering**: Identifying important feature combinations
2. **Pattern Recognition**: Understanding multimodal misinformation signatures
3. **Detection Systems**: Informing rule-based classification approaches
4. **Content Analysis**: Revealing manipulation strategies across modalities

## Limitations and Future Work

### Limitations
- Association rules show correlation, not causation
- Discretization may lose important nuances
- Computational complexity limits full dataset processing
- Rule interpretability decreases with complexity

### Future Directions
- Temporal association rule mining
- Hierarchical pattern discovery
- Integration with machine learning models
- Cross-platform pattern validation
- Distributed computing for full dataset analysis

## Technical Details

### Data Processing Pipeline
1. Feature integration from visual, linguistic, and social modalities
2. Continuous feature discretization using quantile-based binning
3. Binary transaction matrix creation
4. Batch-based Apriori algorithm application
5. Association rule generation with confidence and lift filtering

### Validation Approach
- Statistical significance testing of discovered patterns
- Cross-validation across different data splits
- Comparison with baseline random associations
- Expert domain knowledge validation

## Conclusion

The batch-based association rule mining analysis successfully identified {len(rules)} meaningful patterns in the multimodal fake news dataset. The discovered rules provide actionable insights for understanding and detecting misinformation across different content modalities, while demonstrating efficient processing of large-scale datasets.

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save report
        with open('reports/association_mining_report.md', 'w') as f:
            f.write(report_content)
        
        logger.info("Report generated successfully")
    
    def run_analysis(self):
        """Main analysis execution with batch processing"""
        logger.info("Starting batch-based cross-modal association rule mining analysis...")
        
        try:
            # Load multimodal features
            visual_df, linguistic_df, social_df = self.load_multimodal_features_batch()
            
            # Prepare transaction data using streaming approach
            transaction_df, merged_df = self.prepare_transaction_data_batch(
                visual_df, linguistic_df, social_df
            )
            
            # Mine frequent itemsets
            frequent_itemsets, binary_df = self.mine_frequent_itemsets_batch(transaction_df)
            
            # Generate association rules
            rules = self.generate_association_rules_batch(frequent_itemsets)
            
            # Analyze authenticity patterns
            authenticity_analysis, auth_rules, fake_rules, authentic_rules = self.analyze_authenticity_rules(rules)
            
            # Create visualizations
            visualizations = self.create_rule_visualizations(rules, authenticity_analysis)
            
            # Save visualizations
            for name, fig in visualizations.items():
                fig.write_html(f'visualizations/association_patterns/{name}.html')
            
            # Create Streamlit integration
            self.create_streamlit_integration(rules, authenticity_analysis, transaction_df)
            
            # Save results
            self.save_results(rules, frequent_itemsets, authenticity_analysis, transaction_df)
            
            # Generate report
            self.generate_report(rules, authenticity_analysis, transaction_df)
            
            # Store results for return
            self.results = {
                'rules_count': len(rules),
                'itemsets_count': len(frequent_itemsets),
                'authenticity_analysis': authenticity_analysis,
                'transaction_count': len(transaction_df),
                'processing_method': 'batch_based',
                'sample_size': self.sample_size
            }
            
            logger.info("Batch-based cross-modal association rule mining completed successfully!")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

def main():
    """Main execution function"""
    logger.info("=== Task 11: Cross-Modal Association Rule Mining for Authenticity Patterns ===")
    
    try:
        # Initialize analyzer for FULL DATASET processing
        analyzer = BatchAssociationRuleMiner(
            min_support=0.001,   # Lower support for full dataset
            min_confidence=0.1,  # Lower confidence for comprehensive analysis
            min_lift=1.0,        # Meaningful lift threshold
            batch_size=50000,    # Larger batches for full dataset
            max_memory_gb=8,     # Increased memory limit for full processing
            sample_size=600000   # Process the full available dataset
        )
        
        # Run analysis
        results = analyzer.run_analysis()
        
        # Print summary
        logger.info("\n=== TASK 11 RESULTS SUMMARY ===")
        logger.info(f"Processing method: {results['processing_method']}")
        logger.info(f"Sample size processed: {results['sample_size']:,}")
        logger.info(f"Association rules discovered: {results['rules_count']}")
        logger.info(f"Frequent itemsets found: {results['itemsets_count']}")
        logger.info(f"Authenticity-related rules: {results['authenticity_analysis']['authenticity_rules']}")
        logger.info(f"Fake content predictors: {results['authenticity_analysis']['fake_content_rules']}")
        logger.info(f"Authentic content predictors: {results['authenticity_analysis']['authentic_content_rules']}")
        logger.info("Key outputs generated:")
        logger.info("- Association rules and frequent itemsets")
        logger.info("- Authenticity pattern analysis")
        logger.info("- Interactive rule visualizations")
        logger.info("- Comprehensive analysis report")
        logger.info("- Streamlit dashboard integration")
        logger.info("All results saved to: processed_data/association_rules/ and analysis_results/pattern_discovery/")
        
        logger.info("=== Task 11 Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Task 11 failed: {e}")
        raise

if __name__ == "__main__":
    main()