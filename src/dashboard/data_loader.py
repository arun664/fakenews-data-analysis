"""
Data Loading Components for Dashboard

Handles loading and caching of analysis results and sample data.
"""

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path

@st.cache_data
def load_eda_data():
    """Load EDA results from main analysis_results folder."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        analysis_results_dir = os.getenv('ANALYSIS_OUTPUT_DIR', 'analysis_results')
        
        if os.path.exists(f'{analysis_results_dir}/multimodal_eda_report.json'):
            with open(f'{analysis_results_dir}/multimodal_eda_report.json', 'r') as f:
                return json.load(f)
        elif os.path.exists(f'{analysis_results_dir}/eda_comprehensive_report.json'):
            with open(f'{analysis_results_dir}/eda_comprehensive_report.json', 'r') as f:
                return json.load(f)
        elif os.path.exists('analysis_results/multimodal_eda_report.json'):
            with open('analysis_results/multimodal_eda_report.json', 'r') as f:
                return json.load(f)
        elif os.path.exists('analysis_results/eda_comprehensive_report.json'):
            with open('analysis_results/eda_comprehensive_report.json', 'r') as f:
                return json.load(f)
        else:
            st.error("EDA results not found. Please run the optimized multimodal EDA first.")
            return None
    except Exception as e:
        st.error(f"Error loading EDA data: {e}")
        return None

@st.cache_data
def load_sample_data():
    """Load a sample of the clean dataset from processed_data structure."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        processed_text_dir = os.getenv('PROCESSED_TEXT_DATA_DIR', 'processed_data/text_data')
        
        if os.path.exists(f'{processed_text_dir}/train_clean.parquet'):
            df = pd.read_parquet(f'{processed_text_dir}/train_clean.parquet')
            return df.head(100)
        elif os.path.exists('analysis_results/train_clean.parquet'):
            df = pd.read_parquet('analysis_results/train_clean.parquet')
            return df.head(100)
        else:
            return pd.DataFrame({
                'clean_title': [
                    'Scientists discover new treatment for cancer',
                    'SHOCKING: Aliens found in government facility!!!',
                    'Local community raises funds for school',
                    'You won\'t believe what happens next in this video',
                    'Research shows benefits of exercise'
                ],
                '2_way_label': [0, 1, 0, 1, 0],
                'title_length': [45, 52, 38, 48, 35]
            })
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

def detect_completed_analyses():
    """Detect which analyses have been completed based on organized structure."""
    from dotenv import load_dotenv
    load_dotenv()
    
    analysis_dir = os.getenv('ANALYSIS_OUTPUT_DIR', 'analysis_results')
    text_data_dir = os.getenv('PROCESSED_TEXT_DATA_DIR', 'processed_data/text_data')
    
    analyses = {
        'basic_eda': False,
        'multimodal_eda': False,
        'sample_data': False,
        'leakage_analysis': False,
        'task1_report': False,
        'task2_report': False
    }
    
    if (os.path.exists(f'{analysis_dir}/eda_comprehensive_report.json') or 
        os.path.exists('analysis_results/eda_comprehensive_report.json')):
        analyses['basic_eda'] = True
    
    if (os.path.exists(f'{analysis_dir}/multimodal_eda_report.json') or 
        os.path.exists('analysis_results/multimodal_eda_report.json')):
        analyses['multimodal_eda'] = True
    
    if (os.path.exists(f'{text_data_dir}/train_clean.parquet') or 
        os.path.exists('analysis_results/train_clean.parquet')):
        analyses['sample_data'] = True
    
    if (os.path.exists(f'{analysis_dir}/leakage_detection_report.txt') or 
        os.path.exists('analysis_results/leakage_detection_report.txt')):
        analyses['leakage_analysis'] = True
    
    if (os.path.exists(f'{analysis_dir}/task1_dataset_loading_report.json') or 
        os.path.exists('analysis_results/task1_dataset_loading_report.json')):
        analyses['task1_report'] = True
    
    if (os.path.exists(f'{analysis_dir}/task2_leakage_analysis_report.json') or 
        os.path.exists('analysis_results/task2_leakage_analysis_report.json')):
        analyses['task2_report'] = True
    
    return analyses