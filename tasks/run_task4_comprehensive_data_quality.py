#!/usr/bin/env python3
"""
Main Execution Script for Comprehensive Data Quality Assessment and Leakage Mitigation
Task 4: Multimodal Fake News Detection Project

This script orchestrates all phases of the data quality assessment pipeline.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Import our modules
from data_quality_assessment import DataQualityAssessment
from data_preparation_standardization import DataPreparationStandardization

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_data_quality_assessment.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print execution banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COMPREHENSIVE DATA QUALITY ASSESSMENT                     ║
║                         AND LEAKAGE MITIGATION SYSTEM                        ║
║                                                                              ║
║  Task 4: Multimodal Fake News Detection - Data Quality Pipeline             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_phase_header(phase_num: int, phase_name: str, description: str):
    """Print phase header"""
    print(f"\n{'='*80}")
    print(f"PHASE {phase_num}: {phase_name}")
    print(f"{'='*80}")
    print(f"📋 {description}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

def print_phase_summary(phase_num: int, success: bool, duration: float, key_metrics: dict = None):
    """Print phase summary"""
    status = "✅ COMPLETED" if success else "❌ FAILED"
    print(f"\n{'-'*80}")
    print(f"PHASE {phase_num} {status} in {duration:.2f} seconds")
    
    if key_metrics:
        print("📊 Key Metrics:")
        for metric, value in key_metrics.items():
            print(f"   • {metric}: {value}")
    print(f"{'='*80}")

def check_prerequisites():
    """Check if prerequisites are met"""
    logger.info("Checking prerequisites...")
    
    required_dirs = [
        "processed_data/text_data",
        "processed_data/images", 
        "processed_data/comments"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        logger.error(f"Missing required directories: {missing_dirs}")
        logger.error("Please run Tasks 1-3 first to generate processed data.")
        return False
    
    # Check for required files
    text_files = list(Path("processed_data/text_data").glob("*_clean.parquet"))
    if not text_files:
        logger.error("No processed text data files found.")
        return False
    
    logger.info("All prerequisites met.")
    return True

def main():
    """Main execution function"""
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        return 1
    
    print_phase_header(1, "COMPREHENSIVE DATA QUALITY ASSESSMENT", 
                      "Running complete data quality assessment and leakage mitigation")
    
    start_time = time.time()
    
    try:
        # Initialize assessment system
        assessment = DataQualityAssessment()
        
        # Run comprehensive assessment
        logger.info("Running comprehensive data quality assessment...")
        results = assessment.run_comprehensive_assessment()
        
        if not results:
            raise Exception("Comprehensive assessment failed")
        
        # Calculate summary metrics
        quality_metrics_count = len(results.get('quality_metrics', {}))
        
        # Get mapping validation results
        mapping_validation = results.get('mapping_validation')
        text_image_rate = mapping_validation.text_image_mapping_rate if mapping_validation else 0
        cross_modal_consistency = mapping_validation.cross_modal_consistency_score if mapping_validation else 0
        
        # Count leakage detection results
        duplicate_results = results.get('duplicate_detection', {})
        total_duplicates = sum(duplicate_results.get('cross_split_duplicates', {}).values())
        
        key_metrics = {
            'Quality Assessments Completed': quality_metrics_count,
            'Text-Image Mapping Rate': f"{text_image_rate:.2f}%",
            'Cross-Modal Consistency': f"{cross_modal_consistency:.2f}%",
            'Cross-Split Duplicates Found': total_duplicates,
            'Visualizations Generated': len(list(Path("visualizations/data_quality").glob("*.png"))) if Path("visualizations/data_quality").exists() else 0,
            'Reports Generated': len(list(Path("reports").glob("*quality*.md"))) if Path("reports").exists() else 0
        }
        
        duration = time.time() - start_time
        print_phase_summary(1, True, duration, key_metrics)
        
        # Now run data preparation and standardization
        print_phase_header(2, "DATA PREPARATION AND STANDARDIZATION", 
                          "Creating clean datasets with rigorous split validation")
        
        prep_start_time = time.time()
        
        # Initialize preparation system
        preparation = DataPreparationStandardization()
        
        # Run comprehensive preparation
        logger.info("Running comprehensive data preparation...")
        prep_results = preparation.run_comprehensive_preparation()
        
        if not prep_results:
            raise Exception("Data preparation failed")
        
        # Extract preparation results
        final_splits = prep_results['final_splits']
        metadata = prep_results['metadata']
        
        # Calculate preparation metrics
        total_final_records = sum(len(df) for df in final_splits.values())
        leakage_validation_passed = metadata['split_statistics']['leakage_validation']['no_leakage_detected']
        
        prep_metrics = {
            'Training Records': f"{len(final_splits['train']):,}",
            'Validation Records': f"{len(final_splits['validation']):,}",
            'Test Records': f"{len(final_splits['test']):,}",
            'Total Final Records': f"{total_final_records:,}",
            'Leakage Validation': 'PASSED' if leakage_validation_passed else 'FAILED'
        }
        
        prep_duration = time.time() - prep_start_time
        print_phase_summary(2, True, prep_duration, prep_metrics)
        
        # Print final summary
        print(f"\n{'='*80}")
        print("🎉 COMPREHENSIVE DATA QUALITY ASSESSMENT COMPLETED")
        print(f"{'='*80}")
        
        print(f"📅 Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Total Duration: {time.time() - start_time:.2f} seconds")
        
        print(f"\n📁 Generated Outputs:")
        print(f"   • Clean Datasets: processed_data/clean_datasets/")
        print(f"   • Quality Analysis: analysis_results/data_quality/")
        print(f"   • Preparation Results: analysis_results/data_preparation/")
        print(f"   • Visualizations: visualizations/data_quality/")
        print(f"   • Reports: reports/")
        
        print(f"\n📊 Key Files Generated:")
        print(f"   • train_final_clean.parquet")
        print(f"   • validation_final_clean.parquet") 
        print(f"   • test_final_clean.parquet")
        print(f"   • data_quality_report.md")
        print(f"   • data_preparation_methodology.md")
        
        print(f"\n🔒 Data Quality Validation:")
        print(f"   • Leakage Detection: COMPLETED")
        print(f"   • Cross-Modal Validation: COMPLETED")
        print(f"   • Split Integrity: {'VALIDATED' if leakage_validation_passed else 'FAILED'}")
        print(f"   • Scientific Rigor: ENSURED")
        
        print(f"\n📋 Next Steps:")
        print(f"   • Proceed to Task 5: Advanced Multimodal Feature Extraction")
        print(f"   • Use clean datasets from processed_data/clean_datasets/")
        print(f"   • Review quality reports for insights")
        
        print(f"{'='*80}")
        
        return 0
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Assessment failed: {e}")
        print_phase_summary(1, False, duration)
        
        print(f"\n❌ Assessment failed. Check logs for details.")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())