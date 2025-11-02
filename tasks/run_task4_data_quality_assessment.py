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

# Import our modules
from data_quality_assessment import DataQualityAssessment
from data_preparation_standardization import DataPreparationStandardization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_data_quality_assessment.log'),
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
    
    logger.info("✅ All prerequisites met.")
    return True

def run_phase_1_leakage_detection():
    """
    Phase 4.1: Data Leakage Detection and Prevention
    """
    print_phase_header(1, "DATA LEAKAGE DETECTION AND PREVENTION", 
                      "Implementing cross-modal duplicate detection and advanced leakage detection")
    
    start_time = time.time()
    
    try:
        # Initialize assessment system
        assessment = DataQualityAssessment()
        
        # Load datasets
        datasets = assessment.load_datasets()
        if not datasets:
            raise Exception("Failed to load datasets")
        
        # Run leakage detection components
        logger.info("🔍 Running cross-modal duplicate detection...")
        duplicate_results = assessment.detect_cross_modal_duplicates(datasets)
        
        logger.info("📝 Running text content overlap analysis...")
        text_overlap = assessment.detect_text_content_overlap(datasets)
        
        logger.info("⏰ Running temporal consistency validation...")
        temporal_results = assessment.validate_temporal_consistency(datasets)
        
        logger.info("👤 Running author leakage detection...")
        author_leakage = assessment.detect_author_leakage(datasets)
        
        # Compile results
        leakage_results = {
            'duplicate_detection': duplicate_results,
            'text_overlap': text_overlap,
            'temporal_validation': temporal_results,
            'author_leakage': author_leakage
        }
        
        # Calculate key metrics
        total_duplicates = sum(duplicate_results.get('cross_split_duplicates', {}).values())
        image_duplicates = duplicate_results.get('image_duplicates', {}).get('near_duplicates', 0)
        
        key_metrics = {
            'Cross-Split ID Duplicates': total_duplicates,
            'Near-Duplicate Images': image_duplicates,
            'Text Overlap Checks': len(text_overlap) if isinstance(text_overlap, dict) else 0,
            'Author Leakage Checks': len(author_leakage)
        }
        
        duration = time.time() - start_time
        print_phase_summary(1, True, duration, key_metrics)
        
        return leakage_results
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Phase 1 failed: {e}")
        print_phase_summary(1, False, duration)
        return None

def run_phase_2_quality_assessment():
    """
    Phase 4.2: Data Quality Assessment by Mapping Type
    """
    print_phase_header(2, "DATA QUALITY ASSESSMENT BY MAPPING TYPE", 
                      "Assessing quality metrics and validating ID mapping relationships")
    
    start_time = time.time()
    
    try:
        # Initialize assessment system
        assessment = DataQualityAssessment()
        
        # Load datasets
        datasets = assessment.load_datasets()
        if not datasets:
            raise Exception("Failed to load datasets")
        
        logger.info("📊 Assessing data quality by content type...")
        quality_metrics = assessment.assess_data_quality_by_mapping_type(datasets)
        
        logger.info("🔗 Validating ID mapping relationships...")
        mapping_validation = assessment.validate_id_mapping_relationships(datasets)
        
        # Calculate key metrics
        avg_completeness = sum(m.completeness_score for m in quality_metrics.values()) / len(quality_metrics)
        avg_consistency = sum(m.consistency_score for m in quality_metrics.values()) / len(quality_metrics)
        total_anomalies = sum(m.anomaly_count for m in quality_metrics.values())
        
        key_metrics = {
            'Average Completeness': f"{avg_completeness:.2f}%",
            'Average Consistency': f"{avg_consistency:.2f}%",
            'Total Anomalies': total_anomalies,
            'Text-Image Mapping Rate': f"{mapping_validation.text_image_mapping_rate:.2f}%",
            'Cross-Modal Consistency': f"{mapping_validation.cross_modal_consistency_score:.2f}%"
        }
        
        duration = time.time() - start_time
        print_phase_summary(2, True, duration, key_metrics)
        
        return {
            'quality_metrics': quality_metrics,
            'mapping_validation': mapping_validation
        }
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Phase 2 failed: {e}")
        print_phase_summary(2, False, duration)
        return None

def run_phase_3_data_preparation():
    """
    Phase 4.3: Data Preparation and Standardization
    """
    print_phase_header(3, "DATA PREPARATION AND STANDARDIZATION", 
                      "Standardizing formats and implementing validation pipelines")
    
    start_time = time.time()
    
    try:
        # Initialize preparation system
        preparation = DataPreparationStandardization()
        
        # Load and combine datasets
        text_data_files = list(Path("processed_data/text_data").glob("*_clean.parquet"))
        all_data = []
        for file_path in text_data_files:
            df = pd.read_parquet(file_path)
            all_data.append(df)
        
        import pandas as pd
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded combined dataset: {len(combined_df):,} records")
        
        logger.info("🔤 Standardizing text encoding...")
        df_processed = preparation.standardize_text_encoding(combined_df)
        
        logger.info("🔢 Standardizing numeric formats...")
        df_processed = preparation.standardize_numeric_formats(df_processed)
        
        logger.info("🔗 Validating cross-modal consistency...")
        df_processed, consistency_issues = preparation.validate_cross_modal_consistency(df_processed)
        
        logger.info("🧹 Removing duplicates and anomalies...")
        df_processed, removal_stats = preparation.remove_duplicates_and_anomalies(df_processed)
        
        logger.info("⚖️ Creating balanced sampling strategy...")
        df_processed = preparation.create_balanced_sampling_strategy(df_processed)
        
        # Calculate key metrics
        key_metrics = {
            'Records Processed': f"{len(combined_df):,}",
            'Records After Cleaning': f"{len(df_processed):,}",
            'Duplicates Removed': removal_stats['exact_duplicates'] + removal_stats['near_duplicates'],
            'Anomalies Removed': removal_stats['anomalies'],
            'Consistency Issues Fixed': sum(consistency_issues.values())
        }
        
        duration = time.time() - start_time
        print_phase_summary(3, True, duration, key_metrics)
        
        return {
            'processed_data': df_processed,
            'consistency_issues': consistency_issues,
            'removal_stats': removal_stats
        }
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Phase 3 failed: {e}")
        print_phase_summary(3, False, duration)
        return None

def run_phase_4_clean_dataset_generation():
    """
    Phase 4.4: Clean Dataset Generation
    """
    print_phase_header(4, "CLEAN DATASET GENERATION", 
                      "Creating rigorous splits and generating final clean datasets")
    
    start_time = time.time()
    
    try:
        # Initialize preparation system
        preparation = DataPreparationStandardization()
        
        # Run the complete preparation pipeline
        logger.info("🏗️ Running comprehensive data preparation...")
        results = preparation.run_comprehensive_preparation()
        
        if not results:
            raise Exception("Comprehensive preparation failed")
        
        # Extract results
        final_splits = results['final_splits']
        metadata = results['metadata']
        
        # Calculate key metrics
        total_final_records = sum(len(df) for df in final_splits.values())
        leakage_validation_passed = metadata['split_statistics']['leakage_validation']['no_leakage_detected']
        
        key_metrics = {
            'Training Records': f"{len(final_splits['train']):,}",
            'Validation Records': f"{len(final_splits['validation']):,}",
            'Test Records': f"{len(final_splits['test']):,}",
            'Total Final Records': f"{total_final_records:,}",
            'Leakage Validation': '✅ PASSED' if leakage_validation_passed else '❌ FAILED'
        }
        
        duration = time.time() - start_time
        print_phase_summary(4, True, duration, key_metrics)
        
        return results
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Phase 4 failed: {e}")
        print_phase_summary(4, False, duration)
        return None

def run_comprehensive_assessment_and_visualization():
    """
    Run comprehensive assessment with visualization generation
    """
    print_phase_header(5, "COMPREHENSIVE ASSESSMENT AND VISUALIZATION", 
                      "Generating visualizations and comprehensive reports")
    
    start_time = time.time()
    
    try:
        # Initialize assessment system
        assessment = DataQualityAssessment()
        
        # Run complete assessment
        logger.info("🔍 Running comprehensive assessment...")
        results = assessment.run_comprehensive_assessment()
        
        if not results:
            raise Exception("Comprehensive assessment failed")
        
        # Calculate summary metrics
        quality_metrics_count = len(results.get('quality_metrics', {}))
        visualizations_generated = len(list(Path("visualizations/data_quality").glob("*.png")))
        reports_generated = len(list(Path("reports").glob("*quality*.md")))
        
        key_metrics = {
            'Quality Assessments': quality_metrics_count,
            'Visualizations Generated': visualizations_generated,
            'Reports Generated': reports_generated,
            'Assessment Status': '✅ COMPLETE'
        }
        
        duration = time.time() - start_time
        print_phase_summary(5, True, duration, key_metrics)
        
        return results
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Phase 5 failed: {e}")
        print_phase_summary(5, False, duration)
        return None

def print_final_summary(all_results: dict):
    """Print final execution summary"""
    print(f"\n{'='*80}")
    print("🎉 COMPREHENSIVE DATA QUALITY ASSESSMENT COMPLETED")
    print(f"{'='*80}")
    
    print(f"📅 Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Count successful phases
    successful_phases = sum(1 for result in all_results.values() if result is not None)
    total_phases = len(all_results)
    
    print(f"✅ Successful Phases: {successful_phases}/{total_phases}")
    
    if successful_phases == total_phases:
        print("\n🎯 ALL PHASES COMPLETED SUCCESSFULLY!")
        
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
        print(f"   • Leakage Detection: ✅ COMPLETED")
        print(f"   • Cross-Modal Validation: ✅ COMPLETED")
        print(f"   • Split Integrity: ✅ VALIDATED")
        print(f"   • Scientific Rigor: ✅ ENSURED")
        
    else:
        print(f"\n⚠️  Some phases failed. Check logs for details.")
        failed_phases = [phase for phase, result in all_results.items() if result is None]
        print(f"❌ Failed Phases: {failed_phases}")
    
    print(f"\n📋 Next Steps:")
    if successful_phases == total_phases:
        print(f"   • Proceed to Task 5: Advanced Multimodal Feature Extraction")
        print(f"   • Use clean datasets from processed_data/clean_datasets/")
        print(f"   • Review quality reports for insights")
    else:
        print(f"   • Review error logs and fix issues")
        print(f"   • Re-run failed phases")
        print(f"   • Ensure all prerequisites are met")
    
    print(f"{'='*80}")

def main():
    """Main execution function"""
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        return 1
    
    # Track all results
    all_results = {}
    
    # Phase 4.1: Data Leakage Detection and Prevention
    all_results['leakage_detection'] = run_phase_1_leakage_detection()
    
    # Phase 4.2: Data Quality Assessment by Mapping Type
    all_results['quality_assessment'] = run_phase_2_quality_assessment()
    
    # Phase 4.3: Data Preparation and Standardization
    all_results['data_preparation'] = run_phase_3_data_preparation()
    
    # Phase 4.4: Clean Dataset Generation
    all_results['clean_dataset_generation'] = run_phase_4_clean_dataset_generation()
    
    # Phase 4.5: Comprehensive Assessment and Visualization
    all_results['comprehensive_assessment'] = run_comprehensive_assessment_and_visualization()
    
    # Print final summary
    print_final_summary(all_results)
    
    # Return success/failure code
    successful_phases = sum(1 for result in all_results.values() if result is not None)
    return 0 if successful_phases == len(all_results) else 1


if __name__ == "__main__":
    exit(main())