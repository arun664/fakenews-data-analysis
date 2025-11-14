#!/usr/bin/env python3
"""
Task Template with Automatic Cleanup and Performance Tracking
============================================================

Template for creating new tasks with built-in cleanup and performance tracking.
Copy this template and modify for new tasks.

Author: Graduate Research Project
Date: November 2025
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import json
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add dashboard to Python path
project_root = Path(__file__).parent
dashboard_path = project_root / 'dashboard'
sys.path.insert(0, str(dashboard_path))

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Import utilities
from utils.task_cleanup import clean_task_results
from utils.performance_tracker import get_task_tracker

# Setup logging with performance tracking
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('taskX_template.log'),  # Change X to task number
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TaskXProcessor:  # Change X to task number
    """
    Template Task Processor with automatic cleanup and performance tracking
    """
    
    def __init__(self, task_id: str, task_name: str):
        self.task_id = task_id
        self.task_name = task_name
        
        # Initialize performance tracking
        self.performance = get_task_tracker(task_id, task_name)
        
        # Define output directories
        self.output_dir = Path(f'processed_data/{task_id}_data')  # Customize
        self.analysis_dir = Path(f'analysis_results/{task_id}_analysis')  # Customize
        self.viz_dir = Path(f'visualizations/{task_id}_viz')  # Customize
        self.reports_dir = Path('reports')
        
        # Clear previous results before starting
        self._clear_previous_results()
        
        # Create directories
        for dir_path in [self.output_dir, self.analysis_dir, self.viz_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {}
    
    def _clear_previous_results(self):
        """Clear previous task results to ensure clean run"""
        clean_task_results(self.task_id, self.task_name)
    
    def run_complete_processing(self) -> Dict:
        """Run the complete task processing pipeline"""
        
        self.performance.start_task()
        
        try:
            # Phase 1: Data Loading
            self.performance.start_phase("Load Data")
            self.load_data()
            
            # Phase 2: Data Processing
            self.performance.start_phase("Process Data")
            self.process_data()
            
            # Phase 3: Analysis
            self.performance.start_phase("Analyze Data")
            self.analyze_data()
            
            # Phase 4: Generate Visualizations
            self.performance.start_phase("Create Visualizations")
            self.create_visualizations()
            
            # Phase 5: Generate Report
            self.performance.start_phase("Generate Report")
            self.generate_report()
            
            # Phase 6: Save Results
            self.performance.start_phase("Save Results")
            self.save_results()
            
            return self.performance.end_task(success=True)
            
        except Exception as e:
            logger.error(f"Error in {self.task_name}: {e}")
            import traceback
            traceback.print_exc()
            return self.performance.end_task(success=False, error_message=str(e))
    
    def load_data(self):
        """Load data for processing"""
        logger.info("Loading data...")
        
        # TODO: Implement data loading logic
        # Example:
        # self.data = pd.read_csv('data.csv')
        # self.performance.update_processing_stats(records_processed=len(self.data))
        
        logger.info("[OK] Data loading completed")
    
    def process_data(self):
        """Process the loaded data"""
        logger.info("Processing data...")
        
        # TODO: Implement data processing logic
        
        logger.info("[OK] Data processing completed")
    
    def analyze_data(self):
        """Analyze the processed data"""
        logger.info("Analyzing data...")
        
        # TODO: Implement analysis logic
        
        logger.info("[OK] Data analysis completed")
    
    def create_visualizations(self):
        """Create visualizations for the analysis"""
        logger.info("Creating visualizations...")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # TODO: Implement visualization logic
            # Example:
            # plt.figure(figsize=(10, 6))
            # plt.plot(data)
            # plt.savefig(self.viz_dir / 'analysis_chart.png')
            # plt.close()
            
            logger.info("[OK] Visualizations created")
            
        except Exception as e:
            logger.warning(f"Visualization creation failed: {e}")
    
    def generate_report(self):
        """Generate comprehensive report"""
        logger.info("Generating report...")
        
        report_content = f"""# {self.task_name} Report

## Executive Summary

This report presents the analysis results for {self.task_name}.

## Key Findings

TODO: Add key findings

## Methodology

TODO: Describe methodology

## Results

TODO: Present results

## Conclusion

TODO: Add conclusion

---

*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Task execution time: {self.performance.get_current_stats().get('current_duration_formatted', 'N/A')}*
"""
        
        # Save report
        report_path = self.reports_dir / f'{self.task_id}_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"[OK] Report saved: {report_path}")
    
    def save_results(self):
        """Save processed results"""
        logger.info("Saving results...")
        
        # TODO: Implement result saving logic
        # Example:
        # results_path = self.analysis_dir / 'results.json'
        # with open(results_path, 'w', encoding='utf-8') as f:
        #     json.dump(self.results, f, indent=2, default=str)
        
        # Save performance metrics
        perf_path = self.analysis_dir / f'{self.task_id}_performance_metrics.json'
        with open(perf_path, 'w', encoding='utf-8') as f:
            json.dump(self.performance.get_current_stats(), f, indent=2, default=str)
        
        logger.info(f"[OK] Performance metrics saved: {perf_path}")

def main():
    """Main execution function"""
    
    # TODO: Update these for your specific task
    TASK_ID = "taskX"  # Change to actual task ID (e.g., "task3", "task4")
    TASK_NAME = "Task X: Template Task"  # Change to actual task name
    
    print(f"{TASK_NAME}")
    print("=" * 80)
    print("This task runs independently and can execute in parallel with other tasks")
    print("=" * 80)
    
    # Create processor
    processor = TaskXProcessor(TASK_ID, TASK_NAME)  # Change class name
    
    try:
        # Run complete processing
        performance_metrics = processor.run_complete_processing()
        
        print("\n" + "=" * 80)
        print(f"{TASK_NAME.upper()} COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Results organized in:")
        print(f"  - {processor.output_dir}: Processed data")
        print(f"  - {processor.analysis_dir}: Analysis results")
        print(f"  - {processor.viz_dir}: Visualizations")
        print(f"  - {processor.reports_dir}: Reports")
        print("=" * 80)
        print(f"Total execution time: {performance_metrics['total_duration_formatted']}")
        print(f"Records processed: {performance_metrics['processing_metrics']['records_processed']:,}")
        print(f"Processing rate: {performance_metrics['processing_metrics']['processing_rate_per_second']:.1f} records/second")
        print(f"Memory usage: {performance_metrics['system_metrics']['final_memory_mb']:.1f} MB")
        print("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] {TASK_NAME} interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n[ERROR] {TASK_NAME} failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)