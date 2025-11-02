# Tasks Directory

This directory contains all task execution scripts and utilities for the Multimodal Fake News Detection project.

## Task Scripts

### Core Task Execution Scripts
- `run_task1_image_catalog.py` - Task 1: Image Catalog Creation and ID Mapping
- `run_task2_text_integration.py` - Task 2: Text Data Integration and Processing
- `run_task3_comment_integration.py` - Task 3: Comments Integration and Social Data
- `run_task4_comprehensive_data_quality.py` - Task 4: Comprehensive Data Quality Assessment
- `run_task4_data_quality_assessment.py` - Task 4: Alternative Data Quality Assessment
- `run_task5_social_engagement_analysis.py` - Task 5: Social Engagement Analysis
- `run_task7__visual_feature_engineering.py` - Task 7: Visual Feature Engineering

### Dashboard and Utility Scripts
- `dashboard_data_loader.py` - Dashboard data processing and integration utility
- `chart_generator.py` - Dashboard chart generation utility
- `run_dashboard_tasks.py` - Dashboard task execution script
- `data_preparation_standardization.py` - Data preparation and standardization utility
- `data_quality_assessment.py` - Data quality assessment utility
- `task_template.py` - Template for creating new task scripts

## Usage

### Running Individual Tasks
```bash
# Run from project root directory
python tasks/run_task1_image_catalog.py
python tasks/run_task2_text_integration.py
python tasks/run_task3_comment_integration.py
# ... etc
```

### Running Dashboard Tasks
```bash
python tasks/run_dashboard_tasks.py
```

### Generating Dashboard Data
```bash
python tasks/dashboard_data_loader.py
python tasks/chart_generator.py
```

## Task Dependencies

1. **Task 1** (Image Catalog) - No dependencies
2. **Task 2** (Text Integration) - Depends on Task 1
3. **Task 3** (Comment Integration) - Depends on Tasks 1 & 2
4. **Task 4** (Data Quality) - Depends on Tasks 1, 2 & 3
5. **Task 5** (Social Analysis) - Depends on Tasks 1, 2, 3 & 4
6. **Task 7** (Dashboard Enhancement) - Depends on Tasks 1-5

## Output Locations

- **Analysis Results**: `../analysis_results/`
- **Processed Data**: `../processed_data/`
- **Reports**: `../reports/`
- **Visualizations**: `../visualizations/`
- **Logs**: `../logs/`

## Configuration

Most tasks use the main `config.yaml` and `.env` files in the project root for configuration settings.