# Multimodal Fake News Detection using Data Mining Techniques

## 🚀 Live Dashboard

**View the interactive dashboard here:** [https://fakenews-data-analysis.streamlit.app/](https://fakenews-data-analysis.streamlit.app/)

---

## 📋 Overview

This project provides a comprehensive multimodal analysis system for fake news detection, processing 620,665+ social media posts with text, images, and social engagement data. The system consists of two phases:

1. **Phase 1:** Data processing and analysis (Python tasks)
2. **Phase 2:** Interactive dashboard (Streamlit application)

---

## 🎯 System Requirements

### Hardware Requirements
- **RAM:** 16GB minimum (32GB recommended)
- **Storage:** 120GB free space minimum (for dataset + processing)
  - Input data: ~111 GB
  - Processing output: ~3 GB
  - Temporary files: ~5 GB
- **CPU:** Multi-core processor (4+ cores recommended)
- **OS:** Windows, macOS, or Linux

### Software Requirements
- **Python:** 3.8 or higher
- **Git:** For cloning the repository

---

## 📥 Dataset Download

### Source
Download the **Fakeddit dataset** from GitHub:
- **Repository:** https://github.com/entitize/Fakeddit

### Required Files
Download the following files:
1. **multimodal_train.tsv** (148 MB)
2. **multimodal_validate.tsv** (16 MB)
3. **multimodal_test_public.tsv** (16 MB)
4. **all_comments.tsv** (1.9 GB)
5. **public_image_set/** (107 GB - 773,563 images)

### Dataset Statistics
- **Total Posts:** 1,063,106
- **Images:** 773,563 files
- **Comments:** 13,800,000+
- **Total Input Size:** ~109 GB (TSV files + images)

---

## 📁 Project Structure

After cloning the repository and downloading the dataset, organize as follows:

```
data-mining-project/
│
├── multimodal_train.tsv              # Place downloaded file here
├── multimodal_validate.tsv           # Place downloaded file here
├── multimodal_test_public.tsv        # Place downloaded file here
├── all_comments.tsv                  # Place downloaded file here
├── public_image_set/                 # Extract images here
│   ├── 0a1b2c.jpg
│   ├── 0a1b2d.jpg
│   └── ... (1.5M+ images)
│
├── .env                              # Create from sample-env.txt
├── sample-env.txt                    # Template for .env
├── requirements.txt                  # Python dependencies
├── app.py                            # Streamlit dashboard
│
├── tasks/                            # Processing scripts
│   ├── generate_all_dashboard_data.py
│   ├── run_task1_image_catalog.py
│   ├── run_task2_text_integration.py
│   └── ... (other task files)
│
├── src/                              # Dashboard source code
│   ├── pages/                        # Dashboard pages
│   └── utils/                        # Utility functions
│
├── processed_data/                   # Generated during Phase 1
│   ├── clean_datasets/
│   ├── visual_features/
│   ├── linguistic_features/
│   └── social_engagement/
│
├── analysis_results/                 # Generated during Phase 1
│   └── dashboard_data/               # JSON files for dashboard
│
├── visualizations/                   # Generated charts
└── reports/                          # Generated reports
```

---

## ⚙️ Initial Setup

### Step 1: Clone Repository

```bash
git clone <your-repository-url>
cd data-mining-project
```

### Step 2: Place Dataset Files

**IMPORTANT:** Based on the `.env` configuration, dataset files should be placed in the **parent directory** (`../`) of the project:

```bash
# Correct folder structure:
parent-folder/
├── data-mining-project/          # This repository
│   ├── .env
│   ├── app.py
│   ├── tasks/
│   └── src/
│
├── multimodal_train.tsv          # 148 MB
├── multimodal_validate.tsv       # 16 MB
├── multimodal_test_public.tsv    # 16 MB
├── all_comments.tsv              # 1.9 GB
└── public_image_set/             # 107 GB (773,563 images)
    ├── 0a1b2c.jpg
    ├── 0a1b2d.jpg
    └── ...
```

**Alternative:** If you prefer to keep dataset files inside the project folder, update the `.env` file:

```bash
# Edit .env and change paths from:
TRAIN_TSV_PATH=../multimodal_train.tsv

# To:
TRAIN_TSV_PATH=multimodal_train.tsv

# Do this for all dataset paths
```

### Step 3: Configure Environment

Create `.env` file from the template:

```bash
# Copy the sample environment file
cp sample-env.txt .env
```

**Default Configuration:** The `.env` file is pre-configured to look for dataset files in the parent directory (`../`):

```bash
# Default paths in .env (dataset in parent folder)
TRAIN_TSV_PATH=../multimodal_train.tsv
VALIDATION_TSV_PATH=../multimodal_validate.tsv
TEST_TSV_PATH=../multimodal_test_public.tsv
COMMENTS_TSV_PATH=../all_comments.tsv
IMAGES_FOLDER_PATH=../public_image_set
```

**Verify Dataset Location:**

```bash
# Check if files are accessible (Windows)
dir ..\multimodal_train.tsv
dir ..\public_image_set

# Check if files are accessible (Linux/Mac)
ls -lh ../multimodal_train.tsv
ls -lh ../public_image_set | head
```

**If files are in project root instead:** Edit `.env` and remove `../`:

```bash
# For files inside project folder
TRAIN_TSV_PATH=multimodal_train.tsv
VALIDATION_TSV_PATH=multimodal_validate.tsv
TEST_TSV_PATH=multimodal_test_public.tsv
COMMENTS_TSV_PATH=all_comments.tsv
IMAGES_FOLDER_PATH=public_image_set
```

### Step 4: Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Key Dependencies:**
- pandas, numpy - Data processing
- scikit-learn - Machine learning
- opencv-python, Pillow - Image processing
- nltk, textblob - Text analysis
- plotly - Visualizations
- streamlit - Dashboard framework

---

## 🔬 PHASE 1: Data Processing & Analysis

This phase processes the raw dataset and generates analysis results. **Estimated time: 26-36 hours (CPU only)**

**Note:** GPU acceleration can reduce time to 14-22 hours, but is not required.

### Overview

Phase 1 consists of running **9 core Python task scripts in sequence** to generate analysis results for the dashboard:

1. **Image Catalog** (5-10 min) - Map 773,563 images to text records
2. **Text Integration** (3-5 min) - Process and clean 682,661 text records  
3. **Comments Integration** (5-8 min) - Integrate 13.8M comments from 1.9 GB file
4. **Data Quality & Preparation** (8-12 min) - Clean datasets, quality checks, and leakage detection
5. **Social Engagement Analysis** (8-12 min) - Analyze comment sentiment and engagement patterns
6. **Visual Feature Extraction** (25-35 hours) - Process 773,563 images with 23 features each
7. **Linguistic Pattern Mining** (15-25 min) - Extract 26 linguistic features from 682,661 text records
8. **Final Integration** (3-5 min) - Merge all features into final multimodal dataset
9. **Dashboard Data Generation** (30-60 sec) - Create compressed JSON summaries (0.30 MB)

### Data Quality & Leakage Detection

**Important:** The data preparation step includes comprehensive data quality assessment:

- **Data Leakage Detection:** Checks for cross-split contamination
- **Duplicate Detection:** Identifies duplicate records across train/val/test splits
- **Temporal Validation:** Ensures chronological consistency
- **Author Leakage:** Prevents user information leakage across splits
- **Cross-Modal Consistency:** Validates image-text-comment mappings

These checks ensure scientific rigor and prevent data leakage that could inflate model performance.

### Step-by-Step Execution (Required Order)

Run these **9 core tasks** one after another in this exact order:

```bash
# Step 1: Image Catalog Creation (5-10 min)
# Creates image-to-text ID mappings for 773,563 images
python tasks/run_task1_image_catalog.py

# Step 2: Text Data Integration (3-5 min)
# Processes and cleans 682,661 text records across train/val/test splits
python tasks/run_task2_text_integration.py

# Step 3: Comments Integration (5-8 min)
# Integrates 13.8M comments from 1.9 GB file
python tasks/run_task3_comment_integration.py

# Step 4: Data Quality & Preparation (8-12 min)
# Comprehensive data quality checks, leakage detection, and clean dataset generation
python tasks/run_task4_comprehensive_data_quality.py

# Step 5: Social Engagement Analysis (8-12 min)
# Analyzes comment sentiment and engagement patterns
python tasks/run_task5_social_engagement_analysis.py

# Step 6: Visual Feature Engineering (25-35 hours - LONGEST STEP)
# Processes 773,563 images with 23 features each
python tasks/run_task8_visual_feature_engineering.py

# Step 7: Linguistic Pattern Mining (15-25 min)
# Processes 682,661 text records with 26 linguistic features
python tasks/run_task9_linguistic_pattern_mining.py

# Step 8: Final Integration (3-5 min)
# Integrates all features into final multimodal dataset
python tasks/run_task15_final_integration.py

# Step 9: Generate Dashboard Data (30-60 sec)
# Creates compressed JSON summaries (0.30 MB) for dashboard
python tasks/generate_all_dashboard_data.py
```

**Important:** Each task must complete successfully before running the next one. Tasks depend on outputs from previous tasks.

**Optional Advanced Analysis Tasks** (not required for dashboard):
- `run_task6_comprehensive_visualization.py` - Additional visualizations
- `run_task10_multimodal_clustering.py` - Clustering analysis
- `run_task11_association_rule_mining.py` - Association rule mining
- `run_task12_cross_modal_analysis.py` - Cross-modal comparative analysis

### What Each Task Does

| Step | Task | Input Data | Output Location | Time (CPU) | Time (GPU) |
|------|------|-----------|-----------------|------------|------------|
| 1 | **run_task1_image_catalog** | 773,563 images | `analysis_results/image_catalog/` | 5-10 min | 5-10 min |
| 2 | **run_task2_text_integration** | 2.1 GB TSV (682,661 records) | `processed_data/text_data/` | 3-5 min | 3-5 min |
| 3 | **run_task3_comment_integration** | 1.9 GB TSV (13.8M comments) | `processed_data/comments/` | 5-8 min | 5-8 min |
| 4 | **run_task4_comprehensive_data_quality** | All processed data | `processed_data/clean_datasets/` | 8-12 min | 8-12 min |
| 5 | **run_task5_social_engagement_analysis** | Comments + text data | `processed_data/social_engagement/` | 8-12 min | 8-12 min |
| 6 | **run_task8_visual_feature_engineering** | 773,563 images (107 GB) | `processed_data/visual_features/` | **25-35 hrs** | **13-20 hrs** |
| 7 | **run_task9_linguistic_pattern_mining** | 682,661 text records | `processed_data/linguistic_features/` | 15-25 min | 15-25 min |
| 8 | **run_task15_final_integration** | All feature datasets | `processed_data/final_integrated_dataset/` | 3-5 min | 3-5 min |
| 9 | **generate_all_dashboard_data** | All processed data | `analysis_results/dashboard_data/` | 30-60 sec | 30-60 sec |
| | **TOTAL PIPELINE TIME** | | | **26-36 hrs** | **14-22 hrs** |

**Key Processing Details:**

**Step 6 - Visual Feature Engineering (Bottleneck):**
- Extracts 23 features per image: brightness, contrast, sharpness, texture, manipulation scores, etc.
- Processing rate: ~350-500 images/min (CPU) or ~650-1,000 images/min (GPU)
- Memory usage: 4-8 GB RAM
- GPU provides 1.8-2× speedup using CUDA acceleration

**Step 7 - Linguistic Pattern Mining:**
- Extracts 26 features per text: readability, sentiment, linguistic patterns, clickbait indicators
- Processing rate: ~700-1,000 records/sec with parallel processing
- Uses multi-core CPU parallelization (8+ cores recommended)

**Steps 1-5, 8-9 - Data Preparation & Integration:**
- Fast I/O operations with minimal computation
- Combined time: ~30-40 minutes
- Primarily disk-bound operations (SSD recommended)

### Expected Output (Final Step)

After running `generate_all_dashboard_data.py`, you should see:

```
======================================================================
GENERATING COMPLETE DASHBOARD DATA
======================================================================

INFO: Creating visual features summary from FULL dataset...
INFO: Loaded 618828 visual feature records (using ALL data)
INFO: ✓ Visual features summary created: 0.12 MB

INFO: Creating linguistic features summary from FULL dataset...
INFO: Loaded 682661 linguistic feature records (using ALL data)
INFO: ✓ Linguistic features summary created: 0.11 MB

INFO: Creating social engagement summary from FULL dataset...
INFO: ✓ Social engagement summary created: 0.02 MB

INFO: Creating dataset overview summary...
INFO: ✓ Dataset overview summary created: 0.00 MB

======================================================================
SUMMARY
======================================================================
Total files: 12
Total dashboard data size: 0.30 MB
✓ SUCCESS! Data size is under 50 MB target
Completed in 25.0 seconds
======================================================================
```

### Monitoring Progress

Each task will:
- Show progress bars for long operations
- Log completion status
- Create output files in `processed_data/` or `analysis_results/`
- Display error messages if something fails

**If a task fails:**
1. Check the error message
2. Verify input files exist (from previous tasks)
3. Check available disk space and RAM
4. Review the `.env` file paths
5. Re-run the failed task after fixing the issue

### Verify Phase 1 Completion

Check that the following files were created:

```bash
# List generated dashboard data files
ls -lh analysis_results/dashboard_data/

# Expected files (total ~0.30 MB):
# - dataset_overview_summary.json
# - linguistic_features_summary.json
# - visual_features_summary.json
# - social_engagement_summary.json
# - clustering_dashboard_data.json
# - association_mining_dashboard_data.json
# - authenticity_analysis_summary.json
# - processed_dashboard_data.json
# - (and 4 more JSON files)
```

### Total Processing Time Summary

| Configuration | Step 6 (Visual) | Steps 1-5, 7-9 | Total Time |
|---------------|-----------------|----------------|------------|
| **CPU Only (Standard)** | 25-35 hours | 45-60 min | **26-36 hours** |
| **With GPU (Optional)** | 13-20 hours | 45-60 min | **14-22 hours** |

**Time Breakdown by Phase:**

**CPU-Only Processing (26-36 hours total):**
- **Visual Feature Extraction (Step 6):** 25-35 hours (96% of time)
  - Bottleneck: Processing 773,563 images at ~350-500 images/min
  - Extracts 23 features per image (brightness, sharpness, texture, manipulation scores)
- **Linguistic Pattern Mining (Step 7):** 15-25 minutes (2% of time)
  - Parallel processing 682,661 text records at ~700-1,000 records/sec
- **Data Preparation (Steps 1-5):** 30-40 minutes (2% of time)
  - Image catalog, text/comment integration, quality checks, social engagement
- **Integration & Dashboard (Steps 8-9):** 4-6 minutes (<1% of time)
  - Final dataset integration and JSON summary generation

**GPU-Accelerated Processing (14-22 hours total):**
- **Visual Feature Extraction (Step 6):** 13-20 hours (95% of time)
  - GPU acceleration: ~650-1,000 images/min (1.8-2× faster than CPU)
  - Requires NVIDIA GPU with CUDA support
- **All Other Steps:** Same as CPU (45-60 minutes combined)

**Critical Planning Notes:** 
- **Plan for 1-2 day processing** - full pipeline takes 27-38 hours on CPU
- **Run overnight or over weekend** - visual processing is very time-intensive
- **GPU is optional** - provides 1.8-2× speedup but not required
- **Stable environment required** - interruptions require restarting failed tasks
- **Monitor disk space** - ensure 120GB+ free space throughout processing

**System Requirements:**
- **CPU:** 4+ cores recommended (8+ cores optimal for parallel processing)
- **RAM:** 16GB minimum, 32GB recommended for clustering/mining tasks
- **Storage:** SSD strongly recommended (10× faster I/O for 773K+ image files)
- **GPU (Optional):** NVIDIA GPU with CUDA support reduces time by ~40-50%

**Performance Tips:**
- Close other applications to free up RAM during processing
- Use SSD for faster image loading (Step 2 reads 773K+ files)
- Enable all CPU cores for parallel processing (Steps 3, 6, 7)
- Monitor system resources to avoid memory swapping

### Storage Requirements After Phase 1

| Component | Size | Location |
|-----------|------|----------|
| **INPUT (Required)** | | |
| Raw Dataset (TSV files) | 2.1 GB | `../multimodal_*.tsv` |
| Raw Comments | 1.9 GB | `../all_comments.tsv` |
| Raw Images | 107 GB | `../public_image_set/` |
| **Subtotal Input** | **~111 GB** | |
| | | |
| **OUTPUT (Generated)** | | |
| Processed Data | 2.75 GB | `processed_data/` |
| Analysis Results | 0.11 GB | `analysis_results/` |
| Visualizations | 0.08 GB | `visualizations/` |
| Reports | <0.01 GB | `reports/` |
| Dashboard Data | 0.30 MB | `analysis_results/dashboard_data/` |
| **Subtotal Output** | **~3 GB** | |
| | | |
| **TOTAL STORAGE REQUIRED** | **~114 GB** | |

---

## 🎨 PHASE 2: Interactive Dashboard

This phase launches the Streamlit dashboard to visualize and explore the analysis results.

### Prerequisites

✅ Phase 1 must be completed  
✅ Dashboard data files exist in `analysis_results/dashboard_data/`  
✅ All dependencies installed

### Launch Dashboard

```bash
# Start the Streamlit application
streamlit run app.py
```

### What Happens Next

1. **Streamlit starts** and opens your default browser
2. **Dashboard loads** at `http://localhost:8501`
3. **Navigation sidebar** appears on the left
4. **Data loads** in <1 second per page

### Dashboard Features

The dashboard includes **9 comprehensive analysis sections**:

#### 1. 📊 Dataset Overview
- Total records: 620,665
- Class distribution (57.5% fake, 42.5% real)
- Data splits visualization
- Feature availability analysis

#### 2. 📝 Text Patterns
- Linguistic feature distributions
- Readability analysis (Flesch scores)
- Sentiment analysis
- Statistical comparisons

#### 3. 🖼️ Visual Patterns
- Image quality metrics
- Manipulation detection
- Brightness, sharpness, entropy analysis
- Visual feature distributions

#### 4. 💬 Social Patterns
- Engagement score analysis
- Comment count distributions
- Statistical significance tests
- Community detection patterns

#### 5. 😊 Sentiment Analysis
- Emotion distribution
- Polarity vs subjectivity
- Sentiment by authenticity
- Radar charts and heatmaps

#### 6. ⏰ Temporal Trends
- Posting time patterns
- Day-of-week analysis
- Temporal authenticity patterns
- Automation detection

#### 7. 🔬 Advanced Analytics
- **Clustering Analysis**
  - K-means (k=6)
  - Hierarchical (k=8)
  - Silhouette score: 0.0217
- **Association Rule Mining**
  - 969 rules discovered
  - Top fake indicators (4.21× lift)
  - Multimodal patterns

#### 8. 🔗 Cross-Modal Insights
- Text-visual correlations
- Multimodal feature interactions
- Cross-modal patterns

#### 9. ✅ Authenticity Analysis
- Comprehensive fake vs real comparison
- Statistical summaries
- Key discriminative features

### Navigation

Use the **sidebar** to navigate between sections:
- Click on any section name
- Sections load instantly (<1 second)
- All visualizations are interactive
- Hover over charts for details

### Performance

- **Page Load Time:** <1 second
- **Chart Rendering:** <0.5 seconds
- **Data Coverage:** 100% (no sampling)
- **Dashboard Size:** 0.30 MB
- **Memory Usage:** <4 GB

### Streamlit Configuration

The dashboard uses the following Streamlit settings:

```python
# Configured in app.py
st.set_page_config(
    page_title="Multimodal Fake News Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

**Cache Settings:**
- Data cached for 10 minutes (600 seconds)
- Automatic cache refresh
- Press `C` in dashboard to clear cache manually

---

## 🔍 Verification & Testing

### Input Files Verification

Before starting Phase 1, verify all input files are accessible:

```bash
# Run verification script
python -c "
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

print('='*60)
print('INPUT FILES VERIFICATION')
print('='*60)

# Check TSV files
tsv_files = [
    ('TRAIN_TSV_PATH', 'Train dataset'),
    ('VALIDATION_TSV_PATH', 'Validation dataset'),
    ('TEST_TSV_PATH', 'Test dataset'),
    ('COMMENTS_TSV_PATH', 'Comments dataset')
]

all_ok = True
for env_var, desc in tsv_files:
    path = os.getenv(env_var, '')
    exists = os.path.exists(path)
    status = '✓' if exists else '✗'
    size = f'{os.path.getsize(path)/1024/1024:.0f} MB' if exists else 'N/A'
    print(f'{status} {desc}: {path} ({size})')
    all_ok = all_ok and exists

# Check images folder
img_path = os.getenv('IMAGES_FOLDER_PATH', '')
img_exists = os.path.exists(img_path)
status = '✓' if img_exists else '✗'
if img_exists:
    img_count = len([f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))])
    print(f'{status} Images folder: {img_path} ({img_count:,} files)')
else:
    print(f'{status} Images folder: {img_path} (NOT FOUND)')
all_ok = all_ok and img_exists

print('='*60)
if all_ok:
    print('✅ All input files found! Ready for Phase 1.')
else:
    print('❌ Some files missing. Check paths in .env file.')
print('='*60)
"
```

### Phase 1 Verification

After running Phase 1, verify dashboard data was generated:

```bash
# Check if all dashboard data files exist
python -c "
import os
from pathlib import Path

data_dir = Path('analysis_results/dashboard_data')
required_files = [
    'dataset_overview_summary.json',
    'linguistic_features_summary.json',
    'visual_features_summary.json',
    'social_engagement_summary.json',
    'clustering_dashboard_data.json',
    'association_mining_dashboard_data.json'
]

print('Checking Phase 1 completion...')
all_exist = True
for file in required_files:
    exists = (data_dir / file).exists()
    status = '✓' if exists else '✗'
    print(f'{status} {file}')
    all_exist = all_exist and exists

if all_exist:
    print('\n✅ Phase 1 Complete! Ready for Phase 2.')
else:
    print('\n❌ Phase 1 Incomplete. Run: python tasks/generate_all_dashboard_data.py')
"
```

### Phase 2 Verification

Once the dashboard is running:

1. **Check Dataset Overview:**
   - Navigate to "Dataset Overview"
   - Verify: Total Records = 620,665
   - Verify: Fake = 356,715 (57.5%)

2. **Check Text Patterns:**
   - Navigate to "Text Patterns"
   - Verify: Histograms display
   - Verify: Statistics show fake vs real

3. **Check Visual Patterns:**
   - Navigate to "Visual Patterns"
   - Verify: Sharpness difference ~115%
   - Verify: Heatmaps display

4. **Check Advanced Analytics:**
   - Navigate to "Advanced Analytics"
   - Verify: Clustering tab loads
   - Verify: Association rules tab loads

---

## ⚠️ Troubleshooting

### Phase 1 Issues

#### Issue: "File not found" error
**Solution:**
```bash
# Verify dataset files are in correct location
ls -l multimodal_train.tsv
ls -l multimodal_validate.tsv
ls -l multimodal_test_public.tsv
ls -l all_comments.tsv
ls -l public_image_set/

# Check .env file paths
cat .env
```

#### Issue: "Out of memory" error
**Solution:**
- Close other applications
- Increase system RAM
- Process in smaller batches (modify BATCH_SIZE in .env)

#### Issue: Processing takes too long
**Solution:**
- Use the quick start script: `python tasks/generate_all_dashboard_data.py`
- Ensure multi-core processing is enabled in .env
- Check system resources (CPU, RAM usage)

### Phase 2 Issues

#### Issue: "Module not found" error
**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Verify streamlit installation
streamlit --version
```

#### Issue: Dashboard won't start
**Solution:**
```bash
# Check if port 8501 is available
# Windows:
netstat -ano | findstr :8501

# Linux/Mac:
lsof -i :8501

# Use different port if needed:
streamlit run app.py --server.port 8502
```

#### Issue: Dashboard loads slowly
**Solution:**
```bash
# Clear Streamlit cache
streamlit cache clear

# Restart dashboard
# Press Ctrl+C to stop
streamlit run app.py
```

#### Issue: Visualizations not displaying
**Solution:**
- Check browser console for errors (F12)
- Try different browser (Chrome recommended)
- Clear browser cache
- Verify dashboard data files exist

---

## 📊 Expected Results

### Key Statistics

After completing both phases, you should see:

**Dataset Overview:**
- Total: 620,665 records
- Fake: 356,715 (57.5%)
- Real: 263,950 (42.5%)

**Text Analysis:**
- Fake content 32% shorter
- 454% more exclamation marks
- Lower readability (5.8 vs 6.8 grade level)

**Visual Analysis:**
- Fake images 115% sharper (over-processing)
- 1.4% darker
- 2.8% lower entropy

**Social Engagement:**
- Fake content: 72% fewer comments
- Fake content: 65% lower engagement scores

**Clustering:**
- Silhouette score: 0.0217 (poor separation)
- 6 K-means clusters
- 2 extreme fake clusters (>99% fake)

**Association Rules:**
- 969 rules discovered
- Strongest indicator: 4.21× lift
- Highest confidence: 93.8%

---

## 📞 Additional Resources

### Documentation
- **Task Documentation:** `tasks/README.md`
- **Environment Template:** `sample-env.txt`

### Dataset Information
- **Fakeddit Repository:** https://github.com/entitize/Fakeddit
- **Dataset Paper:** https://arxiv.org/abs/1911.03854

### Streamlit Documentation
- **Official Docs:** https://docs.streamlit.io
- **API Reference:** https://docs.streamlit.io/library/api-reference

---

## 📄 Dataset Citation

If using the Fakeddit dataset, please cite:

```bibtex
@article{nakamura2019fakeddit,
  title={Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection},
  author={Nakamura, Kai and Levy, Sharon and Wang, William Yang},
  journal={arXiv preprint arXiv:1911.03854},
  year={2019}
}
```

---

**Last Updated:** November 2025  
**Status:** Production Ready ✅
