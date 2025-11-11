# Multimodal Fake News Detection System

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

This phase processes the raw dataset and generates analysis results. **Estimated time: 2-4 hours**

### Overview

Phase 1 consists of running Python task scripts **in sequence**. Each task builds on the previous one:

1. **Data Preparation** - Clean and standardize dataset
2. **Visual Feature Extraction** - Process 773,563 images (23 features per image)
3. **Linguistic Feature Extraction** - Analyze text (26 features per post)
4. **Social Engagement Analysis** - Process comments and engagement
5. **Clustering Analysis** - Discover patterns in multimodal data
6. **Association Rule Mining** - Find feature combinations
7. **Dashboard Data Generation** - Create compressed JSON summaries (0.30 MB)

### Step-by-Step Execution (Required Order)

Run these tasks **one after another** in this exact order:

```bash
# Step 1: Data Preparation and Cleaning (5-10 min)
python tasks/data_preparation_standardization.py

# Step 2: Visual Feature Engineering (30-60 min)
python tasks/run_task8_visual_feature_engineering.py

# Step 3: Linguistic Pattern Mining (10-15 min)
python tasks/run_task9_linguistic_pattern_mining.py

# Step 4: Social Engagement Analysis (5-10 min)
python tasks/run_task5_social_engagement_analysis.py

# Step 5: Final Integration (5-10 min)
python tasks/run_task15_final_integration.py

# Step 6: Clustering Analysis (15-20 min)
python tasks/run_task10_multimodal_clustering.py

# Step 7: Association Rule Mining (20-30 min)
python tasks/run_task11_association_rule_mining.py

# Step 8: Generate Dashboard Data (1-2 min)
# This reads from processed_data/ and creates JSON summaries
python tasks/generate_all_dashboard_data.py
```

**Important:** Each task must complete successfully before running the next one. Tasks depend on outputs from previous tasks.

### What Each Task Does

| Task | Input | Output | Time |
|------|-------|--------|------|
| **data_preparation_standardization** | Raw TSV files | `processed_data/clean_datasets/` | 5-10 min |
| **run_task8_visual_feature_engineering** | Images + clean data | `processed_data/visual_features/` | 30-60 min |
| **run_task9_linguistic_pattern_mining** | Clean datasets | `processed_data/linguistic_features/` | 10-15 min |
| **run_task5_social_engagement_analysis** | Comments + clean data | `processed_data/social_engagement/` | 5-10 min |
| **run_task15_final_integration** | All features | `processed_data/final_integrated_dataset/` | 5-10 min |
| **run_task10_multimodal_clustering** | Integrated data | `processed_data/clustering_results/` | 15-20 min |
| **run_task11_association_rule_mining** | All features | `processed_data/association_rules/` | 20-30 min |
| **generate_all_dashboard_data** | All processed_data | `analysis_results/dashboard_data/` (0.30 MB) | 1-2 min |

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

### Total Processing Time

| Phase | Time Estimate |
|-------|---------------|
| Steps 1-7 (Analysis Tasks) | 90-150 minutes |
| Step 8 (Dashboard Data) | 1-2 minutes |
| **TOTAL** | **2-4 hours** |

**Note:** Times vary based on:
- CPU speed and core count
- Available RAM
- Disk I/O speed
- System load

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
