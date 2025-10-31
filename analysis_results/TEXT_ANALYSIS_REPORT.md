# Text Analysis Report

**Analysis Date:** 2025-10-31 12:08:53

## Dataset Overview

- **Total Records:** 9,837
- **True Content:** 5,997
- **False Content:** 3,840

## Linguistic Features Summary

### Text Length Analysis

**True Content:**
- Average character length: 34.2
- Average word count: 6.1
- Sample count: 5,997

**False Content:**
- Average character length: 53.9
- Average word count: 9.5
- Sample count: 3,840

### Readability Analysis

**True Content:**
- Flesch Reading Ease: 58.9
- Flesch-Kincaid Grade: 6.7
- Sample size: 259

**False Content:**
- Flesch Reading Ease: 63.6
- Flesch-Kincaid Grade: 6.7
- Sample size: 197

## Authenticity Patterns

### Key Findings

- False content is on average 19.7 characters longer than true content
- This represents a 57.7% difference in length

### Distinctive Language Features

**True Content Characteristics:**
- discussions (score: 0.024)
- cutouts (score: 0.022)
- happy (score: 0.010)
- just (score: 0.011)

**False Content Characteristics:**
- dog (score: 0.020)
- cat (score: 0.018)
- man (score: 0.019)
- way (score: 0.011)
- looks like (score: 0.011)

## Visualizations

The following visualizations have been generated:
- `text_basic_statistics.png` - Basic text statistics comparison
- `text_readability_metrics.png` - Readability analysis
- `text_sentiment_analysis.png` - Sentiment comparison
- `text_keyword_analysis.png` - Top keywords by category
- `text_distinguishing_features.png` - Key distinguishing features

## Methodology

This analysis used the following approaches:
- **Linguistic Feature Extraction:** Character/word counts, readability metrics, sentiment analysis
- **Pattern Detection:** TF-IDF analysis, keyword frequency
- **Statistical Comparison:** Category differences analysis
- **Visualization:** Comparative charts, distribution plots
