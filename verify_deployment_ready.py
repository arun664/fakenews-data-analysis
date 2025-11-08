"""
Verify Deployment Readiness

This script verifies that all required files are present and the deployment
size is within acceptable limits before pushing to GitHub.
"""

import json
from pathlib import Path
import sys

def check_dashboard_data():
    """Verify all required dashboard data files exist"""
    print("=" * 70)
    print("DEPLOYMENT READINESS CHECK")
    print("=" * 70)
    print()
    
    dashboard_dir = Path('analysis_results/dashboard_data')
    
    if not dashboard_dir.exists():
        print("❌ ERROR: Dashboard data directory not found!")
        print(f"   Expected: {dashboard_dir}")
        print()
        print("   Run: python tasks/generate_all_dashboard_data.py")
        return False
    
    # Required files
    required_files = [
        'visual_features_summary.json',
        'linguistic_features_summary.json',
        'social_engagement_summary.json',
        'dataset_overview_summary.json',
        'authenticity_analysis_summary.json'
    ]
    
    print("Checking required files:")
    all_present = True
    
    for filename in required_files:
        filepath = dashboard_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024 / 1024
            print(f"  ✓ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {filename} - MISSING!")
            all_present = False
    
    print()
    
    if not all_present:
        print("❌ ERROR: Some required files are missing!")
        print()
        print("   Run: python tasks/generate_all_dashboard_data.py")
        return False
    
    # Check total size
    total_size = 0
    file_count = 0
    
    for json_file in dashboard_dir.glob('*.json'):
        total_size += json_file.stat().st_size
        file_count += 1
    
    total_size_mb = total_size / 1024 / 1024
    
    print(f"Total dashboard data:")
    print(f"  Files: {file_count}")
    print(f"  Size: {total_size_mb:.2f} MB")
    print()
    
    # Size check
    max_size_mb = 50
    if total_size_mb > max_size_mb:
        print(f"⚠️  WARNING: Total size ({total_size_mb:.2f} MB) exceeds {max_size_mb} MB target")
        print()
        return False
    else:
        print(f"✓ Size is within {max_size_mb} MB target")
        print()
    
    # Verify JSON validity
    print("Verifying JSON files:")
    all_valid = True
    
    for json_file in sorted(dashboard_dir.glob('*.json'))[:5]:  # Check first 5
        try:
            with open(json_file) as f:
                data = json.load(f)
            print(f"  ✓ {json_file.name} - Valid JSON")
        except json.JSONDecodeError as e:
            print(f"  ❌ {json_file.name} - Invalid JSON: {e}")
            all_valid = False
    
    print()
    
    if not all_valid:
        print("❌ ERROR: Some JSON files are invalid!")
        return False
    
    # Check .gitignore
    gitignore = Path('.gitignore')
    if gitignore.exists():
        content = gitignore.read_text()
        if 'processed_data/' in content and '*.parquet' in content:
            print("✓ .gitignore properly configured to exclude large files")
        else:
            print("⚠️  WARNING: .gitignore may not be properly configured")
    else:
        print("⚠️  WARNING: .gitignore not found")
    
    print()
    
    # Final verdict
    print("=" * 70)
    print("✅ DEPLOYMENT READY!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. git add .")
    print("  2. git commit -m 'Add lightweight dashboard data for deployment'")
    print("  3. git push origin main")
    print()
    print("Then deploy on Streamlit Cloud:")
    print("  https://share.streamlit.io/")
    print()
    
    return True


if __name__ == "__main__":
    success = check_dashboard_data()
    sys.exit(0 if success else 1)
