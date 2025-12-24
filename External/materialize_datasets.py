# -*- coding: utf-8 -*-
"""
Materialize external datasets from statsmodels and sklearn

This script downloads and saves datasets from statsmodels and sklearn libraries
into the StatQA data directory structure with proper metadata.
"""

import sys
import os
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_folder_path)

import pandas as pd
import numpy as np
import path
import utils
from pathlib import Path


# Create directories if they don't exist
EXTERNAL_ORIGIN_DIR = 'Data/External Dataset/Origin/'
EXTERNAL_PROCESSED_DIR = 'Data/External Dataset/Processed/'
os.makedirs(EXTERNAL_ORIGIN_DIR, exist_ok=True)
os.makedirs(EXTERNAL_PROCESSED_DIR, exist_ok=True)


def materialize_statsmodels_datasets():
    """
    Materialize selected datasets from statsmodels.
    These datasets are well-studied and have known statistical properties.
    """
    print("[*] Materializing statsmodels datasets...")
    
    try:
        import statsmodels.api as sm
    except ImportError:
        print("[!] statsmodels not installed. Install with: pip install statsmodels")
        return []
    
    materialized = []
    
    # 1. Duncan's Occupational Prestige Data
    # Good for: regression, correlation, potential outliers
    try:
        duncan = sm.datasets.get_rdataset("Duncan", "carData")
        df = duncan.data.reset_index()
        output_path = os.path.join(EXTERNAL_ORIGIN_DIR, "duncan_prestige.csv")
        df.to_csv(output_path, index=False)
        print(f"[+] Saved: duncan_prestige.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
        materialized.append({
            'name': 'duncan_prestige',
            'path': output_path,
            'description': 'Duncan\'s occupational prestige data with income and education',
            'source': 'statsmodels/carData',
            'columns': list(df.columns),
            'n_rows': len(df),
            'suitable_for': ['regression', 'correlation', 'outlier_detection']
        })
    except Exception as e:
        print(f"[!] Could not load Duncan dataset: {e}")
    
    # 2. Longley Economic Data
    # Good for: multicollinearity, regression
    try:
        longley = sm.datasets.longley.load_pandas()
        df = longley.data
        output_path = os.path.join(EXTERNAL_ORIGIN_DIR, "longley_economic.csv")
        df.to_csv(output_path, index=False)
        print(f"[+] Saved: longley_economic.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
        materialized.append({
            'name': 'longley_economic',
            'path': output_path,
            'description': 'Longley economic data with high multicollinearity',
            'source': 'statsmodels',
            'columns': list(df.columns),
            'n_rows': len(df),
            'suitable_for': ['regression', 'multicollinearity']
        })
    except Exception as e:
        print(f"[!] Could not load Longley dataset: {e}")
    
    # 3. Grunfeld Investment Data
    # Good for: panel data, mixed effects (but we'll use it for basic regression)
    try:
        grunfeld = sm.datasets.grunfeld.load_pandas()
        df = grunfeld.data
        output_path = os.path.join(EXTERNAL_ORIGIN_DIR, "grunfeld_investment.csv")
        df.to_csv(output_path, index=False)
        print(f"[+] Saved: grunfeld_investment.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
        materialized.append({
            'name': 'grunfeld_investment',
            'path': output_path,
            'description': 'Grunfeld investment data for firms over time',
            'source': 'statsmodels',
            'columns': list(df.columns),
            'n_rows': len(df),
            'suitable_for': ['regression', 'clustered_data']
        })
    except Exception as e:
        print(f"[!] Could not load Grunfeld dataset: {e}")
    
    # 4. Stackloss Data
    # Good for: regression with outliers
    try:
        stackloss = sm.datasets.stackloss.load_pandas()
        df = stackloss.data
        output_path = os.path.join(EXTERNAL_ORIGIN_DIR, "stackloss.csv")
        df.to_csv(output_path, index=False)
        print(f"[+] Saved: stackloss.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
        materialized.append({
            'name': 'stackloss',
            'path': output_path,
            'description': 'Stack loss data from a chemical plant',
            'source': 'statsmodels',
            'columns': list(df.columns),
            'n_rows': len(df),
            'suitable_for': ['regression', 'outliers']
        })
    except Exception as e:
        print(f"[!] Could not load Stackloss dataset: {e}")
    
    # 5. Copper Data
    # Good for: regression, variance tests
    try:
        copper = sm.datasets.copper.load_pandas()
        df = copper.data
        output_path = os.path.join(EXTERNAL_ORIGIN_DIR, "copper.csv")
        df.to_csv(output_path, index=False)
        print(f"[+] Saved: copper.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
        materialized.append({
            'name': 'copper',
            'path': output_path,
            'description': 'Copper concentrations in different locations',
            'source': 'statsmodels',
            'columns': list(df.columns),
            'n_rows': len(df),
            'suitable_for': ['anova', 'variance_test']
        })
    except Exception as e:
        print(f"[!] Could not load Copper dataset: {e}")
    
    # 6. Engel Food Expenditure Data
    # Good for: regression, heteroscedasticity
    try:
        engel = sm.datasets.engel.load_pandas()
        df = engel.data
        output_path = os.path.join(EXTERNAL_ORIGIN_DIR, "engel_food.csv")
        df.to_csv(output_path, index=False)
        print(f"[+] Saved: engel_food.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
        materialized.append({
            'name': 'engel_food',
            'path': output_path,
            'description': 'Belgian household food expenditure data',
            'source': 'statsmodels',
            'columns': list(df.columns),
            'n_rows': len(df),
            'suitable_for': ['regression', 'heteroscedasticity']
        })
    except Exception as e:
        print(f"[!] Could not load Engel dataset: {e}")
    
    # 7. Star98 Data
    # Good for: larger dataset with multiple predictors
    try:
        star98 = sm.datasets.star98.load_pandas()
        df = star98.data
        output_path = os.path.join(EXTERNAL_ORIGIN_DIR, "star98_schools.csv")
        df.to_csv(output_path, index=False)
        print(f"[+] Saved: star98_schools.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
        materialized.append({
            'name': 'star98_schools',
            'path': output_path,
            'description': 'California schools test score data',
            'source': 'statsmodels',
            'columns': list(df.columns),
            'n_rows': len(df),
            'suitable_for': ['regression', 'multiple_predictors']
        })
    except Exception as e:
        print(f"[!] Could not load Star98 dataset: {e}")
    
    return materialized


def materialize_sklearn_datasets():
    """
    Materialize selected datasets from sklearn.
    These datasets are clean and well-structured for various analyses.
    """
    print("[*] Materializing sklearn datasets...")
    
    try:
        from sklearn import datasets
    except ImportError:
        print("[!] scikit-learn not installed. Install with: pip install scikit-learn")
        return []
    
    materialized = []
    
    # 1. Iris Dataset
    # Good for: ANOVA, classification, discriminant analysis
    try:
        iris = datasets.load_iris(as_frame=True)
        df = iris.frame
        output_path = os.path.join(EXTERNAL_ORIGIN_DIR, "iris.csv")
        df.to_csv(output_path, index=False)
        print(f"[+] Saved: iris.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
        materialized.append({
            'name': 'iris',
            'path': output_path,
            'description': 'Iris flower measurements with species',
            'source': 'sklearn',
            'columns': list(df.columns),
            'n_rows': len(df),
            'suitable_for': ['anova', 'group_comparison', 'multivariate']
        })
    except Exception as e:
        print(f"[!] Could not load Iris dataset: {e}")
    
    # 2. Wine Dataset
    # Good for: ANOVA, correlation, normality tests
    try:
        wine = datasets.load_wine(as_frame=True)
        df = wine.frame
        output_path = os.path.join(EXTERNAL_ORIGIN_DIR, "wine.csv")
        df.to_csv(output_path, index=False)
        print(f"[+] Saved: wine.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
        materialized.append({
            'name': 'wine',
            'path': output_path,
            'description': 'Wine chemical analysis with cultivar type',
            'source': 'sklearn',
            'columns': list(df.columns),
            'n_rows': len(df),
            'suitable_for': ['anova', 'group_comparison', 'correlation']
        })
    except Exception as e:
        print(f"[!] Could not load Wine dataset: {e}")
    
    # 3. Diabetes Dataset
    # Good for: regression, multiple predictors
    try:
        diabetes = datasets.load_diabetes(as_frame=True)
        df = diabetes.frame
        output_path = os.path.join(EXTERNAL_ORIGIN_DIR, "diabetes.csv")
        df.to_csv(output_path, index=False)
        print(f"[+] Saved: diabetes.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
        materialized.append({
            'name': 'diabetes',
            'path': output_path,
            'description': 'Diabetes progression data with clinical measurements',
            'source': 'sklearn',
            'columns': list(df.columns),
            'n_rows': len(df),
            'suitable_for': ['regression', 'multiple_predictors']
        })
    except Exception as e:
        print(f"[!] Could not load Diabetes dataset: {e}")
    
    # 4. Breast Cancer Dataset (for creating categorical outcomes)
    # Good for: contingency tables, chi-square (after binning continuous vars)
    try:
        cancer = datasets.load_breast_cancer(as_frame=True)
        df = cancer.frame
        output_path = os.path.join(EXTERNAL_ORIGIN_DIR, "breast_cancer.csv")
        df.to_csv(output_path, index=False)
        print(f"[+] Saved: breast_cancer.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
        materialized.append({
            'name': 'breast_cancer',
            'path': output_path,
            'description': 'Breast cancer diagnosis data',
            'source': 'sklearn',
            'columns': list(df.columns),
            'n_rows': len(df),
            'suitable_for': ['group_comparison', 'contingency_table']
        })
    except Exception as e:
        print(f"[!] Could not load Breast Cancer dataset: {e}")
    
    return materialized


def generate_metadata_for_dataset(dataset_info):
    """
    Generate column metadata for a materialized dataset in StatQA format.
    """
    df = pd.read_csv(dataset_info['path'])
    dataset_name = dataset_info['name']
    
    metadata_rows = []
    
    for col in df.columns:
        series = df[col]
        
        # Determine data type
        data_type = utils.determine_data_type(series, dataset_name, threshold=0.2)
        
        # Check normality for quantitative columns
        is_normality = False
        if data_type == 'quant' and len(series.dropna()) > 3:
            try:
                # Use Anderson-Darling test
                is_normality = utils.is_normality_ad(series.dropna())
            except:
                is_normality = False
        
        # Build metadata row
        metadata_rows.append({
            'dataset': dataset_name,
            'column_header': col,
            'data_type': data_type,
            'num_of_rows': len(series.dropna()),
            'is_normality': is_normality,
            'column_description': '',  # Will be filled manually if needed
        })
    
    # Save metadata
    metadata_path = f"{path.meta_dir}{path.col_meta_dir}{dataset_name}_col_meta.csv"
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(metadata_path, index=False)
    print(f"[+] Generated metadata: {metadata_path}")
    
    return metadata_path


def materialize_all_external_datasets():
    """
    Main function to materialize all external datasets and generate metadata.
    """
    print("=" * 60)
    print("Materializing External Datasets for StressQA")
    print("=" * 60)
    
    all_datasets = []
    
    # Materialize from statsmodels
    statsmodels_datasets = materialize_statsmodels_datasets()
    all_datasets.extend(statsmodels_datasets)
    
    print()
    
    # Materialize from sklearn
    sklearn_datasets = materialize_sklearn_datasets()
    all_datasets.extend(sklearn_datasets)
    
    print()
    print("=" * 60)
    print(f"[*] Total datasets materialized: {len(all_datasets)}")
    print("=" * 60)
    
    # Generate metadata for each dataset
    print()
    print("[*] Generating metadata...")
    for dataset_info in all_datasets:
        try:
            generate_metadata_for_dataset(dataset_info)
        except Exception as e:
            print(f"[!] Error generating metadata for {dataset_info['name']}: {e}")
    
    # Save dataset inventory
    inventory_df = pd.DataFrame(all_datasets)
    inventory_path = os.path.join(EXTERNAL_ORIGIN_DIR, "_dataset_inventory.csv")
    inventory_df.to_csv(inventory_path, index=False)
    print(f"\n[+] Dataset inventory saved: {inventory_path}")
    
    return all_datasets


if __name__ == "__main__":
    materialized = materialize_all_external_datasets()
    print("\n[✓] External dataset materialization complete!")

