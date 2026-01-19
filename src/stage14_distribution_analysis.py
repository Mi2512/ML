
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from feature_metadata import (
    FEATURE_METADATA,
    get_numerical_features,
    get_categorical_features,
    get_feature_info
)
from distribution_analyzer import (
    NormalityTester,
    MultimodalityDetector,
    DistributionClassifier,
    TransformationRecommender
)
from distribution_visualizer import DistributionVisualizer



DATA_FILE = Path("data/output/stage13_features_extracted.csv")
OUTPUT_DIR = Path("data/output/stage14")
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)



def analyze_all_features():
    print("="*80)
    print("STAGE 1.4: Distribution analysis")
    print("="*80)
    
    print(f"\n Loading data from: {DATA_FILE}")
    if not DATA_FILE.exists():
        print(f" File not found: {DATA_FILE}")
        return
    
    df = pd.read_csv(DATA_FILE)
    print(f" Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    
    numerical_features = get_numerical_features()
    categorical_features = get_categorical_features()
    
    print(f"\n Features to analyze:")
    print(f"   - Numerical: {len(numerical_features)}")
    print(f"   - Categorical: {len(categorical_features)}")
    print(f"   - Total: {len(numerical_features) + len(categorical_features)}")
    
    normality_tester = NormalityTester(alpha=0.05)
    multimodality_detector = MultimodalityDetector()
    distribution_classifier = DistributionClassifier()
    transformation_recommender = TransformationRecommender()
    visualizer = DistributionVisualizer(output_dir=str(VISUALIZATIONS_DIR))
    
    normality_results = []
    multimodality_results = []
    classification_results = []
    transformation_results = []
    
    
    print("\n" + "="*80)
    print("Analyzing numerical FEATURES")
    print("="*80)
    
    for feature_name in tqdm(numerical_features, desc="Numerical features"):
        if feature_name not in df.columns:
            print(f" Feature {feature_name} not found in data")
            continue
        
        data = df[feature_name].values
        metadata = get_feature_info(feature_name)
        
        is_normal, norm_tests = normality_tester.consensus_is_normal(data)
        normality_results.append({
            'feature': feature_name,
            'feature_type': 'numerical',
            'is_normal': is_normal,
            'shapiro_wilk_stat': norm_tests['shapiro_wilk'].statistic,
            'shapiro_wilk_p': norm_tests['shapiro_wilk'].p_value,
            'anderson_darling_stat': norm_tests['anderson_darling'].statistic,
            'jarque_bera_stat': norm_tests['jarque_bera'].statistic,
            'jarque_bera_p': norm_tests['jarque_bera'].p_value,
            'dagostino_stat': norm_tests['dagostino_pearson'].statistic,
            'dagostino_p': norm_tests['dagostino_pearson'].p_value
        })
        
        is_multimodal, n_modes, multi_tests = multimodality_detector.consensus_is_multimodal(data)
        multimodality_results.append({
            'feature': feature_name,
            'feature_type': 'numerical',
            'is_multimodal': is_multimodal,
            'n_modes': n_modes,
            'dip_test_multimodal': multi_tests['dip_test'].is_multimodal,
            'gmm_bic_n_modes': multi_tests['gmm_bic'].n_modes,
            'kde_peaks_n_modes': multi_tests['kde_peaks'].n_modes
        })
        
        distribution_class = distribution_classifier.classify(data, feature_name)
        classification_results.append({
            'feature': feature_name,
            'feature_type': 'numerical',
            'distribution_type': distribution_class.distribution_type.value,
            'confidence': distribution_class.confidence,
            'skewness': distribution_class.skewness,
            'kurtosis': distribution_class.kurtosis,
            'is_normal': distribution_class.is_normal,
            'is_multimodal': distribution_class.is_multimodal,
            'expected_distribution': metadata.get('expected_distribution', 'unknown') if metadata else 'unknown'
        })
        
        transformations = transformation_recommender.recommend(data, distribution_class)
        for i, trans in enumerate(transformations['transformations'][:3]):
            transformation_results.append({
                'feature': feature_name,
                'feature_type': 'numerical',
                'rank': i + 1,
                'method': trans.get('method', 'unknown'),
                'formula': trans.get('formula', 'N/A'),
                'achieves_normality': trans.get('achieves_normality', False),
                'reason': trans.get('reason', 'N/A')
            })
        
        visualizer.plot_full_diagnostic(data, feature_name, save=True)
    
    
    print("\n" + "="*80)
    print("Analyzing categorical FEATURES")
    print("="*80)
    
    for feature_name in tqdm(categorical_features, desc="Categorical features"):
        if feature_name not in df.columns:
            print(f" Feature {feature_name} not found in data")
            continue
        
        data = df[feature_name].values
        metadata = get_feature_info(feature_name)
        
        unique_values = pd.Series(data).dropna().unique()
        n_categories = len(unique_values)
        
        value_counts = pd.Series(data).value_counts()
        mode_value = value_counts.index[0] if len(value_counts) > 0 else None
        mode_frequency = value_counts.iloc[0] if len(value_counts) > 0 else 0
        
        classification_results.append({
            'feature': feature_name,
            'feature_type': 'categorical',
            'distribution_type': 'categorical',
            'confidence': 1.0,
            'n_categories': n_categories,
            'mode': str(mode_value),
            'mode_frequency': int(mode_frequency),
            'mode_percentage': float(mode_frequency / len(data) * 100),
            'expected_distribution': metadata.get('expected_distribution', 'categorical') if metadata else 'categorical'
        })
        
        categories = metadata.get('categories', None) if metadata else None
        visualizer.plot_categorical_distribution(data, feature_name, categories=categories, save=True)
    
    
    print("\n" + "="*80)
    print("Saving results")
    print("="*80)
    
    pd.DataFrame(normality_results).to_csv(
        OUTPUT_DIR / "normality_tests.csv", index=False
    )
    print(f" Saved: normality_tests.csv")
    
    pd.DataFrame(multimodality_results).to_csv(
        OUTPUT_DIR / "multimodality_tests.csv", index=False
    )
    print(f" Saved: multimodality_tests.csv")
    
    pd.DataFrame(classification_results).to_csv(
        OUTPUT_DIR / "distribution_classification.csv", index=False
    )
    print(f" Saved: distribution_classification.csv")
    
    pd.DataFrame(transformation_results).to_csv(
        OUTPUT_DIR / "transformation_recommendations.csv", index=False
    )
    print(f" Saved: transformation_recommendations.csv")
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_file': str(DATA_FILE),
        'n_samples': int(df.shape[0]),
        'n_features': int(df.shape[1]),
        'numerical_features_analyzed': len(numerical_features),
        'categorical_features_analyzed': len(categorical_features),
        'normality_summary': {
            'n_normal': sum(1 for r in normality_results if r['is_normal']),
            'n_non_normal': sum(1 for r in normality_results if not r['is_normal'])
        },
        'multimodality_summary': {
            'n_multimodal': sum(1 for r in multimodality_results if r['is_multimodal']),
            'n_unimodal': sum(1 for r in multimodality_results if not r['is_multimodal'])
        },
        'distribution_types': pd.DataFrame(classification_results)['distribution_type'].value_counts().to_dict(),
        'output_directory': str(OUTPUT_DIR),
        'visualizations_directory': str(VISUALIZATIONS_DIR)
    }
    
    with open(OUTPUT_DIR / "analysis_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f" Saved: analysis_summary.json")
    
    
    print("\n" + "="*80)
    print("Analysis summary")
    print("="*80)
    print(f"\n Total features analyzed: {len(numerical_features) + len(categorical_features)}")
    print(f"   - Numerical: {len(numerical_features)}")
    print(f"   - Categorical: {len(categorical_features)}")
    
    print(f"\n Normality tests:")
    print(f"   - Normal: {summary['normality_summary']['n_normal']}")
    print(f"   - Non-normal: {summary['normality_summary']['n_non_normal']}")
    
    print(f"\n Multimodality tests:")
    print(f"   - Multimodal: {summary['multimodality_summary']['n_multimodal']}")
    print(f"   - Unimodal: {summary['multimodality_summary']['n_unimodal']}")
    
    print(f"\n Distribution types:")
    for dist_type, count in summary['distribution_types'].items():
        print(f"   - {dist_type}: {count}")
    
    print(f"\n Results saved to: {OUTPUT_DIR}")
    print(f" Visualizations saved to: {VISUALIZATIONS_DIR}")
    
    print("\n" + "="*80)
    print(" STAGE 1.4 ANALYSIS Готово")
    print("="*80)



if __name__ == "__main__":
    analyze_all_features()
