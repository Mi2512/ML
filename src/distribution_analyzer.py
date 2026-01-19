
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, anderson, jarque_bera, normaltest, kurtosis, skew
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.neighbors import KernelDensity
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')


@dataclass
class NormalityTestResult:
    test_name: str
    statistic: float
    p_value: Optional[float]
    is_normal: bool
    significance_level: float = 0.05
    critical_values: Optional[Dict] = None
    
    def __repr__(self):
        if self.p_value is not None:
            return f"{self.test_name}: stat={self.statistic:.4f}, p={self.p_value:.4f}, normal={self.is_normal}"
        else:
            return f"{self.test_name}: stat={self.statistic:.4f}, normal={self.is_normal}"


@dataclass
class MultimodalityResult:
    is_multimodal: bool
    n_modes: int
    method: str
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    details: Optional[Dict] = None
    
    def __repr__(self):
        return f"{self.method}: multimodal={self.is_multimodal}, n_modes={self.n_modes}"


class DistributionType(Enum):
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"
    UNIFORM = "uniform"
    BIMODAL = "bimodal"
    MULTIMODAL = "multimodal"
    SKEWED_RIGHT = "skewed_right"
    SKEWED_LEFT = "skewed_left"
    ZERO_INFLATED = "zero_inflated"
    UNKNOWN = "unknown"


@dataclass
class DistributionClassification:
    feature_name: str
    distribution_type: DistributionType
    confidence: float
    skewness: float
    kurtosis: float
    is_normal: bool
    is_multimodal: bool
    best_fit_params: Optional[Dict] = None
    
    def __repr__(self):
        return f"{self.feature_name}: {self.distribution_type.value} (conf={self.confidence:.2f})"



class NormalityTester:
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def shapiro_wilk_test(self, data: np.ndarray) -> NormalityTestResult:
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 3:
            return NormalityTestResult(
                test_name="Shapiro-Wilk",
                statistic=np.nan,
                p_value=np.nan,
                is_normal=False,
                significance_level=self.alpha
            )
        
        if len(data_clean) > 5000:
            data_clean = np.random.choice(data_clean, 5000, replace=False)
        
        stat, p_value = shapiro(data_clean)
        
        return NormalityTestResult(
            test_name="Shapiro-Wilk",
            statistic=stat,
            p_value=p_value,
            is_normal=p_value > self.alpha,
            significance_level=self.alpha
        )
    
    def anderson_darling_test(self, data: np.ndarray) -> NormalityTestResult:
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 3:
            return NormalityTestResult(
                test_name="Anderson-Darling",
                statistic=np.nan,
                p_value=None,
                is_normal=False,
                significance_level=self.alpha
            )
        
        result = anderson(data_clean, dist='norm')
        
        critical_values = {
            '15%': result.critical_values[0],
            '10%': result.critical_values[1],
            '5%': result.critical_values[2],
            '2.5%': result.critical_values[3],
            '1%': result.critical_values[4]
        }
        
        is_normal = result.statistic < result.critical_values[2]
        
        return NormalityTestResult(
            test_name="Anderson-Darling",
            statistic=result.statistic,
            p_value=None,
            is_normal=is_normal,
            significance_level=self.alpha,
            critical_values=critical_values
        )
    
    def jarque_bera_test(self, data: np.ndarray) -> NormalityTestResult:
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 3:
            return NormalityTestResult(
                test_name="Jarque-Bera",
                statistic=np.nan,
                p_value=np.nan,
                is_normal=False,
                significance_level=self.alpha
            )
        
        stat, p_value = jarque_bera(data_clean)
        
        return NormalityTestResult(
            test_name="Jarque-Bera",
            statistic=stat,
            p_value=p_value,
            is_normal=p_value > self.alpha,
            significance_level=self.alpha
        )
    
    def dagostino_pearson_test(self, data: np.ndarray) -> NormalityTestResult:
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 8:
            return NormalityTestResult(
                test_name="D'Agostino-Pearson",
                statistic=np.nan,
                p_value=np.nan,
                is_normal=False,
                significance_level=self.alpha
            )
        
        stat, p_value = normaltest(data_clean)
        
        return NormalityTestResult(
            test_name="D'Agostino-Pearson",
            statistic=stat,
            p_value=p_value,
            is_normal=p_value > self.alpha,
            significance_level=self.alpha
        )
    
    def test_all(self, data: np.ndarray) -> Dict[str, NormalityTestResult]:
        return {
            'shapiro_wilk': self.shapiro_wilk_test(data),
            'anderson_darling': self.anderson_darling_test(data),
            'jarque_bera': self.jarque_bera_test(data),
            'dagostino_pearson': self.dagostino_pearson_test(data)
        }
    
    def consensus_is_normal(self, data: np.ndarray) -> Tuple[bool, Dict]:
        results = self.test_all(data)
        
        votes = [r.is_normal for r in results.values() if not np.isnan(r.statistic)]
        
        if len(votes) == 0:
            return False, results
        
        is_normal = sum(votes) >= len(votes) / 2
        
        return is_normal, results



class MultimodalityDetector:
    
    def __init__(self):
        pass
    
    def hartigan_dip_test(self, data: np.ndarray, alpha: float = 0.05) -> MultimodalityResult:
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 10:
            return MultimodalityResult(
                is_multimodal=False,
                n_modes=1,
                method="Hartigan's Dip Test",
                details={'error': 'Insufficient data'}
            )
        
        try:
            import diptest
            dip_stat, p_value = diptest.diptest(data_clean)
            
            is_multimodal = p_value < alpha
            n_modes = 2 if is_multimodal else 1
            
            return MultimodalityResult(
                is_multimodal=is_multimodal,
                n_modes=n_modes,
                method="Hartigan's Dip Test",
                statistic=dip_stat,
                p_value=p_value,
                details={'alpha': alpha}
            )
        except ImportError:
            return MultimodalityResult(
                is_multimodal=False,
                n_modes=1,
                method="Hartigan's Dip Test",
                details={'error': 'diptest library not installed'}
            )
    
    def gmm_bic_test(self, data: np.ndarray, max_components: int = 5) -> MultimodalityResult:
        data_clean = data[~np.isnan(data)].reshape(-1, 1)
        
        if len(data_clean) < 10:
            return MultimodalityResult(
                is_multimodal=False,
                n_modes=1,
                method="GMM-BIC",
                details={'error': 'Insufficient data'}
            )
        
        bics = []
        n_components_range = range(1, min(max_components + 1, len(data_clean) // 10))
        
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(data_clean)
            bics.append(gmm.bic(data_clean))
        
        best_n_components = n_components_range[np.argmin(bics)]
        
        is_multimodal = best_n_components > 1
        
        return MultimodalityResult(
            is_multimodal=is_multimodal,
            n_modes=best_n_components,
            method="GMM-BIC",
            details={
                'bics': list(bics),
                'n_components_tested': list(n_components_range),
                'best_bic': min(bics)
            }
        )
    
    def kde_peak_detection(self, data: np.ndarray, bandwidth: float = None) -> MultimodalityResult:
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 10:
            return MultimodalityResult(
                is_multimodal=False,
                n_modes=1,
                method="KDE Peak Detection",
                details={'error': 'Insufficient data'}
            )
        
        data_std = np.std(data_clean)
        data_range = data_clean.max() - data_clean.min()
        
        if data_std < 1e-10 or data_range < 1e-10:
            return MultimodalityResult(
                is_multimodal=False,
                n_modes=1,
                method="KDE Peak Detection",
                details={'error': 'Data is nearly constant (std={:.2e}, range={:.2e})'.format(data_std, data_range)}
            )
        
        if bandwidth is None:
            bandwidth = 1.06 * data_std * len(data_clean) ** (-1/5)
        
        bandwidth = max(bandwidth, 1e-6)
        
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(data_clean.reshape(-1, 1))
        
        x_grid = np.linspace(data_clean.min(), data_clean.max(), 200)
        log_dens = kde.score_samples(x_grid.reshape(-1, 1))
        dens = np.exp(log_dens)
        
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(dens, prominence=0.01 * dens.max())
        
        n_peaks = len(peaks)
        is_multimodal = n_peaks > 1
        
        return MultimodalityResult(
            is_multimodal=is_multimodal,
            n_modes=max(n_peaks, 1),
            method="KDE Peak Detection",
            details={
                'n_peaks': n_peaks,
                'peak_positions': x_grid[peaks].tolist() if n_peaks > 0 else [],
                'peak_heights': dens[peaks].tolist() if n_peaks > 0 else [],
                'bandwidth': bandwidth
            }
        )
    
    def detect_all(self, data: np.ndarray) -> Dict[str, MultimodalityResult]:
        return {
            'dip_test': self.hartigan_dip_test(data),
            'gmm_bic': self.gmm_bic_test(data),
            'kde_peaks': self.kde_peak_detection(data)
        }
    
    def consensus_is_multimodal(self, data: np.ndarray) -> Tuple[bool, int, Dict]:
        results = self.detect_all(data)
        
        votes = [r.is_multimodal for r in results.values() 
                 if 'error' not in r.details]
        
        n_modes_estimates = [r.n_modes for r in results.values() 
                            if 'error' not in r.details and r.is_multimodal]
        
        if len(votes) == 0:
            return False, 1, results
        
        is_multimodal = sum(votes) >= len(votes) / 2
        
        n_modes = int(np.median(n_modes_estimates)) if n_modes_estimates else 1
        
        return is_multimodal, n_modes, results



class DistributionClassifier:
    
    def __init__(self):
        self.normality_tester = NormalityTester()
        self.multimodality_detector = MultimodalityDetector()
    
    def classify(self, data: np.ndarray, feature_name: str = "unknown") -> DistributionClassification:
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 3:
            return DistributionClassification(
                feature_name=feature_name,
                distribution_type=DistributionType.UNKNOWN,
                confidence=0.0,
                skewness=np.nan,
                kurtosis=np.nan,
                is_normal=False,
                is_multimodal=False
            )
        
        is_normal, normality_results = self.normality_tester.consensus_is_normal(data_clean)
        
        is_multimodal, n_modes, multimodal_results = self.multimodality_detector.consensus_is_multimodal(data_clean)
        
        skewness = skew(data_clean)
        kurt = kurtosis(data_clean)
        
        zero_ratio = np.sum(data_clean == 0) / len(data_clean)
        is_zero_inflated = zero_ratio > 0.3
        
        is_lognormal = False
        if np.all(data_clean > 0) and skewness > 0.5:
            log_data = np.log(data_clean)
            is_lognormal, _ = self.normality_tester.consensus_is_normal(log_data)
        
        distribution_type = DistributionType.UNKNOWN
        confidence = 0.5
        
        if is_zero_inflated:
            distribution_type = DistributionType.ZERO_INFLATED
            confidence = 0.9
        elif is_normal:
            distribution_type = DistributionType.NORMAL
            confidence = 0.9
        elif is_lognormal:
            distribution_type = DistributionType.LOGNORMAL
            confidence = 0.85
        elif is_multimodal:
            if n_modes == 2:
                distribution_type = DistributionType.BIMODAL
            else:
                distribution_type = DistributionType.MULTIMODAL
            confidence = 0.85
        else:
            if abs(skewness) < 0.5:
                distribution_type = DistributionType.UNIFORM
                confidence = 0.7
            elif skewness > 1.0:
                distribution_type = DistributionType.SKEWED_RIGHT
                confidence = 0.8
            elif skewness < -1.0:
                distribution_type = DistributionType.SKEWED_LEFT
                confidence = 0.8
            else:
                distribution_type = DistributionType.UNKNOWN
                confidence = 0.5
        
        return DistributionClassification(
            feature_name=feature_name,
            distribution_type=distribution_type,
            confidence=confidence,
            skewness=skewness,
            kurtosis=kurt,
            is_normal=is_normal,
            is_multimodal=is_multimodal,
            best_fit_params={
                'zero_ratio': zero_ratio,
                'n_modes': n_modes if is_multimodal else 1
            }
        )



class TransformationRecommender:
    
    def __init__(self):
        self.normality_tester = NormalityTester()
    
    def recommend(self, data: np.ndarray, distribution_class: DistributionClassification) -> Dict:
        data_clean = data[~np.isnan(data)]
        
        recommendations = {
            'original_distribution': distribution_class.distribution_type.value,
            'is_normal': distribution_class.is_normal,
            'transformations': []
        }
        
        if distribution_class.is_normal:
            recommendations['transformations'].append({
                'method': 'none',
                'formula': 'x (no transformation)',
                'achieves_normality': True,
                'reason': 'Data already normally distributed',
                'priority': 0
            })
            return recommendations
        
        if distribution_class.skewness > 1.0 and np.all(data_clean > 0):
            log_data = np.log(data_clean)
            is_normal, _ = self.normality_tester.consensus_is_normal(log_data)
            recommendations['transformations'].append({
                'method': 'log',
                'formula': 'log(x)',
                'achieves_normality': is_normal,
                'reason': 'Right-skewed distribution with positive values',
                'priority': 1
            })
        
        if distribution_class.skewness > 0.5 and np.all(data_clean >= 0):
            sqrt_data = np.sqrt(data_clean)
            is_normal, _ = self.normality_tester.consensus_is_normal(sqrt_data)
            recommendations['transformations'].append({
                'method': 'sqrt',
                'formula': 'sqrt(x)',
                'achieves_normality': is_normal,
                'reason': 'Moderate right skewness',
                'priority': 2
            })
        
        if np.all(data_clean > 0):
            try:
                boxcox_data, lambda_param = stats.boxcox(data_clean)
                is_normal, _ = self.normality_tester.consensus_is_normal(boxcox_data)
                recommendations['transformations'].append({
                    'method': 'box-cox',
                    'formula': f'box-cox(x, λ={lambda_param:.3f})',
                    'lambda': lambda_param,
                    'achieves_normality': is_normal,
                    'reason': 'Optimal power transformation for positive data',
                    'priority': 1
                })
            except:
                pass
        
        try:
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            yj_data = pt.fit_transform(data_clean.reshape(-1, 1)).flatten()
            is_normal, _ = self.normality_tester.consensus_is_normal(yj_data)
            recommendations['transformations'].append({
                'method': 'yeo-johnson',
                'formula': 'yeo-johnson(x)',
                'achieves_normality': is_normal,
                'reason': 'Universal power transformation (works with negative values)',
                'priority': 2
            })
        except:
            pass
        
        recommendations['transformations'].sort(key=lambda x: (not x.get('achieves_normality', False), x['priority']))
        
        return recommendations



if __name__ == "__main__":
    print("Testing Distribution Analyzer")
    
    np.random.seed(42)
    
    normal_data = np.random.normal(50, 10, 500)
    
    lognormal_data = np.random.lognormal(2, 0.5, 500)
    
    bimodal_data = np.concatenate([
        np.random.normal(30, 5, 250),
        np.random.normal(70, 5, 250)
    ])
    
    print("\n" + "="*80)
    print("1. Normality testing")
    print("="*80)
    
    tester = NormalityTester()
    is_normal, results = tester.consensus_is_normal(normal_data)
    print(f"\nNormal data - Is Normal: {is_normal}")
    for name, result in results.items():
        print(f"  {result}")
    
    print("\n" + "="*80)
    print("2. Multimodality detection")
    print("="*80)
    
    detector = MultimodalityDetector()
    is_multi, n_modes, results = detector.consensus_is_multimodal(bimodal_data)
    print(f"\nBimodal data - Is Multimodal: {is_multi}, N_modes: {n_modes}")
    for name, result in results.items():
        print(f"  {result}")
    
    print("\n" + "="*80)
    print("3. Distribution classification")
    print("="*80)
    
    classifier = DistributionClassifier()
    
    for data, name in [(normal_data, "normal"), (lognormal_data, "lognormal"), (bimodal_data, "bimodal")]:
        classification = classifier.classify(data, name)
        print(f"\n{classification}")
        print(f"  Skewness: {classification.skewness:.3f}, Kurtosis: {classification.kurtosis:.3f}")
    
    print("\n" + "="*80)
    print("4. Transformation recommendations")
    print("="*80)
    
    recommender = TransformationRecommender()
    
    lognormal_class = classifier.classify(lognormal_data, "lognormal")
    recommendations = recommender.recommend(lognormal_data, lognormal_class)
    
    print(f"\nRecommendations for lognormal data:")
    print(f"  Original distribution: {recommendations['original_distribution']}")
    print(f"  Is normal: {recommendations['is_normal']}")
    print(f"\n  Suggested transformations:")
    for i, trans in enumerate(recommendations['transformations'], 1):
        print(f"    {i}. {trans['method']}: {trans['formula']}")
        print(f"       Achieves normality: {trans['achieves_normality']}")
        print(f"       Reason: {trans['reason']}")
    
    print("\n" + "="*80)
    print(" Distribution analyzer TEST Готово")
    print("="*80)

DistributionAnalyzer = DistributionClassifier
