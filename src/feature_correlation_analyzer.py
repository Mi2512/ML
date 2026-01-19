
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from scipy.spatial.distance import pdist, squareform
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)



@dataclass
class CorrelationResult:
    feature_1: str
    feature_2: str
    method: str
    correlation: float
    p_value: float
    significant: bool
    n_samples: int
    
    def __str__(self):
        sig_marker = "[SIG]" if self.significant else "[NS]"
        return f"{sig_marker} {self.feature_1} <-> {self.feature_2}: {self.correlation:.4f} (p={self.p_value:.4f})"


@dataclass
class CorrelationMatrix:
    method: str
    matrix: np.ndarray
    p_values: np.ndarray
    feature_names: List[str]
    feature_types: Dict[str, str]
    
    def get_correlation(self, feat1: str, feat2: str) -> float:
        idx1 = self.feature_names.index(feat1)
        idx2 = self.feature_names.index(feat2)
        return self.matrix[idx1, idx2]
    
    def get_significant_pairs(self, threshold: float = 0.05, min_corr: float = 0.3) -> List[Tuple[str, str, float]]:
        pairs = []
        n = len(self.feature_names)
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.p_values[i, j] < threshold and abs(self.matrix[i, j]) >= min_corr:
                    pairs.append((
                        self.feature_names[i],
                        self.feature_names[j],
                        self.matrix[i, j]
                    ))
        
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs


@dataclass
class ConfounderInfo:
    confounder: str
    feature_1: str
    feature_2: str
    corr_12: float
    corr_1c: float
    corr_2c: float
    partial_corr: float
    mediation_score: float
    action: str
    
    def __str__(self):
        return (f"Confounder: {self.confounder}\n"
                f"  {self.feature_1} <-> {self.feature_2}: r={self.corr_12:.4f}\n"
                f"  {self.feature_1} <-> {self.confounder}: r={self.corr_1c:.4f}\n"
                f"  {self.feature_2} <-> {self.confounder}: r={self.corr_2c:.4f}\n"
                f"  Partial correlation: {self.partial_corr:.4f}\n"
                f"  Mediation score: {self.mediation_score:.4f}\n"
                f"  Action: {self.action}")


@dataclass
class CorrelationAnalysisResult:
    pearson_matrix: CorrelationMatrix
    spearman_matrix: CorrelationMatrix
    cramers_matrix: Optional[CorrelationMatrix]
    mi_matrix: Optional[CorrelationMatrix]
    
    significant_pairs: List[Tuple[str, str, float]]
    confounders: List[ConfounderInfo]
    redundant_features: Set[str]
    
    feature_importance_by_correlation: Dict[str, float]



class PearsonCalculator:
    
    @staticmethod
    def calculate(data: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        n = len(features)
        corr_matrix = np.zeros((n, n))
        p_matrix = np.zeros((n, n))
        
        valid_features = [f for f in features if f in data.columns]
        feature_to_idx = {f: i for i, f in enumerate(features)}
        
        for i, feat1 in enumerate(valid_features):
            for j, feat2 in enumerate(valid_features):
                idx1 = feature_to_idx[feat1]
                idx2 = feature_to_idx[feat2]
                
                if idx1 == idx2:
                    corr_matrix[idx1, idx2] = 1.0
                    p_matrix[idx1, idx2] = 0.0
                elif idx1 < idx2:
                    try:
                        mask = ~(data[feat1].isna() | data[feat2].isna())
                        x = data.loc[mask, feat1].values
                        y = data.loc[mask, feat2].values
                        
                        if len(x) > 2:
                            corr, p_val = pearsonr(x, y)
                            corr_matrix[idx1, idx2] = corr
                            corr_matrix[idx2, idx1] = corr
                            p_matrix[idx1, idx2] = p_val
                            p_matrix[idx2, idx1] = p_val
                        else:
                            corr_matrix[idx1, idx2] = np.nan
                            p_matrix[idx1, idx2] = 1.0
                    except Exception as e:
                        logger.warning(f"Error calculating Pearson correlation {feat1}-{feat2}: {str(e)}")
                        corr_matrix[idx1, idx2] = np.nan
                        p_matrix[idx1, idx2] = 1.0
        
        return corr_matrix, p_matrix


class SpearmanCalculator:
    
    @staticmethod
    def calculate(data: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        n = len(features)
        corr_matrix = np.zeros((n, n))
        p_matrix = np.zeros((n, n))
        
        valid_features = [f for f in features if f in data.columns]
        feature_to_idx = {f: i for i, f in enumerate(features)}
        
        for i, feat1 in enumerate(valid_features):
            for j, feat2 in enumerate(valid_features):
                idx1 = feature_to_idx[feat1]
                idx2 = feature_to_idx[feat2]
                
                if idx1 == idx2:
                    corr_matrix[idx1, idx2] = 1.0
                    p_matrix[idx1, idx2] = 0.0
                elif idx1 < idx2:
                    try:
                        mask = ~(data[feat1].isna() | data[feat2].isna())
                        x = data.loc[mask, feat1].values
                        y = data.loc[mask, feat2].values
                        
                        if len(x) > 2:
                            corr, p_val = spearmanr(x, y)
                            corr_matrix[idx1, idx2] = corr
                            corr_matrix[idx2, idx1] = corr
                            p_matrix[idx1, idx2] = p_val
                            p_matrix[idx2, idx1] = p_val
                        else:
                            corr_matrix[idx1, idx2] = np.nan
                            p_matrix[idx1, idx2] = 1.0
                    except Exception as e:
                        logger.warning(f"Error calculating Spearman correlation {feat1}-{feat2}: {str(e)}")
                        corr_matrix[idx1, idx2] = np.nan
                        p_matrix[idx1, idx2] = 1.0
        
        return corr_matrix, p_matrix

class CramersCalculator:
    
    @staticmethod
    def calculate(data: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        n = len(features)
        corr_matrix = np.zeros((n, n))
        p_matrix = np.zeros((n, n))
        
        valid_features = [f for f in features if f in data.columns]
        
        feature_to_idx = {f: i for i, f in enumerate(features)}
        
        for i, feat1 in enumerate(valid_features):
            for j, feat2 in enumerate(valid_features):
                idx1 = feature_to_idx[feat1]
                idx2 = feature_to_idx[feat2]
                
                if idx1 == idx2:
                    corr_matrix[idx1, idx2] = 1.0
                    p_matrix[idx1, idx2] = 0.0
                elif idx1 < idx2:
                    try:
                        if data[feat1].isna().all() or data[feat2].isna().all():
                            corr_matrix[idx1, idx2] = 0.0
                            p_matrix[idx1, idx2] = 1.0
                            corr_matrix[idx2, idx1] = 0.0
                            p_matrix[idx2, idx1] = 1.0
                            continue
                        
                        contingency = pd.crosstab(data[feat1].fillna('missing'), 
                                                  data[feat2].fillna('missing'))
                        chi2, p_val, dof, expected = chi2_contingency(contingency)
                        
                        n_samples = contingency.sum().sum()
                        min_dim = min(contingency.shape) - 1
                        if min_dim > 0 and n_samples > 0:
                            cramers_v = np.sqrt(chi2 / (n_samples * min_dim))
                        else:
                            cramers_v = 0.0
                        
                        corr_matrix[idx1, idx2] = cramers_v
                        corr_matrix[idx2, idx1] = cramers_v
                        p_matrix[idx1, idx2] = p_val
                        p_matrix[idx2, idx1] = p_val
                    except Exception as e:
                        logger.warning(f"Error calculating Cramér's V {feat1}-{feat2}: {str(e)}")
                        corr_matrix[idx1, idx2] = 0.0
                        p_matrix[idx1, idx2] = 1.0
                        corr_matrix[idx2, idx1] = 0.0
                        p_matrix[idx2, idx1] = 1.0
        
        return corr_matrix, p_matrix


class MutualInformationCalculator:
    
    @staticmethod
    def _entropy(data: np.ndarray, bins: int = 10) -> float:
        try:
            if isinstance(data, pd.Series):
                data = data.values
            
            if data.dtype.kind in ['f', 'i']:
                data_clean = data[~np.isnan(data.astype(float))]
            else:
                data_clean = data[pd.notna(data)]
            
            if len(data_clean) == 0:
                return 0.0
            
            if data_clean.dtype.kind in ['f', 'i']:
                try:
                    data_binned = pd.cut(data_clean, bins=bins, labels=False, duplicates='drop').values
                except:
                    data_binned = data_clean
            else:
                data_binned = data_clean
            
            unique, counts = np.unique(data_binned, return_counts=True)
            probabilities = counts / len(data_binned)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            return entropy
        except:
            return 0.0
    
    @staticmethod
    def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
        try:
            if isinstance(x, pd.Series):
                x = x.values
            if isinstance(y, pd.Series):
                y = y.values
            
            x_is_numeric = np.issubdtype(x.dtype, np.number)
            y_is_numeric = np.issubdtype(y.dtype, np.number)
            
            if x_is_numeric and y_is_numeric:
                mask = ~(np.isnan(x.astype(float)) | np.isnan(y.astype(float)))
                x_clean = x[mask].astype(float)
                y_clean = y[mask].astype(float)
            elif x_is_numeric and not y_is_numeric:
                mask = ~(np.isnan(x.astype(float)) | pd.isna(y))
                x_clean = x[mask].astype(float)
                y_clean = y[mask]
            elif not x_is_numeric and y_is_numeric:
                mask = ~(pd.isna(x) | np.isnan(y.astype(float)))
                x_clean = x[mask]
                y_clean = y[mask].astype(float)
            else:
                mask = ~(pd.isna(x) | pd.isna(y))
                x_clean = x[mask]
                y_clean = y[mask]
            
            if len(x_clean) < 2:
                return 0.0
            
            if x_is_numeric:
                try:
                    x_binned = pd.cut(x_clean, bins=bins, labels=False, duplicates='drop').values
                except:
                    x_binned = x_clean
            else:
                x_binned = x_clean
            
            if y_is_numeric:
                try:
                    y_binned = pd.cut(y_clean, bins=bins, labels=False, duplicates='drop').values
                except:
                    y_binned = y_clean
            else:
                y_binned = y_clean
            
            try:
                joint_data = np.column_stack([x_binned, y_binned])
                unique_pairs, counts = np.unique(joint_data, axis=0, return_counts=True)
                pxy = counts / len(joint_data)
                h_xy = -np.sum(pxy * np.log2(pxy + 1e-10))
            except:
                return 0.0
            
            h_x = MutualInformationCalculator._entropy(x_clean, bins)
            h_y = MutualInformationCalculator._entropy(y_clean, bins)
            
            mi = h_x + h_y - h_xy
            
            if max(h_x, h_y) > 0:
                mi_normalized = mi / max(h_x, h_y)
            else:
                mi_normalized = 0.0
            
            return np.clip(mi_normalized, 0, 1)
        except Exception as e:
            return 0.0
    
    @staticmethod
    def calculate(data: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        n = len(features)
        corr_matrix = np.zeros((n, n))
        p_matrix = np.zeros((n, n))
        
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_matrix[i, j] = 0.0
                elif i < j:
                    try:
                        x = data[feat1].values
                        y = data[feat2].values
                        mi = MutualInformationCalculator._mutual_information(x, y)
                        
                        corr_matrix[i, j] = mi
                        corr_matrix[j, i] = mi
                        p_matrix[i, j] = 0.0
                        p_matrix[j, i] = 0.0
                    except Exception as e:
                        logger.warning(f"Error calculating MI {feat1}-{feat2}: {e}")
                        corr_matrix[i, j] = 0.0
                        p_matrix[i, j] = 1.0
        
        return corr_matrix, p_matrix



class FeatureCorrelationAnalyzer:
    
    def __init__(self):
        self.pearson_calc = PearsonCalculator()
        self.spearman_calc = SpearmanCalculator()
        self.cramers_calc = CramersCalculator()
        self.mi_calc = MutualInformationCalculator()
        
        logger.info("[OK] FeatureCorrelationAnalyzer Запуск")
    
    def analyze(self,
                df: pd.DataFrame,
                numerical_features: Set[str],
                categorical_features: Set[str],
                p_value_threshold: float = 0.05,
                min_correlation: float = 0.3) -> CorrelationAnalysisResult:
        logger.info(f"\nStep 1: Computing Pearson correlation ({len(numerical_features)} numerical features)...")
        pearson_corr, pearson_p = self.pearson_calc.calculate(df, list(numerical_features))
        pearson_matrix = CorrelationMatrix(
            method='pearson',
            matrix=pearson_corr,
            p_values=pearson_p,
            feature_names=list(numerical_features),
            feature_types={f: 'numerical' for f in numerical_features}
        )
        logger.info(f"   [OK] Pearson correlation computed")
        
        logger.info(f"\nStep 2: Computing Spearman correlation ({len(numerical_features)} features)...")
        spearman_corr, spearman_p = self.spearman_calc.calculate(df, list(numerical_features))
        spearman_matrix = CorrelationMatrix(
            method='spearman',
            matrix=spearman_corr,
            p_values=spearman_p,
            feature_names=list(numerical_features),
            feature_types={f: 'numerical' for f in numerical_features}
        )
        logger.info(f"   [OK] Spearman correlation computed")
        
        cramers_matrix = None
        if len(categorical_features) > 1:
            logger.info(f"\nStep 3: Computing Cramér's V ({len(categorical_features)} categorical features)...")
            cramers_corr, cramers_p = self.cramers_calc.calculate(df, list(categorical_features))
            cramers_matrix = CorrelationMatrix(
                method='cramers',
                matrix=cramers_corr,
                p_values=cramers_p,
                feature_names=list(categorical_features),
                feature_types={f: 'categorical' for f in categorical_features}
            )
            logger.info(f"   [OK] Cramér's V computed")
        
        mi_matrix = None
        if len(df.columns) > 2:
            logger.info(f"\nStep 4: Computing Mutual Information (all {len(df.columns)} features)...")
            mi_corr, mi_p = self.mi_calc.calculate(df, list(df.columns))
            mi_matrix = CorrelationMatrix(
                method='mutual_information',
                matrix=mi_corr,
                p_values=mi_p,
                feature_names=list(df.columns),
                feature_types={f: 'numerical' if f in numerical_features else 'categorical' for f in df.columns}
            )
            logger.info(f"   [OK] Mutual Information computed")
        
        logger.info(f"\nStep 5: Identifying significant correlations (p < {p_value_threshold})...")
        significant_pairs = pearson_matrix.get_significant_pairs(
            threshold=p_value_threshold,
            min_corr=min_correlation
        )
        logger.info(f"   [OK] Found {len(significant_pairs)} significant pairs")
        
        logger.info(f"\nStep 6: Detecting potential confounders...")
        confounders = self._detect_confounders(
            pearson_matrix,
            df,
            significant_pairs,
            threshold=p_value_threshold
        )
        logger.info(f"   [OK] Detected {len(confounders)} potential confounders")
        
        logger.info(f"\nStep 7: Identifying redundant features...")
        redundant_features = self._identify_redundant_features(
            pearson_matrix,
            threshold=0.9
        )
        logger.info(f"   [OK] Identified {len(redundant_features)} redundant features")
        
        logger.info(f"\nStep 8: Calculating feature importance...")
        feature_importance = self._calculate_feature_importance(pearson_matrix)
        logger.info(f"   [OK] Feature importance calculated")
        
        logger.info("\n[DONE] Correlation analysis Готово!")
        
        return CorrelationAnalysisResult(
            pearson_matrix=pearson_matrix,
            spearman_matrix=spearman_matrix,
            cramers_matrix=cramers_matrix,
            mi_matrix=mi_matrix,
            significant_pairs=significant_pairs,
            confounders=confounders,
            redundant_features=redundant_features,
            feature_importance_by_correlation=feature_importance
        )
    
    def _detect_confounders(self,
                            corr_matrix: CorrelationMatrix,
                            df: pd.DataFrame,
                            significant_pairs: List[Tuple[str, str, float]],
                            threshold: float = 0.05) -> List[ConfounderInfo]:
        confounders = []
        features = set(corr_matrix.feature_names)
        
        for feat1, feat2, corr_12 in significant_pairs[:10]:
            if feat1 not in features or feat2 not in features:
                continue
            
            for confounder in features:
                if confounder == feat1 or confounder == feat2:
                    continue
                
                try:
                    corr_1c = corr_matrix.get_correlation(feat1, confounder)
                    corr_2c = corr_matrix.get_correlation(feat2, confounder)
                    
                    if abs(corr_1c) > 0.3 and abs(corr_2c) > 0.3:
                        partial_corr = self._calculate_partial_correlation(
                            df, feat1, feat2, confounder
                        )
                        
                        mediation = abs(corr_12 - partial_corr) / (abs(corr_12) + 1e-6)
                        
                        if mediation > 0.5:
                            action = "investigate"
                        elif abs(partial_corr) < 0.1:
                            action = "control"
                        else:
                            action = "monitor"
                        
                        confounders.append(ConfounderInfo(
                            confounder=confounder,
                            feature_1=feat1,
                            feature_2=feat2,
                            corr_12=corr_12,
                            corr_1c=corr_1c,
                            corr_2c=corr_2c,
                            partial_corr=partial_corr,
                            mediation_score=mediation,
                            action=action
                        ))
                except Exception as e:
                    logger.debug(f"Error detecting confounder {confounder}: {e}")
        
        return confounders
    
    def _calculate_partial_correlation(self, df: pd.DataFrame, x: str, y: str, z: str) -> float:
        try:
            from scipy.stats import linregress
            
            mask = ~(df[x].isna() | df[y].isna() | df[z].isna())
            x_vals = df.loc[mask, x].values
            y_vals = df.loc[mask, y].values
            z_vals = df.loc[mask, z].values
            
            if len(x_vals) < 3:
                return 0.0
            
            slope_xz, intercept_xz, _, _, _ = linregress(z_vals, x_vals)
            x_residuals = x_vals - (slope_xz * z_vals + intercept_xz)
            
            slope_yz, intercept_yz, _, _, _ = linregress(z_vals, y_vals)
            y_residuals = y_vals - (slope_yz * z_vals + intercept_yz)
            
            partial_corr, _ = pearsonr(x_residuals, y_residuals)
            return partial_corr
        except:
            return 0.0
    
    def _identify_redundant_features(self,
                                     corr_matrix: CorrelationMatrix,
                                     threshold: float = 0.9) -> Set[str]:
        redundant = set()
        n = len(corr_matrix.feature_names)
        
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr_matrix.matrix[i, j]) > threshold:
                    avg_corr_i = np.nanmean(np.abs(corr_matrix.matrix[i, :]))
                    avg_corr_j = np.nanmean(np.abs(corr_matrix.matrix[j, :]))
                    
                    if avg_corr_i < avg_corr_j:
                        redundant.add(corr_matrix.feature_names[i])
                    else:
                        redundant.add(corr_matrix.feature_names[j])
        
        return redundant
    
    def _calculate_feature_importance(self, corr_matrix: CorrelationMatrix) -> Dict[str, float]:
        importance = {}
        
        for feat in corr_matrix.feature_names:
            idx = corr_matrix.feature_names.index(feat)
            avg_corr = np.nanmean(np.abs(corr_matrix.matrix[idx, :]))
            importance[feat] = avg_corr
        
        return importance
    
    def print_summary(self, result: CorrelationAnalysisResult):
        logger.info("\n" + "=" * 80)
        logger.info("Correlation analysis SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"\nTop 10 Correlated Pairs:")
        for feat1, feat2, corr in result.significant_pairs[:10]:
            logger.info(f"  {feat1} <-> {feat2}: r={corr:.4f}")
        
        if result.confounders:
            logger.info(f"\nDetected Confounders ({len(result.confounders)}):")
            for conf in result.confounders[:5]:
                logger.info(f"  [{conf.action}] {conf.confounder} affects {conf.feature_1}<->{conf.feature_2}")
        
        if result.redundant_features:
            logger.info(f"\nRedundant Features ({len(result.redundant_features)}):")
            for feat in sorted(result.redundant_features)[:10]:
                logger.info(f"  - {feat}")
        
        logger.info(f"\nTop 10 Important Features (by correlation):")
        sorted_importance = sorted(
            result.feature_importance_by_correlation.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for feat, importance in sorted_importance[:10]:
            logger.info(f"  {feat}: {importance:.4f}")


if __name__ == "__main__":
    logger.info("Feature Correlation Analyzer Module")
