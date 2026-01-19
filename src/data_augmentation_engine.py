
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    smote_k_neighbors: int = 5
    smote_sampling_strategy: float = 0.5
    vae_latent_dim: int = 8
    vae_epochs: int = 50
    vae_batch_size: int = 16
    kriging_method: str = 'ordinary'
    kriging_variogram_model: str = 'exponential'
    min_cluster_size: int = 3
    max_synthetic_samples: int = 1000
    correlation_preservation_threshold: float = 0.85


class SMOTEAugmenter:
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'SMOTEAugmenter':
        self.feature_names = X.columns.tolist()
        self.X_fit = X.copy()
        self.y_fit = y
        
        if y is not None:
            self.class_distribution = y.value_counts()
            self.minority_class = self.class_distribution.idxmin()
            self.majority_class = self.class_distribution.idxmax()
        
        self.logger.info(f"[OK] SMOTE fitted on {len(X)} samples, {len(self.feature_names)} features")
        return self
    
    def _get_neighbors(self, X: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
        distances = np.sqrt(((X - x) ** 2).sum(axis=1))
        return np.argsort(distances)[1:k+1]
    
    def generate(self, n_samples: int = 100) -> pd.DataFrame:
        X_minority = self.X_fit.copy()
        
        X_numeric = X_minority.select_dtypes(include=[np.number]).values
        X_mean = X_numeric.mean(axis=0)
        X_std = X_numeric.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_normalized = (X_numeric - X_mean) / X_std
        
        synthetic_samples = []
        
        for _ in range(n_samples):
            idx = np.random.randint(0, len(X_minority))
            x = X_normalized[idx]
            
            neighbors_idx = self._get_neighbors(X_normalized, x, self.config.smote_k_neighbors)
            neighbor_idx = neighbors_idx[np.random.randint(0, len(neighbors_idx))]
            
            lambda_coef = np.random.uniform(0, 1, size=x.shape)
            x_synthetic_normalized = x + lambda_coef * (X_normalized[neighbor_idx] - x)
            
            x_synthetic = x_synthetic_normalized * X_std + X_mean
            
            synthetic_sample = X_minority.iloc[idx].copy()
            for i, col in enumerate(X_minority.select_dtypes(include=[np.number]).columns):
                col_idx = list(X_minority.columns).index(col)
                synthetic_sample.iloc[col_idx] = x_synthetic[i]
            
            synthetic_samples.append(synthetic_sample)
        
        synthetic_df = pd.DataFrame(synthetic_samples, columns=self.feature_names)
        self.logger.info(f"[OK] Generated {len(synthetic_df)} SMOTE samples")
        return synthetic_df


class VAEAugmenter:
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.encoder = None
        self.decoder = None
        self.vae = None
    
    def fit(self, X: pd.DataFrame) -> 'VAEAugmenter':
        try:
            from tensorflow.keras import layers, Model, Input
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.losses import MeanSquaredError, KLDivergence
            from sklearn.preprocessing import StandardScaler
            
            self.feature_names = X.columns.tolist()
            self.input_dim = X.shape[1]
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            encoder_input = Input(shape=(self.input_dim,))
            x = layers.Dense(32, activation='relu')(encoder_input)
            x = layers.Dense(16, activation='relu')(x)
            z_mean = layers.Dense(self.config.vae_latent_dim)(x)
            z_log_var = layers.Dense(self.config.vae_latent_dim)(x)
            
            self.encoder = Model(encoder_input, [z_mean, z_log_var], name='encoder')
            
            decoder_input = Input(shape=(self.config.vae_latent_dim,))
            x = layers.Dense(16, activation='relu')(decoder_input)
            x = layers.Dense(32, activation='relu')(x)
            decoder_output = layers.Dense(self.input_dim, activation='linear')(x)
            
            self.decoder = Model(decoder_input, decoder_output, name='decoder')
            
            vae_input = Input(shape=(self.input_dim,))
            z_mean, z_log_var = self.encoder(vae_input)
            
            def sampling(args):
                z_mean_layer, z_log_var_layer = args
                batch = layers.backend.shape(z_mean_layer)[0]
                dim = layers.backend.int_shape(z_mean_layer)[1]
                epsilon = layers.backend.random_normal(shape=(batch, dim))
                return z_mean_layer + layers.backend.exp(0.5 * z_log_var_layer) * epsilon
            
            z = layers.Lambda(sampling, output_shape=(self.config.vae_latent_dim,))([z_mean, z_log_var])
            vae_output = self.decoder(z)
            
            self.vae = Model(vae_input, vae_output, name='vae')
            
            def vae_loss(y_true, y_pred):
                reconstruction_loss = MeanSquaredError()(y_true, y_pred)
                kl_loss = -0.5 * layers.backend.sum(1 + z_log_var - layers.backend.square(z_mean) 
                                                      - layers.backend.exp(z_log_var), axis=-1)
                return reconstruction_loss + 0.001 * kl_loss
            
            self.vae.compile(optimizer=Adam(learning_rate=0.001), loss=vae_loss)
            
            self.vae.fit(
                X_scaled, X_scaled,
                epochs=self.config.vae_epochs,
                batch_size=self.config.vae_batch_size,
                validation_split=0.1,
                verbose=0
            )
            
            self.logger.info(f"[OK] VAE fitted on {len(X)} samples, latent_dim={self.config.vae_latent_dim}")
            return self
        
        except ImportError:
            self.logger.warning("TensorFlow not installed, VAE disabled")
            self.vae = None
            return self
    
    def generate(self, n_samples: int = 100) -> pd.DataFrame:
        if self.vae is None:
            self.logger.warning("VAE not available, returning empty DataFrame")
            return pd.DataFrame()
        
        try:
            z_samples = np.random.normal(0, 1, (n_samples, self.config.vae_latent_dim))
            
            X_synthetic_scaled = self.decoder.predict(z_samples, verbose=0)
            
            X_synthetic = self.scaler.inverse_transform(X_synthetic_scaled)
            
            synthetic_df = pd.DataFrame(X_synthetic, columns=self.feature_names)
            self.logger.info(f"[OK] Generated {len(synthetic_df)} VAE samples")
            return synthetic_df
        
        except Exception as e:
            self.logger.error(f"Error in VAE generation: {e}")
            return pd.DataFrame()


class KrigingAugmenter:
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fit(self, X: pd.DataFrame, spatial_features: List[str] = None) -> 'KrigingAugmenter':
        self.feature_names = X.columns.tolist()
        self.X_fit = X.copy()
        
        if spatial_features is None:
            spatial_candidates = [col for col in X.columns 
                                 if any(x in col.lower() for x in ['lat', 'lon', 'x', 'y', 'easting', 'northing'])]
            self.spatial_features = spatial_candidates if spatial_candidates else []
        else:
            self.spatial_features = spatial_features
        
        self.logger.info(f"[OK] Kriging fitted, spatial features: {self.spatial_features}")
        return self
    
    def _inverse_distance_weighting(self, X_train: np.ndarray, X_test: np.ndarray, 
                                    y_train: np.ndarray, power: float = 2) -> np.ndarray:
        distances = np.sqrt(((X_train[:, np.newaxis, :] - X_test[np.newaxis, :, :]) ** 2).sum(axis=2))
        distances[distances == 0] = 1e-10
        
        weights = 1.0 / (distances ** power)
        weights /= weights.sum(axis=0, keepdims=True)
        
        return (weights.T @ y_train)
    
    def generate(self, n_samples: int = 100, spatial_extent: float = 0.1) -> pd.DataFrame:
        if len(self.spatial_features) == 0:
            self.logger.warning("No spatial features detected, using random sampling")
            return self.X_fit.sample(n=min(n_samples, len(self.X_fit)), replace=True).reset_index(drop=True)
        
        synthetic_samples = []
        
        spatial_cols = [col for col in self.spatial_features if col in self.X_fit.columns]
        non_spatial_cols = [col for col in self.feature_names if col not in spatial_cols]
        
        if len(spatial_cols) == 0:
            self.logger.warning("Spatial features not found in data")
            return self.X_fit.sample(n=min(n_samples, len(self.X_fit)), replace=True).reset_index(drop=True)
        
        X_spatial = self.X_fit[spatial_cols].values
        spatial_range = X_spatial.max(axis=0) - X_spatial.min(axis=0)
        
        for _ in range(n_samples):
            spatial_extent_vec = spatial_range * spatial_extent
            center = X_spatial.mean(axis=0)
            random_location = center + np.random.uniform(-spatial_extent_vec, spatial_extent_vec)
            
            distances = np.sqrt(((X_spatial - random_location) ** 2).sum(axis=1))
            k = min(self.config.smote_k_neighbors, len(X_spatial))
            nearest_idx = np.argsort(distances)[:k]
            
            spatial_interpolated = self._inverse_distance_weighting(
                X_spatial[nearest_idx], 
                random_location.reshape(1, -1),
                X_spatial[nearest_idx]
            ).flatten()
            
            synthetic_sample = {}
            for col in spatial_cols:
                col_idx = spatial_cols.index(col)
                synthetic_sample[col] = spatial_interpolated[col_idx]
            
            weights = 1.0 / (distances[nearest_idx] ** 2 + 1e-10)
            weights /= weights.sum()
            
            for col in non_spatial_cols:
                if col in self.X_fit.columns:
                    if self.X_fit[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                        synthetic_sample[col] = np.average(self.X_fit.iloc[nearest_idx][col].values, weights=weights)
                    else:
                        synthetic_sample[col] = self.X_fit.iloc[nearest_idx][col].mode()[0] if len(self.X_fit.iloc[nearest_idx][col].mode()) > 0 else self.X_fit.iloc[nearest_idx[0]][col]
            
            synthetic_samples.append(synthetic_sample)
        
        synthetic_df = pd.DataFrame(synthetic_samples, columns=self.feature_names)
        self.logger.info(f"[OK] Generated {len(synthetic_df)} Kriging samples")
        return synthetic_df


class DataAugmentationEngine:
    
    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.smote = SMOTEAugmenter(self.config)
        self.vae = VAEAugmenter(self.config)
        self.kriging = KrigingAugmenter(self.config)
        
        self.redundant_features: Set[str] = set()
        self.confounders: Dict = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
    
    def set_redundant_features(self, features: Set[str]) -> 'DataAugmentationEngine':
        self.redundant_features = features
        self.logger.info(f"[OK] Set {len(features)} redundant features for exclusion")
        return self
    
    def set_confounders(self, confounders: List[Dict]) -> 'DataAugmentationEngine':
        self.confounders = {c.get('confounder', ''): c for c in confounders}
        self.logger.info(f"[OK] Set {len(confounders)} confounders for control")
        return self
    
    def set_correlation_matrix(self, corr_matrix: pd.DataFrame) -> 'DataAugmentationEngine':
        self.correlation_matrix = corr_matrix
        self.logger.info(f"[OK] Set correlation matrix ({corr_matrix.shape})")
        return self
    
    def augment(self, X: pd.DataFrame, 
                method: str = 'hybrid',
                n_samples: int = 100,
                spatial_features: List[str] = None) -> pd.DataFrame:
        
        if self.redundant_features:
            X_aug = X.drop(columns=list(self.redundant_features & set(X.columns)))
        else:
            X_aug = X.copy()
        
        n_samples = min(n_samples, self.config.max_synthetic_samples)
        
        self.logger.info(f"\n================================================================================")
        self.logger.info(f"Starting data augmentation ({method.upper()})")
        self.logger.info(f"================================================================================")
        self.logger.info(f"Original data: {len(X)} samples, {X.shape[1]} features")
        self.logger.info(f"Augmentation data: {X_aug.shape[1]} features ({len(self.redundant_features)} excluded)")
        self.logger.info(f"Target synthetic samples: {n_samples}")
        
        synthetic_data = pd.DataFrame()
        
        if method in ['smote', 'hybrid']:
            self.smote.fit(X_aug)
            synthetic_smote = self.smote.generate(n_samples // 3)
            synthetic_data = pd.concat([synthetic_data, synthetic_smote], ignore_index=True)
            self.logger.info(f"Step 1: SMOTE generated {len(synthetic_smote)} samples")
        
        if method in ['vae', 'hybrid']:
            self.vae.fit(X_aug)
            synthetic_vae = self.vae.generate(n_samples // 3)
            if len(synthetic_vae) > 0:
                synthetic_data = pd.concat([synthetic_data, synthetic_vae], ignore_index=True)
                self.logger.info(f"Step 2: VAE generated {len(synthetic_vae)} samples")
            else:
                self.logger.info(f"Step 2: VAE skipped (not available)")
        
        if method in ['kriging', 'hybrid']:
            self.kriging.fit(X_aug, spatial_features)
            synthetic_kriging = self.kriging.generate(n_samples // 3)
            synthetic_data = pd.concat([synthetic_data, synthetic_kriging], ignore_index=True)
            self.logger.info(f"Step 3: Kriging generated {len(synthetic_kriging)} samples")
        
        if self.redundant_features:
            for col in self.redundant_features & set(X.columns):
                if col not in synthetic_data.columns:
                    sampled_vals = X[col].sample(n=len(synthetic_data), replace=True).values
                    synthetic_data[col] = sampled_vals
        
        synthetic_data = synthetic_data[X.columns]
        
        synthetic_data = synthetic_data.reset_index(drop=True)
        
        self.logger.info(f"\n[DONE] Augmentation complete!")
        self.logger.info(f"================================================================================")
        self.logger.info(f"Generated synthetic data: {len(synthetic_data)} samples, {synthetic_data.shape[1]} features")
        self.logger.info(f"================================================================================\n")
        
        return synthetic_data
    
    def validate_augmentation(self, X_original: pd.DataFrame, X_synthetic: pd.DataFrame) -> Dict:
        validation = {
            'n_original': len(X_original),
            'n_synthetic': len(X_synthetic),
            'n_features': X_synthetic.shape[1],
            'feature_distribution_match': {},
            'correlation_preservation': {},
            'constraints_violated': 0
        }
        
        numeric_cols = X_original.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in X_synthetic.columns:
                original_mean = X_original[col].mean()
                synthetic_mean = X_synthetic[col].mean()
                original_std = X_original[col].std()
                synthetic_std = X_synthetic[col].std()
                
                mean_diff = abs(original_mean - synthetic_mean) / (abs(original_mean) + 1e-10)
                std_diff = abs(original_std - synthetic_std) / (original_std + 1e-10)
                
                validation['feature_distribution_match'][col] = {
                    'mean_diff': mean_diff,
                    'std_diff': std_diff,
                    'acceptable': mean_diff < 0.2 and std_diff < 0.3
                }
        
        if self.correlation_matrix is not None:
            synthetic_corr = X_synthetic.corr()
            for col1 in numeric_cols:
                for col2 in numeric_cols:
                    if col1 < col2 and col1 in synthetic_corr.columns and col2 in synthetic_corr.columns:
                        if col1 in self.correlation_matrix.index and col2 in self.correlation_matrix.columns:
                            original_corr = self.correlation_matrix.loc[col1, col2]
                            synthetic_corr_val = synthetic_corr.loc[col1, col2]
                            corr_diff = abs(original_corr - synthetic_corr_val)
                            
                            validation['correlation_preservation'][f"{col1}-{col2}"] = {
                                'original': original_corr,
                                'synthetic': synthetic_corr_val,
                                'difference': corr_diff,
                                'preserved': corr_diff < (1 - self.config.correlation_preservation_threshold)
                            }
        
        self.logger.info(f"\n[VALIDATION] Augmented data quality check:")
        self.logger.info(f"  Distribution matches: {sum(1 for v in validation['feature_distribution_match'].values() if v.get('acceptable', False))}/{len(numeric_cols)}")
        self.logger.info(f"  Correlations preserved: {len(validation['correlation_preservation'])} checked")
        
        return validation
