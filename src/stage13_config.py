
from typing import Dict, List, Set


FEATURE_CATEGORIES = {
    'temporal': [
        'T_month', 'T_season', 'T_day_of_week', 'T_day_of_year', 
        'T_year', 'T_is_winter', 'T_is_expedition_month', 'T_quarter'
    ],
    'geographic': [
        'G_latitude', 'G_longitude', 'G_region', 'G_latitude_band', 
        'G_longitude_band', 'G_utm_zone', 'G_distance_to_pole_km',
        'G_latitude_sin', 'G_longitude_sin'
    ],
    'meteorological': [
        'M_temperature', 'M_temperature_class', 'M_is_cold', 
        'M_is_freezing', 'M_is_hot'
    ],
    'terrain': [
        'TR_terrain_type', 'TR_terrain_confidence', 'TR_altitude',
        'TR_altitude_class', 'TR_is_forest', 'TR_is_water', 
        'TR_is_open', 'TR_is_mountain', 'TR_is_urban'
    ],
    'trajectory': [
        'TJ_step_frequency'
    ],
    'environmental': [
        'OBJ_count_nearby', 'OBJ_shelter_count', 'OBJ_water_count',
        'OBJ_landmark_count', 'OBJ_nearest_distance_m', 'OBJ_has_shelter',
        'OBJ_has_water', 'OBJ_has_landmark', 'OBJ_poi_density_per_km2'
    ],
    'quality': [
        'Q_data_completeness'
    ]
}

NUMERICAL_FEATURES = {
    'T_month', 'T_day_of_week', 'T_day_of_year', 'T_year', 'T_quarter',
    'G_latitude', 'G_longitude', 'G_utm_zone', 'G_distance_to_pole_km',
    'G_latitude_sin', 'G_longitude_sin',
    'M_temperature',
    'TR_altitude',
    'OBJ_count_nearby', 'OBJ_shelter_count', 'OBJ_water_count', 
    'OBJ_landmark_count', 'OBJ_poi_density_per_km2',
    'Q_data_completeness'
}

CATEGORICAL_FEATURES = {
    'T_season', 'T_is_winter', 'T_is_expedition_month',
    'G_region', 'G_latitude_band', 'G_longitude_band',
    'M_temperature_class', 'M_is_cold', 'M_is_freezing', 'M_is_hot',
    'TR_terrain_type', 'TR_altitude_class', 'TR_is_forest', 'TR_is_water',
    'TR_is_open', 'TR_is_mountain', 'TR_is_urban',
    'OBJ_has_shelter', 'OBJ_has_water', 'OBJ_has_landmark'
}


SEASONAL_DISTRIBUTION = {
    'winter': 0.25,
    'spring': 0.25,
    'summer': 0.25,
    'autumn': 0.25,
}

MIN_COVERAGE_POINTS = 15


CORRELATION_ANALYSIS = {
    'methods': ['pearson', 'spearman', 'mutual_information'],
    
    'p_value_threshold': 0.05,
    
    'min_correlation': 0.1,
    
    'permutation_n_repeats': 10,
    'permutation_random_state': 42,
    
    'shap_sample_size': 1000,
    'shap_n_samples': 100,
}


POTENTIAL_CONFOUNDERS = [
    'T_season',
    'T_month',
    'G_latitude_band',
    'G_region',
]

CONFOUNDING_THRESHOLD = 0.15


AUGMENTATION_PARAMETERS = {
    'smote': {
        'k_neighbors': 5,
        'random_state': 42,
        'sampling_strategy': 'not majority',
    },
    
    'vae': {
        'input_dim': None,
        'latent_dim': 10,
        'hidden_dim': 20,
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'validation_split': 0.1,
        'random_state': 42,
    },
    
    'kriging': {
        'variogram_model': 'linear',
        'nlags': 6,
    },
    
    'noise_std': 0.05,
    'outlier_rate': 0.005,
}


PHYSICAL_CONSTRAINTS = {
    'TJ_step_frequency': {
        'min': 0.5,
        'max': 3.0,
        'description': 'Steps per meter (0.5-3.0 is realistic for humans)'
    },
    
    'TR_altitude': {
        'min': -500,
        'max': 7000,
        'description': 'Altitude in meters above sea level'
    },
    
    'M_temperature': {
        'min': -60,
        'max': 50,
        'description': 'Temperature in Celsius'
    },
    
    'TJ_distance_to_prev': {
        'min': 0,
        'max': 1000,
        'description': 'Distance to previous point in meters'
    },
    
    'G_latitude': {
        'min': 45.0,
        'max': 75.0,
        'description': 'Latitude in degrees (45-75°N for Russia)'
    },
    
    'G_longitude': {
        'min': 15.0,
        'max': 180.0,
        'description': 'Longitude in degrees (15-180°E for Russia)'
    },
}

CONSTRAINT_VIOLATION_TOLERANCE = 0.01


STATISTICAL_TESTS = {
    'normality': {
        'test': 'shapiro',
        'threshold': 0.05,
    },
    
    'homoscedasticity': {
        'test': 'breusch_pagan',
        'threshold': 0.05,
    },
    
    'ks_test': {
        'threshold': 0.05,
        'description': 'Kolmogorov-Smirnov test for distribution match'
    },
    
    'permutation': {
        'n_permutations': 1000,
        'random_state': 42,
    },
}


OUTPUT_PATHS = {
    'extracted_features': 'data/output/stage13_extracted_features.csv',
    'feature_importance': 'data/output/stage13_feature_importance.csv',
    'correlation_matrix': 'data/output/stage13_correlation_matrix.csv',
    'confounding_analysis': 'data/output/stage13_confounding_analysis.json',
    'seasonal_analysis': 'data/output/stage13_seasonal_analysis.json',
    'augmented_dataset': 'data/output/stage13_augmented_dataset.csv',
    'synthetic_points': 'data/output/stage13_synthetic_points.csv',
    'validation_report': 'data/output/stage13_validation_report.json',
    'metadata': 'data/output/stage13_metadata.json',
}


LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'data/temp/stage13.log',
}


IMPORTANCE_WEIGHTS = {
    'permutation': 0.40,
    'shap': 0.35,
    'pearson': 0.15,
    'mutual_information': 0.10,
}


RARE_COMBINATION_FACTORS = [
    'T_season',
    'TR_altitude_class',
    'G_latitude_band',
    'G_region',
]

