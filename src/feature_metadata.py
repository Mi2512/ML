
from typing import Dict, Any, Optional, List
from enum import Enum


class FeatureCategory(Enum):
    TEMPORAL = "temporal"
    GEOGRAPHIC = "geographic"
    METEOROLOGICAL = "meteorological"
    TERRAIN = "terrain"
    TRAJECTORY = "trajectory"
    ENVIRONMENTAL = "environmental"
    QUALITY = "quality"


class DataType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    BINARY = "binary"
    CATEGORICAL = "categorical"


class ExpectedDistribution(Enum):
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    BIMODAL = "bimodal"
    MULTIMODAL = "multimodal"
    ZERO_INFLATED = "zero_inflated"
    BERNOULLI = "bernoulli"
    CATEGORICAL_DIST = "categorical"
    SKEWED_RIGHT = "skewed_right"
    SKEWED_LEFT = "skewed_left"



FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    
    
    'T_month': {
        'name_ru': 'Месяц года',
        'name_en': 'Month of Year',
        'category': FeatureCategory.TEMPORAL,
        'data_type': DataType.DISCRETE,
        'units': 'month',
        'range': (1, 12),
        'description': 'Календарный месяц, когда была записана точка маршрута. Используется для анализа сезонных паттернов и временной динамики маршрутов.',
        'physical_meaning': 'Порядковый номер месяца в году (1=Январь, 12=Декабрь)',
        'expected_distribution': ExpectedDistribution.UNIFORM,
        'constraints': {'min': 1, 'max': 12, 'integer': True},
        'missing_allowed': False,
        'related_features': ['T_season', 'T_quarter', 'T_day_of_year', 'T_is_winter'],
        'use_case': 'Сезонная сегментация, временной анализ, выявление сезонных трендов'
    },
    
    'T_season': {
        'name_ru': 'Сезон года',
        'name_en': 'Season of Year',
        'category': FeatureCategory.TEMPORAL,
        'data_type': DataType.CATEGORICAL,
        'units': 'category',
        'range': None,
        'categories': ['winter', 'spring', 'summer', 'autumn'],
        'description': 'Метеорологический сезон, к которому относится точка маршрута. Зима: декабрь-февраль, весна: март-май, лето: июнь-август, осень: сентябрь-ноябрь.',
        'physical_meaning': 'Сезонная классификация на основе календарного месяца',
        'expected_distribution': ExpectedDistribution.CATEGORICAL_DIST,
        'constraints': {'categories': ['winter', 'spring', 'summer', 'autumn']},
        'missing_allowed': False,
        'related_features': ['T_month', 'M_temperature', 'T_is_expedition_month'],
        'use_case': 'Сезонная стратификация, анализ сезонных различий в маршрутах'
    },
    
    'T_day_of_week': {
        'name_ru': 'День недели',
        'name_en': 'Day of Week',
        'category': FeatureCategory.TEMPORAL,
        'data_type': DataType.DISCRETE,
        'units': 'day',
        'range': (0, 6),
        'description': 'День недели (0=понедельник, 6=воскресенье). Используется для анализа недельных паттернов активности.',
        'physical_meaning': 'Порядковый номер дня в неделе',
        'expected_distribution': ExpectedDistribution.UNIFORM,
        'constraints': {'min': 0, 'max': 6, 'integer': True},
        'missing_allowed': False,
        'related_features': ['T_month', 'T_day_of_year'],
        'use_case': 'Анализ недельной активности, выявление выходных vs будних дней'
    },
    
    'T_day_of_year': {
        'name_ru': 'День года',
        'name_en': 'Day of Year',
        'category': FeatureCategory.TEMPORAL,
        'data_type': DataType.DISCRETE,
        'units': 'day',
        'range': (1, 366),
        'description': 'Порядковый номер дня в году (1=1 января, 365/366=31 декабря). Обеспечивает детальную временную гранулярность для анализа сезонных трендов.',
        'physical_meaning': 'Номер дня от начала года',
        'expected_distribution': ExpectedDistribution.UNIFORM,
        'constraints': {'min': 1, 'max': 366, 'integer': True},
        'missing_allowed': False,
        'related_features': ['T_month', 'T_season', 'T_quarter'],
        'use_case': 'Высокоточный временной анализ, выявление внутрисезонных паттернов'
    },
    
    'T_year': {
        'name_ru': 'Год',
        'name_en': 'Year',
        'category': FeatureCategory.TEMPORAL,
        'data_type': DataType.DISCRETE,
        'units': 'year',
        'range': (2004, 2024),
        'description': 'Календарный год регистрации точки маршрута. Охватывает 20-летний период наблюдений (2004-2024).',
        'physical_meaning': 'Год в формате YYYY',
        'expected_distribution': ExpectedDistribution.UNIFORM,
        'constraints': {'min': 2004, 'max': 2024, 'integer': True},
        'missing_allowed': False,
        'related_features': ['T_month', 'T_day_of_year'],
        'use_case': 'Межгодовой анализ, выявление долгосрочных трендов'
    },
    
    'T_is_winter': {
        'name_ru': 'Индикатор зимы',
        'name_en': 'Winter Indicator',
        'category': FeatureCategory.TEMPORAL,
        'data_type': DataType.BINARY,
        'units': 'boolean',
        'range': (0, 1),
        'description': 'Бинарный индикатор зимнего сезона (1=декабрь-февраль, 0=остальные месяцы). Используется для дихотомической классификации зима/не-зима.',
        'physical_meaning': 'Флаг зимнего периода',
        'expected_distribution': ExpectedDistribution.BERNOULLI,
        'constraints': {'values': [0, 1]},
        'missing_allowed': False,
        'related_features': ['T_season', 'M_is_cold', 'M_is_freezing'],
        'use_case': 'Бинарная классификация сезонов, анализ зимних vs незимних маршрутов'
    },
    
    'T_is_expedition_month': {
        'name_ru': 'Индикатор экспедиционного месяца',
        'name_en': 'Expedition Month Indicator',
        'category': FeatureCategory.TEMPORAL,
        'data_type': DataType.BINARY,
        'units': 'boolean',
        'range': (0, 1),
        'description': 'Бинарный индикатор оптимального периода для горных экспедиций (1=июнь-август, 0=остальные месяцы). Основан на метеорологических условиях для безопасного прохождения маршрутов.',
        'physical_meaning': 'Флаг оптимального экспедиционного сезона',
        'expected_distribution': ExpectedDistribution.BERNOULLI,
        'constraints': {'values': [0, 1]},
        'missing_allowed': False,
        'related_features': ['T_season', 'T_month', 'M_temperature'],
        'use_case': 'Сегментация на экспедиционные и межсезонные периоды'
    },
    
    'T_quarter': {
        'name_ru': 'Квартал года',
        'name_en': 'Quarter of Year',
        'category': FeatureCategory.TEMPORAL,
        'data_type': DataType.DISCRETE,
        'units': 'quarter',
        'range': (1, 4),
        'description': 'Квартал года (1=янв-мар, 2=апр-июн, 3=июл-сен, 4=окт-дек). Промежуточная временная агрегация между месяцем и сезоном.',
        'physical_meaning': 'Номер квартала в календарном году',
        'expected_distribution': ExpectedDistribution.UNIFORM,
        'constraints': {'min': 1, 'max': 4, 'integer': True},
        'missing_allowed': False,
        'related_features': ['T_month', 'T_season'],
        'use_case': 'Квартальный анализ, агрегация по кварталам'
    },
    
    
    'G_latitude': {
        'name_ru': 'Широта',
        'name_en': 'Latitude',
        'category': FeatureCategory.GEOGRAPHIC,
        'data_type': DataType.CONTINUOUS,
        'units': 'degrees (°N)',
        'range': (45.0, 75.0),
        'description': 'Географическая широта точки маршрута в системе WGS84. Охватывает регионы России от южной Сибири до Арктики.',
        'physical_meaning': 'Угловое расстояние от экватора в градусах (положительные значения - северная широта)',
        'expected_distribution': ExpectedDistribution.MULTIMODAL,
        'constraints': {'min': 45.0, 'max': 75.0},
        'missing_allowed': False,
        'related_features': ['G_longitude', 'G_region', 'G_latitude_band', 'G_distance_to_pole_km'],
        'use_case': 'Пространственный анализ, кластеризация маршрутов, климатическая зональность'
    },
    
    'G_longitude': {
        'name_ru': 'Долгота',
        'name_en': 'Longitude',
        'category': FeatureCategory.GEOGRAPHIC,
        'data_type': DataType.CONTINUOUS,
        'units': 'degrees (°E)',
        'range': (15.0, 180.0),
        'description': 'Географическая долгота точки маршрута в системе WGS84. Охватывает территорию России от западных границ до Дальнего Востока.',
        'physical_meaning': 'Угловое расстояние от нулевого меридиана в градусах (положительные значения - восточная долгота)',
        'expected_distribution': ExpectedDistribution.MULTIMODAL,
        'constraints': {'min': 15.0, 'max': 180.0},
        'missing_allowed': False,
        'related_features': ['G_latitude', 'G_region', 'G_longitude_band'],
        'use_case': 'Пространственный анализ, региональная сегментация'
    },
    
    'G_region': {
        'name_ru': 'Регион',
        'name_en': 'Region',
        'category': FeatureCategory.GEOGRAPHIC,
        'data_type': DataType.CATEGORICAL,
        'units': 'category',
        'range': None,
        'description': 'Административный регион/субъект РФ, в котором находится точка маршрута. Определяется на основе геокодирования координат.',
        'physical_meaning': 'Название региона России (область, край, республика, автономный округ)',
        'expected_distribution': ExpectedDistribution.CATEGORICAL_DIST,
        'constraints': None,
        'missing_allowed': False,
        'related_features': ['G_latitude', 'G_longitude', 'G_latitude_band'],
        'use_case': 'Региональная кластеризация, анализ межрегиональных различий'
    },
    
    'G_latitude_band': {
        'name_ru': 'Широтный пояс',
        'name_en': 'Latitude Band',
        'category': FeatureCategory.GEOGRAPHIC,
        'data_type': DataType.CATEGORICAL,
        'units': 'category',
        'range': None,
        'categories': ['southern', 'middle', 'northern'],
        'description': 'Широтная классификация: южный пояс (45-55°N), средний пояс (55-60°N), северный пояс (60-75°N). Отражает климатическую зональность.',
        'physical_meaning': 'Климатический широтный пояс',
        'expected_distribution': ExpectedDistribution.CATEGORICAL_DIST,
        'constraints': {'categories': ['southern', 'middle', 'northern']},
        'missing_allowed': False,
        'related_features': ['G_latitude', 'M_temperature', 'G_distance_to_pole_km'],
        'use_case': 'Климатическая стратификация, анализ широтных эффектов'
    },
    
    'G_longitude_band': {
        'name_ru': 'Долготный пояс',
        'name_en': 'Longitude Band',
        'category': FeatureCategory.GEOGRAPHIC,
        'data_type': DataType.CATEGORICAL,
        'units': 'category',
        'range': None,
        'categories': ['western', 'eastern'],
        'description': 'Долготная классификация: западная Россия (<70°E), восточная Россия (≥70°E). Отражает континентальность климата.',
        'physical_meaning': 'Западный или восточный макрорегион',
        'expected_distribution': ExpectedDistribution.CATEGORICAL_DIST,
        'constraints': {'categories': ['western', 'eastern']},
        'missing_allowed': False,
        'related_features': ['G_longitude', 'G_region'],
        'use_case': 'Анализ континентальности, региональная сегментация запад-восток'
    },
    
    'G_utm_zone': {
        'name_ru': 'Зона UTM',
        'name_en': 'UTM Zone',
        'category': FeatureCategory.GEOGRAPHIC,
        'data_type': DataType.DISCRETE,
        'units': 'zone number',
        'range': (32, 61),
        'description': 'Номер зоны в системе координат Universal Transverse Mercator (UTM). Используется для локальных метрических вычислений расстояний.',
        'physical_meaning': 'Номер 6-градусной меридианной зоны UTM',
        'expected_distribution': ExpectedDistribution.UNIFORM,
        'constraints': {'min': 32, 'max': 61, 'integer': True},
        'missing_allowed': False,
        'related_features': ['G_longitude'],
        'use_case': 'Геодезические вычисления, локальная метрическая система координат'
    },
    
    'G_distance_to_pole_km': {
        'name_ru': 'Расстояние до Северного полюса',
        'name_en': 'Distance to North Pole',
        'category': FeatureCategory.GEOGRAPHIC,
        'data_type': DataType.CONTINUOUS,
        'units': 'kilometers (km)',
        'range': (1665.0, 5000.0),
        'description': 'Ортодромическое расстояние от точки маршрута до Северного полюса в километрах. Альтернативная метрика широты, удобная для моделирования климатических эффектов.',
        'physical_meaning': 'Кратчайшее расстояние по геодезической линии до полюса',
        'expected_distribution': ExpectedDistribution.NORMAL,
        'constraints': {'min': 1665.0, 'max': 5000.0},
        'missing_allowed': False,
        'related_features': ['G_latitude', 'G_latitude_band'],
        'use_case': 'Климатическое моделирование, альтернатива широте для регрессионных моделей'
    },
    
    'G_latitude_sin': {
        'name_ru': 'Синус широты',
        'name_en': 'Sine of Latitude',
        'category': FeatureCategory.GEOGRAPHIC,
        'data_type': DataType.CONTINUOUS,
        'units': 'dimensionless',
        'range': (-1.0, 1.0),
        'description': 'Синус широты (sin(latitude)). Циклическое преобразование координаты для устранения граничных эффектов и обеспечения периодичности.',
        'physical_meaning': 'Тригонометрическое преобразование широты',
        'expected_distribution': ExpectedDistribution.NORMAL,
        'constraints': {'min': -1.0, 'max': 1.0},
        'missing_allowed': False,
        'related_features': ['G_latitude', 'G_longitude_sin'],
        'use_case': 'Циклическое кодирование географических координат для ML'
    },
    
    'G_longitude_sin': {
        'name_ru': 'Синус долготы',
        'name_en': 'Sine of Longitude',
        'category': FeatureCategory.GEOGRAPHIC,
        'data_type': DataType.CONTINUOUS,
        'units': 'dimensionless',
        'range': (-1.0, 1.0),
        'description': 'Синус долготы (sin(longitude)). Циклическое преобразование для корректной обработки переходов через 0°/360° меридиан.',
        'physical_meaning': 'Тригонометрическое преобразование долготы',
        'expected_distribution': ExpectedDistribution.NORMAL,
        'constraints': {'min': -1.0, 'max': 1.0},
        'missing_allowed': False,
        'related_features': ['G_longitude', 'G_latitude_sin'],
        'use_case': 'Циклическое кодирование долготы для ML'
    },
    
    
    'M_temperature': {
        'name_ru': 'Температура воздуха',
        'name_en': 'Air Temperature',
        'category': FeatureCategory.METEOROLOGICAL,
        'data_type': DataType.CONTINUOUS,
        'units': 'degrees Celsius (°C)',
        'range': (-60.0, 50.0),
        'description': 'Температура воздуха в точке маршрута в градусах Цельсия. Получена из OpenWeatherMap API. Критический параметр для оценки условий прохождения маршрута.',
        'physical_meaning': 'Температура воздуха на высоте 2м над поверхностью',
        'expected_distribution': ExpectedDistribution.BIMODAL,
        'constraints': {'min': -60.0, 'max': 50.0},
        'missing_allowed': False,
        'related_features': ['M_temperature_class', 'M_is_cold', 'M_is_freezing', 'T_season'],
        'use_case': 'Оценка метеоусловий, классификация сезонов, безопасность маршрутов',
        'distribution_notes': 'Ожидается биомодальность: зимний режим (отрицательные температуры) и летний режим (положительные температуры)'
    },
    
    'M_temperature_class': {
        'name_ru': 'Класс температуры',
        'name_en': 'Temperature Class',
        'category': FeatureCategory.METEOROLOGICAL,
        'data_type': DataType.CATEGORICAL,
        'units': 'category',
        'range': None,
        'categories': ['very_cold', 'cold', 'cool', 'warm', 'hot'],
        'description': 'Категориальная классификация температуры: очень холодно (<-20°C), холодно (-20...-5°C), прохладно (-5...10°C), тепло (10...25°C), жарко (>25°C).',
        'physical_meaning': 'Качественная оценка теплового комфорта',
        'expected_distribution': ExpectedDistribution.CATEGORICAL_DIST,
        'constraints': {'categories': ['very_cold', 'cold', 'cool', 'warm', 'hot']},
        'missing_allowed': False,
        'related_features': ['M_temperature', 'M_is_cold', 'T_season'],
        'use_case': 'Категориальный анализ температурных условий'
    },
    
    'M_is_cold': {
        'name_ru': 'Индикатор холода',
        'name_en': 'Cold Indicator',
        'category': FeatureCategory.METEOROLOGICAL,
        'data_type': DataType.BINARY,
        'units': 'boolean',
        'range': (0, 1),
        'description': 'Бинарный индикатор холодной погоды (1 если температура < -5°C, иначе 0). Порог выбран как граница комфортного движения без специального снаряжения.',
        'physical_meaning': 'Флаг температуры ниже -5°C',
        'expected_distribution': ExpectedDistribution.BERNOULLI,
        'constraints': {'values': [0, 1]},
        'missing_allowed': False,
        'related_features': ['M_temperature', 'M_is_freezing', 'T_is_winter'],
        'use_case': 'Бинарная классификация температурных условий'
    },
    
    'M_is_freezing': {
        'name_ru': 'Индикатор мороза',
        'name_en': 'Freezing Indicator',
        'category': FeatureCategory.METEOROLOGICAL,
        'data_type': DataType.BINARY,
        'units': 'boolean',
        'range': (0, 1),
        'description': 'Бинарный индикатор отрицательной температуры (1 если температура < 0°C, иначе 0). Критический порог для наличия снега и льда.',
        'physical_meaning': 'Флаг температуры ниже точки замерзания воды',
        'expected_distribution': ExpectedDistribution.BERNOULLI,
        'constraints': {'values': [0, 1]},
        'missing_allowed': False,
        'related_features': ['M_temperature', 'M_is_cold', 'T_is_winter'],
        'use_case': 'Индикатор снежно-ледовых условий'
    },
    
    'M_is_hot': {
        'name_ru': 'Индикатор жары',
        'name_en': 'Hot Indicator',
        'category': FeatureCategory.METEOROLOGICAL,
        'data_type': DataType.BINARY,
        'units': 'boolean',
        'range': (0, 1),
        'description': 'Бинарный индикатор жаркой погоды (1 если температура > 20°C, иначе 0). Порог дискомфорта при физических нагрузках.',
        'physical_meaning': 'Флаг температуры выше 20°C',
        'expected_distribution': ExpectedDistribution.BERNOULLI,
        'constraints': {'values': [0, 1]},
        'missing_allowed': False,
        'related_features': ['M_temperature', 'M_temperature_class'],
        'use_case': 'Индикатор высокотемпературных условий'
    },
    
    
    'TR_terrain_type': {
        'name_ru': 'Тип местности',
        'name_en': 'Terrain Type',
        'category': FeatureCategory.TERRAIN,
        'data_type': DataType.CATEGORICAL,
        'units': 'category',
        'range': None,
        'categories': ['forest', 'water', 'mountain', 'swamp', 'plain', 'urban', 'desert', 'road', 'unknown'],
        'description': 'Классификация типа местности на основе анализа карт Yandex Maps. Определяет характер поверхности и растительности.',
        'physical_meaning': 'Категориальный тип подстилающей поверхности',
        'expected_distribution': ExpectedDistribution.CATEGORICAL_DIST,
        'constraints': {'categories': ['forest', 'water', 'mountain', 'swamp', 'plain', 'urban', 'desert', 'road', 'unknown']},
        'missing_allowed': False,
        'related_features': ['TR_altitude', 'TR_is_forest', 'TR_is_water', 'TR_is_mountain'],
        'use_case': 'Классификация типов маршрутов, оценка проходимости'
    },
    
    'TR_terrain_confidence': {
        'name_ru': 'Уверенность классификации местности',
        'name_en': 'Terrain Classification Confidence',
        'category': FeatureCategory.TERRAIN,
        'data_type': DataType.CONTINUOUS,
        'units': 'probability (0-1)',
        'range': (0.0, 1.0),
        'description': 'Уверенность алгоритма классификации типа местности (0=низкая, 1=высокая). Отражает качество данных геокодирования.',
        'physical_meaning': 'Вероятностная оценка корректности классификации',
        'expected_distribution': ExpectedDistribution.SKEWED_LEFT,
        'constraints': {'min': 0.0, 'max': 1.0},
        'missing_allowed': False,
        'related_features': ['TR_terrain_type', 'Q_data_completeness'],
        'use_case': 'Оценка качества данных, фильтрация низкокачественных записей'
    },
    
    'TR_altitude': {
        'name_ru': 'Высота над уровнем моря',
        'name_en': 'Altitude Above Sea Level',
        'category': FeatureCategory.TERRAIN,
        'data_type': DataType.CONTINUOUS,
        'units': 'meters (m)',
        'range': (-500.0, 7000.0),
        'description': 'Высота точки маршрута над уровнем моря в метрах. Получена из данных GPS и цифровых моделей рельефа. Критический параметр для оценки сложности маршрута.',
        'physical_meaning': 'Вертикальная координата относительно геоида WGS84',
        'expected_distribution': ExpectedDistribution.SKEWED_RIGHT,
        'constraints': {'min': -500.0, 'max': 7000.0},
        'missing_allowed': False,
        'related_features': ['TR_altitude_class', 'TR_is_mountain', 'TR_terrain_type'],
        'use_case': 'Оценка высотной зональности, сложности маршрута, горной акклиматизации',
        'distribution_notes': 'Правосторонняя скошенность: большинство точек на низких высотах, редкие высокогорные пики'
    },
    
    'TR_altitude_class': {
        'name_ru': 'Высотный класс',
        'name_en': 'Altitude Class',
        'category': FeatureCategory.TERRAIN,
        'data_type': DataType.CATEGORICAL,
        'units': 'category',
        'range': None,
        'categories': ['lowland', 'medium', 'highland', 'alpine'],
        'description': 'Высотная классификация: низкогорье (<500м), среднегорье (500-1500м), высокогорье (1500-3000м), альпийская зона (>3000м).',
        'physical_meaning': 'Категориальная высотная поясность',
        'expected_distribution': ExpectedDistribution.CATEGORICAL_DIST,
        'constraints': {'categories': ['lowland', 'medium', 'highland', 'alpine']},
        'missing_allowed': False,
        'related_features': ['TR_altitude', 'TR_is_mountain'],
        'use_case': 'Высотная стратификация, анализ горных vs равнинных маршрутов'
    },
    
    'TR_is_forest': {
        'name_ru': 'Индикатор леса',
        'name_en': 'Forest Indicator',
        'category': FeatureCategory.TERRAIN,
        'data_type': DataType.BINARY,
        'units': 'boolean',
        'range': (0, 1),
        'description': 'Бинарный индикатор лесной местности (1=лес, 0=не лес). Определяется на основе TR_terrain_type.',
        'physical_meaning': 'Флаг лесного покрова',
        'expected_distribution': ExpectedDistribution.BERNOULLI,
        'constraints': {'values': [0, 1]},
        'missing_allowed': False,
        'related_features': ['TR_terrain_type', 'TR_is_open'],
        'use_case': 'Бинарная классификация лес/не-лес для анализа проходимости'
    },
    
    'TR_is_water': {
        'name_ru': 'Индикатор воды',
        'name_en': 'Water Indicator',
        'category': FeatureCategory.TERRAIN,
        'data_type': DataType.BINARY,
        'units': 'boolean',
        'range': (0, 1),
        'description': 'Бинарный индикатор водной поверхности (1=вода, 0=суша). Включает реки, озера, водохранилища.',
        'physical_meaning': 'Флаг водной поверхности',
        'expected_distribution': ExpectedDistribution.BERNOULLI,
        'constraints': {'values': [0, 1]},
        'missing_allowed': False,
        'related_features': ['TR_terrain_type'],
        'use_case': 'Выявление водных участков, анализ переправ'
    },
    
    'TR_is_open': {
        'name_ru': 'Индикатор открытой местности',
        'name_en': 'Open Terrain Indicator',
        'category': FeatureCategory.TERRAIN,
        'data_type': DataType.BINARY,
        'units': 'boolean',
        'range': (0, 1),
        'description': 'Бинарный индикатор открытой местности (1=степь/тундра/луга, 0=закрытая). Местность без высокой растительности и застройки.',
        'physical_meaning': 'Флаг открытого пространства',
        'expected_distribution': ExpectedDistribution.BERNOULLI,
        'constraints': {'values': [0, 1]},
        'missing_allowed': False,
        'related_features': ['TR_terrain_type', 'TR_is_forest'],
        'use_case': 'Оценка видимости, ветровой нагрузки'
    },
    
    'TR_is_mountain': {
        'name_ru': 'Индикатор горной местности',
        'name_en': 'Mountain Indicator',
        'category': FeatureCategory.TERRAIN,
        'data_type': DataType.BINARY,
        'units': 'boolean',
        'range': (0, 1),
        'description': 'Бинарный индикатор горной местности (1=горы, 0=не горы). Определяется комбинацией высоты и типа рельефа.',
        'physical_meaning': 'Флаг горного рельефа',
        'expected_distribution': ExpectedDistribution.BERNOULLI,
        'constraints': {'values': [0, 1]},
        'missing_allowed': False,
        'related_features': ['TR_terrain_type', 'TR_altitude', 'TR_altitude_class'],
        'use_case': 'Классификация горные/равнинные маршруты'
    },
    
    'TR_is_urban': {
        'name_ru': 'Индикатор урбанизированной территории',
        'name_en': 'Urban Indicator',
        'category': FeatureCategory.TERRAIN,
        'data_type': DataType.BINARY,
        'units': 'boolean',
        'range': (0, 1),
        'description': 'Бинарный индикатор городской/населенной местности (1=город/поселок, 0=дикая природа). Определяется наличием застройки.',
        'physical_meaning': 'Флаг урбанизированной территории',
        'expected_distribution': ExpectedDistribution.BERNOULLI,
        'constraints': {'values': [0, 1]},
        'missing_allowed': False,
        'related_features': ['TR_terrain_type', 'OBJ_count_nearby'],
        'use_case': 'Классификация городские/дикие маршруты, доступность инфраструктуры'
    },
    
    
    'TJ_step_frequency': {
        'name_ru': 'Частота шагов',
        'name_en': 'Step Frequency',
        'category': FeatureCategory.TRAJECTORY,
        'data_type': DataType.CONTINUOUS,
        'units': 'steps per meter',
        'range': (0.5, 3.0),
        'description': 'Частота шагов в точке маршрута, измеренная в шагах на метр. Индикатор скорости движения и сложности рельефа.',
        'physical_meaning': 'Плотность шагов на единицу пройденного расстояния',
        'expected_distribution': ExpectedDistribution.NORMAL,
        'constraints': {'min': 0.5, 'max': 3.0},
        'missing_allowed': False,
        'related_features': ['TR_altitude', 'TR_terrain_type'],
        'use_case': 'Оценка темпа движения, энергозатрат, сложности участка',
        'distribution_notes': 'Ожидается нормальное распределение с центром ~1.5-2.0 шагов/метр'
    },
    
    
    'OBJ_count_nearby': {
        'name_ru': 'Количество объектов поблизости',
        'name_en': 'Nearby Objects Count',
        'category': FeatureCategory.ENVIRONMENTAL,
        'data_type': DataType.DISCRETE,
        'units': 'count',
        'range': (0, 50),
        'description': 'Общее количество объектов инфраструктуры в радиусе 500м от точки маршрута. Включает убежища, источники воды, достопримечательности.',
        'physical_meaning': 'Число точек интереса (POI) в буферной зоне 500м',
        'expected_distribution': ExpectedDistribution.ZERO_INFLATED,
        'constraints': {'min': 0, 'integer': True},
        'missing_allowed': False,
        'related_features': ['OBJ_poi_density_per_km2', 'OBJ_shelter_count', 'OBJ_water_count'],
        'use_case': 'Оценка доступности инфраструктуры, безопасности маршрута',
        'distribution_notes': 'Zero-inflated: много точек без объектов (дикая природа), редкие скопления объектов (населенные пункты)'
    },
    
    'OBJ_shelter_count': {
        'name_ru': 'Количество убежищ',
        'name_en': 'Shelter Count',
        'category': FeatureCategory.ENVIRONMENTAL,
        'data_type': DataType.DISCRETE,
        'units': 'count',
        'range': (0, 20),
        'description': 'Количество убежищ, хижин, домов в радиусе 500м. Критично для планирования ночевок и аварийных ситуаций.',
        'physical_meaning': 'Число укрытий для ночлега/отдыха',
        'expected_distribution': ExpectedDistribution.ZERO_INFLATED,
        'constraints': {'min': 0, 'integer': True},
        'missing_allowed': False,
        'related_features': ['OBJ_has_shelter', 'OBJ_count_nearby'],
        'use_case': 'Планирование ночевок, оценка безопасности'
    },
    
    'OBJ_water_count': {
        'name_ru': 'Количество источников воды',
        'name_en': 'Water Source Count',
        'category': FeatureCategory.ENVIRONMENTAL,
        'data_type': DataType.DISCRETE,
        'units': 'count',
        'range': (0, 20),
        'description': 'Количество источников воды (реки, ручьи, озера, родники) в радиусе 500м. Критический ресурс для автономных маршрутов.',
        'physical_meaning': 'Число водных источников',
        'expected_distribution': ExpectedDistribution.ZERO_INFLATED,
        'constraints': {'min': 0, 'integer': True},
        'missing_allowed': False,
        'related_features': ['OBJ_has_water', 'TR_is_water'],
        'use_case': 'Планирование водоснабжения, логистика маршрута'
    },
    
    'OBJ_landmark_count': {
        'name_ru': 'Количество достопримечательностей',
        'name_en': 'Landmark Count',
        'category': FeatureCategory.ENVIRONMENTAL,
        'data_type': DataType.DISCRETE,
        'units': 'count',
        'range': (0, 20),
        'description': 'Количество достопримечательностей (вершины, перевалы, смотровые площадки) в радиусе 500м. Ориентиры и точки интереса.',
        'physical_meaning': 'Число ориентиров и достопримечательностей',
        'expected_distribution': ExpectedDistribution.ZERO_INFLATED,
        'constraints': {'min': 0, 'integer': True},
        'missing_allowed': False,
        'related_features': ['OBJ_has_landmark'],
        'use_case': 'Навигация, туристическая привлекательность маршрута'
    },
    
    'OBJ_nearest_distance_m': {
        'name_ru': 'Расстояние до ближайшего объекта',
        'name_en': 'Distance to Nearest Object',
        'category': FeatureCategory.ENVIRONMENTAL,
        'data_type': DataType.CONTINUOUS,
        'units': 'meters (m)',
        'range': (0.0, 500.0),
        'description': 'Расстояние в метрах до ближайшего объекта инфраструктуры. Индикатор удаленности от цивилизации.',
        'physical_meaning': 'Метрическое расстояние до ближайшего POI',
        'expected_distribution': ExpectedDistribution.ZERO_INFLATED,
        'constraints': {'min': 0.0, 'max': 500.0},
        'missing_allowed': True,
        'related_features': ['OBJ_count_nearby', 'OBJ_poi_density_per_km2'],
        'use_case': 'Оценка изолированности, доступности инфраструктуры'
    },
    
    'OBJ_has_shelter': {
        'name_ru': 'Наличие убежища',
        'name_en': 'Shelter Available',
        'category': FeatureCategory.ENVIRONMENTAL,
        'data_type': DataType.BINARY,
        'units': 'boolean',
        'range': (0, 1),
        'description': 'Бинарный индикатор наличия убежища в радиусе 500м (1=есть, 0=нет).',
        'physical_meaning': 'Флаг доступности укрытия',
        'expected_distribution': ExpectedDistribution.BERNOULLI,
        'constraints': {'values': [0, 1]},
        'missing_allowed': False,
        'related_features': ['OBJ_shelter_count'],
        'use_case': 'Бинарная классификация доступности убежища'
    },
    
    'OBJ_has_water': {
        'name_ru': 'Наличие воды',
        'name_en': 'Water Available',
        'category': FeatureCategory.ENVIRONMENTAL,
        'data_type': DataType.BINARY,
        'units': 'boolean',
        'range': (0, 1),
        'description': 'Бинарный индикатор наличия источника воды в радиусе 500м (1=есть, 0=нет).',
        'physical_meaning': 'Флаг доступности воды',
        'expected_distribution': ExpectedDistribution.BERNOULLI,
        'constraints': {'values': [0, 1]},
        'missing_allowed': False,
        'related_features': ['OBJ_water_count'],
        'use_case': 'Бинарная классификация доступности воды'
    },
    
    'OBJ_has_landmark': {
        'name_ru': 'Наличие достопримечательности',
        'name_en': 'Landmark Available',
        'category': FeatureCategory.ENVIRONMENTAL,
        'data_type': DataType.BINARY,
        'units': 'boolean',
        'range': (0, 1),
        'description': 'Бинарный индикатор наличия достопримечательности в радиусе 500м (1=есть, 0=нет).',
        'physical_meaning': 'Флаг наличия ориентира',
        'expected_distribution': ExpectedDistribution.BERNOULLI,
        'constraints': {'values': [0, 1]},
        'missing_allowed': False,
        'related_features': ['OBJ_landmark_count'],
        'use_case': 'Бинарная классификация наличия ориентиров'
    },
    
    'OBJ_poi_density_per_km2': {
        'name_ru': 'Плотность объектов на км²',
        'name_en': 'POI Density per km²',
        'category': FeatureCategory.ENVIRONMENTAL,
        'data_type': DataType.CONTINUOUS,
        'units': 'count per km²',
        'range': (0.0, 200.0),
        'description': 'Плотность объектов инфраструктуры на квадратный километр. Рассчитывается как OBJ_count_nearby / площадь круга радиусом 500м.',
        'physical_meaning': 'Пространственная плотность POI',
        'expected_distribution': ExpectedDistribution.SKEWED_RIGHT,
        'constraints': {'min': 0.0},
        'missing_allowed': False,
        'related_features': ['OBJ_count_nearby', 'TR_is_urban'],
        'use_case': 'Оценка урбанизированности территории, плотности инфраструктуры'
    },
    
    
    'Q_data_completeness': {
        'name_ru': 'Полнота данных',
        'name_en': 'Data Completeness',
        'category': FeatureCategory.QUALITY,
        'data_type': DataType.CONTINUOUS,
        'units': 'ratio (0-1)',
        'range': (0.0, 1.0),
        'description': 'Коэффициент полноты данных для точки маршрута (доля заполненных полей от общего числа). Метрика качества данных.',
        'physical_meaning': 'Доля непустых полей / общее число полей',
        'expected_distribution': ExpectedDistribution.SKEWED_LEFT,
        'constraints': {'min': 0.0, 'max': 1.0},
        'missing_allowed': False,
        'related_features': ['TR_terrain_confidence'],
        'use_case': 'Фильтрация низкокачественных записей, взвешивание данных',
        'distribution_notes': 'Ожидается левосторонняя скошенность: большинство записей с высокой полнотой (>0.9)'
    },
}



def get_feature_info(feature_name: str) -> Optional[Dict[str, Any]]:
    return FEATURE_METADATA.get(feature_name)


def get_features_by_category(category: FeatureCategory) -> List[str]:
    return [
        name for name, meta in FEATURE_METADATA.items()
        if meta['category'] == category
    ]


def get_features_by_data_type(data_type: DataType) -> List[str]:
    return [
        name for name, meta in FEATURE_METADATA.items()
        if meta['data_type'] == data_type
    ]


def get_numerical_features() -> List[str]:
    return get_features_by_data_type(DataType.CONTINUOUS) + \
           get_features_by_data_type(DataType.DISCRETE)


def get_categorical_features() -> List[str]:
    return get_features_by_data_type(DataType.CATEGORICAL) + \
           get_features_by_data_type(DataType.BINARY)


def get_feature_summary() -> Dict[str, int]:
    summary = {
        'total_features': len(FEATURE_METADATA),
        'by_category': {},
        'by_data_type': {},
        'by_expected_distribution': {}
    }
    
    for meta in FEATURE_METADATA.values():
        cat = meta['category'].value
        summary['by_category'][cat] = summary['by_category'].get(cat, 0) + 1
        
        dtype = meta['data_type'].value
        summary['by_data_type'][dtype] = summary['by_data_type'].get(dtype, 0) + 1
        
        dist = meta['expected_distribution'].value
        summary['by_expected_distribution'][dist] = \
            summary['by_expected_distribution'].get(dist, 0) + 1
    
    return summary


if __name__ == '__main__':
    print("=" * 80)
    print("Feature metadata Database summary")
    print("=" * 80)
    
    summary = get_feature_summary()
    print(f"\nTotal features: {summary['total_features']}")
    
    print("\nBy Category:")
    for cat, count in sorted(summary['by_category'].items()):
        print(f"  {cat:20s}: {count:2d}")
    
    print("\nBy Data Type:")
    for dtype, count in sorted(summary['by_data_type'].items()):
        print(f"  {dtype:20s}: {count:2d}")
    
    print("\nBy Expected Distribution:")
    for dist, count in sorted(summary['by_expected_distribution'].items()):
        print(f"  {dist:20s}: {count:2d}")
    
    print("\n" + "=" * 80)
    print("Numerical features:", len(get_numerical_features()))
    print("Categorical features:", len(get_categorical_features()))
    print("=" * 80)
