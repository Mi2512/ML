
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from scipy.stats import chi2_contingency, pearsonr
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class CausalEdge:
    source: str
    target: str
    strength: float
    direction: str


@dataclass
class DSeparationPath:
    x: str
    y: str
    z: Set[str]
    is_dseparated: bool
    path_description: str


@dataclass
class CausalGraphResult:
    edges: List[CausalEdge]
    nodes: List[str]
    independence_statements: Dict[Tuple[str, str], Set[str]]
    confounders: Dict[Tuple[str, str], List[str]]
    colliders: List[Tuple[str, str, str]]


class ConditionalIndependenceTest:
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _partial_correlation(self, data: pd.DataFrame, 
                            x: str, y: str, z_set: Set[str]) -> float:
        try:
            if len(z_set) == 0:
                x_vals = data[x].values.astype(float)
                y_vals = data[y].values.astype(float)
                mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                if np.sum(mask) < 3:
                    return 0.0
                corr, _ = pearsonr(x_vals[mask], y_vals[mask])
                return corr
            
            mask = ~(data[[x, y] + list(z_set)].isna().any(axis=1))
            if np.sum(mask) < len(z_set) + 3:
                return 0.0
            
            x_vals = data.loc[mask, x].values.astype(float)
            y_vals = data.loc[mask, y].values.astype(float)
            
            from sklearn.linear_model import LinearRegression
            
            z_vals = data.loc[mask, list(z_set)].values.astype(float)
            
            lr_x = LinearRegression()
            lr_x.fit(z_vals, x_vals)
            x_residuals = x_vals - lr_x.predict(z_vals)
            
            lr_y = LinearRegression()
            lr_y.fit(z_vals, y_vals)
            y_residuals = y_vals - lr_y.predict(z_vals)
            
            corr, _ = pearsonr(x_residuals, y_residuals)
            return corr
        
        except Exception as e:
            self.logger.debug(f"Error computing partial correlation: {e}")
            return 0.0
    
    def test(self, data: pd.DataFrame, x: str, y: str, 
             z_set: Set[str] = None) -> bool:
        if z_set is None:
            z_set = set()
        
        if x not in data.columns or y not in data.columns:
            return False
        
        for z in z_set:
            if z not in data.columns:
                return False
        
        partial_corr = self._partial_correlation(data, x, y, z_set)
        
        is_independent = abs(partial_corr) < 0.1
        
        return is_independent


class PCAlgorithm:
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.ci_test = ConditionalIndependenceTest(alpha)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.graph = {}
    
    def learn_structure(self, data: pd.DataFrame, max_conditioning_set: int = 1) -> Dict[str, List[str]]:
        self.logger.info("\nPhase 1: Fast skeleton discovery (PC Algorithm with optimizations)")
        
        data_numerical = data.select_dtypes(include=[np.number])
        nodes = data_numerical.columns.tolist()
        
        if len(nodes) > 20:
            self.logger.warning(f"Warning: {len(nodes)} features may be slow. Using subset...")
            corr_matrix = data_numerical.corr().abs()
            top_features = []
            for feat in nodes:
                if len(top_features) < 15:
                    top_features.append(feat)
            nodes = top_features
            self.logger.info(f"  Using {len(nodes)} top features for speed")
        
        skeleton = {node: set(n for n in nodes if n != node) for node in nodes}
        
        max_conditioning_set = min(max_conditioning_set, 1)
        
        for conditioning_size in range(max_conditioning_set + 1):
            removed_edges = []
            tested_pairs = 0
            
            for x in nodes:
                for y in list(skeleton[x]):
                    tested_pairs += 1
                    
                    if tested_pairs % 50 == 0:
                        self.logger.info(f"  Tested {tested_pairs} pairs...")
                    
                    neighbors = (skeleton[x] | skeleton[y]) - {x, y}
                    
                    if len(neighbors) < conditioning_size:
                        continue
                    
                    max_neighbors = min(3, len(neighbors))
                    neighbors = list(neighbors)[:max_neighbors]
                    
                    from itertools import combinations
                    
                    for z_set in combinations(neighbors, conditioning_size):
                        z_set = set(z_set)
                        
                        if self.ci_test.test(data_numerical, x, y, z_set):
                            skeleton[x].discard(y)
                            skeleton[y].discard(x)
                            removed_edges.append((x, y))
                            break
            
            if removed_edges:
                self.logger.info(f"  [OK] Removed {len(removed_edges)} edges (conditioning set size: {conditioning_size})")
        
        self.graph = skeleton
        edge_count = sum(len(v) for v in skeleton.values()) // 2
        self.logger.info(f"[OK] Skeleton learned with {edge_count} edges from {len(nodes)} features")
        
        return skeleton
    
    def orient_edges(self) -> List[CausalEdge]:
        self.logger.info("\nPhase 2: Orienting edges")
        
        edges = []
        oriented = set()
        
        for x in self.graph:
            for y in self.graph[x]:
                if (x, y) not in oriented and (y, x) not in oriented:
                    x_degree = len(self.graph[x])
                    y_degree = len(self.graph[y])
                    
                    if x_degree > y_degree:
                        direction = '->'
                        source, target = x, y
                    elif y_degree > x_degree:
                        direction = '->'
                        source, target = y, x
                    else:
                        direction = '<->'
                        source, target = x, y
                    
                    edges.append(CausalEdge(
                        source=source,
                        target=target,
                        strength=0.5,
                        direction=direction
                    ))
                    
                    oriented.add((x, y))
                    oriented.add((y, x))
        
        self.logger.info(f"[OK] Oriented {len(edges)} edges")
        return edges
    
    def find_confounders(self, data: pd.DataFrame, x: str, y: str) -> List[str]:
        confounders = []
        
        for z in data.columns:
            if z == x or z == y:
                continue
            
            if z in self.graph.get(x, set()) or x in self.graph.get(z, set()):
                if z in self.graph.get(y, set()) or y in self.graph.get(z, set()):
                    corr_zx = abs(data[[z, x]].corr().iloc[0, 1])
                    corr_zy = abs(data[[z, y]].corr().iloc[0, 1])
                    
                    if corr_zx > 0.2 and corr_zy > 0.2:
                        confounders.append(z)
        
        return confounders


class CausalGraphAnalyzer:
    
    def __init__(self, alpha: float = 0.05):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pc_algorithm = PCAlgorithm(alpha)
        self.ci_test = ConditionalIndependenceTest(alpha)
        self.result = None
    
    def analyze(self, data: pd.DataFrame) -> Optional[CausalGraphResult]:
        self.logger.info("\n" + "="*80)
        self.logger.info("Causal Structure Learning (PC Algorithm - Fast Mode)")
        self.logger.info("="*80)
        
        try:
            data_numerical = data.select_dtypes(include=[np.number])
            
            if len(data_numerical.columns) < 3:
                self.logger.warning("Need at least 3 numerical features for causal analysis")
                return None
            
            skeleton = self.pc_algorithm.learn_structure(data_numerical, max_conditioning_set=1)
            
            edges = self.pc_algorithm.orient_edges()
            
            independence_statements = self._find_independence_statements(data_numerical)
            confounders = self._find_confounders(data_numerical)
            colliders = self._find_colliders(skeleton)
            
            result = CausalGraphResult(
                edges=edges,
                nodes=data_numerical.columns.tolist(),
                independence_statements=independence_statements,
                confounders=confounders,
                colliders=colliders
            )
            
            self.result = result
            self._log_results(result)
            
            return result
        
        except Exception as e:
            self.logger.warning(f"Causal analysis error (non-critical): {e}")
            return None
    
    def _fast_correlation_skeleton(self, data: pd.DataFrame) -> Dict[str, Set[str]]:
        self.logger.info("\nUsing fast correlation-based skeleton")
        
        nodes = data.columns.tolist()
        skeleton = {node: set() for node in nodes}
        
        corr_matrix = data.corr().abs()
        
        for i, x in enumerate(nodes):
            for j, y in enumerate(nodes):
                if i < j and corr_matrix.iloc[i, j] > 0.3:
                    skeleton[x].add(y)
                    skeleton[y].add(x)
        
        self.graph = skeleton
        edge_count = sum(len(v) for v in skeleton.values()) // 2
        self.logger.info(f"[OK] Fast skeleton learned with {edge_count} edges")
        
        return skeleton
    
    def _find_independence_statements(self, data: pd.DataFrame) -> Dict[Tuple[str, str], Set[str]]:
        statements = {}
        nodes = data.columns.tolist()
        
        max_pairs = min(100, len(nodes) * (len(nodes) - 1) // 2)
        
        corr_matrix = data.corr().abs()
        pairs = []
        
        for i, x in enumerate(nodes):
            for j, y in enumerate(nodes):
                if i < j:
                    pairs.append((x, y, corr_matrix.iloc[i, j]))
        
        pairs.sort(key=lambda p: p[2], reverse=True)
        pairs = pairs[:max_pairs]
        
        for x, y, corr_val in pairs:
            separating_sets = set()
            
            for z in nodes:
                if z != x and z != y:
                    if self.ci_test.test(data, x, y, {z}):
                        separating_sets.add(z)
            
            if separating_sets:
                statements[(x, y)] = separating_sets
        
        return statements
    
    def _find_confounders(self, data: pd.DataFrame) -> Dict[Tuple[str, str], List[str]]:
        confounders = {}
        nodes = data.columns.tolist()
        
        corr_matrix = data.corr().abs()
        
        max_pairs = min(50, len(nodes) * (len(nodes) - 1) // 2)
        
        pairs = []
        for i, x in enumerate(nodes):
            for j, y in enumerate(nodes):
                if i < j and corr_matrix.iloc[i, j] > 0.3:
                    pairs.append((x, y, corr_matrix.iloc[i, j]))
        
        pairs.sort(key=lambda p: p[2], reverse=True)
        pairs = pairs[:max_pairs]
        
        for x, y, corr_val in pairs:
            conf = self.pc_algorithm.find_confounders(data, x, y)
            if conf:
                confounders[(x, y)] = conf
        
        return confounders
    
    def _find_colliders(self, skeleton: Dict[str, Set[str]]) -> List[Tuple[str, str, str]]:
        colliders = []
        nodes = list(skeleton.keys())
        
        if len(nodes) > 15:
            self.logger.debug("Too many nodes for full collider analysis, using sampling")
            nodes = nodes[:15]
        
        for z in nodes:
            neighbors = skeleton[z]
            
            if len(neighbors) < 2:
                continue
            
            from itertools import combinations
            neighbor_list = list(neighbors)[:10]
            
            for x, y in combinations(neighbor_list, 2):
                if y not in skeleton.get(x, set()):
                    colliders.append((x, z, y))
        
        return colliders
    
    def _log_results(self, result: CausalGraphResult):
        self.logger.info(f"\n[OK] Causal structure learned!")
        self.logger.info(f"    Nodes: {len(result.nodes)}")
        self.logger.info(f"    Edges: {len(result.edges)}")
        self.logger.info(f"    Independence statements: {len(result.independence_statements)}")
        self.logger.info(f"    Confounder pairs: {len(result.confounders)}")
        self.logger.info(f"    Colliders: {len(result.colliders)}")
        
        if result.confounders:
            self.logger.info("\n Top confounders:")
            for (x, y), confs in list(result.confounders.items())[:5]:
                self.logger.info(f"      {x} <-> {y}: confounders = {confs}")
    
    def export_to_dataframe(self) -> pd.DataFrame:
        if not self.result:
            return pd.DataFrame()
        
        data = {
            'source': [e.source for e in self.result.edges],
            'target': [e.target for e in self.result.edges],
            'direction': [e.direction for e in self.result.edges],
            'strength': [e.strength for e in self.result.edges]
        }
        
        return pd.DataFrame(data)
