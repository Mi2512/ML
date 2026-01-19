
import logging
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

logger = logging.getLogger(__name__)


class CausalAnalyzer:
    
    def __init__(self):
        logger.info("CausalAnalyzer Запуск")
        try:
            from causal_graph_analyzer import CausalGraphAnalyzer
            self.causal = CausalGraphAnalyzer()
        except (ImportError, Exception) as e:
            logger.warning(f"Could not import causal analyzer: {e}")
            self.causal = None
    
    def learn_causal_structure(self, features_df) -> Dict:
        if self.causal:
            return self.causal.learn_causal_structure(features_df)
        
        return {"status": "error", "message": "Causal analyzer unavailable"}
    
    def detect_d_separation(self, features_df) -> Dict:
        if self.causal:
            return self.causal.detect_d_separation(features_df)
        
        return {}
    
    def analyze_confounding(self, features_df, treatment: str, 
                           outcome: str) -> Dict:
        if self.causal:
            return self.causal.analyze_confounding(features_df, treatment, outcome)
        
        return {}
    
    def get_causal_graph(self) -> Dict:
        if self.causal:
            return self.causal.get_graph()
        
        return {}
    
    def visualize_dag(self) -> str:
        if self.causal:
            return self.causal.visualize_dag()
        
        return "DAG visualization unavailable"
    
    def estimate_causal_effects(self, features_df, treatment: str) -> Dict:
        if self.causal:
            return self.causal.estimate_causal_effects(features_df, treatment)
        
        return {}


DAGAnalyzer = CausalAnalyzer
