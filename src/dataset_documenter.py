
import logging
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)


class DatasetDocumenter:
    
    def __init__(self):
        logger.info("DatasetDocumenter Запуск")
        self.documentation = {}
    
    def document_dataset(self, dataset_df, name: str = "dataset") -> Dict:
        doc = {
            "name": name,
            "row_count": len(dataset_df),
            "column_count": len(dataset_df.columns),
            "columns": {},
            "summary_statistics": {}
        }
        
        for col in dataset_df.columns:
            doc["columns"][col] = {
                "dtype": str(dataset_df[col].dtype),
                "non_null_count": dataset_df[col].notna().sum(),
                "null_count": dataset_df[col].isna().sum(),
                "unique_values": dataset_df[col].nunique(),
                "sample_values": dataset_df[col].head(3).tolist()
            }
            
            if dataset_df[col].dtype in ['int64', 'float64']:
                doc["summary_statistics"][col] = {
                    "min": float(dataset_df[col].min()),
                    "max": float(dataset_df[col].max()),
                    "mean": float(dataset_df[col].mean()),
                    "std": float(dataset_df[col].std())
                }
        
        self.documentation[name] = doc
        return doc
    
    def document_attribute(self, attr_name: str, description: str, 
                          dtype: str, unit: Optional[str] = None,
                          range_min: Optional[float] = None,
                          range_max: Optional[float] = None) -> Dict:
        return {
            "name": attr_name,
            "description": description,
            "dtype": dtype,
            "unit": unit,
            "range": {
                "min": range_min,
                "max": range_max
            }
        }
    
    def generate_metadata_report(self) -> str:
        report = "# Dataset Documentation Report\n\n"
        
        for name, doc in self.documentation.items():
            report += f"## {name}\n\n"
            report += f"- **Rows**: {doc['row_count']}\n"
            report += f"- **Columns**: {doc['column_count']}\n\n"
            report += "### Columns\n\n"
            
            for col_name, col_info in doc['columns'].items():
                report += f"#### {col_name}\n"
                report += f"- **Type**: {col_info['dtype']}\n"
                report += f"- **Non-null**: {col_info['non_null_count']}\n"
                report += f"- **Unique**: {col_info['unique_values']}\n\n"
        
        return report
    
    def export_documentation(self, output_path: str) -> bool:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.documentation, f, indent=2, ensure_ascii=False)
            logger.info(f"Documentation exported to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export documentation: {e}")
            return False


DatasetMetadataDocumenter = DatasetDocumenter
