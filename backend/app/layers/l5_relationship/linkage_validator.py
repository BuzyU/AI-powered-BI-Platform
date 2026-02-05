# Linkage Validator - Checks relationships between uploaded datasets
# Validates if files relate to each other and calculates link strength

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LinkStrength(Enum):
    """Link strength classification."""
    STRONG = "strong"        # 80%+ overlap
    MODERATE = "moderate"    # 50-80% overlap
    WEAK = "weak"           # 20-50% overlap
    BROKEN = "broken"       # <20% overlap
    NO_LINK = "no_link"     # No linkable columns found


@dataclass
class LinkageResult:
    """Result of a linkage check between two datasets."""
    source_dataset: str
    target_dataset: str
    source_column: str
    target_column: str
    strength: LinkStrength
    overlap_percentage: float
    matched_count: int
    source_unique: int
    target_unique: int
    confidence: float
    issues: List[str]


class LinkageValidator:
    """
    Validates relationships between datasets.
    Checks if uploaded files relate to each other through common identifiers.
    """
    
    # Common column patterns that indicate linkable fields
    LINKABLE_PATTERNS = {
        "id": ["id", "key", "code", "number", "no", "num"],
        "customer": ["customer", "client", "user", "account", "buyer", "member"],
        "product": ["product", "item", "sku", "goods", "article"],
        "order": ["order", "transaction", "purchase", "po"],
        "employee": ["employee", "staff", "worker", "emp"],
        "ticket": ["ticket", "case", "issue", "incident"],
        "invoice": ["invoice", "bill", "receipt"],
        "date": ["date", "time", "timestamp", "created", "updated"],
    }
    
    def __init__(self):
        self.results: List[LinkageResult] = []
    
    def validate_linkages(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate linkages between all datasets.
        
        Args:
            datasets: Dict mapping dataset_id to DataFrame
        
        Returns:
            Comprehensive linkage analysis report
        """
        self.results = []
        dataset_ids = list(datasets.keys())
        
        if len(dataset_ids) < 2:
            return {
                "status": "single_dataset",
                "message": "Only one dataset uploaded. Upload multiple files to check linkages.",
                "links": [],
                "warnings": [],
                "graph": {"nodes": [], "edges": []}
            }
        
        # Check all pairs
        for i, source_id in enumerate(dataset_ids):
            for target_id in dataset_ids[i+1:]:
                source_df = datasets[source_id]
                target_df = datasets[target_id]
                
                links = self._find_links(source_id, target_id, source_df, target_df)
                self.results.extend(links)
        
        # Generate summary
        return self._generate_report(datasets)
    
    def _find_links(
        self, 
        source_id: str, 
        target_id: str, 
        source_df: pd.DataFrame, 
        target_df: pd.DataFrame
    ) -> List[LinkageResult]:
        """Find potential links between two datasets."""
        links = []
        
        # Get linkable columns from both datasets
        source_linkable = self._get_linkable_columns(source_df)
        target_linkable = self._get_linkable_columns(target_df)
        
        # Try to match columns
        for source_col, source_type in source_linkable.items():
            for target_col, target_type in target_linkable.items():
                # Check if types are compatible
                if self._types_compatible(source_type, target_type, source_col, target_col):
                    link = self._check_value_overlap(
                        source_id, target_id,
                        source_df, target_df,
                        source_col, target_col
                    )
                    if link:
                        links.append(link)
        
        return links
    
    def _get_linkable_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Identify columns that could be used for linking."""
        linkable = {}
        
        for col in df.columns:
            col_lower = col.lower().replace("_", " ").replace("-", " ")
            
            for link_type, patterns in self.LINKABLE_PATTERNS.items():
                if any(pattern in col_lower for pattern in patterns):
                    linkable[col] = link_type
                    break
            else:
                # Check if it looks like an ID column (high cardinality, mostly unique)
                if df[col].dtype == 'object' or df[col].dtype.name.startswith('int'):
                    unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                    if unique_ratio > 0.8:  # Likely an ID column
                        linkable[col] = "id"
        
        return linkable
    
    def _types_compatible(
        self, 
        source_type: str, 
        target_type: str,
        source_col: str,
        target_col: str
    ) -> bool:
        """Check if two column types are compatible for linking."""
        # Same type is always compatible
        if source_type == target_type:
            return True
        
        # ID type is compatible with specific entity types
        if source_type == "id" or target_type == "id":
            return True
        
        # Check column name similarity
        source_lower = source_col.lower()
        target_lower = target_col.lower()
        
        # Direct name match
        if source_lower == target_lower:
            return True
        
        # Partial match (e.g., customer_id matches customer)
        if source_lower in target_lower or target_lower in source_lower:
            return True
        
        return False
    
    def _check_value_overlap(
        self,
        source_id: str,
        target_id: str,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_col: str,
        target_col: str
    ) -> Optional[LinkageResult]:
        """Check the value overlap between two columns."""
        try:
            # Get unique values
            source_values = set(source_df[source_col].dropna().astype(str).unique())
            target_values = set(target_df[target_col].dropna().astype(str).unique())
            
            if not source_values or not target_values:
                return None
            
            # Calculate overlap
            overlap = source_values & target_values
            overlap_count = len(overlap)
            
            # Calculate percentages
            source_overlap_pct = (overlap_count / len(source_values)) * 100 if source_values else 0
            target_overlap_pct = (overlap_count / len(target_values)) * 100 if target_values else 0
            
            # Use the higher overlap percentage (one side might be a subset)
            max_overlap_pct = max(source_overlap_pct, target_overlap_pct)
            
            if overlap_count == 0:
                return None  # No actual link
            
            # Determine strength
            if max_overlap_pct >= 80:
                strength = LinkStrength.STRONG
            elif max_overlap_pct >= 50:
                strength = LinkStrength.MODERATE
            elif max_overlap_pct >= 20:
                strength = LinkStrength.WEAK
            else:
                strength = LinkStrength.BROKEN
            
            # Identify issues
            issues = []
            if max_overlap_pct < 50:
                missing_in_target = len(source_values - target_values)
                if missing_in_target > 0:
                    issues.append(f"{missing_in_target} values from source not found in target")
            
            if source_overlap_pct < 20 and target_overlap_pct < 20:
                issues.append("Very low overlap in both directions - possible data mismatch")
            
            # Calculate confidence based on overlap and data quality
            confidence = min(max_overlap_pct, 95)
            
            return LinkageResult(
                source_dataset=source_id,
                target_dataset=target_id,
                source_column=source_col,
                target_column=target_col,
                strength=strength,
                overlap_percentage=round(max_overlap_pct, 1),
                matched_count=overlap_count,
                source_unique=len(source_values),
                target_unique=len(target_values),
                confidence=round(confidence, 1),
                issues=issues
            )
            
        except Exception as e:
            logger.error(f"Error checking overlap for {source_col} <-> {target_col}: {e}")
            return None
    
    def _generate_report(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive linkage report."""
        
        # Build graph structure
        nodes = []
        edges = []
        warnings = []
        
        # Add nodes for each dataset
        for dataset_id in datasets.keys():
            nodes.append({
                "id": dataset_id,
                "label": dataset_id[:20],  # Truncate for display
                "rows": len(datasets[dataset_id]),
                "columns": len(datasets[dataset_id].columns)
            })
        
        # Add edges for links
        best_links = {}  # Keep only the best link between each pair
        
        for link in self.results:
            pair_key = tuple(sorted([link.source_dataset, link.target_dataset]))
            
            if pair_key not in best_links or link.overlap_percentage > best_links[pair_key].overlap_percentage:
                best_links[pair_key] = link
        
        for link in best_links.values():
            edge_color = {
                LinkStrength.STRONG: "#10b981",
                LinkStrength.MODERATE: "#3b82f6",
                LinkStrength.WEAK: "#f59e0b",
                LinkStrength.BROKEN: "#ef4444",
            }.get(link.strength, "#9ca3af")
            
            edges.append({
                "source": link.source_dataset,
                "target": link.target_dataset,
                "strength": link.strength.value,
                "overlap": link.overlap_percentage,
                "source_column": link.source_column,
                "target_column": link.target_column,
                "color": edge_color
            })
            
            # Generate warnings for weak/broken links
            if link.strength == LinkStrength.BROKEN:
                warnings.append({
                    "type": "broken_link",
                    "severity": "high",
                    "message": f"Weak connection between datasets: Only {link.overlap_percentage}% overlap between {link.source_column} and {link.target_column}",
                    "datasets": [link.source_dataset, link.target_dataset]
                })
            elif link.strength == LinkStrength.WEAK:
                warnings.append({
                    "type": "weak_link",
                    "severity": "medium",
                    "message": f"Limited connection: {link.overlap_percentage}% overlap between {link.source_column} and {link.target_column}",
                    "datasets": [link.source_dataset, link.target_dataset]
                })
        
        # Check for isolated datasets (no links)
        linked_datasets = set()
        for link in best_links.values():
            linked_datasets.add(link.source_dataset)
            linked_datasets.add(link.target_dataset)
        
        for dataset_id in datasets.keys():
            if dataset_id not in linked_datasets:
                warnings.append({
                    "type": "isolated_dataset",
                    "severity": "high",
                    "message": f"Dataset has no connections to other uploaded files",
                    "datasets": [dataset_id]
                })
        
        # Calculate overall linkage health
        if not best_links:
            health_score = 0
            health_status = "no_links"
        else:
            avg_overlap = sum(l.overlap_percentage for l in best_links.values()) / len(best_links)
            strong_links = sum(1 for l in best_links.values() if l.strength == LinkStrength.STRONG)
            health_score = (avg_overlap * 0.7) + (strong_links / len(best_links) * 30)
            
            if health_score >= 80:
                health_status = "excellent"
            elif health_score >= 60:
                health_status = "good"
            elif health_score >= 40:
                health_status = "fair"
            else:
                health_status = "poor"
        
        return {
            "status": "validated",
            "health_score": round(health_score, 1),
            "health_status": health_status,
            "total_links": len(best_links),
            "strong_links": sum(1 for l in best_links.values() if l.strength == LinkStrength.STRONG),
            "warnings": warnings,
            "links": [
                {
                    "source": l.source_dataset,
                    "target": l.target_dataset,
                    "source_column": l.source_column,
                    "target_column": l.target_column,
                    "strength": l.strength.value,
                    "overlap": l.overlap_percentage,
                    "matched_count": l.matched_count,
                    "confidence": l.confidence,
                    "issues": l.issues
                }
                for l in best_links.values()
            ],
            "graph": {
                "nodes": nodes,
                "edges": edges
            }
        }


# Singleton instance
linkage_validator = LinkageValidator()


def validate_dataset_linkages(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Convenience function for linkage validation."""
    return linkage_validator.validate_linkages(datasets)
