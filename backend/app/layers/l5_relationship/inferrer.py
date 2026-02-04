# L5: Relationship Inference Layer
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from rapidfuzz import fuzz
from collections import defaultdict


def infer_relationships(
    datasets: Dict[str, pd.DataFrame],
    dataset_roles: Dict[str, str]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Infer relationships between datasets based on:
    - Column name similarity
    - Value overlap
    - Cardinality patterns
    
    Returns a dictionary of relationships.
    """
    relationships = []
    
    dataset_names = list(datasets.keys())
    
    # Compare each pair of datasets
    for i, ds1_name in enumerate(dataset_names):
        for ds2_name in dataset_names[i+1:]:
            ds1 = datasets[ds1_name]
            ds2 = datasets[ds2_name]
            
            # Find potential join columns
            join_candidates = find_join_candidates(ds1, ds2, ds1_name, ds2_name)
            
            for candidate in join_candidates:
                relationships.append(candidate)
    
    return {"relationships": relationships}


def find_join_candidates(
    df1: pd.DataFrame, 
    df2: pd.DataFrame,
    df1_name: str,
    df2_name: str
) -> List[Dict[str, Any]]:
    """Find potential join columns between two dataframes."""
    candidates = []
    
    for col1 in df1.columns:
        for col2 in df2.columns:
            # Check name similarity
            name_score = fuzz.ratio(str(col1).lower(), str(col2).lower()) / 100
            
            # Check for common ID patterns
            col1_lower = str(col1).lower()
            col2_lower = str(col2).lower()
            
            id_pattern_match = (
                ("id" in col1_lower and "id" in col2_lower) or
                (col1_lower.endswith("_id") and col2_lower.endswith("_id")) or
                (col1_lower == col2_lower)
            )
            
            if name_score > 0.7 or id_pattern_match:
                # Check value overlap
                overlap_score = calculate_value_overlap(df1[col1], df2[col2])
                
                if overlap_score > 0.3:
                    # Determine cardinality
                    cardinality = determine_cardinality(df1[col1], df2[col2])
                    
                    # Calculate combined confidence
                    confidence = (name_score * 0.4) + (overlap_score * 0.4) + (0.2 if id_pattern_match else 0)
                    
                    candidates.append({
                        "source_dataset": df1_name,
                        "source_column": col1,
                        "target_dataset": df2_name,
                        "target_column": col2,
                        "confidence": round(min(confidence, 0.99), 2),
                        "cardinality": cardinality,
                        "overlap_score": round(overlap_score, 2),
                        "name_similarity": round(name_score, 2)
                    })
    
    # Sort by confidence, return top candidates
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates[:5]  # Top 5 per pair


def calculate_value_overlap(s1: pd.Series, s2: pd.Series) -> float:
    """Calculate the overlap between values in two series."""
    # Sample for large datasets
    sample_size = min(1000, len(s1), len(s2))
    
    set1 = set(s1.dropna().astype(str).head(sample_size))
    set2 = set(s2.dropna().astype(str).head(sample_size))
    
    if not set1 or not set2:
        return 0.0
    
    intersection = set1.intersection(set2)
    smaller_set = min(len(set1), len(set2))
    
    return len(intersection) / smaller_set if smaller_set > 0 else 0.0


def determine_cardinality(s1: pd.Series, s2: pd.Series) -> str:
    """Determine the cardinality of the relationship."""
    unique1 = s1.nunique()
    unique2 = s2.nunique()
    count1 = len(s1)
    count2 = len(s2)
    
    # One-to-one: both columns have mostly unique values
    if unique1 / max(count1, 1) > 0.9 and unique2 / max(count2, 1) > 0.9:
        return "one_to_one"
    
    # One-to-many or many-to-one
    if unique1 / max(count1, 1) > 0.9:
        return "one_to_many"
    if unique2 / max(count2, 1) > 0.9:
        return "many_to_one"
    
    # Many-to-many
    return "many_to_many"


def build_entity_graph(
    relationships: List[Dict[str, Any]],
    dataset_roles: Dict[str, str]
) -> Dict[str, Any]:
    """Build an entity relationship graph."""
    nodes = []
    edges = []
    
    # Add datasets as nodes
    dataset_set = set()
    for rel in relationships:
        dataset_set.add(rel["source_dataset"])
        dataset_set.add(rel["target_dataset"])
    
    for ds in dataset_set:
        nodes.append({
            "id": ds,
            "type": "dataset",
            "role": dataset_roles.get(ds, "unknown")
        })
    
    # Add relationships as edges
    for rel in relationships:
        if rel["confidence"] > 0.5:  # Only include confident relationships
            edges.append({
                "source": rel["source_dataset"],
                "target": rel["target_dataset"],
                "source_column": rel["source_column"],
                "target_column": rel["target_column"],
                "cardinality": rel["cardinality"],
                "confidence": rel["confidence"]
            })
    
    return {
        "nodes": nodes,
        "edges": edges
    }
