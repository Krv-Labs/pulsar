"""
FastMCP Server for Pulsar.

Exposes "Thick Tools" for topological data analysis and interpretation.
"""

import os
from typing import Dict, Optional

import pandas as pd
from fastmcp import FastMCP

from pulsar.pipeline import ThemaRS
from pulsar.mcp.interpreter import resolve_clusters, build_dossier, dossier_to_markdown

# Initialize FastMCP
mcp = FastMCP("Pulsar")

# Global state to keep track of the last fitted model and data
# In a production environment, this might be handled via a session store or database
_last_model: Optional[ThemaRS] = None
_last_data: Optional[pd.DataFrame] = None
_last_clusters: Optional[pd.Series] = None


@mcp.tool()
def run_topological_sweep(config_path: str) -> str:
    """
    Runs the ThemaRS topological sweep pipeline on a dataset.
    
    Args:
        config_path: Path to the params.yaml configuration file.
    
    Returns:
        A summary of the generated Cosmic Graph.
    """
    global _last_model, _last_data
    
    if not os.path.exists(config_path):
        return f"Error: Config file not found at {config_path}"
    
    try:
        model = ThemaRS(config_path)
        model.fit()
        
        _last_model = model
        # We need to retrieve the original data used for fitting for interpretation
        # ThemaRS stores it in _data (after copy)
        _last_data = model._data
        
        graph = model.cosmic_graph
        return (
            f"Successfully ran topological sweep.\n"
            f"- Nodes: {graph.number_of_nodes()}\n"
            f"- Edges: {graph.number_of_edges()}\n"
            f"- Resolution: {model.resolved_threshold:.4f}\n"
            f"- Data Shape: {_last_data.shape}"
        )
    except Exception as e:
        return f"Error running sweep: {str(e)}"


@mcp.tool()
def generate_cluster_dossier() -> str:
    """
    Analyzes the topological graph, finds stable clusters, and generates 
     a statistical dossier for semantic interpretation.
    
    Returns:
        A Markdown-formatted dossier describing the relative shifts, 
        homogeneity, and defining features of each cluster.
    """
    global _last_model, _last_data, _last_clusters
    
    if _last_model is None or _last_data is None:
        return "Error: No model found. Run run_topological_sweep() first."
    
    try:
        # 1. Resolve Clusters (Fallback to Spectral Silhouette if needed)
        clusters = resolve_clusters(_last_model)
        _last_clusters = clusters
        
        # 2. Build Statistical Dossier
        dossier = build_dossier(_last_model, _last_data, clusters)
        
        # 3. Convert to Markdown
        return dossier_to_markdown(dossier)
    except Exception as e:
        return f"Error generating dossier: {str(e)}"


@mcp.tool()
def export_labeled_data(cluster_names: Dict[int, str], output_path: str) -> str:
    """
    Assigns human-readable names to clusters and exports the labeled dataset to CSV.
    
    Args:
        cluster_names: A dictionary mapping cluster IDs (ints) to semantic names (strings).
        output_path: Path where the labeled CSV should be saved.
    
    Returns:
        Confirmation message.
    """
    global _last_data, _last_clusters
    
    if _last_data is None or _last_clusters is None:
        return "Error: No data or clusters found. Run generate_cluster_dossier() first."
    
    try:
        df = _last_data.copy()
        df['topological_cluster_id'] = _last_clusters
        
        # Map IDs to names
        # Ensure keys are ints (LLMs sometimes send strings in JSON)
        names_map = {int(k): v for k, v in cluster_names.items()}
        df['topological_cluster_name'] = df['topological_cluster_id'].map(names_map)
        
        df.to_csv(output_path, index=False)
        return f"Successfully exported labeled data to {output_path}"
    except Exception as e:
        return f"Error exporting data: {str(e)}"


def main():
    """Entry point for the pulsar-mcp CLI."""
    mcp.run()


if __name__ == "__main__":
    main()
