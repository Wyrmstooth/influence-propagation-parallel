import os
import time
import random
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm

def load_graph(filepath):
    """Load graph from various formats"""
    _, ext = os.path.splitext(filepath)
    
    if ext == '.edgelist':
        G = nx.read_edgelist(filepath, nodetype=int, create_using=nx.DiGraph())
    elif ext == '.txt':
        # Check if this is a .txt file with specific P2P-Gnutella format
        if 'Gnutella' in filepath:
            G = load_gnutella_graph(filepath)
        else:
            # Assuming space-separated edge list
            G = nx.read_edgelist(filepath, nodetype=int, create_using=nx.DiGraph())
    elif ext == '.gml':
        G = nx.read_gml(filepath)
    elif ext == '.pkl':
        with open(filepath, 'rb') as f:
            G = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Set default edge weights if not available
    for u, v, d in G.edges(data=True):
        if 'weight' not in d:
            G[u][v]['weight'] = 0.1  # Default propagation probability
    
    return G

def load_gnutella_graph(filepath):
    """
    Load P2P-Gnutella graph from specified format:
    # Nodes: <n> Edges: <m>
    # FromNodeId    ToNodeId
    ...
    """
    G = nx.DiGraph()
    
    with open(filepath, 'r') as f:
        # Skip header lines that start with #
        for line in f:
            if not line.startswith('#'):
                # Parse edges
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        source = int(parts[0])
                        target = int(parts[1])
                        G.add_edge(source, target, weight=0.1)  # Default weight
                    except ValueError:
                        # Skip lines that can't be parsed as integers
                        continue
    
    return G

def independent_cascade(G, seed_nodes, mc_simulations=1000):
    """
    Run the Independent Cascade model for influence propagation
    
    Parameters:
    -----------
    G : networkx.DiGraph
        Input graph with weighted edges
    seed_nodes : list
        Initial set of activated nodes
    mc_simulations : int
        Number of Monte Carlo simulations
    
    Returns:
    --------
    float : Average number of activated nodes
    """
    # Precompute neighbors and weights for faster access
    neighbors_dict = {}
    weights_dict = {}
    for node in G:
        neighbors_dict[node] = list(G.neighbors(node))
        weights_dict[node] = [G[node][v]['weight'] for v in neighbors_dict[node]]
    
    spread = 0
    
    for _ in range(mc_simulations):
        # Simulate propagation
        activated = set(seed_nodes)
        front = list(seed_nodes)
        
        while front:
            new_front = []
            for node in front:
                neighbors = neighbors_dict[node]
                weights = weights_dict[node]
                
                for i, neighbor in enumerate(neighbors):
                    if neighbor not in activated:
                        # Probability of influence
                        prob = weights[i]
                        if random.random() < prob:
                            activated.add(neighbor)
                            new_front.append(neighbor)
            front = new_front
            
        spread += len(activated)
    
    return spread / mc_simulations

def partition_graph(G, n_parts):
    """
    Partition the graph using METIS with robust error handling
    
    Parameters:
    -----------
    G : networkx.Graph
        Input graph (must be undirected for METIS)
    n_parts : int
        Number of partitions
    
    Returns:
    --------
    dict : Mapping of node to partition ID
    """
    try:
        import metis
        
        # Check if graph is too small for partitioning
        if len(G) <= n_parts:
            # Return simple partitioning if graph is smaller than requested parts
            partition = {}
            for i, node in enumerate(G.nodes()):
                partition[node] = i % n_parts
            return partition
            
        # Add self-loops to avoid empty adjacency lists which cause METIS errors
        for node in G.nodes():
            if len(list(G.neighbors(node))) == 0:
                G.add_edge(node, node)
        
        # Convert graph to format needed by METIS
        # METIS requires consecutive integer node labels starting from 0
        node_map = {n: i for i, n in enumerate(G.nodes())}
        reverse_map = {i: n for n, i in node_map.items()}
        
        # Create adjacency list for METIS
        adjacency = [[] for _ in range(len(G))]
        for u, v in G.edges():
            adjacency[node_map[u]].append(node_map[v])
            if u != v:  # Don't add self-loops twice
                adjacency[node_map[v]].append(node_map[u])  # For undirected graph
        
        # Remove duplicate edges
        adjacency = [list(set(neighbors)) for neighbors in adjacency]
        
        # Check for empty adjacency lists
        for i, adj in enumerate(adjacency):
            if not adj:
                # Add self-loop to avoid METIS errors
                adjacency[i].append(i)
        
        # Perform partitioning
        _, partition_vector = metis.part_graph(adjacency, n_parts)
        
        # Map back to original node IDs
        partition = {reverse_map[i]: part for i, part in enumerate(partition_vector)}
        
        return partition
    except Exception as e:
        print(f"METIS partitioning failed with error: {e}")
        print("Using NetworkX's spectral partitioning instead.")
        
        # Fallback to NetworkX's partitioning
        try:
            # Try spectral clustering first
            import numpy as np
            from sklearn.cluster import SpectralClustering
            
            # Get adjacency matrix
            adj_matrix = nx.to_numpy_array(G)
            
            # Apply spectral clustering
            clustering = SpectralClustering(n_clusters=n_parts, 
                                           affinity='precomputed', 
                                           assign_labels='discretize',
                                           random_state=42)
            labels = clustering.fit_predict(adj_matrix)
            
            # Convert to node-to-partition mapping
            node_partition = {}
            for i, node in enumerate(G.nodes()):
                node_partition[node] = int(labels[i])
                
            return node_partition
        except:
            print("Spectral clustering failed. Using simple partitioning.")
            # Simple partitioning as last resort
            partition = {}
            for i, node in enumerate(G.nodes()):
                partition[node] = i % n_parts
            return partition

def evaluate_partition(G, partition):
    """
    Evaluate the quality of a graph partition
    
    Parameters:
    -----------
    G : networkx.Graph
        Input graph
    partition : dict
        Mapping of node to partition ID
    
    Returns:
    --------
    float : Edge cut ratio (fraction of edges that cross partitions)
    """
    total_edges = G.number_of_edges()
    cut_edges = 0
    
    for u, v in G.edges():
        if partition[u] != partition[v]:
            cut_edges += 1
    
    return cut_edges / total_edges