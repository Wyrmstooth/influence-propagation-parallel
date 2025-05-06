import time
import numpy as np
import networkx as nx
from utils import independent_cascade

def greedy_influence_maximization(G, k, mc_simulations=1000):
    """
    Implements the greedy algorithm for influence maximization
    
    Parameters:
    -----------
    G : networkx.DiGraph
        Input graph with weighted edges
    k : int
        Number of seed nodes to select
    mc_simulations : int
        Number of Monte Carlo simulations for influence estimation
    
    Returns:
    --------
    list : Selected seed nodes
    float : Estimated influence spread
    float : Execution time
    """
    start_time = time.time()
    
    # Initialize empty seed set
    seed_set = []
    
    # Track nodes that have been considered
    remaining_nodes = list(G.nodes())
    
    # Greedy algorithm: select nodes incrementally
    for _ in range(k):
        best_node = None
        best_spread = 0
        
        # Try adding each remaining node to the seed set
        for node in remaining_nodes:
            spread = independent_cascade(G, seed_set + [node], mc_simulations)
            
            if spread > best_spread:
                best_spread = spread
                best_node = node
        
        if best_node is not None:
            seed_set.append(best_node)
            remaining_nodes.remove(best_node)
            print(f"Selected node {best_node} with marginal gain {best_spread}")
        
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Calculate final influence spread
    final_spread = independent_cascade(G, seed_set, mc_simulations)
    
    return seed_set, final_spread, execution_time

def celf_influence_maximization(G, k, mc_simulations=1000):
    """
    Implements the CELF algorithm for influence maximization
    
    Parameters:
    -----------
    G : networkx.DiGraph
        Input graph with weighted edges
    k : int
        Number of seed nodes to select
    mc_simulations : int
        Number of Monte Carlo simulations for influence estimation
    
    Returns:
    --------
    list : Selected seed nodes
    float : Estimated influence spread
    float : Execution time
    """
    start_time = time.time()
    
    # Initialize empty seed set
    seed_set = []
    
    # Initialize priority queue with marginal gains
    # Format: (marginal_gain, node_id, flag)
    # flag: 0 = need to recompute, 1 = up-to-date
    queue = []
    
    # Calculate initial marginal gains
    for node in G.nodes():
        spread = independent_cascade(G, [node], mc_simulations)
        queue.append((spread, node, 0))
    
    # Sort in descending order of marginal gain
    queue.sort(reverse=True)
    
    # Select first node
    best_node_gain, best_node, _ = queue[0]
    seed_set.append(best_node)
    queue.pop(0)
    
    # Mark all nodes for recomputation
    for i in range(len(queue)):
        queue[i] = (queue[i][0], queue[i][1], 0)
    
    # Greedy algorithm with lazy evaluations
    for _ in range(k - 1):
        best_node = None
        
        while queue:
            # Get node with highest marginal gain
            top_gain, top_node, top_flag = queue[0]
            queue.pop(0)
            
            # If marginal gain is up-to-date, select this node
            if top_flag == 1:
                best_node = top_node
                break
            
            # Recompute marginal gain
            new_gain = independent_cascade(G, seed_set + [top_node], mc_simulations) - \
                       independent_cascade(G, seed_set, mc_simulations)
            
            # Update queue with recomputed gain
            queue.append((new_gain, top_node, 1))
            queue.sort(reverse=True)
        
        if best_node is not None:
            seed_set.append(best_node)
            print(f"Selected node {best_node} with marginal gain {top_gain}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Calculate final influence spread
    final_spread = independent_cascade(G, seed_set, mc_simulations)
    
    return seed_set, final_spread, execution_time

if __name__ == "__main__":
    import sys
    import os
    import argparse
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import load_graph
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Sequential Influence Maximization")
    parser.add_argument("--dataset", type=str, default="../datasets/sample_graph.pkl",
                        help="Path to graph dataset")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of seed nodes to select")
    parser.add_argument("--mc", type=int, default=100,
                        help="Monte Carlo simulations for influence estimation")
    parser.add_argument("--use_gnutella", action="store_true",
                        help="Use the P2P-Gnutella04 dataset")
    args = parser.parse_args()
    
    # Load graph
    if args.use_gnutella:
        G = load_graph("../datasets/p2p-Gnutella04.txt")
        # Reduce simulations for large graph
        args.mc = max(20, args.mc // 2)
        print(f"Using reduced MC simulations: {args.mc} for Gnutella dataset")
    else:
        G = load_graph(args.dataset)
    
    print(f"Graph loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Set parameters
    k = args.k
    mc_simulations = args.mc
    
    # Run CELF algorithm
    seed_nodes, influence, runtime = celf_influence_maximization(G, k, mc_simulations)
    
    print(f"\nSelected seed set: {seed_nodes}")
    print(f"Estimated influence spread: {influence:.2f}")
    print(f"Execution time: {runtime:.2f} seconds")