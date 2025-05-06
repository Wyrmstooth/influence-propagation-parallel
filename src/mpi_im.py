import time
import numpy as np
import networkx as nx
from mpi4py import MPI
import pickle
import random
from utils import independent_cascade, partition_graph

def mpi_influence_maximization(G, k, mc_simulations=1000):
    """
    Implements a parallel version of the greedy algorithm for influence maximization using MPI
    
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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    start_time = time.time()
    
    # All processes have the entire graph
    # Partition nodes among processes
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    # Each process gets a subset of nodes to evaluate
    local_nodes = nodes[rank:n_nodes:size]
    
    # Initialize empty seed set
    seed_set = []
    
    # Greedy algorithm: select nodes incrementally
    for _ in range(k):
        best_local_node = None
        best_local_spread = 0
        
        # Try adding each remaining node to the seed set
        for node in local_nodes:
            if node not in seed_set:
                spread = independent_cascade(G, seed_set + [node], mc_simulations // size)
                
                if spread > best_local_spread:
                    best_local_spread = spread
                    best_local_node = node
        
        # Gather results from all processes
        best_nodes = comm.gather((best_local_spread, best_local_node), root=0)
        
        # Process 0 selects the best node
        if rank == 0:
            best_node = None
            best_spread = 0
            
            for spread, node in best_nodes:
                if spread > best_spread and node is not None:
                    best_spread = spread
                    best_node = node
        else:
            best_node = None
            best_spread = 0
        
        # Broadcast the selected node to all processes
        best_node = comm.bcast(best_node, root=0)
        best_spread = comm.bcast(best_spread, root=0)
        
        if best_node is not None:
            seed_set.append(best_node)
            if rank == 0:
                print(f"Selected node {best_node} with marginal gain {best_spread}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Calculate final influence spread (only process 0)
    if rank == 0:
        final_spread = independent_cascade(G, seed_set, mc_simulations)
    else:
        final_spread = 0
        
    return seed_set, final_spread, execution_time

def mpi_celf(G, k, mc_simulations=1000):
    """
    Implements a parallel version of the CELF algorithm for influence maximization using MPI
    
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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    start_time = time.time()
    
    # Adjust MC simulations for larger graphs
    if len(G) > 1000:
        mc_simulations = max(100, mc_simulations // 2)
    
    # All processes have the entire graph
    # Partition nodes among processes for initial evaluation
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    # Each process gets a subset of nodes to evaluate initially
    local_nodes = nodes[rank:n_nodes:size]
    
    # Initialize empty seed set
    seed_set = []
    
    # Calculate initial marginal gains for local nodes
    local_queue = []
    for node in local_nodes:
        spread = independent_cascade(G, [node], mc_simulations // size)
        local_queue.append((spread, node, 0))  # (marginal_gain, node_id, flag)
    
    # Gather initial results from all processes
    all_queues = comm.gather(local_queue, root=0)
    
    # Process 0 aggregates and sorts the queue
    if rank == 0:
        queue = []
        for q in all_queues:
            queue.extend(q)
        queue.sort(reverse=True)
    else:
        queue = None
    
    # Select first node
    if rank == 0:
        best_node_gain, best_node, _ = queue[0]
        seed_set.append(best_node)
        queue.pop(0)
        
        # Mark all nodes for recomputation
        for i in range(len(queue)):
            queue[i] = (queue[i][0], queue[i][1], 0)
    
    # Broadcast the selected seed node to all processes
    seed_set = comm.bcast(seed_set, root=0)
    
    # Greedy algorithm with lazy evaluations
    for _ in range(k - 1):
        best_node = None
        
        if rank == 0:
            # Process 0 manages the queue
            while queue:
                # Get node with highest marginal gain
                top_gain, top_node, top_flag = queue[0]
                queue.pop(0)
                
                # If marginal gain is up-to-date, select this node
                if top_flag == 1:
                    best_node = top_node
                    print(f"Selected node {best_node} with marginal gain {top_gain}")
                    break
                
                # Distribute recomputation tasks to worker processes
                node_to_evaluate = top_node
                nodes_to_evaluate = [node_to_evaluate] * size
            
            # If no node with up-to-date marginal gain, select top node for recomputation
            if best_node is None and queue:
                node_to_evaluate = queue[0][1]
                nodes_to_evaluate = [node_to_evaluate] * size
            else:
                nodes_to_evaluate = [None] * size
        else:
            nodes_to_evaluate = None
            
        # Broadcast node to evaluate
        node_to_evaluate = comm.scatter(nodes_to_evaluate, root=0)
        
        # If all workers have found up-to-date marginal gains, move to next iteration
        if node_to_evaluate is None:
            best_node = comm.bcast(best_node, root=0)
            if best_node is not None:
                seed_set.append(best_node)
            continue
        
        # Workers compute marginal gain
        local_gain = 0
        if node_to_evaluate is not None and node_to_evaluate not in seed_set:
            # Compute marginal gain
            mc_part = max(10, mc_simulations // size)
            gain = independent_cascade(G, seed_set + [node_to_evaluate], mc_part) - \
                   independent_cascade(G, seed_set, mc_part)
            local_gain = gain
        
        # Gather results from all workers
        all_gains = comm.gather(local_gain, root=0)
        
        # Process 0 aggregates results
        if rank == 0:
            avg_gain = sum(all_gains) / size
            
            # Update queue with recomputed gain
            queue.append((avg_gain, node_to_evaluate, 1))
            queue.sort(reverse=True)
            
            # Check if we can select this node
            if queue and queue[0][2] == 1:
                best_node = queue[0][1]
                best_gain = queue[0][0]
                queue.pop(0)
                print(f"Selected node {best_node} with marginal gain {best_gain}")
            else:
                best_node = None
        
        # Broadcast selected node to all processes
        best_node = comm.bcast(best_node, root=0)
        
        # Update seed set
        if best_node is not None:
            seed_set.append(best_node)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Calculate final influence spread (only process 0)
    if rank == 0:
        final_spread = independent_cascade(G, seed_set, mc_simulations)
    else:
        final_spread = 0
        
    return seed_set, final_spread, execution_time

if __name__ == "__main__":
    import sys
    import os
    import argparse
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import load_graph
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Running with {size} MPI processes")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="MPI Influence Maximization")
    parser.add_argument("--dataset", type=str, default="../datasets/sample_graph.pkl",
                        help="Path to graph dataset")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of seed nodes to select")
    parser.add_argument("--mc", type=int, default=100,
                        help="Monte Carlo simulations for influence estimation")
    parser.add_argument("--use_gnutella", action="store_true",
                        help="Use the P2P-Gnutella04 dataset")
    args = parser.parse_args()
    
    # Load graph (only on root process)
    if rank == 0:
        if args.use_gnutella:
            G = load_graph("../datasets/p2p-Gnutella04.txt")
            # Reduce simulations for large graph
            args.mc = max(20, args.mc // 2)
            print(f"Using reduced MC simulations: {args.mc} for Gnutella dataset")
        else:
            G = load_graph(args.dataset)
        print(f"Graph loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")
    else:
        G = None
    
    # Broadcast graph to all processes
    G = comm.bcast(G, root=0)
    
    # Set parameters
    k = args.k
    mc_simulations = args.mc
    
    # Run MPI CELF algorithm
    seed_nodes, influence, runtime = mpi_celf(G, k, mc_simulations)
    
    if rank == 0:
        print(f"\nSelected seed set: {seed_nodes}")
        print(f"Estimated influence spread: {influence:.2f}")
        print(f"Execution time: {runtime:.2f} seconds")