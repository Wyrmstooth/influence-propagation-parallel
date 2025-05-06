import time
import numpy as np
import networkx as nx
from mpi4py import MPI
import pickle
from multiprocessing import Pool
import random
import os
from utils import independent_cascade, partition_graph

def _ic_worker(args):
    """Worker function for parallel IC simulation"""
    G, seed_nodes, num_simulations = args
    
    # Precompute neighbors and weights for faster access
    neighbors_dict = {}
    weights_dict = {}
    for node in G:
        neighbors_dict[node] = list(G.neighbors(node))
        weights_dict[node] = [G[node][v]['weight'] for v in neighbors_dict[node]]
    
    total_spread = 0
    for _ in range(num_simulations):
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
            
        total_spread += len(activated)
    
    return total_spread

def parallel_independent_cascade(G, seed_nodes, mc_simulations=1000):
    """
    Run the Independent Cascade model with OpenMP-like parallelism using Python's multiprocessing
    
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
    # Determine number of processes based on available cores (mimicking OpenMP)
    num_processes = min(os.cpu_count(), 8)  # Limit to 8 cores
    
    # Adjust for very small simulation counts
    if mc_simulations < num_processes:
        num_processes = 1
    
    # Split simulations among processes
    chunk_size = mc_simulations // num_processes
    remainder = mc_simulations % num_processes
    
    # Create tasks with adjusted simulation counts
    tasks = []
    for i in range(num_processes):
        sim_count = chunk_size + (1 if i < remainder else 0)
        if sim_count > 0:
            tasks.append((G, seed_nodes, sim_count))
    
    # Execute in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(_ic_worker, tasks)
    
    # Calculate average spread
    total_spread = sum(results)
    return total_spread / mc_simulations

def mpi_omp_celf(G, k, mc_simulations=1000):
    """
    Implements a hybrid MPI+OpenMP version of the CELF algorithm for influence maximization
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    start_time = time.time()
    
    # Adjust MC simulations for larger graphs
    if len(G) > 1000:
        mc_simulations = max(100, mc_simulations // 2)
    
    # Partition the graph for MPI processes
    if rank == 0:
        G_undir = G.to_undirected()
        node_partition = partition_graph(G_undir, size)
        partitioned_nodes = [[] for _ in range(size)]
        for node, part in node_partition.items():
            partitioned_nodes[part].append(node)
    else:
        partitioned_nodes = None
    
    # Broadcast partition information to all processes
    local_nodes = comm.scatter(partitioned_nodes, root=0)
    
    # Initialize empty seed set
    seed_set = []
    
    # Calculate initial marginal gains for local nodes
    local_queue = []
    for node in local_nodes:
        # Use sequential IC simulation inside each MPI process for speed
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
        for i in range(len(queue)):
            queue[i] = (queue[i][0], queue[i][1], 0)
    
    # Broadcast the selected seed node to all processes
    seed_set = comm.bcast(seed_set, root=0)
    
    # Greedy algorithm with lazy evaluations
    for _ in range(k - 1):
        best_node = None
        
        if rank == 0:
            while queue:
                top_gain, top_node, top_flag = queue[0]
                queue.pop(0)
                if top_flag == 1:
                    best_node = top_node
                    print(f"Selected node {best_node} with marginal gain {top_gain}")
                    break
                node_to_evaluate = top_node
                nodes_to_evaluate = [node_to_evaluate] * size
            if best_node is None and queue:
                node_to_evaluate = queue[0][1]
                nodes_to_evaluate = [node_to_evaluate] * size
            else:
                nodes_to_evaluate = [None] * size
        else:
            nodes_to_evaluate = None
            
        node_to_evaluate = comm.scatter(nodes_to_evaluate, root=0)
        
        if node_to_evaluate is None:
            best_node = comm.bcast(best_node, root=0)
            if best_node is not None:
                seed_set.append(best_node)
            continue
        
        # Workers compute marginal gain using sequential IC simulation
        local_gain = 0
        if node_to_evaluate is not None and node_to_evaluate not in seed_set:
            mc_part = max(10, mc_simulations // size)
            gain = independent_cascade(G, seed_set + [node_to_evaluate], mc_part) - \
                   independent_cascade(G, seed_set, mc_part)
            local_gain = gain
        
        all_gains = comm.gather(local_gain, root=0)
        
        if rank == 0:
            avg_gain = sum(all_gains) / size
            queue.append((avg_gain, node_to_evaluate, 1))
            queue.sort(reverse=True)
            if queue and queue[0][2] == 1:
                best_node = queue[0][1]
                best_gain = queue[0][0]
                queue.pop(0)
                print(f"Selected node {best_node} with marginal gain {best_gain}")
            else:
                best_node = None
        
        best_node = comm.bcast(best_node, root=0)
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
    parser = argparse.ArgumentParser(description="Hybrid MPI+OpenMP Influence Maximization")
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
    
    # Run hybrid MPI+OpenMP CELF algorithm
    seed_nodes, influence, runtime = mpi_omp_celf(G, k, mc_simulations)
    
    if rank == 0:
        print(f"\nSelected seed set: {seed_nodes}")
        print(f"Estimated influence spread: {influence:.2f}")
        print(f"Execution time: {runtime:.2f} seconds")