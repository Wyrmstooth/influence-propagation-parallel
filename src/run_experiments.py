import os
import time
import argparse
import pickle
import numpy as np
import networkx as nx
import sys
import matplotlib.pyplot as plt
import threading

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from utils import load_graph, partition_graph, evaluate_partition
from sequential import celf_influence_maximization
from visualization import (
    visualize_seed_nodes, plot_execution_times, plot_speedup, 
    plot_strong_scaling, plot_weak_scaling, visualize_partitioning
)

def create_sample_graph(n_nodes=100, edge_prob=0.05, seed=42):
    """Create a sample graph for testing"""
    G = nx.gnp_random_graph(n_nodes, edge_prob, seed=seed, directed=True)
    
    # Add weights to edges
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.01, 0.1)
    
    return G

def load_or_create_datasets(datasets_dir):
    """Load existing or create new datasets"""
    os.makedirs(datasets_dir, exist_ok=True)
    
    datasets = {}
    sample_sizes = [100, 500, 1000, 2000]
    
    for size in sample_sizes:
        filename = os.path.join(datasets_dir, f'random_graph_{size}.pkl')
        
        if os.path.exists(filename):
            print(f"Loading dataset: {filename}")
            with open(filename, 'rb') as f:
                G = pickle.load(f)
        else:
            print(f"Creating new dataset: {filename}")
            G = create_sample_graph(n_nodes=size, edge_prob=5/size)
            with open(filename, 'wb') as f:
                pickle.dump(G, f)
                
        datasets[size] = G
    
    return datasets

def time_bounded_execution(func, *args, max_time=3600, **kwargs):
    """Run a function with a time limit"""
    start_time = time.time()
    result = None
    
    def target():
        nonlocal result
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print(f"Error during execution: {e}")
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    
    thread.join(timeout=max_time)
    if thread.is_alive():
        print(f"Execution time limit ({max_time} seconds) exceeded. Terminating.")
        return None, 0, max_time
    
    return result if result else (None, 0, time.time() - start_time)

def run_strong_scaling_experiments(G, k, mc_simulations, output_dir, max_processes=8):
    """Run strong scaling experiments"""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        if size > 1 and rank == 0:
            print("Error: Strong scaling experiments must be started with a single process")
            print("Please run this script without mpirun first")
            return
    except ImportError:
        pass
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run sequential version first
    print("Running sequential CELF...")
    seed_nodes, influence, seq_runtime = celf_influence_maximization(G, k, mc_simulations)
    
    results = {
        1: seq_runtime
    }
    
    # Run with different numbers of processes
    for n_procs in range(2, max_processes + 1):
        print(f"Running with {n_procs} processes...")
        
        # Execute mpirun command programmatically
        cmd = f"mpirun -np {n_procs} python -m src.mpi_im {G} {k} {mc_simulations}"
        
        start_time = time.time()
        os.system(cmd)
        runtime = time.time() - start_time
        
        results[n_procs] = runtime
    
    # Save results
    with open(os.path.join(output_dir, 'strong_scaling_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Plot results
    num_processors = list(results.keys())
    execution_times = [results[p] for p in num_processors]
    
    plot_strong_scaling(
        num_processors, 
        execution_times,
        output_path=os.path.join(output_dir, 'strong_scaling.png')
    )

def run_weak_scaling_experiments(datasets, k, mc_simulations, output_dir, max_processes=8):
    """Run weak scaling experiments"""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        if size > 1 and rank == 0:
            print("Error: Weak scaling experiments must be started with a single process")
            print("Please run this script without mpirun first")
            return
    except ImportError:
        pass
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    problem_sizes = []
    
    # Get sorted dataset sizes
    sizes = sorted(datasets.keys())
    
    # Run with different numbers of processes and corresponding dataset sizes
    for i, n_procs in enumerate(range(1, min(len(sizes), max_processes) + 1)):
        size = sizes[i]
        G = datasets[size]
        problem_sizes.append(size)
        
        print(f"Running with {n_procs} processes on graph with {size} nodes...")
        
        if n_procs == 1:
            # Run sequential version
            _, _, runtime = celf_influence_maximization(G, k, mc_simulations)
        else:
            # Execute mpirun command programmatically
            cmd = f"mpirun -np {n_procs} python -m src.mpi_im {G} {k} {mc_simulations}"
            
            start_time = time.time()
            os.system(cmd)
            runtime = time.time() - start_time
        
        results[n_procs] = runtime
    
    # Save results
    with open(os.path.join(output_dir, 'weak_scaling_results.pkl'), 'wb') as f:
        pickle.dump((results, problem_sizes), f)
    
    # Plot results
    num_processors = list(results.keys())
    execution_times = [results[p] for p in num_processors]
    
    plot_weak_scaling(
        num_processors, 
        execution_times,
        problem_sizes,
        output_path=os.path.join(output_dir, 'weak_scaling.png')
    )

def print_banner(title):
    print("\n" + "=" * 60)
    print(f"{title.center(60)}")
    print("=" * 60)

def print_section(title):
    print("\n" + "-" * 60)
    print(f"{title}")
    print("-" * 60)

def print_result_table(results):
    print_section("Summary of Results")
    print(f"{'Implementation':<18} {'Time (s)':<12} {'Influence':<12} {'Seed Nodes'}")
    print("-" * 60)
    for impl, res in results.items():
        time_val = f"{res['time']:.2f}" if res['time'] else "-"
        influence_val = f"{res['influence']:.2f}" if res['influence'] else "-"
        seeds = ', '.join(str(x) for x in res['seed_nodes']) if res['seed_nodes'] else "-"
        print(f"{impl:<18} {time_val:<12} {influence_val:<12} {seeds}")
    print("-" * 60)

def run_implementation_comparison(G, k, mc_simulations, output_dir, max_time=3600):
    """Compare different implementations"""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        mpi_available = True
    except ImportError:
        print("MPI not available. Running in sequential mode only.")
        rank = 0
        size = 1
        mpi_available = False

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Adjust MC simulations for large graphs
    if len(G) > 1000:
        mc_simulations = max(50, mc_simulations // 2)

    if rank == 0:
        print_banner("Influence Maximization Implementation Comparison")
        print(f"Graph size: {len(G)} nodes, k: {k}, MC simulations: {mc_simulations}")
        print(f"Output directory: {output_dir}")

    # Run sequential implementation
    if rank == 0:
        print_section("Sequential CELF Implementation")
        seed_nodes_seq, influence_seq, time_seq = time_bounded_execution(
            celf_influence_maximization, G, k, mc_simulations, max_time=max_time
        )
        results['Sequential'] = {
            'seed_nodes': seed_nodes_seq,
            'influence': influence_seq,
            'time': time_seq
        }
        print(f"  Done. Time: {time_seq:.2f}s, Influence: {influence_seq:.2f}")

    # Run MPI-only implementation if MPI is available and we have more than 1 process
    if mpi_available and size > 1:
        from mpi_im import mpi_celf
        if rank == 0:
            print_section("MPI-only CELF Implementation")
        seed_nodes_mpi, influence_mpi, time_mpi = mpi_celf(G, k, mc_simulations)
        if rank == 0:
            results['MPI'] = {
                'seed_nodes': seed_nodes_mpi,
                'influence': influence_mpi,
                'time': time_mpi
            }
            print(f"  Done. Time: {time_mpi:.2f}s, Influence: {influence_mpi:.2f}")

        from hybrid_im import mpi_omp_celf
        if rank == 0:
            print_section("Hybrid MPI+OpenMP CELF Implementation")
        seed_nodes_hybrid, influence_hybrid, time_hybrid = mpi_omp_celf(G, k, mc_simulations)
        if rank == 0:
            results['MPI+OpenMP'] = {
                'seed_nodes': seed_nodes_hybrid,
                'influence': influence_hybrid,
                'time': time_hybrid
            }
            print(f"  Done. Time: {time_hybrid:.2f}s, Influence: {influence_hybrid:.2f}")

    elif rank == 0 and mpi_available:
        print_section("MPI Implementations Skipped")
        print("  MPI implementations require at least 2 processes.")
        print("  Run with: mpirun -np <N> python run_experiments.py")

    if rank == 0:
        # Save results
        with open(os.path.join(output_dir, 'implementation_comparison.pkl'), 'wb') as f:
            pickle.dump(results, f)

        # Print summary table
        print_result_table(results)

        # Plot execution times if we have more than one implementation
        if len(results) > 1:
            plot_execution_times(
                results,
                output_path=os.path.join(output_dir, 'execution_times.png')
            )
            plot_speedup(
                results,
                output_path=os.path.join(output_dir, 'speedup.png')
            )

        # Visualize seed nodes for each implementation
        for impl in results:
            if results[impl]['seed_nodes'] is not None:
                visualize_seed_nodes(
                    G,
                    results[impl]['seed_nodes'],
                    output_path=os.path.join(output_dir, f'seed_nodes_{impl}.png')
                )

        # Visualize graph partitioning if we have multiple processes
        if mpi_available and size > 1:
            try:
                G_undir = G.to_undirected()
                partition = partition_graph(G_undir, size)
                edge_cut = evaluate_partition(G, partition)
                print_section("Graph Partitioning")
                print(f"  Number of partitions: {size}")
                print(f"  Edge cut ratio: {edge_cut:.4f}")
                visualize_partitioning(
                    G_undir,
                    partition,
                    output_path=os.path.join(output_dir, f'partitioning_{size}_parts.png')
                )
            except Exception as e:
                print(f"Failed to visualize partitioning: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='Run influence maximization experiments')
    parser.add_argument('--experiment', type=str, choices=['comparison', 'strong', 'weak', 'all'], 
                        default='comparison', help='Type of experiment to run')
    parser.add_argument('--k', type=int, default=5, help='Number of seed nodes to select')
    parser.add_argument('--mc', type=int, default=100, help='Number of Monte Carlo simulations')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--max_procs', type=int, default=8, help='Maximum number of processes for scaling experiments')
    parser.add_argument('--graph_size', type=int, default=500, help='Size of graph for comparison experiment')
    parser.add_argument('--max_time', type=int, default=3600, help='Maximum execution time in seconds')
    parser.add_argument('--reduced_mc', action='store_true', help='Use reduced Monte Carlo simulations for large graphs')
    parser.add_argument('--quick', action='store_true', help='Run in quick mode with minimal parameters')
    parser.add_argument('--skip_mpi', action='store_true', help='Skip MPI implementations and run sequential only')
    parser.add_argument('--dataset', type=str, default=None, 
                        help='Path to custom dataset (e.g., datasets/p2p-Gnutella04.txt)')
    parser.add_argument('--use_gnutella', action='store_true', 
                        help='Use the P2P-Gnutella04 dataset')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Quick mode for faster testing
    if args.quick:
        args.k = 2
        args.mc = 20
        args.graph_size = min(100, args.graph_size)
    
    # Reduced MC for large graphs
    if args.reduced_mc and args.graph_size > 1000:
        args.mc = max(50, args.mc // 5)
    
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except ImportError:
        rank = 0
        size = 1
    
    # Load either custom dataset or random graphs
    if rank == 0:
        if args.dataset:
            # Load custom dataset
            custom_path = args.dataset
            print(f"Loading custom dataset from: {custom_path}")
            G = load_graph(custom_path)
            datasets = {'custom': G}
        elif args.use_gnutella:
            # Load the P2P-Gnutella04 dataset
            gnutella_path = os.path.join('datasets', 'p2p-Gnutella04.txt')
            print(f"Loading P2P-Gnutella04 dataset from: {gnutella_path}")
            G = load_graph(gnutella_path)
            # For very large graphs, reduce MC simulations even further
            if len(G) > 5000:
                args.mc = max(20, args.mc // 10)
                print(f"Adjusted MC simulations to {args.mc} for large graph ({len(G)} nodes)")
            datasets = {'gnutella': G}
        else:
            # Load or create random datasets
            datasets = load_or_create_datasets('datasets')
            G = datasets.get(args.graph_size)
            if G is None:
                print(f"No dataset with size {args.graph_size}. Using the closest available.")
                sizes = sorted(datasets.keys())
                closest_size = min(sizes, key=lambda x: abs(x - args.graph_size))
                G = datasets[closest_size]
                print(f"Using graph with {closest_size} nodes.")
    else:
        datasets = None
        G = None
    
    try:
        # Broadcast the graph to all processes
        G = comm.bcast(G, root=0)
    except:
        pass
    
    # Run experiments based on the specified type
    if args.experiment == 'comparison' or args.experiment == 'all':
        output_dir = os.path.join(args.output_dir, 'comparison')
        if args.dataset:
            output_dir += '_custom'
        elif args.use_gnutella:
            output_dir += '_gnutella'
        run_implementation_comparison(G, args.k, args.mc, output_dir, args.max_time)
    
    if (args.experiment == 'strong' or args.experiment == 'all') and rank == 0 and not args.skip_mpi:
        output_dir = os.path.join(args.output_dir, 'strong_scaling')
        if args.dataset:
            output_dir += '_custom'
        elif args.use_gnutella:
            output_dir += '_gnutella'
        run_strong_scaling_experiments(G, args.k, args.mc, output_dir, args.max_procs)
    
    if (args.experiment == 'weak' or args.experiment == 'all') and rank == 0 and not args.skip_mpi \
       and not args.dataset and not args.use_gnutella:
        # Weak scaling only works with random datasets
        run_weak_scaling_experiments(datasets, args.k, args.mc, 
                                    os.path.join(args.output_dir, 'weak_scaling'), 
                                    args.max_procs)