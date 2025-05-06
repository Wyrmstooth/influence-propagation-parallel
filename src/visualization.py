import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import os
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.2)

def visualize_seed_nodes(G, seed_nodes, output_path=None):
    """Visualize the graph with seed nodes highlighted"""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    node_colors = ['#e41a1c' if node in seed_nodes else '#377eb8' for node in G.nodes()]
    node_sizes = [350 + 10 * G.degree(node) for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85, linewidths=1, edgecolors='k')
    nx.draw_networkx_edges(G, pos, alpha=0.25, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='black')
    plt.title(f"Seed Nodes Highlighted ({len(seed_nodes)} selected)", fontsize=17, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
    else:
        plt.show()
    plt.close()

def plot_execution_times(results, output_path=None):
    """Plot execution times for different implementations with table"""
    implementations = list(results.keys())
    times = [results[imp]['time'] for imp in implementations]
    influences = [results[imp]['influence'] for imp in implementations]
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("Set2", len(implementations))
    bars = ax.bar(implementations, times, color=palette)
    for bar, t, inf in zip(bars, times, influences):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                f'{t:.2f}s\nInf: {inf:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('Execution Time Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Implementation', fontsize=13)
    ax.set_ylabel('Execution Time (seconds)', fontsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
    else:
        plt.show()
    plt.close()

def plot_speedup(results, base_implementation='Sequential', output_path=None):
    """Plot speedup for different implementations with annotations"""
    implementations = [imp for imp in results.keys() if imp != base_implementation]
    base_time = results[base_implementation]['time']
    speedups = [base_time / results[imp]['time'] if results[imp]['time'] else 0 for imp in implementations]
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("Set1", len(implementations))
    bars = ax.bar(implementations, speedups, color=palette)
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                f'{s:.2f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title(f'Speedup Relative to {base_implementation}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Implementation', fontsize=13)
    ax.set_ylabel('Speedup Factor', fontsize=13)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No Speedup')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
    else:
        plt.show()
    plt.close()

def plot_strong_scaling(num_processors, execution_times, output_path=None):
    """Plot strong scaling: speedup and efficiency with annotations and ideal lines"""
    base_time = execution_times[0]
    speedups = [base_time / t if t else 0 for t in execution_times]
    efficiencies = [speedups[i] / num_processors[i] if num_processors[i] else 0 for i in range(len(num_processors))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Speedup plot
    ax1.plot(num_processors, speedups, 'o-', color='#377eb8', linewidth=2, markersize=8, label='Measured Speedup')
    ax1.plot(num_processors, num_processors, '--', color='#e41a1c', alpha=0.7, label='Ideal Speedup')
    for x, y in zip(num_processors, speedups):
        ax1.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0,7), ha='center', fontsize=10)
    ax1.set_xlabel('Number of Processors', fontsize=13)
    ax1.set_ylabel('Speedup', fontsize=13)
    ax1.set_title('Strong Scaling: Speedup', fontsize=15, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(num_processors)
    # Efficiency plot
    ax2.plot(num_processors, efficiencies, 'o-', color='#4daf4a', linewidth=2, markersize=8, label='Measured Efficiency')
    ax2.axhline(y=1, color='#e41a1c', linestyle='--', alpha=0.7, label='Ideal Efficiency')
    for x, y in zip(num_processors, efficiencies):
        ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0,7), ha='center', fontsize=10)
    ax2.set_xlabel('Number of Processors', fontsize=13)
    ax2.set_ylabel('Efficiency', fontsize=13)
    ax2.set_title('Strong Scaling: Efficiency', fontsize=15, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(num_processors)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
    else:
        plt.show()
    plt.close()

def plot_weak_scaling(num_processors, execution_times, problem_sizes, output_path=None):
    """Plot weak scaling efficiency with annotations"""
    base_time = execution_times[0]
    efficiencies = [(base_time / execution_times[i]) if execution_times[i] else 0 for i in range(len(execution_times))]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(num_processors, efficiencies, 'o-', color='#984ea3', linewidth=2, markersize=8, label='Measured Efficiency')
    ax.axhline(y=1, color='#e41a1c', linestyle='--', alpha=0.7, label='Ideal Efficiency')
    for x, y, sz in zip(num_processors, efficiencies, problem_sizes):
        ax.annotate(f"{y:.2f}\nN={sz}", (x, y), textcoords="offset points", xytext=(0,7), ha='center', fontsize=10)
    ax.set_xlabel('Number of Processors', fontsize=13)
    ax.set_ylabel('Efficiency', fontsize=13)
    ax.set_title('Weak Scaling Efficiency', fontsize=15, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(num_processors)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
    else:
        plt.show()
    plt.close()

def visualize_partitioning(G, partition, output_path=None):
    """Visualize graph partitioning with improved color and legend"""
    plt.figure(figsize=(12, 10))
    n_partitions = max(partition.values()) + 1
    colors = sns.color_palette("hsv", n_partitions)
    node_colors = [colors[partition[node]] for node in G.nodes()]
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=120, alpha=0.85, linewidths=1, edgecolors='k')
    internal_edges = [(u, v) for u, v in G.edges() if partition[u] == partition[v]]
    cross_edges = [(u, v) for u, v in G.edges() if partition[u] != partition[v]]
    nx.draw_networkx_edges(G, pos, edgelist=internal_edges, alpha=0.3, edge_color='gray')
    nx.draw_networkx_edges(G, pos, edgelist=cross_edges, alpha=0.8, edge_color='#e41a1c', width=2)
    if len(G) <= 100:
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
    # Add legend for partitions
    for i in range(n_partitions):
        plt.scatter([], [], c=[colors[i]], label=f'Partition {i}', s=120)
    plt.legend(scatterpoints=1, frameon=True, labelspacing=0.8, fontsize=11)
    plt.title(f"Graph Partitioning into {n_partitions} Parts", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
    else:
        plt.show()
    plt.close()

def export_performance_csv(results, output_path):
    """Export performance results to CSV for further analysis"""
    rows = []
    for impl, res in results.items():
        rows.append({
            'Implementation': impl,
            'Time (s)': res['time'],
            'Influence': res['influence'],
            'Seed Nodes': ','.join(str(x) for x in res['seed_nodes']) if res['seed_nodes'] else ''
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)