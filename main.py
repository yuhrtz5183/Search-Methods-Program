import sys
import random
import matplotlib.pyplot as plt

from graph_loader import load_graph, load_graph_set2, make_grid_world, make_random_weighted_graph
from algorithms import euclidean
from algorithms import bfs_steps, dfs_steps, iddfs_steps, greedy_steps, astar_steps, run_bfs, run_dfs, run_iddfs, run_greedy, run_astar
from visualize import SearchAnimator
from benchmark import benchmark_single, write_csv

DATASETS = {
    "set1": ("data/coordinates.csv", "data/Adjacencies.txt"),
    "set2": ("data/KC_Metro_100_Cities___Nodes.csv", "data/KC_Metro_100_Cities___Edges.csv"),
}

def choose_dataset():
    print("\nChoose graph source:")
    print("1) Set 1 (coordinates and adjacencies)")
    print("2) Set 2 (KC_Metro_100_Cities)")
    print("3) Random: grid world")
    print("4) Random: general weighted graph")
    
    while True:
        c = input("Choose 1/2/3/4: ").strip()
        if c == "1":
            return load_graph(*DATASETS["set1"])
        elif c == "2":
            if "set2" not in DATASETS:
                print(" Set 2 not configured. Edit DATASETS in main.py.")
                continue
            return load_graph_set2(*DATASETS["set2"])
        elif c == "3":
            return get_grid_world_params()
        elif c == "4":
            return get_random_graph_params()
        else:
            print(" Invalid choice. Please enter 1, 2, 3, or 4.")

def get_grid_world_params():
    while True:
        try:
            N = int(input("Grid size (ie: 20): ") or "20")
            if N <= 0:
                print(" Grid size must be positive. Try again.")
                continue
            break
        except ValueError:
            print(" Please enter a valid integer.")
    
    while True:
        try:
            p = float(input("Obstacle density p [0-1] (ie: 0.2): ") or "0.2")
            if not 0 <= p <= 1:
                print(" Obstacle density must be between 0 and 1. Try again.")
                continue
            break
        except ValueError:
            print(" Please enter a valid number.")
    
    while True:
        conn = input("Connectivity (4 or 8): ").strip()
        if conn in ["4", "8"]:
            eight = conn == "8"
            break
        else:
            print(" Please enter 4 or 8.")
    
    while True:
        try:
            seed = int(input("Seed (int): ") or "0")
            break
        except ValueError:
            print(" Please enter a valid integer.")
    
    return make_grid_world(N=N, p=p, eight_conn=eight, weighted=True, seed=seed)

def get_random_graph_params():
    while True:
        try:
            N = int(input("Nodes N (ie: 80): ") or "80")
            if N <= 0:
                print(" Number of nodes must be positive. Try again.")
                continue
            break
        except ValueError:
            print(" Please enter a valid integer.")
    
    while True:
        try:
            b = float(input("Expected branching factor b (ie: 3.0): ") or "3.0")
            if b <= 0:
                print(" Branching factor must be positive. Try again.")
                continue
            break
        except ValueError:
            print(" Please enter a valid number.")
    
    while True:
        try:
            seed = int(input("Seed (int): ") or "0")
            break
        except ValueError:
            print(" Please enter a valid integer.")
    
    return make_random_weighted_graph(N=N, exp_b=b, seed=seed)

def choose_pair(G):
    print(f"\nGraph loaded. {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    while True:
        s = input("Start node/city (exact name): ").strip()
        if s in G:
            break
        else:
            sample_nodes = random.sample(list(G.nodes()), min(5, len(G.nodes())))
            print(f" Node '{s}' not found in graph. Example nodes: {sample_nodes}{'' if len(G.nodes()) > 5 else ''}")
    
    while True:
        t = input("Goal node/city (exact name): ").strip()
        if t in G:
            if t == s:
                print(" Goal cannot be the same as start. Choose a different goal.")
                continue
            break
        else:
            sample_nodes = random.sample(list(G.nodes()), min(5, len(G.nodes())))
            print(f" Node '{t}' not found in graph. Example nodes: {sample_nodes}{'' if len(G.nodes()) > 5 else ''}")
    
    return s, t

def visualize_mode(G, coords):
    s, t = choose_pair(G)
    
    while True:
        print("\nChoose algorithm:")
        print("1) BFS  2) DFS  3) IDDFS  4) Greedy  5) A*")
        c = input("Choose 1/2/3/4/5: ").strip()
        
        if c == "1":
            steps = bfs_steps(G, s, t)
            res = run_bfs(G, s, t, None)
            break
        elif c == "2":
            steps = dfs_steps(G, s, t)
            res = run_dfs(G, s, t, None)
            break
        elif c == "3":
            steps = iddfs_steps(G, s, t)
            res = run_iddfs(G, s, t, None)
            break
        elif c == "4":
            steps = greedy_steps(G, s, t, coords, euclidean)
            res = run_greedy(G, s, t, coords, euclidean)
            break
        elif c == "5":
            steps = astar_steps(G, s, t, coords, euclidean)
            res = run_astar(G, s, t, coords, euclidean)
            break
        else:
            print(" Invalid choice. Please enter 1, 2, 3, 4, or 5.")

    anim = SearchAnimator(G, coords, s, t, steps, res)
    anim.show()

    print("\n--- Result Summary ---")
    print(f"Path length: {len(res.path) if res.path else 0}")
    print(f"Path cost:   {res.path_cost:.3f}")
    print(f"Time (ms):   {res.time_ms:.2f}")
    print(f"Nodes expanded: {res.nodes_expanded}")
    print(f"Footprint peak (|frontier|+|explored|): {res.footprint_peak}")
    print(f"Process peak memory (KB): {res.peak_proc_kb:.1f}")

def benchmark_mode(G, coords):
    s, t = choose_pair(G)
    
    while True:
        try:
            reps_input = input("Repeats per algorithm: ") or "5"
            reps = int(reps_input)
            if reps < 1:
                print(" Number of repeats must be at least 1. Try again.")
                continue
            break
        except ValueError:
            print(" Please enter a valid integer.")

    rows = benchmark_single(G, coords, s, t, repeats=reps)
    
    # Sort by fastest time
    sorted_rows = sorted(rows, key=lambda x: x['time_ms_mean'])
    
    print("\n--- Benchmark Results ---")
    for i, r in enumerate(sorted_rows, 1):
        print(f"{i}. {r['algorithm']:7s} | time {r['time_ms_mean']:.2f}±{r['time_ms_std']:.2f} ms | "
              f"mem {r['mem_kb_mean']:.1f} KB | nodes {r['nodes_expanded_mean']:.1f} | "
              f"depth {r['solution_depth_mean']:.1f} | cost {r['path_cost_mean']:.2f} | "
              f"foot {r['footprint_peak_mean']:.1f}")
    
    create_benchmark_chart(sorted_rows, s, t)

def create_benchmark_chart(rows, start, goal):
    # Create a chart showing benchmark comparison
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract data for charting
    algorithms = [r['algorithm'] for r in rows]
    times = [r['time_ms_mean'] for r in rows]
    time_errors = [r['time_ms_std'] for r in rows]
    nodes = [r['nodes_expanded_mean'] for r in rows]
    costs = [r['path_cost_mean'] for r in rows]
    memory = [r['mem_kb_mean'] for r in rows]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Benchmark Results: {start} → {goal}', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # 1. Runtime comparison (bar chart)
    bars1 = ax1.bar(algorithms, times, color=colors[:len(algorithms)], alpha=0.8, edgecolor='black')
    ax1.set_title('Runtime (ms)', fontweight='bold')
    ax1.set_ylabel('Time (milliseconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, time in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{time:.1f}', ha='center', va='center', fontsize=9, 
                color='black', fontweight='bold')
    
    bars2 = ax2.bar(algorithms, nodes, color=colors[:len(algorithms)], alpha=0.8, edgecolor='black')
    ax2.set_title('Nodes Expanded', fontweight='bold')
    ax2.set_ylabel('Number of Nodes')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, node_count in zip(bars2, nodes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{node_count:.0f}', ha='center', va='center', fontsize=9, 
                color='black', fontweight='bold')
    
    bars3 = ax3.bar(algorithms, memory, color=colors[:len(algorithms)], alpha=0.8, edgecolor='black')
    ax3.set_title('Memory Usage (KB)', fontweight='bold')
    ax3.set_ylabel('Memory (KB)')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, mem in zip(bars3, memory):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{mem:.1f}', ha='center', va='center', fontsize=9, 
                color='black', fontweight='bold')
    
    bars4 = ax4.bar(algorithms, costs, color=colors[:len(algorithms)], alpha=0.8, edgecolor='black')
    ax4.set_title('Path Cost', fontweight='bold')
    ax4.set_ylabel('Total Path Cost')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, cost in zip(bars4, costs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{cost:.2f}', ha='center', va='center', fontsize=9, 
                color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

def main():
    print("\n=== AI Search Methods ===\n")
    
    while True:
        while True:
            print("1) Visualizer")
            print("2) Benchmark Testing")
            print("3) Quit")
            mode = input("Choose 1/2/3: ").strip()
            
            if mode in ["1", "2", "3"]:
                break
            else:
                print(" Invalid selection. Please enter 1, 2, or 3.")
        
        if mode == "3":
            break
        
        G, coords = choose_dataset()
        if mode == "1":
            visualize_mode(G, coords)
        elif mode == "2":
            benchmark_mode(G, coords)
        
        # Ask if user wants to try a different mode
        while True:
            repeat = input("\nWould you like to test again? (y/n): ").lower().strip()
            if repeat in ["y", "yes", "n", "no"]:
                break
            else:
                print(" Please enter y or n.")
        
        if repeat in ["n", "no"]:
            break

if __name__ == "__main__":
    main()
