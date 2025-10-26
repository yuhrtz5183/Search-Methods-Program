import csv
import statistics
from typing import Dict, Any, List

from algorithms import (
    run_bfs, run_dfs, run_iddfs, run_greedy, run_astar, RunResult, euclidean, manhattan
)

# Run all algorithms multiple times
def benchmark_single(G, coords, start: str, goal: str, repeats: int = 5) -> List[Dict[str, Any]]:
    algos = {
        "BFS":                lambda: run_bfs(G, start, goal, None),
        "DFS":                lambda: run_dfs(G, start, goal, None),
        "IDDFS":              lambda: run_iddfs(G, start, goal, None),
        "Greedy (Euclidean)": lambda: run_greedy(G, start, goal, coords, hfun=None if hasattr(coords, "_noop") else euclidean),
        "A* (Euclidean)":     lambda: run_astar(G, start, goal, coords, hfun=None if hasattr(coords, "_noop") else euclidean),
        "Greedy (Manhattan)": lambda: run_greedy(G, start, goal, coords, hfun=None if hasattr(coords, "_noop") else manhattan),
        "A* (Manhattan)":     lambda: run_astar(G, start, goal, coords, hfun=None if hasattr(coords, "_noop") else manhattan),
    }

    rows = []
    for name, fn in algos.items():
        times, mems, nodes, depths, costs, foots = [], [], [], [], [], []
        for _ in range(repeats):
            res: RunResult = fn()
            times.append(res.time_ms)
            mems.append(res.peak_proc_kb)
            nodes.append(res.nodes_expanded)
            depths.append(res.solution_depth)
            costs.append(res.path_cost)
            foots.append(res.footprint_peak)
        rows.append({
            "algorithm": name,
            "time_ms_mean": statistics.mean(times),
            "time_ms_std": statistics.pstdev(times),
            "mem_kb_mean": statistics.mean(mems),
            "mem_kb_std": statistics.pstdev(mems),
            "nodes_expanded_mean": statistics.mean(nodes),
            "solution_depth_mean": statistics.mean(depths),
            "path_cost_mean": statistics.mean(costs),
            "footprint_peak_mean": statistics.mean(foots),
        })
    return rows

def write_csv(rows: List[Dict[str, Any]], path: str):
    if not rows: return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r[k] for k in keys])
