from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Set, Tuple
from collections import deque
import heapq
import time, tracemalloc, os, psutil
import networkx as nx
import math

Coords = Dict[str, Tuple[float, float]]

def euclidean(a: str, b: str, coords: Coords) -> float:
    (x1, y1), (x2, y2) = coords[a], coords[b]
    return math.hypot(x1 - x2, y1 - y2)

def manhattan(a: str, b: str, coords: Coords) -> float:
    (x1, y1), (x2, y2) = coords[a], coords[b]
    return abs(x1 - x2) + abs(y1 - y2)

@dataclass
class Step:
    algo: str
    current: Optional[str]
    frontier_order: List[Tuple[str, float]]     # (node, priority/index)
    explored: Set[str]
    parents: Dict[str, Optional[str]]
    g: Dict[str, float] = field(default_factory=dict)
    h: Dict[str, float] = field(default_factory=dict)
    f: Dict[str, float] = field(default_factory=dict)
    event: str = "expand"

@dataclass
class RunResult:
    path: Optional[List[str]]
    time_ms: float
    peak_proc_kb: float
    footprint_peak: int
    nodes_expanded: int
    solution_depth: int
    path_cost: float

def reconstruct_path(parents: Dict[str, Optional[str]], start: str, goal: str) -> Optional[List[str]]:
    if goal not in parents:
        return None
    path = [goal]
    while path[-1] != start:
        p = parents.get(path[-1])
        if p is None:
            return None
        path.append(p)
    path.reverse()
    return path

def path_cost_of(G: nx.Graph, path: Optional[List[str]]) -> float:
    if not path or len(path) < 2:
        return 0.0
    return sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))

# ---------- BFS ----------
def bfs_steps(G: nx.Graph, start: str, goal: str) -> Iterable[Step]:
    frontier = deque([start])
    parents = {start: None}
    explored = set()
    while frontier:
        current = frontier.popleft()
        explored.add(current)
        yield Step("BFS", current, [(n, i) for i, n in enumerate(frontier)], set(explored), dict(parents), event="expand")
        if current == goal:
            yield Step("BFS", current, [(n, i) for i, n in enumerate(frontier)], set(explored), dict(parents), event="goal")
            return
        for n in G.neighbors(current):
            if n not in parents:
                parents[n] = current
                frontier.append(n)
                yield Step("BFS", current, [(x, i) for i, x in enumerate(frontier)], set(explored), dict(parents), event="enqueue")

# ---------- DFS ----------
def dfs_steps(G: nx.Graph, start: str, goal: str) -> Iterable[Step]:
    stack = [start]
    parents = {start: None}
    explored = set()
    while stack:
        current = stack.pop()
        if current in explored:
            continue
        explored.add(current)
        yield Step("DFS", current, [(n, i) for i, n in enumerate(stack)], set(explored), dict(parents), event="expand")
        if current == goal:
            yield Step("DFS", current, [(n, i) for i, n in enumerate(stack)], set(explored), dict(parents), event="goal")
            return
        for n in G.neighbors(current):
            if n not in parents:
                parents[n] = current
                stack.append(n)
                yield Step("DFS", current, [(x, i) for i, x in enumerate(stack)], set(explored), dict(parents), event="enqueue")

# ---------- IDDFS ----------
def iddfs_steps(G: nx.Graph, start: str, goal: str, max_depth: int = 100) -> Iterable[Step]:
    def dls(node, depth, parents, explored, g_cost):
        nonlocal found
        explored.add(node)
        yield Step("IDDFS", node, [], set(explored), dict(parents), dict(g_cost), event="expand")
        if node == goal:
            found = True
            if goal not in g_cost:
                g_cost[goal] = g_cost[node]
            yield Step("IDDFS", node, [], set(explored), dict(parents), dict(g_cost), event="goal")
            return
        if depth == 0:
            return
        for n in G.neighbors(node):
            if n not in explored:
                parents[n] = node
                g_cost[n] = g_cost[node] + G[node][n]['weight']
                yield Step("IDDFS", node, [], set(explored), dict(parents), dict(g_cost), event="enqueue")
                yield from dls(n, depth-1, parents, explored, g_cost)
                if found: return

    found = False
    for depth in range(max_depth):
        # Each depth-limited search gets fresh data structures
        parents = {start: None}
        explored = set()
        g_cost = {start: 0.0}
        yield from dls(start, depth, parents, explored, g_cost)
        if found:
            return

# ---------- Greedy Best-First ----------
def greedy_steps(G: nx.Graph, start: str, goal: str, coords, hfun) -> Iterable[Step]:
    frontier = []
    heapq.heappush(frontier, (0.0, start))
    parents = {start: None}
    explored = set()
    h = {start: hfun(start, goal, coords)}
    while frontier:
        _, current = heapq.heappop(frontier)
        explored.add(current)
        fr_view = [(x, p) for p, x in frontier]
        yield Step("Greedy", current, fr_view, set(explored), dict(parents), h=dict(h), event="expand")
        if current == goal:
            yield Step("Greedy", current, fr_view, set(explored), dict(parents), h=dict(h), event="goal")
            return
        for n in G.neighbors(current):
            if n in explored:
                continue
            if n not in parents:
                parents[n] = current
                h[n] = hfun(n, goal, coords)
                heapq.heappush(frontier, (h[n], n))
                fr_view = [(x, p) for p, x in frontier]
                yield Step("Greedy", current, fr_view, set(explored), dict(parents), h=dict(h), event="enqueue")

# ---------- A* ----------
def astar_steps(G: nx.Graph, start: str, goal: str, coords, hfun) -> Iterable[Step]:
    frontier = []
    heapq.heappush(frontier, (0.0, start))
    parents = {start: None}
    g = {start: 0.0}
    h = {start: hfun(start, goal, coords)}
    f = {start: h[start]}
    explored = set()
    while frontier:
        _, current = heapq.heappop(frontier)
        explored.add(current)
        fr_view = [(x, p) for p, x in frontier]
        yield Step("A*", current, fr_view, set(explored), dict(parents), dict(g), dict(h), dict(f), event="expand")
        if current == goal:
            yield Step("A*", current, fr_view, set(explored), dict(parents), dict(g), dict(h), dict(f), event="goal")
            return
        for n in G.neighbors(current):
            if n in explored:
                continue
            tentative_g = g[current] + G[current][n]['weight']
            if n not in g or tentative_g < g[n]:
                parents[n] = current
                g[n] = tentative_g
                h[n] = hfun(n, goal, coords)
                f[n] = g[n] + h[n]
                heapq.heappush(frontier, (f[n], n))
                fr_view = [(x, p) for p, x in frontier]
                yield Step("A*", current, fr_view, set(explored), dict(parents), dict(g), dict(h), dict(f), event="enqueue")

def run_and_measure(G, start, goal, step_iter: Iterable[Step]) -> RunResult:
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    t0 = time.perf_counter()

    nodes_expanded = 0
    footprint_peak = 0
    last_step: Optional[Step] = None
    for st in step_iter:
        last_step = st
        if st.event == "expand":
            nodes_expanded += 1
        footprint_peak = max(footprint_peak, len(st.frontier_order) + len(st.explored))

    t1 = time.perf_counter()
    _, peak_proc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if not last_step:
        return RunResult(None, (t1 - t0) * 1000, peak_proc / 1024, footprint_peak, 0, 0, 0.0)

    path = reconstruct_path(last_step.parents, start, goal)
    depth = len(path) - 1 if path else 0
    
    if last_step.algo == "IDDFS" and last_step.g and goal in last_step.g:
        cost = last_step.g[goal]
    else:
        cost = path_cost_of(G, path)
    
    return RunResult(path, (t1 - t0) * 1000, peak_proc / 1024, footprint_peak, nodes_expanded, depth, cost)

def run_bfs(G, s, g, coords=None):   return run_and_measure(G, s, g, bfs_steps(G, s, g))
def run_dfs(G, s, g, coords=None):   return run_and_measure(G, s, g, dfs_steps(G, s, g))
def run_iddfs(G, s, g, coords=None): return run_and_measure(G, s, g, iddfs_steps(G, s, g))
def run_greedy(G, s, g, coords, hfun): return run_and_measure(G, s, g, greedy_steps(G, s, g, coords, hfun))
def run_astar(G, s, g, coords, hfun):  return run_and_measure(G, s, g, astar_steps(G, s, g, coords, hfun))
