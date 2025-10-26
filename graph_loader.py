from typing import Dict, Tuple, List
import csv
import math
import random
import networkx as nx

Coords = Dict[str, Tuple[float, float]]

# Load graph from coordinate and adjacency files
def load_graph(coord_path: str, adj_path: str) -> Tuple[nx.Graph, Coords]:
    G = nx.Graph()
    coords: Coords = {}

    with open(coord_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                name = row[0].strip()
                try:
                    x, y = float(row[1]), float(row[2])
                except ValueError:
                    continue
                coords[name] = (x, y)
                G.add_node(name, pos=(x, y))

    # Load edges (adjacency pairs)
    with open(adj_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                a, b = parts[0], parts[1]
                if a in coords and b in coords:
                    (x1, y1), (x2, y2) = coords[a], coords[b]
                    w = math.hypot(x1 - x2, y1 - y2)
                    G.add_edge(a, b, weight=w)

    return G, coords

# Load Set2 graph from KC Metro data
def load_graph_set2(nodes_path: str, edges_path: str) -> Tuple[nx.Graph, Coords]:
    G = nx.Graph()
    coords: Coords = {}

    with open(nodes_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                city = row[0].strip('"')
                try:
                    lat, lon = float(row[1]), float(row[2])
                    x, y = lon, lat
                    coords[city] = (x, y)
                    G.add_node(city, pos=(x, y))
                except ValueError:
                    continue

    # Load edges from the edges file
    with open(edges_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                from_city = row[0].strip('"')
                to_city = row[1].strip('"')
                
                if from_city in coords and to_city in coords:
                    (x1, y1), (x2, y2) = coords[from_city], coords[to_city]
                    w = math.hypot(x1 - x2, y1 - y2)
                    G.add_edge(from_city, to_city, weight=w)

    return G, coords

# Generate grid world with obstacles
def make_grid_world(
    N: int = 20,
    p: float = 0.2,
    eight_conn: bool = False,
    weighted: bool = False,
    seed: int = 0,
) -> Tuple[nx.Graph, Coords]:
    rng = random.Random(seed)
    G = nx.Graph()
    coords: Coords = {}

    def node(r: int, c: int) -> str:
        return f"{r},{c}"

    # Blocked vs free cells
    blocked = set()
    for r in range(N):
        for c in range(N):
            if rng.random() < p:
                blocked.add((r, c))
            else:
                name = node(r, c)
                coords[name] = (float(c), float(r))
                G.add_node(name, pos=(float(c), float(r)))

    dirs4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    dirs8 = dirs4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    dirs = dirs8 if eight_conn else dirs4

    # Connect neighbors
    for r in range(N):
        for c in range(N):
            if (r, c) in blocked:
                continue
            a = node(r, c)
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N and (rr, cc) not in blocked:
                    b = node(rr, cc)
                    if a == b:
                        continue
                    if weighted:
                        w = rng.randint(1, 10)
                    else:
                        w = math.hypot(dr, dc)
                    G.add_edge(a, b, weight=w)

    return G, coords

# Generate random weighted graph
def make_random_weighted_graph(
    N: int = 80, exp_b: float = 3.0, seed: int = 0
) -> Tuple[nx.Graph, Coords]:
    rng = random.Random(seed)
    G = nx.Graph()
    coords: Coords = {}

    names: List[str] = [f"v{i}" for i in range(N)]

    # Place nodes
    for name in names:
        x, y = rng.random() * 100.0, rng.random() * 100.0
        coords[name] = (x, y)
        G.add_node(name, pos=(x, y))

    for a in names:
        deg = max(1, int(round(rng.gauss(exp_b, 1.0))))
        neighbors = rng.sample([n for n in names if n != a], k=min(deg, N - 1))
        for b in neighbors:
            (x1, y1), (x2, y2) = coords[a], coords[b]
            w = math.hypot(x1 - x2, y1 - y2)
            G.add_edge(a, b, weight=w)

    return G, coords
