from typing import Dict, Tuple, Iterable, List
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Button, Slider
import networkx as nx

from algorithms import Step, reconstruct_path

class SearchAnimator:
    def __init__(self, G: nx.Graph, coords: Dict[str, Tuple[float, float]], start: str, goal: str, steps_iter: Iterable[Step], final_result=None):
        self.G, self.coords = G, coords
        self.start, self.goal = start, goal
        self.final_result = final_result
        self.steps: List[Step] = []
        for step in steps_iter:
            self.steps.append(step)
            if step.event == "goal":
                break
        self.idx = 0
        self.playing = False
        
        # Set constant base animation speed
        self.interval = 300
            
        # Transform coordinates for visualization
        if coords:
            # Detect coordinate system: Set1 stores (lat, lon), Set2 stores (lon, lat)
            sample_coord = next(iter(coords.values()))
            # If first coordinate is > 0 and second is < 0, it's likely (lat, lon) - Set1
            # If first coordinate is < 0 and second is > 0, it's likely (lon, lat) - Set2
            if sample_coord[0] > 0 and sample_coord[1] < 0:
                # Set1: coords are (lat, lon)
                lats = [coord[0] for coord in coords.values()]
                lons = [coord[1] for coord in coords.values()]
            else:
                # Set2: coords are (lon, lat)
                lons = [coord[0] for coord in coords.values()]
                lats = [coord[1] for coord in coords.values()]
            
            min_lon, max_lon = min(lons), max(lons)
            min_lat, max_lat = min(lats), max(lats)
            
            self.pos = {}
            for city, coord in coords.items():
                if sample_coord[0] > 0 and sample_coord[1] < 0:
                    # Set1: coord is (lat, lon)
                    lat, lon = coord
                else:
                    # Set2: coord is (lon, lat)
                    lon, lat = coord
                x = (lon - min_lon) / (max_lon - min_lon)
                y = (lat - min_lat) / (max_lat - min_lat)
                self.pos[city] = (x, y)
        else:
            self.pos = nx.get_node_attributes(self.G, "pos")
        self.path_overlays = {}
        self.legend_handles = [
            mpl.patches.Patch(color="#66BB6A", label="Start"),
            mpl.patches.Patch(color="#EF5350", label="Goal"),
            mpl.patches.Patch(color="#42A5F5", label="Explored"),
            mpl.patches.Patch(color="#FFEE58", label="Frontier"),
            mpl.patches.Patch(color="#AB47BC", label="Current"),
            mpl.patches.Patch(color="#F4511E", label="Final Path"),
        ]

        self._setup_fig()
        self._render_step()

    def _setup_fig(self):
        # Constant window size for all graphs
        figsize = (14, 10)
            
        self.fig, self.ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(left=0.05, right=0.7, bottom=0.15, top=0.95)
        
        # Center window on screen
        mngr = self.fig.canvas.manager
        if hasattr(mngr, 'window'):
            window = mngr.window
            if hasattr(window, 'wm_geometry'):
                screen_width = window.winfo_screenwidth()
                screen_height = window.winfo_screenheight()
                window_width = int(figsize[0] * 100)
                window_height = int(figsize[1] * 100)
                x = (screen_width - window_width) // 2
                y = (screen_height - window_height) // 2
                window.wm_geometry(f"{window_width}x{window_height}+{x}+{y}")

        self._draw_base()
        
        # Create control buttons
        ax_play    = plt.axes([0.05, 0.05, 0.08, 0.05])
        ax_pause   = plt.axes([0.15, 0.05, 0.08, 0.05])
        ax_back    = plt.axes([0.25, 0.05, 0.08, 0.05])
        ax_step    = plt.axes([0.35, 0.05, 0.08, 0.05])
        ax_restart = plt.axes([0.45, 0.05, 0.10, 0.05])
        ax_speed   = plt.axes([0.62, 0.05, 0.18, 0.03])

        self.btn_play    = Button(ax_play, "Play")
        self.btn_pause   = Button(ax_pause, "Pause")
        self.btn_back    = Button(ax_back, "Back")
        self.btn_step    = Button(ax_step, "Step")
        self.btn_restart = Button(ax_restart, "Restart")
        self.sld_speed   = Slider(ax_speed, "Speed (ms)", 50, 2000, valinit=self.interval, valstep=50)

        self.btn_play.on_clicked(self._on_play)
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_back.on_clicked(self._on_back)
        self.btn_step.on_clicked(self._on_step)
        self.btn_restart.on_clicked(self._on_restart)
        self.sld_speed.on_changed(self._on_speed)

        # Create info panels using organized layout
        self._create_info_panels()

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.timer = self.fig.canvas.new_timer(interval=self.interval)
        self.timer.add_callback(self._tick)

    def _create_info_panels(self):
        # Create info panels with fixed positions to prevent collisions
        panel_width = 0.26
        right_margin = 0.02
        
        # Fixed positions for each panel to prevent overlap
        # Frontier panel - needs more height for 20 nodes
        self.ax_frontier = plt.axes([1 - panel_width - right_margin, 0.70, panel_width, 0.25])
        
        # Info panel - smaller, positioned below frontier
        self.ax_info = plt.axes([1 - panel_width - right_margin, 0.45, panel_width, 0.15])
        
        # Results panel - positioned at bottom, above speed slider
        self.ax_results = plt.axes([1 - panel_width - right_margin, 0.15, panel_width, 0.25])
        
        # Turn off axes for all panels
        for ax in [self.ax_frontier, self.ax_info, self.ax_results]:
            ax.axis("off")

    def _draw_base(self):
        # Draw the static base (edges + labels) and legend
        self.ax.clear()
        
        # Adaptive sizing for different graph densities
        node_count = self.G.number_of_nodes()
        if node_count >= 100:
            node_size = 180
            font_size = 4
            edge_alpha = 0.15
            node_alpha = 0.5
            show_labels = False
        elif node_count >= 500:
            node_size = 120
            font_size = 3
            edge_alpha = 0.1
            node_alpha = 0.4
            show_labels = False
        else:
            node_size = 1500
            font_size = 8
            edge_alpha = 0.6
            node_alpha = 1.0
            show_labels = True
            
        nx.draw(self.G, self.pos, ax=self.ax, node_size=node_size, with_labels=show_labels, font_size=font_size, 
                font_weight='bold', edge_color="#BBBBBB", node_color="#808080", alpha=node_alpha)
        
        if node_count > 100 and show_labels:
            nx.draw_networkx_labels(self.G, self.pos, ax=self.ax, font_size=font_size, font_weight='normal')
        
        edges = nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edge_color="#BBBBBB", alpha=edge_alpha)
        self.ax.legend(handles=self.legend_handles, loc="upper left", bbox_to_anchor=(-0.05, 1.05))

    def _on_play(self, _):   self.playing or self.timer.start(); self.playing = True
    def _on_pause(self, _):  self.playing = False; self.timer.stop()
    def _on_back(self, _):   self._on_pause(_); self._go_back()
    def _on_step(self, _):   self._on_pause(_); self._advance()
    def _on_restart(self, _): self._on_pause(_); self.idx = 0; self._render_step()
    def _on_speed(self, _): 
        self.interval = int(self.sld_speed.val)
        self.timer.interval = self.interval
        if self.playing:
            self.timer.stop()
            self.timer = self.fig.canvas.new_timer(interval=self.interval)
            self.timer.add_callback(self._tick)
            self.timer.start()

    def _tick(self):
        if self.playing:
            self._advance()

    def _advance(self):
        if self.idx < len(self.steps) - 1:
            self.idx += 1
            self._render_step()
            if self.steps[self.idx].event == "goal":
                self._on_pause(None)
        else:
            self._on_pause(None)
    
    def _go_back(self):
        if self.idx > 0:
            self.idx -= 1
            self._render_step()

    def _render_step(self):
        st = self.steps[self.idx] if self.steps else None

        # Choose rendering method based on graph size
        node_count = self.G.number_of_nodes()
        if node_count >= 500:
            self._render_step_efficient(st)
        elif node_count >= 100:
            self._render_step_efficient(st)
        else:
            self._render_step_simple(st)
    
    def _render_step_simple(self, st):
        # Better rendering for small graphs
        self._draw_base()

        # Set node sizes based on graph density
        node_count = self.G.number_of_nodes()
        if node_count >= 100:
            start_goal_size = 350
            current_size = 400
            explored_size = 300
            frontier_size = 250
            path_width = 2
        else:
            start_goal_size = 1800
            current_size = 2000
            explored_size = 1500
            frontier_size = 1700
            path_width = 5
            
        # Draw start and goal nodes
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[self.start], node_color="#66BB6A", node_size=start_goal_size, ax=self.ax)
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[self.goal],  node_color="#EF5350", node_size=start_goal_size, ax=self.ax)

        frontier_nodes = [n for (n, _) in st.frontier_order] if st else []
        if frontier_nodes:
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=frontier_nodes, node_color="#FFEE58", node_size=frontier_size, ax=self.ax)
        if st and st.explored:
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=list(st.explored), node_color="#42A5F5", node_size=explored_size, ax=self.ax)
        if st and st.current:
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[st.current], node_color="#AB47BC", node_size=current_size, ax=self.ax)
        
        self._draw_final_path(st, path_width)
    
    def _render_step_efficient(self, st):
        # Better rendering for large graphs
        if not hasattr(self, '_last_rendered_state') or self._last_rendered_state != (st.current, len(st.explored) if st else 0, len(st.frontier_order) if st else 0):
            self._draw_base()
            self._last_rendered_state = (st.current, len(st.explored) if st else 0, len(st.frontier_order) if st else 0)

        node_count = self.G.number_of_nodes()
        if node_count >= 500:
            start_goal_size = 200
            current_size = 250
            explored_size = 180
            frontier_size = 150
            path_width = 1
        else:
            start_goal_size = 350
            current_size = 400
            explored_size = 300
            frontier_size = 250
            path_width = 2
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[self.start], node_color="#66BB6A", node_size=start_goal_size, ax=self.ax)
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[self.goal],  node_color="#EF5350", node_size=start_goal_size, ax=self.ax)

        frontier_nodes = [n for (n, _) in st.frontier_order] if st else []
        if frontier_nodes:
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=frontier_nodes, node_color="#FFEE58", node_size=frontier_size, ax=self.ax)
        if st and st.explored:
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=list(st.explored), node_color="#42A5F5", node_size=explored_size, ax=self.ax)
        if st and st.current:
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[st.current], node_color="#AB47BC", node_size=current_size, ax=self.ax)
        
        self._draw_final_path(st, path_width)
    
    def _draw_final_path(self, st, path_width=5):
        # Draw final path if goal reached
        if st and st.event == "goal":
            path = reconstruct_path(st.parents, self.start, self.goal)
            if path and len(path) > 1:
                edges = list(zip(path[:-1], path[1:]))
                nx.draw_networkx_edges(self.G, self.pos, edgelist=edges, width=path_width, edge_color="#F4511E", ax=self.ax)
                node_size = 1800 if self.G.number_of_nodes() < 100 else 350
                nx.draw_networkx_nodes(self.G, self.pos, nodelist=path, node_color="#F4511E", node_size=node_size, ax=self.ax)

        self._update_panels(st)
        self.fig.canvas.draw_idle()
    
    def _update_panels(self, st):
        # Update frontier and info panels
        
        # Update frontier panel
        self.ax_frontier.clear(); self.ax_frontier.axis("off")
        if st:
            title = f"{st.algo} Frontier (order/priority)"
            lines = [title, "-" * len(title)]
            for i, (n, p) in enumerate(st.frontier_order[:20]):
                line = f"{i+1:>2}. {n} [{p:.3f}]" if isinstance(p, float) else f"{i+1:>2}. {n} [{p}]"
                lines.append(line)
            lines.append("")
            lines.append("")
            lines.append("")
            self.ax_frontier.text(0, 1, "\n".join(lines), va="top", family="monospace")

        # Update info panel
        self.ax_info.clear(); self.ax_info.axis("off")
        if st:
            info = ["", "", ""]
            info.append(f"Step {self.idx+1}/{len(self.steps)}  Event: {st.event}  Current: {st.current}")
            if st.current and st.current in st.g: info.append(f"Cost from start (g): {st.g[st.current]:.2f}")
            if st.current and st.current in st.h: info.append(f"Heuristic to goal (h): {st.h[st.current]:.2f}")
            if st.current and st.current in st.f: info.append(f"Total estimated (f): {st.f[st.current]:.2f}")
            info.append(f"Explored: {len(st.explored)}  Frontier: {len(st.frontier_order)}")
            self.ax_info.text(0, 1, "\n".join(info), va="top", family="monospace")
        # Update results panel
        self.ax_results.clear(); self.ax_results.axis("off")
        if st and st.event == "goal" and self.final_result:
            results = ["=== FINAL RESULTS ==="]
            if self.final_result.path:
                path_str = " → ".join(self.final_result.path)
                if len(path_str) > 35:
                    path_parts = self.final_result.path
                    results.append("Path found:")
                    for i in range(0, len(path_parts), 5):
                        line = "  " + " → ".join(path_parts[i:i+5])
                        results.append(line)
                else:
                    results.append(f"Path found: {path_str}")
                results.append(f"Path length: {len(self.final_result.path)} nodes")
                results.append(f"Total cost: {self.final_result.path_cost:.2f}")
            else:
                results.append("No path found")
            results.append(f"Nodes expanded: {self.final_result.nodes_expanded}")
            results.append(f"Solution depth: {self.final_result.solution_depth}")
            results.append(f"Time: {self.final_result.time_ms:.2f} ms")
            results.append(f"Algorithm: {st.algo}")
            self.ax_results.text(0, 1, "\n".join(results), va="top", family="monospace")

    def _on_click(self, event):
        # Mouse click for node info
        if event.inaxes != self.ax or not self.steps:
            return
        st = self.steps[self.idx]
        if event.xdata is None or event.ydata is None:
            return

        # Find nearest node
        min_d, min_node = 1e9, None
        for n, (x, y) in self.pos.items():
            d = (event.xdata - x) ** 2 + (event.ydata - y) ** 2
            if d < min_d:
                min_d, min_node = d, n
        if min_node:
            g = st.g.get(min_node)
            h = st.h.get(min_node)
            f = st.f.get(min_node)
            parent = st.parents.get(min_node)
            msg = [f"Node: {min_node}"]
            if g is not None: msg.append(f"Cost from start (g): {g:.3f}")
            if h is not None: msg.append(f"Heuristic to goal (h): {h:.3f}")
            if f is not None: msg.append(f"Total estimated (f): {f:.3f}")
            msg.append(f"Parent: {parent}")
            self.ax_info.clear(); self.ax_info.axis("off")
            self.ax_info.text(0, 1, "\n".join(msg), va="top", family="monospace")
            self.fig.canvas.draw_idle()


    def show(self):
        plt.show()
