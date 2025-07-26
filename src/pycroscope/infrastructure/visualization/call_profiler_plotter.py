"""
Call profiler visualization implementation.

Implements industry-standard cProfile visualizations:
1. Top Functions Bar Chart - Standard performance analysis chart
2. Call Tree Visualization - Function call hierarchy and relationships
"""

from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .interfaces import ProfilerPlotter, StyleManager


class CallProfilerPlotter(ProfilerPlotter):
    """
    Professional cProfile data visualization.

    Generates industry-standard performance analysis charts used by
    performance engineers at major technology companies.
    """

    def can_plot(self, profiler_data: Dict[str, Any]) -> bool:
        """Check if data contains cProfile stats."""
        return "stats" in profiler_data and isinstance(profiler_data["stats"], dict)

    def get_plot_types(self) -> List[str]:
        """Get available plot types for call profiler."""
        return ["top_functions", "call_tree", "flame_graph"]

    def generate_plots(
        self, profiler_data: Dict[str, Any], output_dir: Path
    ) -> Dict[str, Path]:
        """Generate all call profiler visualizations."""
        StyleManager.apply_professional_style()

        if not self.can_plot(profiler_data):
            raise RuntimeError(
                f"Cannot generate call profiler plots: Invalid data structure. "
                f"Expected 'stats' key with dict value, got keys: {list(profiler_data.keys())}"
            )

        stats_dict = profiler_data["stats"]
        plots = {}

        # 1. Top Functions Bar Chart
        top_functions_fig = self._plot_top_functions(stats_dict)
        top_functions_path = output_dir / "call_top_functions.png"
        StyleManager.save_figure(top_functions_fig, top_functions_path)
        plots["top_functions"] = top_functions_path

        # 2. Call Tree Visualization
        call_tree_fig = self._plot_call_tree(stats_dict)
        call_tree_path = output_dir / "call_tree.png"
        StyleManager.save_figure(call_tree_fig, call_tree_path)
        plots["call_tree"] = call_tree_path

        # 3. Flame Graph
        flame_graph_fig = self._plot_flame_graph(stats_dict)
        flame_graph_path = output_dir / "flame_graph.png"
        StyleManager.save_figure(flame_graph_fig, flame_graph_path)
        plots["flame_graph"] = flame_graph_path

        return plots

    def _plot_top_functions(
        self, stats_dict: Dict[str, Any], top_n: int = 20, metric: str = "cumtime"
    ) -> Figure:
        """
        Create top functions bar chart from cProfile stats.

        Industry standard for identifying performance bottlenecks.
        """
        # Parse cProfile stats into structured data
        data = []
        for func_key, func_stats in stats_dict.items():
            # Handle both string keys and tuple keys
            if isinstance(func_key, str):
                filename = func_key.split(":")[0] if ":" in func_key else "unknown"
                funcname = func_key.split(":")[-1] if ":" in func_key else func_key
            else:
                filename = str(func_key)
                funcname = str(func_key)

            # Extract stats with safe defaults
            tottime = func_stats.get("tottime", 0)
            cumtime = func_stats.get("cumtime", 0)
            ncalls = func_stats.get("ncalls", 0)

            data.append(
                {
                    "function": f"{Path(filename).name}:{funcname}"[:50],
                    "ncalls": ncalls,
                    "tottime": tottime,
                    "cumtime": cumtime,
                    "percall_tot": tottime / ncalls if ncalls > 0 else 0,
                    "percall_cum": cumtime / ncalls if ncalls > 0 else 0,
                }
            )

        if not data:
            raise RuntimeError(
                "Call profiler generated no function data. This indicates the profiler "
                "was not properly capturing function calls or no functions were executed."
            )

        df = pd.DataFrame(data)
        top_funcs = df.nlargest(top_n, metric)

        fig, ax = plt.subplots(figsize=(14, 8))

        # Create horizontal bars with professional color scheme
        y_pos = np.arange(len(top_funcs))
        colors = StyleManager.get_color_palette()
        bar_colors = [colors[i % len(colors)] for i in range(len(top_funcs))]

        bars = ax.barh(y_pos, top_funcs[metric], color=bar_colors, alpha=0.8)

        # Customize appearance
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_funcs["function"], fontsize=9)

        xlabel = (
            f"{metric.title()} Time (seconds)" if "time" in metric else metric.title()
        )
        ax.set_xlabel(xlabel)
        ax.set_title(
            f"Top {len(top_funcs)} Functions by {metric.title()}",
            fontweight="bold",
            pad=20,
        )

        # Add value labels on bars
        max_val = top_funcs[metric].max()
        for bar, val in zip(bars, top_funcs[metric]):
            label = f"{val:.4f}s" if "time" in metric else f"{val:,}"
            ax.text(
                bar.get_width() + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontsize=8,
                fontweight="bold",
            )

        # Add summary statistics
        total_time = top_funcs[metric].sum()
        total_functions = len(df)
        ax.text(
            0.02,
            0.98,
            f"Total Functions: {total_functions:,}\nTop {len(top_funcs)} Total: {total_time:.3f}s",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
            fontsize=9,
        )

        plt.tight_layout()
        return fig

    def _plot_call_tree(
        self, stats_dict: Dict[str, Any], max_depth: int = 6
    ) -> plt.Figure:
        """
        Create proper call tree visualization showing actual calling relationships.

        Uses the captured call graph data to show real parent-child relationships
        in the function call hierarchy.
        """
        # Build actual call graph from caller-callee relationships
        call_graph = self._build_tree_from_call_graph(stats_dict)

        if not call_graph:
            raise RuntimeError(
                "Cannot generate call tree: No call graph relationships found in profiler data. "
                "This indicates the call profiler failed to capture caller-callee relationships."
            )

        # Find root nodes (functions with few or no callers)
        root_nodes = self._find_root_nodes(call_graph, stats_dict)

        if not root_nodes:
            raise RuntimeError(
                f"Cannot generate call tree: No root nodes found in call graph with {len(call_graph)} functions. "
                "This indicates invalid or incomplete call relationship data."
            )

        # Create the tree visualization
        fig, ax = plt.subplots(figsize=(16, 12))

        # Calculate tree layout positions
        tree_positions = self._calculate_tree_layout(call_graph, root_nodes, max_depth)

        if not tree_positions:
            raise RuntimeError(
                f"Cannot generate call tree: Failed to layout tree with {len(root_nodes)} root nodes. "
                "This indicates a problem with the tree layout algorithm."
            )

        # Draw the tree
        self._draw_call_tree(ax, call_graph, tree_positions, stats_dict)

        # Customize the plot
        ax.set_title(
            "Call Tree - Function Call Relationships", fontweight="bold", pad=20
        )
        ax.set_xlabel("Call Flow →", fontsize=12)
        ax.set_ylabel("Call Stack Depth ↓", fontsize=12)

        # Add legend
        ax.text(
            0.02,
            0.98,
            "Nodes = Functions\n"
            "Edges = Calls\n"
            "Size ∝ Cumulative time\n"
            "Color = Execution intensity",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
            fontsize=9,
        )

        plt.tight_layout()
        return fig

    def _build_tree_from_call_graph(
        self, stats_dict: Dict[str, Any]
    ) -> Dict[str, Dict]:
        """Build tree structure from actual caller-callee relationships."""
        tree = {}

        for func_id, func_data in stats_dict.items():
            callers = func_data.get("callers", {})
            callees = func_data.get("callees", {})

            # Clean function name
            clean_name = self._clean_tree_function_name(func_id)

            tree[clean_name] = {
                "original_id": func_id,
                "callers": set(
                    self._clean_tree_function_name(caller) for caller in callers.keys()
                ),
                "callees": set(
                    self._clean_tree_function_name(callee) for callee in callees.keys()
                ),
                "cumtime": func_data.get("cumtime", 0),
                "tottime": func_data.get("tottime", 0),
                "ncalls": func_data.get("ncalls", 0),
            }

        return tree

    def _find_root_nodes(
        self, call_graph: Dict[str, Dict], stats_dict: Dict[str, Any]
    ) -> List[str]:
        """Find root nodes (functions with few or no callers)."""
        candidates = []

        # First, try to find functions with no callers (true entry points)
        entry_points = []
        for func_name, data in call_graph.items():
            num_callers = len(data["callers"])
            cumtime = data["cumtime"]

            if num_callers == 0:
                entry_points.append((func_name, cumtime))

        # If we found entry points, use them
        if entry_points:
            entry_points.sort(key=lambda x: x[1], reverse=True)
            return [name for name, _ in entry_points[:5]]

        # Otherwise, find functions with minimal callers
        for func_name, data in call_graph.items():
            num_callers = len(data["callers"])
            cumtime = data["cumtime"]

            # Consider as root if few callers (regardless of execution time for fast functions)
            if num_callers <= 1:
                candidates.append((func_name, cumtime))

        if not candidates:
            # If still no candidates, just take functions with fewest callers
            min_callers = min(len(data["callers"]) for data in call_graph.values())
            for func_name, data in call_graph.items():
                if len(data["callers"]) == min_callers:
                    candidates.append((func_name, data["cumtime"]))

        # Sort by cumulative time and take top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in candidates[:5]]  # Top 5 roots

    def _calculate_tree_layout(
        self, call_graph: Dict[str, Dict], root_nodes: List[str], max_depth: int
    ) -> Dict[str, tuple]:
        """Calculate positions for tree layout using hierarchical positioning."""
        positions = {}
        y_offset = 0

        for root in root_nodes:
            root_positions = self._layout_subtree(
                call_graph, root, 0, 0, y_offset, max_depth, set()
            )
            positions.update(root_positions)

            # Calculate height of this subtree for next root positioning
            if root_positions:
                max_y = max(pos[1] for pos in root_positions.values())
                y_offset = max_y + 2  # Gap between different root trees

        return positions

    def _layout_subtree(
        self,
        call_graph: Dict[str, Dict],
        node: str,
        depth: int,
        x_pos: float,
        y_base: float,
        max_depth: int,
        visited: set,
    ) -> Dict[str, tuple]:
        """Recursively layout a subtree with proper spacing."""
        if depth >= max_depth or node in visited or node not in call_graph:
            return {}

        visited.add(node)
        positions = {node: (x_pos, y_base + depth)}

        # Get children (callees) and sort by cumulative time
        children = list(call_graph[node]["callees"])
        children = [child for child in children if child in call_graph]
        children.sort(key=lambda c: call_graph[c]["cumtime"], reverse=True)

        # Limit number of children to avoid overcrowding
        children = children[:4]

        if children:
            # Space children horizontally
            child_spacing = 2.0
            start_x = x_pos - (len(children) - 1) * child_spacing / 2

            for i, child in enumerate(children):
                child_x = start_x + i * child_spacing
                child_positions = self._layout_subtree(
                    call_graph,
                    child,
                    depth + 1,
                    child_x,
                    y_base,
                    max_depth,
                    visited.copy(),  # Use copy to allow different branches
                )
                positions.update(child_positions)

        return positions

    def _draw_call_tree(
        self,
        ax,
        call_graph: Dict[str, Dict],
        positions: Dict[str, tuple],
        stats_dict: Dict[str, Any],
    ):
        """Draw the actual call tree with nodes and edges."""
        if not positions:
            raise RuntimeError(
                "Cannot draw call tree: No valid positions calculated for any functions. "
                "This indicates the tree layout algorithm failed."
            )

        # Calculate node sizes and colors based on cumulative time
        times = [call_graph[node]["cumtime"] for node in positions.keys()]
        max_time = max(times) if times else 1
        min_time = min(times) if times else 0

        colors = StyleManager.get_color_palette()

        # Draw edges first (so they appear behind nodes)
        for node_name, (x, y) in positions.items():
            if node_name in call_graph:
                callees = call_graph[node_name]["callees"]
                for callee in callees:
                    if callee in positions:
                        callee_x, callee_y = positions[callee]
                        # Draw edge from parent to child
                        ax.plot(
                            [x, callee_x],
                            [y, callee_y],
                            "k-",
                            alpha=0.6,
                            linewidth=1,
                            zorder=1,
                        )

                        # Add arrow to show direction
                        ax.annotate(
                            "",
                            xy=(callee_x, callee_y),
                            xytext=(x, y),
                            arrowprops=dict(
                                arrowstyle="->", color="black", alpha=0.6, lw=1
                            ),
                            zorder=1,
                        )

        # Draw nodes
        for node_name, (x, y) in positions.items():
            node_data = call_graph[node_name]
            cumtime = node_data["cumtime"]

            # Scale node size based on cumulative time
            if max_time > min_time:
                size_factor = (cumtime - min_time) / (max_time - min_time)
            else:
                size_factor = 0.5
            node_size = 300 + size_factor * 1000  # Base size + scaled size

            # Color based on execution intensity
            color_idx = int(size_factor * (len(colors) - 1))
            color = colors[color_idx]

            # Draw node
            circle = Circle((x, y), radius=0.3, color=color, alpha=0.8, zorder=2)
            ax.add_patch(circle)

            # Add function name label
            display_name = node_name[:15] + "..." if len(node_name) > 15 else node_name
            ax.text(
                x,
                y,
                display_name,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                zorder=3,
            )

            # Add timing info below node
            ax.text(
                x,
                y - 0.5,
                f"{cumtime:.3f}s",
                ha="center",
                va="center",
                fontsize=6,
                style="italic",
                zorder=3,
            )

        # Set axis limits with padding
        if positions:
            xs = [pos[0] for pos in positions.values()]
            ys = [pos[1] for pos in positions.values()]
            padding = 1.0
            ax.set_xlim(min(xs) - padding, max(xs) + padding)
            ax.set_ylim(min(ys) - padding, max(ys) + padding)

        # Remove ticks for cleaner appearance
        ax.set_xticks([])
        ax.set_yticks([])

    def _clean_tree_function_name(self, func_id: str) -> str:
        """Clean function name for tree display."""
        if isinstance(func_id, str) and ":" in func_id:
            # Extract just the function name from "filename:line(function)"
            if "(" in func_id and ")" in func_id:
                func_part = func_id.split("(")[1].split(")")[0]
                return func_part
            else:
                return func_id.split(":")[-1]
        return str(func_id)[:20]

    def _plot_flame_graph(self, stats_dict: Dict[str, Any]) -> Figure:
        """
        Create proper flame graph visualization from cProfile stats.

        Implements Brendan Gregg's flame graph specification:
        - Y-axis: Stack depth (root at bottom, leaf at top)
        - X-axis: Proportion of time spent in call paths
        - Width: Time spent in that function context
        - Hierarchical stacking of actual call paths
        """
        # Extract call graph information from cProfile stats
        call_graph = self._build_call_graph(stats_dict)

        if not call_graph:
            raise RuntimeError(
                "Cannot generate flame graph: No call graph data found in profiler statistics. "
                "This indicates the call profiler failed to capture function call data."
            )

        # Build flame graph data structure from call paths
        flame_stacks = self._extract_call_stacks(call_graph)

        if not flame_stacks:
            raise RuntimeError(
                f"Cannot generate flame graph: No call stacks extracted from {len(call_graph)} functions. "
                "This indicates the call graph lacks proper calling relationships."
            )

        # Create the flame graph visualization
        fig, ax = plt.subplots(figsize=(16, 10))

        # Calculate total time for normalization
        total_time = sum(stack["cumtime"] for stack in flame_stacks)

        # Create hierarchical flame graph layout
        self._render_flame_graph(ax, flame_stacks, total_time)

        # Customize the plot
        ax.set_xlabel("Proportion of Total Execution Time", fontsize=12)
        ax.set_ylabel("Call Stack Depth", fontsize=12)
        ax.set_title("Flame Graph - Call Stack Hierarchy", fontweight="bold", pad=20)

        # Set axis limits
        ax.set_xlim(0, 1)
        max_depth = (
            max(len(stack["path"]) for stack in flame_stacks) if flame_stacks else 1
        )
        ax.set_ylim(0, max_depth + 0.5)

        # Add explanatory text
        ax.text(
            0.02,
            0.98,
            "Width ∝ Cumulative time in call path\n"
            "Height = Call stack depth\n"
            "Bottom = Root functions\n"
            "Top = Leaf functions",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
            fontsize=9,
        )

        plt.tight_layout()
        return fig

    def _build_call_graph(self, stats_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Build call graph from cProfile function statistics."""
        call_graph = {}

        for func_name, stats in stats_dict.items():
            # Extract caller information if available
            callers = stats.get("callers", {})
            cumtime = stats.get("cumtime", 0)
            tottime = stats.get("tottime", 0)
            ncalls = stats.get("ncalls", 0)

            call_graph[func_name] = {
                "cumtime": cumtime,
                "tottime": tottime,
                "ncalls": ncalls,
                "callers": callers,
                "callees": stats.get("callees", {}),
            }

        return call_graph

    def _extract_call_stacks(self, call_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract complete call paths from the call graph for flame graph."""
        all_paths = []

        # Find all entry points (functions with no callers)
        entry_points = []
        for func_name, data in call_graph.items():
            callers = data.get("callers", {})
            cumtime = data.get("cumtime", 0)

            if len(callers) == 0 and cumtime > 0:
                entry_points.append(func_name)

        # For each entry point, traverse all possible call paths to leaves
        for entry_func in entry_points:
            self._traverse_all_paths(call_graph, entry_func, [], all_paths)

        return all_paths

    def _traverse_all_paths(
        self,
        call_graph: Dict[str, Any],
        current_func: str,
        current_path: List[str],
        result_paths: List[Dict[str, Any]],
    ):
        """Traverse all possible call paths from current function to all leaves."""
        # Avoid infinite recursion
        if current_func in current_path:
            return

        # Add current function to the path
        new_path = current_path + [current_func]
        func_data = call_graph.get(current_func, {})
        callees = func_data.get("callees", {})

        if not callees:
            # This is a leaf function - we have a complete path
            path_cumtime = sum(
                call_graph.get(func, {}).get("tottime", 0) for func in new_path
            )
            result_paths.append({"path": new_path, "cumtime": path_cumtime})
        else:
            # Continue traversing to all callees
            for callee_func in callees.keys():
                if callee_func in call_graph:
                    self._traverse_all_paths(
                        call_graph, callee_func, new_path, result_paths
                    )

    def _render_flame_graph(
        self, ax, flame_stacks: List[Dict[str, Any]], total_time: float
    ):
        """
        Render hierarchical flame graph following Brendan Gregg's specification.

        Key principles from https://queue.acm.org/detail.cfm?id=2927301:
        - Y-axis: Stack depth (root at bottom, leaf at top)
        - X-axis: Cumulative time (widths are proportional to time spent)
        - Identical function boxes at same level are merged horizontally
        - Width shows frequency function was present in stack traces
        """
        if total_time <= 0:
            raise RuntimeError(
                f"Cannot render flame graph: Total execution time is {total_time}. "
                f"This indicates all functions executed too quickly to measure timing. "
                f"Consider profiling a workload with longer execution time."
            )

        # Build hierarchical flame graph structure with proper merging
        flame_tree = self._build_flame_tree(flame_stacks)

        # Render the hierarchical flame graph
        colors = StyleManager.get_color_palette()
        self._render_flame_tree(ax, flame_tree, colors, total_time)

    def _build_flame_tree(self, flame_stacks: List[Dict[str, Any]]) -> Dict:
        """
        Build hierarchical tree structure for flame graph with proper box merging.

        This implements the core flame graph algorithm where identical functions
        at the same stack level are merged into single boxes.
        """
        # Create hierarchical tree structure
        flame_tree = {"children": {}, "cumtime": 0, "name": "root"}

        # Process each stack trace
        for stack in flame_stacks:
            path = stack["path"]
            cumtime = stack["cumtime"]

            # Navigate/build tree following the call path
            current_node = flame_tree
            current_node["cumtime"] += cumtime

            for level, func_name in enumerate(path):
                if func_name not in current_node["children"]:
                    current_node["children"][func_name] = {
                        "children": {},
                        "cumtime": 0,
                        "name": func_name,
                        "level": level,
                    }

                current_node["children"][func_name]["cumtime"] += cumtime
                current_node = current_node["children"][func_name]

        return flame_tree

    def _render_flame_tree(
        self,
        ax,
        node: Dict,
        colors: List,
        total_time: float,
        x_offset: float = 0,
        level: int = -1,
    ) -> float:
        """
        Recursively render the flame tree as hierarchical boxes.

        Returns the total width used at this level.
        """
        if level == -1:  # Skip root node
            total_width = 0
            for child_name in sorted(node["children"].keys()):  # Sort for consistency
                child = node["children"][child_name]
                width = self._render_flame_tree(
                    ax, child, colors, total_time, x_offset + total_width, 0
                )
                total_width += width
            return total_width

        # Calculate width for this function box
        width = node["cumtime"] / total_time
        func_name = node["name"]

        # Choose consistent color for this function
        color_idx = hash(func_name) % len(colors)
        color = colors[color_idx]

        # Create rectangle following Brendan Gregg standards
        rect = Rectangle(
            (x_offset, level),
            width,
            0.9,  # Standard box height
            facecolor=color,
            edgecolor="white",  # Brendan Gregg uses white edges for clarity
            linewidth=0.1,  # Thin lines for clean separation
            alpha=0.9,  # Slightly transparent for depth perception
        )
        ax.add_patch(rect)

        # Add function name following Brendan Gregg text standards
        display_name = self._clean_function_name(func_name)
        min_width_for_text = (
            0.01  # Show text for smaller boxes (Brendan Gregg standard)
        )

        if width > min_width_for_text:
            # Dynamic font sizing based on box width (Brendan Gregg approach)
            base_fontsize = 9
            fontsize = max(6, min(base_fontsize, width * 120))

            # Truncate text to fit in box width if needed
            max_chars = max(3, int(width * 80))
            if len(display_name) > max_chars:
                display_name = display_name[: max_chars - 2] + ".."

            ax.text(
                x_offset + width / 2,
                level + 0.45,
                display_name,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="normal",  # Brendan Gregg uses normal weight
                color="black",
                clip_on=True,  # Clip text to boundaries
            )

        # Recursively render children
        child_x_offset = x_offset
        for child_name in sorted(node["children"].keys()):
            child = node["children"][child_name]
            child_width = self._render_flame_tree(
                ax, child, colors, total_time, child_x_offset, level + 1
            )
            child_x_offset += child_width

        return width

    def _clean_function_name(self, func_name: str) -> str:
        """Clean function name for display in flame graph."""
        if isinstance(func_name, tuple) and len(func_name) >= 3:
            # Handle (filename, line, function) tuples
            filename, line, function = func_name[:3]
            return f"{Path(filename).name}:{function}"
        elif isinstance(func_name, str):
            # Handle string function names
            if ":" in func_name:
                parts = func_name.split(":")
                if len(parts) >= 2:
                    return f"{Path(parts[0]).name}:{parts[-1]}"
            return func_name.split("/")[-1]  # Just the filename if path
        else:
            return str(func_name)[:30]
