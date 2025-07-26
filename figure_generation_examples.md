```python
"""
Professional Profiler Visualization Functions
============================================

Matplotlib-based figures for real profiler data, modeled after industry-standard visualizations.
Each function takes actual profiler output and creates publication-quality figures.

Dependencies: matplotlib, pandas, numpy
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

# Professional styling
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13

def plot_cprofile_top_functions(stats_dict: Dict, top_n: int = 20, metric: str = 'cumulative'):
    """
    Top functions bar chart from cProfile stats

    Args:
        stats_dict: cProfile stats.get_stats() output or parsed equivalent
        top_n: Number of top functions to show
        metric: 'cumulative', 'tottime', 'ncalls'
    """
    # Parse cProfile stats
    data = []
    for (filename, lineno, funcname), (cc, nc, tt, ct, callers) in stats_dict.items():
        data.append({
            'function': f"{Path(filename).name}:{funcname}",
            'ncalls': cc,
            'tottime': tt,
            'cumulative': ct,
            'percall_tot': tt/cc if cc > 0 else 0,
            'percall_cum': ct/cc if cc > 0 else 0
        })

    df = pd.DataFrame(data)
    top_funcs = df.nlargest(top_n, metric)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create horizontal bars
    y_pos = np.arange(len(top_funcs))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_funcs)))

    bars = ax.barh(y_pos, top_funcs[metric], color=colors, alpha=0.8)

    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[:50] + '...' if len(f) > 50 else f
                       for f in top_funcs['function']], fontsize=9)
    ax.set_xlabel(f'{metric.title()} Time (seconds)' if 'time' in metric else metric.title())
    ax.set_title(f'Top {top_n} Functions by {metric.title()}')

    # Add value labels
    max_val = top_funcs[metric].max()
    for i, (bar, val) in enumerate(zip(bars, top_funcs[metric])):
        label = f'{val:.4f}s' if 'time' in metric else f'{val:,}'
        ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height()/2,
               label, va='center', ha='left', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

def plot_cprofile_call_tree(stats_dict: Dict, root_func: str = None, max_depth: int = 4):
    """
    Call tree visualization showing function relationships

    Args:
        stats_dict: cProfile stats.get_stats() output
        root_func: Root function to start tree from (None = auto-detect)
        max_depth: Maximum tree depth to display
    """
    # Build call relationships
    call_tree = {}
    func_times = {}

    for (filename, lineno, funcname), (cc, nc, tt, ct, callers) in stats_dict.items():
        func_key = f"{Path(filename).name}:{funcname}"
        func_times[func_key] = ct

        if func_key not in call_tree:
            call_tree[func_key] = {'children': set(), 'time': ct}

        for caller_info in callers:
            if len(caller_info) >= 3:
                caller_file, caller_line, caller_func = caller_info[:3]
                caller_key = f"{Path(caller_file).name}:{caller_func}"

                if caller_key not in call_tree:
                    call_tree[caller_key] = {'children': set(), 'time': 0}

                call_tree[caller_key]['children'].add(func_key)

    # Find root function if not specified
    if root_func is None:
        # Find function with most cumulative time and no significant callers
        root_func = max(func_times.items(), key=lambda x: x[1])[0]

    fig, ax = plt.subplots(figsize=(16, 10))

    # Layout tree
    positions = {}
    y_level = 0

    def layout_tree(func, level=0, x_offset=0):
        nonlocal y_level
        if level > max_depth or func not in call_tree:
            return x_offset

        positions[func] = (x_offset, -level)

        children = list(call_tree[func]['children'])
        children.sort(key=lambda f: func_times.get(f, 0), reverse=True)

        child_x = x_offset
        for child in children:
            child_x = layout_tree(child, level + 1, child_x)
            child_x += 1

        return max(x_offset + 1, child_x)

    layout_tree(root_func)

    # Draw nodes and edges
    for func, (x, y) in positions.items():
        time_val = func_times.get(func, 0)
        size = min(max(time_val * 1000, 100), 800)  # Scale node size

        # Node color based on time
        color_intensity = min(time_val / max(func_times.values()), 1.0)
        color = plt.cm.Reds(0.3 + 0.7 * color_intensity)

        circle = plt.Circle((x, y), 0.3, color=color, alpha=0.8)
        ax.add_patch(circle)

        # Function name
        display_name = func.split(':')[-1][:15]
        ax.text(x, y-0.6, display_name, ha='center', fontsize=8, rotation=0)
        ax.text(x, y-0.8, f'{time_val:.3f}s', ha='center', fontsize=7,
                style='italic', color='gray')

        # Draw edges to children
        if func in call_tree:
            for child in call_tree[func]['children']:
                if child in positions:
                    child_x, child_y = positions[child]
                    ax.plot([x, child_x], [y-0.3, child_y+0.3], 'k-', alpha=0.6, linewidth=1)

    ax.set_aspect('equal')
    ax.set_title(f'Call Tree from {root_func.split(":")[-1]} (depth ≤ {max_depth})')
    ax.axis('off')
    plt.tight_layout()
    return fig

def plot_line_profiler_heatmap(line_timings: List[Dict]):
    """
    Line-by-line timing heatmap from line_profiler output

    Args:
        line_timings: [{'line_number': int, 'hits': int, 'time': float,
                       'per_hit': float, 'line_contents': str}, ...]
    """
    df = pd.DataFrame(line_timings)

    fig, ax = plt.subplots(figsize=(16, max(8, len(df) * 0.25)))

    # Create heatmap matrix
    time_values = df['time'].values.reshape(-1, 1)

    # Color mapping
    im = ax.imshow(time_values, cmap='Reds', aspect='auto', alpha=0.8)

    # Set labels
    ax.set_yticks(range(len(df)))
    labels = []
    for _, row in df.iterrows():
        line_content = row['line_contents'][:70] + '...' if len(row['line_contents']) > 70 else row['line_contents']
        label = f"L{row['line_number']:3d}: {line_content}"
        labels.append(label)

    ax.set_yticklabels(labels, fontsize=8, fontfamily='monospace')
    ax.set_xticks([])
    ax.set_title('Line-by-Line Execution Time Heatmap')

    # Add time annotations
    for i, (_, row) in enumerate(df.iterrows()):
        time_val = row['time']
        hits = row['hits']

        # Time annotation
        color = 'white' if time_val > df['time'].max() * 0.5 else 'black'
        ax.text(0, i, f'{time_val:.3f}s', ha='center', va='center',
               color=color, fontweight='bold', fontsize=7)

        # Hits annotation (right side)
        ax.text(0.8, i, f'({hits} hits)', ha='center', va='center',
               color=color, fontsize=6, style='italic')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Execution Time (seconds)', rotation=270, labelpad=20)

    plt.tight_layout()
    return fig

def plot_memory_profiler_timeline(memory_usage: List[Dict]):
    """
    Memory usage timeline from memory_profiler output

    Args:
        memory_usage: [{'line_number': int, 'mem_usage': float, 'increment': float,
                       'line_contents': str}, ...]
    """
    df = pd.DataFrame(memory_usage)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Memory usage plot
    ax1.plot(df.index, df['mem_usage'], 'b-', linewidth=2, label='Memory Usage')
    ax1.fill_between(df.index, df['mem_usage'], alpha=0.3, color='blue')

    # Highlight memory spikes
    spike_threshold = df['mem_usage'].mean() + 2 * df['mem_usage'].std()
    spikes = df[df['mem_usage'] > spike_threshold]
    if not spikes.empty:
        ax1.scatter(spikes.index, spikes['mem_usage'], color='red', s=50,
                   alpha=0.8, label='Memory Spikes', zorder=5)

    ax1.set_ylabel('Memory Usage (MB)')
    ax1.set_title('Memory Usage Over Execution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Memory increments
    positive_inc = df['increment'] > 0
    negative_inc = df['increment'] < 0

    ax2.bar(df[positive_inc].index, df[positive_inc]['increment'],
           color='red', alpha=0.7, label='Memory Increase', width=0.8)
    ax2.bar(df[negative_inc].index, df[negative_inc]['increment'],
           color='green', alpha=0.7, label='Memory Decrease', width=0.8)

    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Line Number / Execution Order')
    ax2.set_ylabel('Memory Change (MB)')
    ax2.set_title('Memory Increments/Decrements')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Annotate significant changes
    significant_changes = df[abs(df['increment']) > df['increment'].std() * 2]
    for idx, row in significant_changes.iterrows():
        ax2.annotate(f"L{row['line_number']}",
                    xy=(idx, row['increment']),
                    xytext=(5, 10), textcoords='offset points',
                    fontsize=8, alpha=0.8)

    plt.tight_layout()
    return fig

def plot_flame_graph(stack_traces: List[Dict], width: int = 1200):
    """
    Real flame graph from stack trace data (py-spy, perf output)

    Args:
        stack_traces: [{'stack': 'main;func1;func2', 'samples': int}, ...]
        width: Graph width in 'pixels' (affects text rendering)
    """
    # Build flame graph tree
    flame_tree = {}
    total_samples = sum(trace['samples'] for trace in stack_traces)

    for trace in stack_traces:
        frames = trace['stack'].split(';')
        samples = trace['samples']

        current = flame_tree
        for frame in frames:
            if frame not in current:
                current[frame] = {'value': 0, 'children': {}}
            current[frame]['value'] += samples
            current = current[frame]['children']

    # Calculate positions
    def calculate_positions(tree, level=0, start=0):
        positions = []
        current_start = start

        for name, data in tree.items():
            width_samples = data['value']
            width_ratio = width_samples / total_samples

            positions.append({
                'name': name,
                'level': level,
                'start': current_start,
                'width': width_samples,
                'samples': width_samples
            })

            # Add children
            if data['children']:
                child_positions = calculate_positions(
                    data['children'], level + 1, current_start
                )
                positions.extend(child_positions)

            current_start += width_samples

        return positions

    rectangles = calculate_positions(flame_tree)

    if not rectangles:
        return plt.figure()

    # Create plot
    max_level = max(rect['level'] for rect in rectangles)
    fig, ax = plt.subplots(figsize=(16, max(8, max_level * 0.8 + 2)))

    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, 20))
    color_map = {}

    for i, rect in enumerate(rectangles):
        func_name = rect['name']
        level = rect['level']
        start = rect['start']
        width_samples = rect['width']

        # Skip very narrow rectangles
        if width_samples < total_samples * 0.001:
            continue

        # Assign color
        if func_name not in color_map:
            color_map[func_name] = colors[len(color_map) % len(colors)]

        # Draw rectangle
        height = 0.8
        rectangle = Rectangle((start, level), width_samples, height,
                            facecolor=color_map[func_name],
                            edgecolor='white', linewidth=0.5, alpha=0.9)
        ax.add_patch(rectangle)

        # Add text if wide enough
        min_width_for_text = total_samples * 0.02
        if width_samples > min_width_for_text:
            display_name = func_name

            # Estimate character width
            char_width = width_samples / max(len(display_name), 1) * 0.6
            if char_width < total_samples * 0.008:  # Text too small
                display_name = func_name[:int(width_samples / (total_samples * 0.008))]
                if len(display_name) < len(func_name):
                    display_name += '...'

            if display_name:
                text_color = 'black' if sum(color_map[func_name][:3]) > 1.5 else 'white'
                ax.text(start + width_samples/2, level + height/2, display_name,
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color=text_color)

    # Format plot
    ax.set_xlim(0, total_samples)
    ax.set_ylim(0, max_level + 1)
    ax.set_xlabel('Sample Count')
    ax.set_ylabel('Stack Depth')
    ax.set_title('Flame Graph - CPU Stack Traces\n(Width ∝ Sample Count, Height = Call Stack)')

    # Clean up appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    return fig

def plot_sampling_hotspots(function_samples: List[Dict], threshold_pct: float = 5.0):
    """
    Performance hotspot identification from sampling profiler

    Args:
        function_samples: [{'function': str, 'samples': int, 'percentage': float}, ...]
        threshold_pct: Percentage threshold for hotspot classification
    """
    df = pd.DataFrame(function_samples)

    # Categorize functions
    hotspots = df[df['percentage'] >= threshold_pct]
    warm = df[(df['percentage'] >= 1.0) & (df['percentage'] < threshold_pct)]
    cold = df[df['percentage'] < 1.0]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Hotspot categories
    categories = [f'Hotspots\n(≥{threshold_pct}%)', 'Warm\n(1-5%)', 'Cold\n(<1%)']
    counts = [len(hotspots), len(warm), len(cold)]
    colors = ['#d62728', '#ff7f0e', '#2ca02c']

    bars = ax1.bar(categories, counts, color=colors, alpha=0.8)
    ax1.set_ylabel('Number of Functions')
    ax1.set_title('Performance Hotspot Classification')

    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontweight='bold')

    # 2. Top hotspots detail
    if not hotspots.empty:
        top_hotspots = hotspots.nlargest(min(10, len(hotspots)), 'percentage')
        y_pos = np.arange(len(top_hotspots))

        bars2 = ax2.barh(y_pos, top_hotspots['percentage'], color='#d62728', alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f[:25] + '...' if len(f) > 25 else f
                           for f in top_hotspots['function']], fontsize=9)
        ax2.set_xlabel('CPU Time (%)')
        ax2.set_title(f'Top Performance Hotspots (≥{threshold_pct}%)')

        for i, (bar, pct) in enumerate(zip(bars2, top_hotspots['percentage'])):
            ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{pct:.1f}%', va='center', ha='left', fontsize=8)
    else:
        ax2.text(0.5, 0.5, f'No hotspots found\n(≥{threshold_pct}% threshold)',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)

    # 3. Cumulative distribution
    sorted_df = df.sort_values('percentage', ascending=False)
    cumulative_pct = sorted_df['percentage'].cumsum()

    ax3.plot(range(len(cumulative_pct)), cumulative_pct, 'b-', linewidth=2)
    ax3.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Line')
    ax3.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90% Line')
    ax3.set_xlabel('Number of Functions (ranked)')
    ax3.set_ylabel('Cumulative CPU Time (%)')
    ax3.set_title('Cumulative Performance Distribution')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Find 80% and 90% points
    func_80 = len(cumulative_pct[cumulative_pct <= 80])
    func_90 = len(cumulative_pct[cumulative_pct <= 90])
    ax3.axvline(x=func_80, color='red', linestyle=':', alpha=0.7)
    ax3.axvline(x=func_90, color='orange', linestyle=':', alpha=0.7)

    # 4. Sample distribution histogram
    ax4.hist(df['percentage'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=threshold_pct, color='red', linestyle='--',
               label=f'Hotspot Threshold ({threshold_pct}%)')
    ax4.set_xlabel('CPU Time Percentage')
    ax4.set_ylabel('Number of Functions')
    ax4.set_title('Function Performance Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

# Professional Profiler Figure Usage Examples

## CallProfiler (cProfile) - 2 Core Figures

### 1. **Top Functions Bar Chart** - `plot_cprofile_top_functions()`

**Purpose**: Standard performance analysis chart showing bottleneck functions
**Used by**: Performance engineers, developers optimizing code

```python
import cProfile
import pstats

# Profile your application
profiler = cProfile.Profile()
profiler.enable()
your_application_code()
profiler.disable()

# Get stats and create figure
stats = pstats.Stats(profiler)
fig = plot_cprofile_top_functions(stats.get_stats(), top_n=20, metric='cumulative')
fig.savefig('performance_bottlenecks.png', dpi=300, bbox_inches='tight')
```

**Metrics**:

- `'cumulative'`: Total time including called functions (most common)
- `'tottime'`: Time excluding called functions
- `'ncalls'`: Number of function calls

### 2. **Call Tree Visualization** - `plot_cprofile_call_tree()`

**Purpose**: Shows function call relationships and execution hierarchy
**Used by**: Architects understanding code flow, debugging performance

```python
# From same cProfile stats
fig = plot_cprofile_call_tree(stats.get_stats(), root_func=None, max_depth=4)
fig.savefig('call_hierarchy.png', dpi=300, bbox_inches='tight')
```

## LineProfiler - 1 Essential Figure

### **Line-by-Line Heatmap** - `plot_line_profiler_heatmap()`

**Purpose**: Industry standard for line-level performance analysis
**Used by**: Developers optimizing specific functions

```python
# From line_profiler output (kernprof -l -v script.py)
line_timings = [
    {
        'line_number': 15,
        'hits': 1000,
        'time': 0.234,
        'per_hit': 0.000234,
        'line_contents': '    result = expensive_computation(data)'
    },
    {
        'line_number': 16,
        'hits': 1000,
        'time': 0.001,
        'per_hit': 0.000001,
        'line_contents': '    return result'
    }
    # ... more lines
]

fig = plot_line_profiler_heatmap(line_timings)
fig.savefig('line_performance.png', dpi=300, bbox_inches='tight')
```

**Data Format**: Direct from `line_profiler` output or parsed `.lprof` files

## MemoryProfiler - 1 Critical Figure

### **Memory Timeline** - `plot_memory_profiler_timeline()`

**Purpose**: Memory leak detection and allocation pattern analysis
**Used by**: SREs, performance engineers tracking memory issues

```python
# From memory_profiler output (@profile decorator + mprof run)
memory_usage = [
    {
        'line_number': 10,
        'mem_usage': 45.2,      # MB
        'increment': 2.1,       # MB change from previous line
        'line_contents': '    data = load_large_dataset()'
    },
    {
        'line_number': 11,
        'mem_usage': 47.3,
        'increment': 2.1,
        'line_contents': '    processed = process_data(data)'
    }
    # ... more measurements
]

fig = plot_memory_profiler_timeline(memory_usage)
fig.savefig('memory_analysis.png', dpi=300, bbox_inches='tight')
```

## SamplingProfiler (py-spy) - 2 Professional Figures

### 1. **Flame Graph** - `plot_flame_graph()`

**Purpose**: Industry-standard CPU profiling visualization
**Used by**: Performance engineers across all major tech companies

```python
# From py-spy raw output or perf script output
stack_traces = [
    {'stack': 'python;main;process_request;json.loads', 'samples': 145},
    {'stack': 'python;main;process_request;validate_input', 'samples': 89},
    {'stack': 'python;main;background_task;cleanup_temp_files', 'samples': 23},
    {'stack': 'python;main;process_request;json.loads;_decode', 'samples': 67}
    # ... more stack traces
]

fig = plot_flame_graph(stack_traces)
fig.savefig('cpu_flame_graph.png', dpi=300, bbox_inches='tight')
```

**Converting py-spy output**:

```bash
# Capture with py-spy
py-spy record -o profile.txt -d 30 -p <PID>

# Parse the output (implement based on py-spy format)
# py-spy outputs: function_name sample_count
# Convert to stack format for flame graph
```

### 2. **Hotspot Analysis** - `plot_sampling_hotspots()`

**Purpose**: Performance hotspot identification and prioritization
**Used by**: Engineering teams prioritizing optimization work

```python
# From aggregated sampling data
function_samples = [
    {'function': 'json.loads', 'samples': 1240, 'percentage': 24.8},
    {'function': 'database.query', 'samples': 890, 'percentage': 17.8},
    {'function': 'template.render', 'samples': 340, 'percentage': 6.8},
    {'function': 'auth.validate', 'samples': 120, 'percentage': 2.4}
    # ... more functions
]

fig = plot_sampling_hotspots(function_samples, threshold_pct=5.0)
fig.savefig('performance_hotspots.png', dpi=300, bbox_inches='tight')
```

## Real Profiler Integration Examples

### cProfile Integration

```python
import cProfile
import pstats
import io

def profile_application():
    profiler = cProfile.Profile()
    profiler.enable()

    # Your application code here
    main_application_function()

    profiler.disable()

    # Generate figures
    stats = pstats.Stats(profiler)

    # Top functions chart
    fig1 = plot_cprofile_top_functions(stats.get_stats(), top_n=25)
    fig1.savefig('reports/top_functions.png', dpi=300, bbox_inches='tight')

    # Call tree
    fig2 = plot_cprofile_call_tree(stats.get_stats(), max_depth=5)
    fig2.savefig('reports/call_tree.png', dpi=300, bbox_inches='tight')

    plt.close('all')  # Clean up
```

### py-spy Integration

```python
import subprocess
import re

def profile_with_pyspy(pid, duration=60):
    """Profile running process and generate flame graph"""

    # Capture stack traces
    cmd = f"py-spy record -o /tmp/profile.txt -d {duration} -p {pid} -r 99"
    subprocess.run(cmd.split())

    # Parse py-spy output to stack format
    stack_traces = []
    with open('/tmp/profile.txt', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                # Parse py-spy format: "function_path samples"
                parts = line.strip().rsplit(' ', 1)
                if len(parts) == 2:
                    stack = parts[0].replace(' ', ';')
                    samples = int(parts[1])
                    stack_traces.append({'stack': stack, 'samples': samples})

    # Generate flame graph
    fig = plot_flame_graph(stack_traces)
    fig.savefig('production_flame_graph.png', dpi=300, bbox_inches='tight')

    return fig
```

### line_profiler Integration

```python
# After running: kernprof -l -v your_script.py > profile_output.txt

def parse_line_profiler_output(output_file):
    """Parse line_profiler text output"""
    line_timings = []

    with open(output_file, 'r') as f:
        lines = f.readlines()

    # Find the line-by-line profile section
    in_profile = False
    for line in lines:
        if 'Line #' in line and 'Hits' in line:
            in_profile = True
            continue

        if in_profile and line.strip():
            # Parse line: "15    1000      0.234    0.000234      result = func()"
            parts = line.split(None, 4)
            if len(parts) >= 5:
                line_timings.append({
                    'line_number': int(parts[0]),
                    'hits': int(parts[1]) if parts[1] != '' else 0,
                    'time': float(parts[2]) if parts[2] != '' else 0.0,
                    'per_hit': float(parts[3]) if parts[3] != '' else 0.0,
                    'line_contents': parts[4]
                })

    return line_timings

# Usage
line_timings = parse_line_profiler_output('profile_output.txt')
fig = plot_line_profiler_heatmap(line_timings)
fig.savefig('line_analysis.png', dpi=300, bbox_inches='tight')
```

## Production Usage Patterns

### 1. **Performance Regression Detection**

```python
# Compare before/after profiles
baseline_stats = load_baseline_profile()
current_stats = profile_current_code()

fig1 = plot_cprofile_top_functions(baseline_stats, top_n=20)
fig1.suptitle('Baseline Performance')
fig1.savefig('baseline.png')

fig2 = plot_cprofile_top_functions(current_stats, top_n=20)
fig2.suptitle('Current Performance')
fig2.savefig('current.png')
```

### 2. **Production Monitoring**

```python
# Regular flame graph generation for production systems
def monitor_production():
    for service_pid in get_service_pids():
        fig = profile_with_pyspy(service_pid, duration=30)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig.savefig(f'monitoring/flame_{service_pid}_{timestamp}.png')
```

### 3. **Optimization Workflow**

```python
def optimization_workflow(target_function):
    # 1. Identify bottlenecks
    stats = profile_function(target_function)
    plot_cprofile_top_functions(stats, top_n=15)

    # 2. Line-level analysis
    line_data = line_profile_function(target_function)
    plot_line_profiler_heatmap(line_data)

    # 3. Memory analysis if needed
    memory_data = memory_profile_function(target_function)
    plot_memory_profiler_timeline(memory_data)
```

These figures represent the industry-standard visualizations used by performance engineers at major technology companies for production performance analysis and optimization work.
