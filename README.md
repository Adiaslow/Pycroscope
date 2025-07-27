# Pycroscope 1.0

[![Tests](https://github.com/Adiaslow/pycroscope/workflows/Tests%20and%20Coverage/badge.svg)](https://github.com/Adiaslow/pycroscope/actions)
[![codecov](https://codecov.io/gh/Adiaslow/pycroscope/branch/main/graph/badge.svg)](https://codecov.io/gh/Adiaslow/pycroscope)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Python performance analysis, pattern detection, and visualization using established tools**

Pycroscope 1.0 is a modern Python profiling framework that combines **performance profiling with intelligent pattern analysis**. Instead of reinventing profiling infrastructure, we leverage battle-tested tools to provide comprehensive performance analysis, anti-pattern detection, and beautiful visualizations in one integrated solution.

## 🎯 Design Philosophy

- **"One Way, Many Options"**: Clean, unified API with extensive configuration for both profiling and analysis
- **Integrated Intelligence**: Performance profiling seamlessly combined with anti-pattern detection
- **No Special Cases**: Architecture naturally handles any use case, including profiling Pycroscope itself
- **Conflict-Free**: Thread isolation and conflict detection prevent interference with other profiling
- **Principled Foundation**: Built on Pydantic V2, established profiling tools, and clean abstractions

## 🚀 Quick Start

### Installation

```bash
pip install pycroscope
```

### Command Line Usage

```bash
# Profile any Python script with all profilers enabled
pycroscope profile my_script.py

# Results saved to ./profiling_results/ with charts and reports
```

### Programmatic Usage

```python
import pycroscope

# Simple decorator usage
@pycroscope.profile()
def my_function():
    # Your code here
    data = [i ** 2 for i in range(1000)]
    return sum(data)

result = my_function()
```

```python
# Context manager usage
with pycroscope.profile() as session:
    # Your code here - gets profiled AND analyzed for patterns
    my_expensive_operation()

print(f"Profiling completed in {session.duration:.3f}s")
print(f"Profilers used: {', '.join(session.get_completed_profilers())}")

# Pattern analysis results are automatically generated and saved
```

### Advanced Configuration

```python
import pycroscope

# Comprehensive profiling with pattern analysis (default behavior)
@pycroscope.profile(
    analyze_patterns=True,                    # Pattern analysis enabled by default
    correlate_patterns_with_profiling=True,  # Link patterns to performance hotspots
    pattern_severity_threshold="medium",     # Report medium+ severity patterns
    detect_nested_loops=True,               # Detect O(n²) complexity issues
    detect_dead_code=True,                  # Find unused code and imports
    detect_complexity_issues=True           # High cyclomatic complexity detection
)
def my_function():
    # Your code gets profiled AND analyzed for anti-patterns
    pass

# Focus on specific types of analysis
config = pycroscope.ProfileConfig().with_performance_focus()  # Algorithmic issues
# or
config = pycroscope.ProfileConfig().with_maintainability_focus()  # Code quality
```

## 🏗️ Architecture

Pycroscope 1.0 is built around established profiling tools:

### Core Profilers

- **CallProfiler**: Custom trace-based function call analysis with caller-callee relationships
- **LineProfiler**: Integrates `line_profiler` for detailed line-by-line timing analysis
- **MemoryProfiler**: Uses `psutil` for comprehensive memory tracking

### Key Components

- **ProfileConfig**: Pydantic-based configuration with validation and type safety
- **ProfileSession**: Manages profiling results and session lifecycle
- **ProfilerOrchestra**: Orchestrates multiple profilers without conflicts
- **TraceMultiplexer**: Coordinates trace-based profilers to prevent conflicts

## 📊 Profiling Tools Integration

| Tool            | Purpose                 | Overhead | Platform |
| --------------- | ----------------------- | -------- | -------- |
| Custom Call     | Function calls & timing | Low      | All      |
| `line_profiler` | Line-by-line timing     | Medium   | All      |
| `psutil`        | Memory usage tracking   | Low      | All      |

## 🛠️ Command Line Interface

```bash
# Profile a Python script (all profilers enabled by default)
pycroscope profile my_script.py

# Disable specific profilers
pycroscope profile my_script.py --no-line --no-memory

# Use minimal overhead
pycroscope profile my_script.py --minimal

# Specify output directory
pycroscope profile my_script.py --output-dir ./my_results

# List saved sessions
pycroscope list-sessions --sessions-dir ./profiling_results
```

## 🔄 Dogfooding: Profiling Pycroscope Itself

The architecture is designed so that Pycroscope can naturally profile itself without special cases:

```python
import pycroscope

# This works without any special handling!
@pycroscope.profile()
def analyze_profiling_data(session_data):
    # Pycroscope analyzing its own profiling results
    return perform_analysis(session_data)

# Or profile Pycroscope's own source files
# pycroscope profile src/pycroscope/infrastructure/profilers/orchestra.py
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run integration tests
python -m pytest tests/integration/ -v

# Run unit tests
python -m pytest tests/unit/ -v
```

## 📋 Dependencies

### Required

- `pydantic>=2.0.0` - Configuration and validation
- `psutil>=5.9.0` - System and process utilities
- `click>=8.0.0` - Command line interface
- `line_profiler>=4.0.0` - Line-by-line profiling

### Visualization

- `matplotlib>=3.5.0` - Professional charts and plots
- `pandas>=1.3.0` - Data analysis and manipulation
- `numpy>=1.21.0` - Numerical computing for visualizations

## 🎨 Features

### ✅ Core Features (Implemented)

- **Integrated Profiling & Analysis**: Performance profiling with automatic pattern detection
- **Multiple Profilers**: 3 specialized profilers with conflict resolution
- **Intelligent Pattern Detection**: Anti-pattern analysis using validated tools (radon, vulture, AST)
- **Hotspot Correlation**: Links detected patterns with actual performance bottlenecks
- **Configuration System**: Pydantic-based validation and type safety
- **Session Management**: Complete profiling session lifecycle
- **Conflict Detection**: Thread isolation and trace multiplexer
- **CLI Interface**: Command-line profiling with integrated analysis
- **Visualization System**: Professional charts for call, line, and memory profiling
- **Comprehensive Reporting**: Integrated reports combining profiling and analysis results
- **Self-Profiling**: Can profile itself without special handling
- **Comprehensive Testing**: Test suite covering core functionality

### 📊 Integrated Analysis Outputs

- **Performance Profiling**: Call graphs, line-by-line timing, memory usage charts
- **Pattern Analysis**: Anti-pattern detection with severity classification
- **Hotspot Correlation**: Visual indicators of patterns found in performance bottlenecks
- **Comprehensive Reports**: Integrated profiling and analysis reports with prioritized recommendations
- **Pattern Distribution**: Breakdown of detected issues by type and severity
- **Actionable Insights**: Specific suggestions for optimization and code improvement

### 🚧 Development Areas

- **Advanced Analysis**: Pattern detection algorithms
- **Interactive Dashboards**: Web-based profiling interface (plotly, rich)
- **Report Templates**: HTML/PDF report generation (jinja2)
- **Comparison Tools**: Session-to-session performance comparison
- **Enhanced Output**: Better terminal formatting (rich, tabulate)

## 🔄 Example Output

After profiling, you'll find in your output directory:

```
profiling_results/
├── session.json                        # Raw profiling data
├── profiling_report.md                 # Performance profiling report
├── pattern_analysis_report.json        # Detected anti-patterns and recommendations
├── integrated_analysis_report.json     # Combined profiling + analysis insights
├── call_top_functions.png             # Top functions bar chart
├── call_tree.png                      # Function call tree
├── line_heatmap.png                   # Line-by-line timing heatmap
└── memory_timeline.png                # Memory usage over time
```

## 🏆 Design Principles Achieved

1. **No Try/Except Import Blocks**: All dependencies are properly declared
2. **No Function-Level Imports**: Clean module-level imports throughout
3. **No Special Cases for Dogfooding**: Architecture naturally handles self-profiling
4. **Conflict-Free Design**: Multiple profiling sessions can coexist safely
5. **Principled Architecture**: Built on established patterns and clean abstractions
6. **SOLID Principles**: Single responsibility, dependency injection, clean interfaces

## 📄 License

MIT License - see LICENSE file for details.

## 👨‍💻 Author

**Adam Murray** ([@Adiaslow](https://github.com/Adiaslow))

Pycroscope is designed and maintained with a focus on clean architecture, principled design patterns, and robust testing practices.

## 🤝 Contributing

This project demonstrates clean architecture principles applied to Python profiling:

- Leveraging existing tools rather than reinventing them
- Clean API design with "One Way, Many Options"
- Conflict-free profiling orchestration
- Type-safe configuration with Pydantic V2
- Comprehensive testing and validation

---

**Pycroscope 1.0: Clean profiling architecture leveraging established tools**
