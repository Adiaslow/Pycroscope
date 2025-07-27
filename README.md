# Pycroscope 1.0

[![Tests](https://github.com/Adiaslow/pycroscope/workflows/Tests%20and%20Coverage/badge.svg)](https://github.com/Adiaslow/pycroscope/actions)
[![codecov](https://codecov.io/gh/Adiaslow/pycroscope/branch/main/graph/badge.svg)](https://codecov.io/gh/Adiaslow/pycroscope)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Production-ready Python performance analysis with integrated scientific computing pattern detection**

Pycroscope 1.0 combines **comprehensive performance profiling with intelligent anti-pattern analysis** in a single, integrated framework. Built on established profiling tools (`line_profiler`, `psutil`, `cProfile`) and research-backed analysis libraries (`radon`, `vulture`), it provides immediate, actionable insights for optimizing scientific computing and general Python code.

**Key Capabilities:**

- 🔬 **Performance Profiling**: Line-by-line timing, call graphs, memory tracking
- 🎯 **Scientific Computing Analysis**: 9+ specialized detectors for vectorization, array operations, linear algebra optimizations
- 📊 **Integrated Reporting**: Single comprehensive report combining all findings
- ⚡ **Production Ready**: 38+ patterns detected with fail-fast architecture and zero error masking

## 🎯 Design Philosophy

- **"One Way, Many Options"**: Clean, unified API with extensive configuration for both profiling and analysis
- **Integrated Intelligence**: Performance profiling seamlessly combined with research-backed anti-pattern detection
- **Fail-Fast Architecture**: Zero tolerance for missing dependencies, no error masking with try-catch blocks
- **Scientific Computing Focus**: Specialized detection for vectorization, array operations, and numerical optimization opportunities
- **Conflict-Free**: Thread isolation and conflict detection prevent interference with other profiling
- **Principled Foundation**: Built on Pydantic V2, established profiling tools, and SOLID architectural principles

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
- `radon>=6.0.1` - Code complexity and maintainability analysis
- `vulture>=2.6` - Dead code detection
- `numpy>=1.21.0` - Numerical computing for scientific pattern analysis

### Visualization

- `matplotlib>=3.5.0` - Professional charts and plots
- `pandas>=1.3.0` - Data analysis and manipulation

## 🎨 Features

### ✅ Core Features (Production Ready)

- **Integrated Profiling & Analysis**: Performance profiling with automatic anti-pattern detection
- **Multiple Profilers**: 3 specialized profilers with conflict resolution via TraceMultiplexer
- **Scientific Computing Pattern Detection**: 9+ detectors for numerical optimization:
  - **Missed Vectorization**: Detects loops that could use NumPy operations
  - **Inefficient Array Operations**: `np.array(range(n))` → `np.arange(n)` optimizations
  - **Suboptimal Matrix Operations**: `np.tensordot()` vs `.dot()` performance differences
  - **Unnecessary Array Copies**: Identifies avoidable `.copy()` calls
  - **Inefficient Broadcasting**: Manual loops vs NumPy automatic broadcasting
  - **Scalar Array Operations**: Element-wise loops that could be vectorized
  - **Wrong Dtype Usage**: `float64` → `float32` optimization opportunities
  - **Inefficient Array Concatenation**: Loop-based concatenation anti-patterns
  - **Suboptimal Linear Algebra**: Matrix inversion vs solve() optimizations
- **General Code Quality Analysis**: Complexity, maintainability, dead code detection using `radon` and `vulture`
- **Hotspot Correlation**: Links detected patterns with actual performance bottlenecks
- **Configuration System**: Pydantic-based validation with fail-fast error handling
- **Session Management**: Complete profiling session lifecycle with `profiling_data.json`
- **CLI Interface**: Command-line profiling with integrated analysis (`pycroscope profile script.py`)
- **Visualization System**: Professional charts for call, line, and memory profiling
- **Comprehensive Reporting**: Single consolidated markdown report with profiling + analysis results
- **Production Testing**: Test suite covering core functionality with strict state rules

### 📊 Integrated Analysis Outputs

- **Performance Profiling**: Call graphs, line-by-line timing, memory usage charts
- **Scientific Computing Analysis**: Vectorization opportunities, array operation optimizations, linear algebra improvements
- **Code Quality Metrics**: Complexity analysis, maintainability scores, dead code identification
- **Hotspot Correlation**: Visual indicators of patterns found in performance bottlenecks
- **Consolidated Reporting**: Single comprehensive markdown report combining all profiling and analysis results
- **Pattern Distribution**: Breakdown of detected issues by type, severity, and performance impact
- **Actionable Insights**: Specific optimization suggestions with code examples and performance estimates
- **Production-Ready Output**: 38+ patterns detected across scientific computing, complexity, and maintainability categories

### 🚧 Development Areas

- **Interactive Dashboards**: Web-based profiling interface (plotly, rich)
- **Report Templates**: HTML/PDF report generation (jinja2)
- **Comparison Tools**: Session-to-session performance comparison
- **Enhanced Output**: Better terminal formatting (rich, tabulate)

## 🔄 Example Output

After profiling, you'll find in your output directory:

```
profiling_results/
├── profiling_data.json                 # Raw profiling data
├── profiling_report.md                 # Comprehensive analysis report (profiling + patterns)
├── call_top_functions.png             # Top functions bar chart
├── call_tree.png                      # Function call tree
├── line_heatmap.png                   # Line-by-line timing heatmap
└── memory_timeline.png                # Memory usage over time
```

**Sample Console Output:**

```
🔍 Running pattern analysis on 2 profiled files...
🎯 Pattern analysis complete - results integrated into comprehensive report
   ⚠️  Found 38 patterns across 2 files
   🏷️  Top patterns: scalar_array_operations(18), inefficient_array_concatenation(6), inefficient_broadcasting(5)
   🔥 Priority issues:
      1. 🚨 nested_loops
      2. ⚠️ too_many_parameters
      3. ⚠️ long_function
```

## 🏆 Design Principles

1. **Zero Tolerance for Missing Dependencies**: All dependencies properly declared, no try-catch import blocks
2. **Fail-Fast Error Handling**: No try-catch blocks masking errors, immediate failure on issues
3. **No Function-Level Imports**: Clean module-level imports throughout
4. **Conflict-Free Design**: Multiple profiling sessions can coexist safely via TraceMultiplexer
5. **SOLID Principles**: Single responsibility, dependency injection, clean interfaces
6. **Production-Ready Pattern Analysis**: 9+ scientific computing detectors with research-backed validation
7. **Integrated Intelligence**: Pattern analysis as core feature, not extension

## 📄 License

MIT License - see LICENSE file for details.

## 👨‍💻 Author

**Adam Murray** ([@Adiaslow](https://github.com/Adiaslow))

Pycroscope is designed and maintained with a focus on clean architecture, principled design patterns, and robust testing practices.
