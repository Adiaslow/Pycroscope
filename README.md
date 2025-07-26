# Pycroscope 2.0

[![Tests](https://github.com/Adiaslow/pycroscope/workflows/Tests%20and%20Coverage/badge.svg)](https://github.com/Adiaslow/pycroscope/actions)
[![codecov](https://codecov.io/gh/Adiaslow/pycroscope/branch/main/graph/badge.svg)](https://codecov.io/gh/Adiaslow/pycroscope)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Python performance analysis and visualization using established profiling tools**

Pycroscope 2.0 is a complete rewrite that focuses on **analysis and visualization** rather than reinventing profiling infrastructure. Instead of custom profilers, we leverage battle-tested profiling packages to provide comprehensive performance analysis.

## 🎯 Design Philosophy

- **"One Way, Many Options"**: Clean, unified API with extensive configuration
- **No Special Cases**: Architecture naturally handles any use case, including profiling Pycroscope itself
- **Conflict-Free**: Thread isolation and conflict detection prevent interference with other profiling
- **Principled Foundation**: Built on Pydantic V2, established profiling tools, and clean abstractions

## 🚀 Quick Start

### Installation

```bash
pip install pycroscope
```

### Basic Usage

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
    # Your code here
    my_expensive_operation()

print(f"Profiling completed in {session.duration:.3f}s")
print(f"Profilers used: {', '.join(session.get_completed_profilers())}")
```

### Advanced Configuration

```python
from pycroscope import ProfileConfig, ProfilerSuite

# Custom configuration
config = ProfileConfig(
    line_profiling=True,        # line_profiler integration
    memory_profiling=True,      # memory_profiler integration
    call_profiling=True,        # cProfile integration
    sampling_profiling=True,    # py-spy integration (Unix only)
    output_dir="./profiling_results",
    create_visualizations=True
)

# For minimal overhead (e.g., profiling Pycroscope itself)
minimal_config = config.with_minimal_overhead()

# Use the profiler suite
suite = ProfilerSuite(config)
with suite.profile() as session:
    # Your code here
    pass
```

## 🏗️ Architecture

Pycroscope 2.0 is built around established profiling tools:

### Core Profilers

- **CallProfiler**: Wraps Python's built-in `cProfile` for function call analysis
- **LineProfiler**: Integrates `line_profiler` for line-by-line timing
- **MemoryProfiler**: Uses `memory_profiler` and `psutil` for memory tracking
- **SamplingProfiler**: Leverages `py-spy` for low-overhead sampling (Unix only)

### Key Components

- **ProfileConfig**: Pydantic-based configuration with validation and type safety
- **ProfileSession**: Manages profiling results and session lifecycle
- **ProfilerSuite**: Orchestrates multiple profilers without conflicts
- **Conflict Detection**: Prevents interference between profiling tools

## 📊 Profiling Tools Integration

| Tool              | Purpose             | Overhead   | Platform  |
| ----------------- | ------------------- | ---------- | --------- |
| `cProfile`        | Function calls      | Low        | All       |
| `line_profiler`   | Line-by-line timing | Medium     | All       |
| `memory_profiler` | Memory usage        | Low-Medium | All       |
| `py-spy`          | Sampling profiler   | Very Low   | Unix only |

## 🛠️ Command Line Interface

```bash
# Profile a Python script
pycroscope profile my_script.py --line --memory --call

# Use minimal overhead
pycroscope profile my_script.py --minimal

# List saved sessions
pycroscope list-sessions

# Run a demo (Pycroscope profiling itself!)
pycroscope demo
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

# Or use minimal overhead for sensitive scenarios
config = pycroscope.ProfileConfig().with_minimal_overhead()
suite = pycroscope.ProfilerSuite(config)

with suite.profile():
    # Profile Pycroscope's internal operations
    pycroscope_internal_function()
```

## 🧪 Testing

```bash
# Run basic functionality tests
python -m pytest tests/test_basic_functionality.py -v

# Run integration test directly
python tests/test_basic_functionality.py
```

## 📋 Dependencies

### Required

- `pydantic>=2.5.0` - Configuration and validation
- `psutil>=5.9.0` - System and process utilities
- `click>=8.0.0` - Command line interface
- `rich>=13.0.0` - Beautiful terminal output

### Profiling Tools

- `line-profiler>=4.0.0` - Line-by-line profiling
- `memory-profiler>=0.61.0` - Memory usage profiling
- `pympler>=0.9` - Advanced memory analysis
- `py-spy>=0.3.0` - Sampling profiler (Unix only)

### Analysis & Visualization

- `matplotlib>=3.5.0` - Plotting and visualization
- `seaborn>=0.11.0` - Statistical data visualization
- `pandas>=1.3.0` - Data analysis and manipulation
- `numpy>=1.21.0` - Numerical computing

## 🎨 Key Features

### ✅ What's Implemented

- **Core Infrastructure**: Complete profiling orchestration system
- **Multiple Profilers**: Integration with 4 established profiling tools
- **Configuration System**: Pydantic-based validation and type safety
- **Session Management**: Complete profiling session lifecycle
- **Conflict Detection**: Thread isolation and conflict prevention
- **CLI Interface**: Command-line tools for basic profiling tasks
- **Comprehensive Testing**: Test suite covering core functionality

### 🚧 Coming Next

- **Analysis Engine**: Advanced pattern detection and performance analysis
- **Visualization System**: Interactive charts and flame graphs
- **Report Generation**: HTML and PDF report generation
- **Web Interface**: Browser-based profiling dashboard
- **Comparison Tools**: Session-to-session performance comparison

## 🔄 Migration from 1.x

Pycroscope 2.0 is a complete rewrite with a cleaner API:

```python
# Old way (1.x)
from pycroscope import enable_profiling
profiler = enable_profiling(config)

# New way (2.0)
import pycroscope
with pycroscope.profile() as session:
    # Your code
    pass
```

## 🏆 Design Principles Achieved

1. **No Try/Except Import Blocks**: All dependencies are properly declared
2. **No Function-Level Imports**: Clean module-level imports throughout
3. **No Special Cases for Dogfooding**: Architecture naturally handles self-profiling
4. **Conflict-Free Design**: Multiple profiling sessions can coexist safely
5. **Principled Architecture**: Built on established patterns and clean abstractions

## 📄 License

MIT License - see LICENSE file for details.

## 👨‍💻 Author

**Adam Murray** ([@Adiaslow](https://github.com/Adiaslow))

Pycroscope is designed and maintained with a focus on clean architecture, principled design patterns, and robust testing practices.

## 🤝 Contributing

This is a demonstration of clean architecture principles applied to Python profiling. The codebase serves as an example of:

- Leveraging existing tools rather than reinventing them
- Clean API design with "One Way, Many Options"
- Conflict-free profiling orchestration
- Type-safe configuration with Pydantic V2
- Comprehensive testing and validation

---

**Pycroscope 2.0: Focus on analysis, leverage established tools, maintain architectural elegance.**
