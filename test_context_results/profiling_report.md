# Pycroscope Profiling Analysis Report

**Generated:** 2025-07-27 14:17:06
**Session ID:** `d9eb9bab-c44e-4803-9057-1a95dc095652`

## Executive Summary

- **Duration:** 0.057 seconds
- **Status:** completed
- **Profilers Used:** line, memory, call
- **Total Results:** 3

## Configuration

| Setting | Value |
|---------|-------|
| Line Profiling | `True` |
| Memory Profiling | `True` |
| Call Profiling | `True` |
| Output Dir | `test_context_results` |
| Session Name | `None` |
| Save Raw Data | `True` |
| Sampling Interval | `0.01` |
| Memory Precision | `3` |
| Max Call Depth | `50` |
| Generate Reports | `True` |
| Create Visualizations | `True` |
| Analyze Patterns | `True` |
| Enabled Patterns | `[<PatternType.NESTED_LOOPS: 'nested_loops'>, <PatternType.INEFFICIENT_DATA_STRUCTURE: 'inefficient_data_structure'>, <PatternType.UNNECESSARY_COMPUTATION: 'unnecessary_computation'>, <PatternType.MEMORY_LEAK_PATTERN: 'memory_leak_pattern'>, <PatternType.QUADRATIC_COMPLEXITY: 'quadratic_complexity'>, <PatternType.EXPONENTIAL_COMPLEXITY: 'exponential_complexity'>, <PatternType.RECURSIVE_WITHOUT_MEMOIZATION: 'recursive_without_memoization'>, <PatternType.DEAD_CODE: 'dead_code'>, <PatternType.UNUSED_IMPORTS: 'unused_imports'>, <PatternType.DUPLICATE_CODE: 'duplicate_code'>, <PatternType.HIGH_CYCLOMATIC_COMPLEXITY: 'high_cyclomatic_complexity'>, <PatternType.LOW_MAINTAINABILITY_INDEX: 'low_maintainability_index'>, <PatternType.LONG_FUNCTION: 'long_function'>, <PatternType.TOO_MANY_PARAMETERS: 'too_many_parameters'>, <PatternType.MISSED_VECTORIZATION: 'missed_vectorization'>, <PatternType.INEFFICIENT_ARRAY_OPERATIONS: 'inefficient_array_operations'>, <PatternType.SUBOPTIMAL_MATRIX_OPERATIONS: 'suboptimal_matrix_operations'>, <PatternType.NON_CONTIGUOUS_MEMORY_ACCESS: 'non_contiguous_memory_access'>, <PatternType.UNNECESSARY_ARRAY_COPY: 'unnecessary_array_copy'>, <PatternType.INEFFICIENT_BROADCASTING: 'inefficient_broadcasting'>, <PatternType.SCALAR_ARRAY_OPERATIONS: 'scalar_array_operations'>, <PatternType.WRONG_DTYPE_USAGE: 'wrong_dtype_usage'>, <PatternType.INEFFICIENT_ARRAY_CONCATENATION: 'inefficient_array_concatenation'>, <PatternType.SUBOPTIMAL_LINEAR_ALGEBRA: 'suboptimal_linear_algebra'>]` |
| Pattern Severity Threshold | `medium` |
| Pattern Confidence Threshold | `0.7` |
| Max Results Per File | `50` |
| Correlate Patterns With Profiling | `True` |
| Include Suggestions | `True` |
| Prioritize Hotspot Patterns | `True` |
| Hotspot Correlation Threshold | `0.1` |
| Pattern Complexity Threshold | `10` |
| Pattern Maintainability Threshold | `20.0` |
| Max Function Lines | `50` |
| Max Function Parameters | `5` |
| Exclude Test Files | `True` |
| Test File Patterns | `['*test*.py', '*_test.py', 'test_*.py']` |
| Generate Detailed Analysis Report | `True` |
| Save Intermediate Analysis Results | `False` |
| Profiler Prefix | `pycroscope` |
| Use Thread Isolation | `True` |
| Cleanup On Exit | `True` |

## Line Profiler Analysis

## Line Profiler Analysis
## Memory Profiler Analysis

### Memory Usage Statistics

| Metric | Value |
|--------|-------|
| Peak Memory Usage | 692.44 MB |
| Average Memory Usage | 692.36 MB |
| Memory Delta | +0.16 MB |
| Sample Count | 6 |
| Initial Memory | 692.25 MB |
| Final Memory | 692.41 MB |

### Memory Timeline Analysis

- **Memory Growth Rate:** 2.8996 MB/second
- **Memory Spikes Detected:** 0 (>1038.54 MB)

## Call Profiler Analysis

### 🕒 Top Functions by Execution Time

| Function | Total Time | Calls | Time/Call | Cumulative |
|----------|------------|-------|-----------|------------|
| `/Users/Adam/Pycroscope/src/pycroscope/infrastructure/profilers/line_profiler.py:0(stop)` | 0.0410s | 1 | 0.041023s | 0.0410s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/line_profiler.py:0(print_stats)` | 0.0400s | 1 | 0.039986s | 0.0400s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/line_profiler.py:0(show_text)` | 0.0399s | 1 | 0.039921s | 0.0399s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/line_profiler.py:0(show_func)` | 0.0244s | 12 | 0.002034s | 0.0244s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/line_profiler.py:0(get_code_block)` | 0.0216s | 12 | 0.001803s | 0.0216s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/inspect.py:0(getblock)` | 0.0194s | 12 | 0.001615s | 0.0194s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/toml_config.py:0(from_config)` | 0.0154s | 1 | 0.015414s | 0.0154s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/toml_config.py:0(find_and_read_config_file)` | 0.0141s | 1 | 0.014127s | 0.0141s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tomllib/_parser.py:0(load)` | 0.0137s | 1 | 0.013674s | 0.0137s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tomllib/_parser.py:0(loads)` | 0.0137s | 1 | 0.013655s | 0.0137s |

### 📞 Most Called Functions

| Function | Calls | Total Time | Avg Time |
|----------|-------|------------|----------|
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tokenize.py:0(_generate_tokens_from_c_tokenizer)` | 1270 | 0.0069s | 0.000005s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/line_profiler.py:0(tokeneater)` | 1258 | 0.0085s | 0.000007s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/inspect.py:0(tokeneater)` | 1258 | 0.0048s | 0.000004s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/collections/__init__.py:0(_make)` | 1258 | 0.0030s | 0.000002s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tomllib/_parser.py:0(skip_chars)` | 516 | 0.0021s | 0.000004s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tomllib/_parser.py:0(skip_comment)` | 235 | 0.0007s | 0.000003s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tomllib/_parser.py:0(skip_comments_and_array_ws)` | 126 | 0.0021s | 0.000017s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tomllib/_parser.py:0(parse_value)` | 101 | 0.0057s | 0.000057s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/copy.py:0(deepcopy)` | 86 | 0.0006s | 0.000007s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tomllib/_parser.py:0(parse_one_line_basic_str)` | 75 | 0.0054s | 0.000071s |

**Total Functions Profiled:** 50

## Performance Insights

- No significant performance issues detected

## Technical Details

### Session Metadata

- **Start Time:** 2025-07-27 14:17:06.148494
- **End Time:** 2025-07-27 14:17:06.205195
- **Output Directory:** `test_context_results`
- **Session Name:** Default

