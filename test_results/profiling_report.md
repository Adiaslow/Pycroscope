# Pycroscope Profiling Analysis Report

**Generated:** 2025-07-27 14:17:03
**Session ID:** `0c4e56d5-faa0-4588-8874-8d8517fc64f0`

## Executive Summary

- **Duration:** 0.076 seconds
- **Status:** completed
- **Profilers Used:** line, memory, call
- **Total Results:** 3

## Configuration

| Setting | Value |
|---------|-------|
| Line Profiling | `True` |
| Memory Profiling | `True` |
| Call Profiling | `True` |
| Output Dir | `test_results` |
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

### Per-Function Line-by-Line Analysis

**Functions Profiled:** 9

#### mkdtemp (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   373            1        226.0    226.0    30.3%  prefix, suffix, dir, output_type = _sanitize_param
   375            1         62.0     62.0     8.3%  names = _get_candidate_names()
   376            1          1.0      1.0     0.1%  if output_type is bytes:
   379            1          1.0      1.0     0.1%  for seq in range(TMP_MAX):
   380            1        202.0    202.0    27.1%  name = next(names)
   381            1         21.0     21.0     2.8%  file = _os.path.join(dir, prefix + name + suffix)
   382            1          1.0      1.0     0.1%  _sys.audit("tempfile.mkdtemp", file)
   383            1          0.0      0.0     0.0%  try:
   384            1        219.0    219.0    29.4%  _os.mkdir(file, 0o700)
   395            1         12.0     12.0     1.6%  return _os.path.abspath(file)
```

**Performance Insights:**
- **Line 373**: 30.3% of function time (1 hits)
- **Line 384**: 29.4% of function time (1 hits)
- **Line 380**: 27.1% of function time (1 hits)

#### _sanitize_params (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   116            1         97.0     97.0    56.1%  output_type = _infer_return_type(prefix, suffix, d
   117            1          1.0      1.0     0.6%  if suffix is None:
   118            1          1.0      1.0     0.6%  suffix = output_type()
   119            1          1.0      1.0     0.6%  if prefix is None:
   124            1          1.0      1.0     0.6%  if dir is None:
   125            1          1.0      1.0     0.6%  if output_type is str:
   126            1         70.0     70.0    40.5%  dir = gettempdir()
   129            1          1.0      1.0     0.6%  return prefix, suffix, dir, output_type
```

**Performance Insights:**
- **Line 116**: 56.1% of function time (1 hits)
- **Line 126**: 40.5% of function time (1 hits)
- **Line 117**: 0.6% of function time (1 hits)

#### _infer_return_type (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    87            1          1.0      1.0     5.9%  return_type = None
    88            4          4.0      1.0    23.5%  for arg in args:
    89            3          3.0      1.0    17.6%  if arg is None:
    90            2          2.0      1.0    11.8%  continue
    92            1          5.0      5.0    29.4%  if isinstance(arg, _os.PathLike):
    95            1          1.0      1.0     5.9%  if isinstance(arg, bytes):
   101            1          0.0      0.0     0.0%  if return_type is bytes:
   104            1          0.0      0.0     0.0%  return_type = str
   105            1          0.0      0.0     0.0%  if return_type is None:
   111            1          1.0      1.0     5.9%  return return_type
```

**Performance Insights:**
- **Line 92**: 29.4% of function time (1 hits)
- **Line 88**: 23.5% of function time (4 hits)
- **Line 89**: 17.6% of function time (3 hits)

#### gettempdir (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   315            1         51.0     51.0   100.0%  return _os.fsdecode(_gettempdir())
```

**Performance Insights:**
- **Line 315**: 100.0% of function time (1 hits)

#### _gettempdir (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   304            1          1.0      1.0    50.0%  if tempdir is None:
   311            1          1.0      1.0    50.0%  return tempdir
```

**Performance Insights:**
- **Line 304**: 50.0% of function time (1 hits)
- **Line 311**: 50.0% of function time (1 hits)

#### RLock (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/threading.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   133            1          1.0      1.0    33.3%  if _CRLock is None:
   135            1          2.0      2.0    66.7%  return _CRLock(*args, **kwargs)
```

**Performance Insights:**
- **Line 135**: 66.7% of function time (1 hits)
- **Line 133**: 33.3% of function time (1 hits)

#### _newname (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/threading.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   840            1          1.0      1.0   100.0%  return name_template % _counter()
```

**Performance Insights:**
- **Line 840**: 100.0% of function time (1 hits)

#### _make_invoke_excepthook (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/threading.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
  1361            1          2.0      2.0    15.4%  old_excepthook = excepthook
  1362            1          1.0      1.0     7.7%  old_sys_excepthook = _sys.excepthook
  1363            1          1.0      1.0     7.7%  if old_excepthook is None:
  1365            1          1.0      1.0     7.7%  if old_sys_excepthook is None:
  1368            1          1.0      1.0     7.7%  sys_exc_info = _sys.exc_info
  1369            1          1.0      1.0     7.7%  local_print = print
  1370            1          2.0      2.0    15.4%  local_sys = _sys
  1372            1          2.0      2.0    15.4%  def invoke_excepthook(thread):
  1404            1          2.0      2.0    15.4%  return invoke_excepthook
```

**Performance Insights:**
- **Line 1361**: 15.4% of function time (1 hits)
- **Line 1370**: 15.4% of function time (1 hits)
- **Line 1372**: 15.4% of function time (1 hits)

#### register_trace_function (/Users/Adam/Pycroscope/src/pycroscope/infrastructure/profilers/trace_multiplexer.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   124            1         44.0     44.0   100.0%  _trace_multiplexer.register_profiler(profiler_name
```

**Performance Insights:**
- **Line 124**: 100.0% of function time (1 hits)

### Line Profiling Summary

- **Total Lines Profiled:** 44
- **Total Hits:** 50
- **Total Time:** 0.001049 seconds
- **Average Time per Hit:** 0.000020980 seconds
## Memory Profiler Analysis

### Memory Usage Statistics

| Metric | Value |
|--------|-------|
| Peak Memory Usage | 107.11 MB |
| Average Memory Usage | 106.94 MB |
| Memory Delta | +0.38 MB |
| Sample Count | 7 |
| Initial Memory | 106.70 MB |
| Final Memory | 107.08 MB |

### Memory Timeline Analysis

- **Memory Growth Rate:** 5.1777 MB/second
- **Memory Spikes Detected:** 0 (>160.40 MB)

## Call Profiler Analysis

### 🕒 Top Functions by Execution Time

| Function | Total Time | Calls | Time/Call | Cumulative |
|----------|------------|-------|-----------|------------|
| `/Users/Adam/Pycroscope/src/pycroscope/infrastructure/profilers/line_profiler.py:0(stop)` | 0.0624s | 1 | 0.062385s | 0.0624s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/line_profiler.py:0(print_stats)` | 0.0597s | 1 | 0.059651s | 0.0597s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/line_profiler.py:0(show_text)` | 0.0595s | 1 | 0.059544s | 0.0595s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/line_profiler.py:0(show_func)` | 0.0444s | 12 | 0.003703s | 0.0444s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/toml_config.py:0(from_config)` | 0.0301s | 3 | 0.010035s | 0.0301s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/toml_config.py:0(find_and_read_config_file)` | 0.0268s | 2 | 0.013395s | 0.0268s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tomllib/_parser.py:0(load)` | 0.0263s | 2 | 0.013134s | 0.0263s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tomllib/_parser.py:0(loads)` | 0.0262s | 2 | 0.013115s | 0.0262s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/line_profiler.py:0(get_code_block)` | 0.0253s | 12 | 0.002110s | 0.0253s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tomllib/_parser.py:0(key_value_rule)` | 0.0217s | 78 | 0.000278s | 0.0217s |

### 📞 Most Called Functions

| Function | Calls | Total Time | Avg Time |
|----------|-------|------------|----------|
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tokenize.py:0(_generate_tokens_from_c_tokenizer)` | 1270 | 0.0073s | 0.000006s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/line_profiler.py:0(tokeneater)` | 1258 | 0.0088s | 0.000007s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/inspect.py:0(tokeneater)` | 1258 | 0.0049s | 0.000004s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/collections/__init__.py:0(_make)` | 1258 | 0.0033s | 0.000003s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tomllib/_parser.py:0(skip_chars)` | 1032 | 0.0041s | 0.000004s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tomllib/_parser.py:0(skip_comment)` | 470 | 0.0013s | 0.000003s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/line_profiler.py:0(<genexpr>)` | 309 | 0.0004s | 0.000001s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/copy.py:0(deepcopy)` | 258 | 0.0017s | 0.000007s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tomllib/_parser.py:0(skip_comments_and_array_ws)` | 252 | 0.0041s | 0.000016s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tomllib/_parser.py:0(parse_value)` | 202 | 0.0108s | 0.000054s |

**Total Functions Profiled:** 50

## Performance Insights

- No significant performance issues detected

## Technical Details

### Session Metadata

- **Start Time:** 2025-07-27 14:17:03.821404
- **End Time:** 2025-07-27 14:17:03.897323
- **Output Directory:** `test_results`
- **Session Name:** Default

