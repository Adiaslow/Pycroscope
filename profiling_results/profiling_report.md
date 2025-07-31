# Pycroscope Profiling Analysis Report

**Generated:** 2025-07-30 22:09:17
**Session ID:** `26b13525-fa85-4dba-8a00-d3f05237571d`

## Executive Summary

- **Duration:** 34.137 seconds
- **Status:** completed
- **Profilers Used:** line, memory, call
- **Total Results:** 3
- **Patterns Detected:** 40

## Configuration

| Setting | Value |
|---------|-------|
| Line Profiling | `True` |
| Memory Profiling | `True` |
| Call Profiling | `True` |
| Output Dir | `profiling_results` |
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

**Functions Profiled:** 30

#### mkdtemp (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   373            2        352.0    176.0    32.4%  prefix, suffix, dir, output_type = _sanitize_param
   375            2         61.0     30.5     5.6%  names = _get_candidate_names()
   376            2          2.0      1.0     0.2%  if output_type is bytes:
   379            2          2.0      1.0     0.2%  for seq in range(TMP_MAX):
   380            2        247.0    123.5    22.7%  name = next(names)
   381            2         41.0     20.5     3.8%  file = _os.path.join(dir, prefix + name + suffix)
   382            2          1.0      0.5     0.1%  _sys.audit("tempfile.mkdtemp", file)
   383            2          1.0      0.5     0.1%  try:
   384            2        346.0    173.0    31.8%  _os.mkdir(file, 0o700)
   395            2         35.0     17.5     3.2%  return _os.path.abspath(file)
```

**Performance Insights:**
- **Line 373**: 32.4% of function time (2 hits)
- **Line 384**: 31.8% of function time (2 hits)
- **Line 380**: 22.7% of function time (2 hits)

#### _sanitize_params (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   116            2        188.0     94.0    65.3%  output_type = _infer_return_type(prefix, suffix, d
   117            2          2.0      1.0     0.7%  if suffix is None:
   118            2          2.0      1.0     0.7%  suffix = output_type()
   119            2          1.0      0.5     0.3%  if prefix is None:
   120            1          1.0      1.0     0.3%  if output_type is str:
   121            1          2.0      2.0     0.7%  prefix = template
   124            2          1.0      0.5     0.3%  if dir is None:
   125            2          2.0      1.0     0.7%  if output_type is str:
   126            2         86.0     43.0    29.9%  dir = gettempdir()
   129            2          3.0      1.5     1.0%  return prefix, suffix, dir, output_type
```

**Performance Insights:**
- **Line 116**: 65.3% of function time (2 hits)
- **Line 126**: 29.9% of function time (2 hits)
- **Line 129**: 1.0% of function time (2 hits)

#### _infer_return_type (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    87            2          3.0      1.5     9.4%  return_type = None
    88            8          9.0      1.1    28.1%  for arg in args:
    89            6          5.0      0.8    15.6%  if arg is None:
    90            5          2.0      0.4     6.2%  continue
    92            1          5.0      5.0    15.6%  if isinstance(arg, _os.PathLike):
    95            1          1.0      1.0     3.1%  if isinstance(arg, bytes):
   101            1          0.0      0.0     0.0%  if return_type is bytes:
   104            1          0.0      0.0     0.0%  return_type = str
   105            2          2.0      1.0     6.2%  if return_type is None:
   106            1          2.0      2.0     6.2%  if tempdir is None or isinstance(tempdir, str):
   107            1          3.0      3.0     9.4%  return str  # tempfile APIs return a str by defaul
   111            1          0.0      0.0     0.0%  return return_type
```

**Performance Insights:**
- **Line 88**: 28.1% of function time (8 hits)
- **Line 89**: 15.6% of function time (6 hits)
- **Line 92**: 15.6% of function time (1 hits)

#### gettempdir (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   315            2         59.0     29.5   100.0%  return _os.fsdecode(_gettempdir())
```

**Performance Insights:**
- **Line 315**: 100.0% of function time (2 hits)

#### _gettempdir (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   304            2          2.0      1.0    50.0%  if tempdir is None:
   311            2          2.0      1.0    50.0%  return tempdir
```

**Performance Insights:**
- **Line 304**: 50.0% of function time (2 hits)
- **Line 311**: 50.0% of function time (2 hits)

#### _get_candidate_names (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   233            1          1.0      1.0    33.3%  if _name_sequence is None:
   240            1          2.0      2.0    66.7%  return _name_sequence
```

**Performance Insights:**
- **Line 240**: 66.7% of function time (1 hits)
- **Line 233**: 33.3% of function time (1 hits)

#### RLock (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/threading.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   133            1          0.0      0.0     0.0%  if _CRLock is None:
   135            1          1.0      1.0   100.0%  return _CRLock(*args, **kwargs)
```

**Performance Insights:**
- **Line 135**: 100.0% of function time (1 hits)
- **Line 133**: 0.0% of function time (1 hits)

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
  1361            1          1.0      1.0    16.7%  old_excepthook = excepthook
  1362            1          1.0      1.0    16.7%  old_sys_excepthook = _sys.excepthook
  1363            1          1.0      1.0    16.7%  if old_excepthook is None:
  1365            1          0.0      0.0     0.0%  if old_sys_excepthook is None:
  1368            1          0.0      0.0     0.0%  sys_exc_info = _sys.exc_info
  1369            1          0.0      0.0     0.0%  local_print = print
  1370            1          1.0      1.0    16.7%  local_sys = _sys
  1372            1          1.0      1.0    16.7%  def invoke_excepthook(thread):
  1404            1          1.0      1.0    16.7%  return invoke_excepthook
```

**Performance Insights:**
- **Line 1361**: 16.7% of function time (1 hits)
- **Line 1362**: 16.7% of function time (1 hits)
- **Line 1363**: 16.7% of function time (1 hits)

#### register_trace_function (/Users/Adam/Pycroscope/src/pycroscope/infrastructure/profilers/trace_multiplexer.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   124            1         42.0     42.0   100.0%  _trace_multiplexer.register_profiler(profiler_name
```

**Performance Insights:**
- **Line 124**: 100.0% of function time (1 hits)

#### run_sample_workload (/Users/Adam/Pycroscope/docs/examples/usage_example.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    43            1       1959.0   1959.0     0.0%  from sample_workload import mixed_workload
    45            1          5.0      5.0     0.0%  print("Executing workload to be profiled...")
    46            1          3.0      3.0     0.0%  print("   (Replace this section with your own code
    47            1          3.0      3.0     0.0%  print()
    50            1   33978520.0 33978520.0   100.0%  results = mixed_workload()
    52            1          2.0      2.0     0.0%  return results
```

**Performance Insights:**
- **Line 50**: 100.0% of function time (1 hits)
- **Line 43**: 0.0% of function time (1 hits)
- **Line 45**: 0.0% of function time (1 hits)

#### _type_check (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/typing.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   187            3          4.0      1.3     4.3%  invalid_generic_forms = (Generic, Protocol)
   188            3          3.0      1.0     3.2%  if not allow_special_forms:
   189            3          4.0      1.3     4.3%  invalid_generic_forms += (ClassVar,)
   190            3          3.0      1.0     3.2%  if is_argument:
   191            3          5.0      1.7     5.4%  invalid_generic_forms += (Final,)
   193            3         54.0     18.0    58.1%  arg = _type_convert(arg, module=module, allow_spec
   194            3          3.0      1.0     3.2%  if (isinstance(arg, _GenericAlias) and
   197            3          3.0      1.0     3.2%  if arg in (Any, LiteralString, NoReturn, Never, Se
   199            3          3.0      1.0     3.2%  if allow_special_forms and arg in (ClassVar, Final
   201            3          3.0      1.0     3.2%  if isinstance(arg, _SpecialForm) or arg in (Generi
   203            3          3.0      1.0     3.2%  if type(arg) is tuple:
   205            3          5.0      1.7     5.4%  return arg
```

**Performance Insights:**
- **Line 193**: 58.1% of function time (3 hits)
- **Line 191**: 5.4% of function time (3 hits)
- **Line 205**: 5.4% of function time (3 hits)

#### _type_convert (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/typing.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   168            3          3.0      1.0    30.0%  if arg is None:
   170            3          3.0      1.0    30.0%  if isinstance(arg, str):
   172            3          4.0      1.3    40.0%  return arg
```

**Performance Insights:**
- **Line 172**: 40.0% of function time (3 hits)
- **Line 168**: 30.0% of function time (3 hits)
- **Line 170**: 30.0% of function time (3 hits)

#### _is_dunder (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/typing.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
  1162           12         20.0      1.7   100.0%  return attr.startswith('__') and attr.endswith('__
```

**Performance Insights:**
- **Line 1162**: 100.0% of function time (12 hits)

#### mixed_workload (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   357            1          2.0      2.0     0.0%  print("Running mixed workload demonstration...")
   359            1          1.0      1.0     0.0%  results = {}
   362            1          1.0      1.0     0.0%  start_time = time.time()
   363            1   15190202.0 15190202.0    44.7%  cpu_result = cpu_intensive_calculation()
   364            1          2.0      2.0     0.0%  results["cpu_time"] = time.time() - start_time
   365            1          2.0      2.0     0.0%  results["cpu_result_sample"] = cpu_result
   368            1          1.0      1.0     0.0%  start_time = time.time()
   369            1    1493286.0 1493286.0     4.4%  memory_data = memory_intensive_operations()
   370            1          3.0      3.0     0.0%  results["memory_time"] = time.time() - start_time
   371            1          1.0      1.0     0.0%  results["memory_objects_created"] = len(memory_dat
   374            1          1.0      1.0     0.0%  start_time = time.time()
   375            1     348440.0 348440.0     1.0%  io_result = file_io_operations()
   376            1          2.0      2.0     0.0%  results["io_time"] = time.time() - start_time
   377            1          1.0      1.0     0.0%  results["io_operations"] = io_result
   380            1          1.0      1.0     0.0%  start_time = time.time()
   381            1        130.0    130.0     0.0%  call_result = nested_function_calls()
   382            1          1.0      1.0     0.0%  results["call_time"] = time.time() - start_time
   383            1          1.0      1.0     0.0%  results["call_result"] = call_result
   386            1          1.0      1.0     0.0%  start_time = time.time()
   387            1      55173.0  55173.0     0.2%  processing_result = data_processing_pipeline()
   388            1          2.0      2.0     0.0%  results["processing_time"] = time.time() - start_t
   389            1          1.0      1.0     0.0%  results["processing_stats"] = processing_result
   392            1          2.0      2.0     0.0%  start_time = time.time()
   393            1   15507625.0 15507625.0    45.7%  antipattern_result = demonstrate_anti_patterns()
   394            1          2.0      2.0     0.0%  results["antipattern_time"] = time.time() - start_
   395            1          1.0      1.0     0.0%  results["antipattern_stats"] = antipattern_result
   398            1          1.0      1.0     0.0%  start_time = time.time()
   399            1    1370047.0 1370047.0     4.0%  fib_recursive = fibonacci_recursive(25)  # Small e
   400            1          2.0      2.0     0.0%  results["fib_recursive_time"] = time.time() - star
   402            1          1.0      1.0     0.0%  start_time = time.time()
   403            1         56.0     56.0     0.0%  fib_iterative = fibonacci_iterative(25)
   404            1          1.0      1.0     0.0%  results["fib_iterative_time"] = time.time() - star
   406            1          1.0      1.0     0.0%  results["fib_results_match"] = fib_recursive == fi
   408            1          3.0      3.0     0.0%  return results
```

**Performance Insights:**
- **Line 393**: 45.7% of function time (1 hits)
- **Line 363**: 44.7% of function time (1 hits)
- **Line 369**: 4.4% of function time (1 hits)

#### cpu_intensive_calculation (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   134            1          4.0      4.0     0.0%  print("Running CPU-intensive calculations...")
   137            1          1.0      1.0     0.0%  size = 200
   138            1      22799.0  22799.0     0.2%  matrix_a = [[random.random() for _ in range(size)]
   139            1      22972.0  22972.0     0.2%  matrix_b = [[random.random() for _ in range(size)]
   141            1          1.0      1.0     0.0%  result = 0.0
   142          201        173.0      0.9     0.0%  for i in range(size):
   143        40200      32493.0      0.8     0.3%  for j in range(size):
   144      8040000    6491779.0      0.8    50.0%  for k in range(size):
   145      8000000    6402371.0      0.8    49.3%  result += matrix_a[i][k] * matrix_b[k][j]
   148        10001       8217.0      0.8     0.1%  for i in range(10000):
   149        10000       8923.0      0.9     0.1%  result += math.sin(i) * math.cos(i) * math.sqrt(i 
   151            1          9.0      9.0     0.0%  return result
```

**Performance Insights:**
- **Line 144**: 50.0% of function time (8,040,000 hits)
- **Line 145**: 49.3% of function time (8,000,000 hits)
- **Line 143**: 0.3% of function time (40,200 hits)

#### memory_intensive_operations (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   156            1          4.0      4.0     0.0%  print("Running memory-intensive operations...")
   159            1          1.0      1.0     0.0%  large_list = []
   160        50001      40941.0      0.8     2.9%  for i in range(50000):
   161       100000      81211.0      0.8     5.8%  large_list.append(
   162        50000      59945.0      1.2     4.3%  {
   163        50000      38443.0      0.8     2.7%  "id": i,
   164        50000     623693.0     12.5    44.5%  "data": [random.random() for _ in range(20)],
   165        50000      47203.0      0.9     3.4%  "metadata": {
   166        50000      41728.0      0.8     3.0%  "timestamp": time.time(),
   167        50000      43950.0      0.9     3.1%  "category": f"category_{i % 10}",
   168        50000     203372.0      4.1    14.5%  "tags": [f"tag_{j}" for j in range(i % 5 + 1)],
   174            1          1.0      1.0     0.0%  lookup_dict = {}
   175        50001      40041.0      0.8     2.9%  for item in large_list:
   176        50000      38924.0      0.8     2.8%  category = item["metadata"]["category"]
   177        50000      40765.0      0.8     2.9%  if category not in lookup_dict:
   178           10          8.0      0.8     0.0%  lookup_dict[category] = []
   179        50000      40082.0      0.8     2.9%  lookup_dict[category].append(item)
   182            1          1.0      1.0     0.0%  temp_lists = []
   183          101         94.0      0.9     0.0%  for i in range(100):
   184          100      59993.0    599.9     4.3%  temp_list = [random.random() for _ in range(1000)]
   185          100         91.0      0.9     0.0%  temp_lists.append(temp_list)
   186          100         95.0      0.9     0.0%  if len(temp_lists) > 10:
   187           90        487.0      5.4     0.0%  temp_lists.pop(0)  # Remove oldest
   189            1          7.0      7.0     0.0%  return large_list
```

**Performance Insights:**
- **Line 164**: 44.5% of function time (50,000 hits)
- **Line 168**: 14.5% of function time (50,000 hits)
- **Line 161**: 5.8% of function time (100,000 hits)

#### file_io_operations (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   194            1          5.0      5.0     0.0%  print("Running file I/O operations...")
   197            1        409.0    409.0     0.1%  temp_dir = tempfile.mkdtemp()
   199            1          0.0      0.0     0.0%  try:
   201            1          1.0      1.0     0.0%  file_data = {}
   202           11         15.0      1.4     0.0%  for i in range(10):
   203           10        237.0     23.7     0.1%  filename = os.path.join(temp_dir, f"test_file_{i}.
   204           10          9.0      0.9     0.0%  data = {
   205           10         15.0      1.5     0.0%  "file_id": i,
   206           10       5477.0    547.7     1.6%  "content": [random.random() for _ in range(1000)],
   207           10         11.0      1.1     0.0%  "metadata": {"created": time.time()},
   210           20       1892.0     94.6     0.5%  with open(filename, "w") as f:
   211           10     336372.0  33637.2    96.7%  json.dump(data, f)
   213           10         13.0      1.3     0.0%  file_data[filename] = data
   216            1          1.0      1.0     0.0%  read_data = {}
   217           11          8.0      0.7     0.0%  for filename in file_data:
   218           20        278.0     13.9     0.1%  with open(filename, "r") as f:
   219           10       2465.0    246.5     0.7%  read_data[filename] = json.load(f)
   222            1          1.0      1.0     0.0%  total_values = 0
   223           11          9.0      0.8     0.0%  for data in read_data.values():
   224           10          9.0      0.9     0.0%  total_values += len(data["content"])
   226            1          1.0      1.0     0.0%  return {"files_processed": len(read_data), "total_
   230           11         29.0      2.6     0.0%  for filename in os.listdir(temp_dir):
   231           10        552.0     55.2     0.2%  os.remove(os.path.join(temp_dir, filename))
   232            1         42.0     42.0     0.0%  os.rmdir(temp_dir)
```

**Performance Insights:**
- **Line 211**: 96.7% of function time (10 hits)
- **Line 206**: 1.6% of function time (10 hits)
- **Line 219**: 0.7% of function time (10 hits)

#### dump (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   165           10         10.0      1.0     0.0%  if (not skipkeys and ensure_ascii and
   166           10         10.0      1.0     0.0%  check_circular and allow_nan and
   167           10         10.0      1.0     0.0%  cls is None and indent is None and separators is N
   168           10         10.0      1.0     0.0%  default is None and not sort_keys and not kw):
   169           10        420.0     42.0     0.1%  iterable = _default_encoder.iterencode(obj)
   179        10180     321509.0     31.6    96.8%  for chunk in iterable:
   180        10170      10127.0      1.0     3.0%  fp.write(chunk)
```

**Performance Insights:**
- **Line 179**: 96.8% of function time (10,180 hits)
- **Line 180**: 3.0% of function time (10,170 hits)
- **Line 169**: 0.1% of function time (10 hits)

#### _make_iterencode (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   275           10         11.0      1.1    18.0%  if _indent is not None and not isinstance(_indent,
   278           10         12.0      1.2    19.7%  def _iterencode_list(lst, _current_indent_level):
   334           10         12.0      1.2    19.7%  def _iterencode_dict(dct, _current_indent_level):
   414           10         11.0      1.1    18.0%  def _iterencode(o, _current_indent_level):
   443           10         15.0      1.5    24.6%  return _iterencode
```

**Performance Insights:**
- **Line 443**: 24.6% of function time (10 hits)
- **Line 278**: 19.7% of function time (10 hits)
- **Line 334**: 19.7% of function time (10 hits)

#### load (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   293           30       2325.0     77.5    99.1%  return loads(fp.read(),
   294           10          7.0      0.7     0.3%  cls=cls, object_hook=object_hook,
   295           10          5.0      0.5     0.2%  parse_float=parse_float, parse_int=parse_int,
   296           20         10.0      0.5     0.4%  parse_constant=parse_constant, object_pairs_hook=o
```

**Performance Insights:**
- **Line 293**: 99.1% of function time (30 hits)
- **Line 296**: 0.4% of function time (20 hits)
- **Line 294**: 0.3% of function time (10 hits)

#### loads (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   333           10         10.0      1.0     0.5%  if isinstance(s, str):
   334           10         10.0      1.0     0.5%  if s.startswith('\ufeff'):
   343           10          7.0      0.7     0.4%  if (cls is None and object_hook is None and
   344           10          8.0      0.8     0.4%  parse_int is None and parse_float is None and
   345           10          8.0      0.8     0.4%  parse_constant is None and object_pairs_hook is No
   346           10       1932.0    193.2    97.8%  return _default_decoder.decode(s)
```

**Performance Insights:**
- **Line 346**: 97.8% of function time (10 hits)
- **Line 333**: 0.5% of function time (10 hits)
- **Line 334**: 0.5% of function time (10 hits)

#### nested_function_calls (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   237            1          3.0      3.0     4.2%  print("Running nested function calls...")
   239            1          2.0      2.0     2.8%  def level_1(n: int) -> int:
   244            1          1.0      1.0     1.4%  def level_2(n: int) -> int:
   249            1          1.0      1.0     1.4%  def level_3(n: int) -> int:
   254            1          1.0      1.0     1.4%  def level_4(n: int) -> int:
   259            1          1.0      1.0     1.4%  def level_5(n: int) -> int:
   262            1         63.0     63.0    87.5%  return level_1(8)
```

**Performance Insights:**
- **Line 262**: 87.5% of function time (1 hits)
- **Line 237**: 4.2% of function time (1 hits)
- **Line 239**: 2.8% of function time (1 hits)

#### data_processing_pipeline (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   267            1          2.0      2.0     0.0%  print("Running data processing pipeline...")
   270            1       5767.0   5767.0    10.5%  raw_data = [random.random() * 1000 for _ in range(
   273            1       5587.0   5587.0    10.2%  filtered_data = [x for x in raw_data if x > 100]
   276            1       5437.0   5437.0     9.9%  transformed_data = [math.log(x) if x > 1 else 0 fo
   279            1         22.0     22.0     0.0%  sum_data = sum(transformed_data)
   280            1          3.0      3.0     0.0%  avg_data = sum_data / len(transformed_data) if tra
   281            1         68.0     68.0     0.1%  max_data = max(transformed_data) if transformed_da
   282            1         66.0     66.0     0.1%  min_data = min(transformed_data) if transformed_da
   285            2      37942.0  18971.0    69.1%  variance = sum((x - avg_data) ** 2 for x in transf
   286            1          1.0      1.0     0.0%  transformed_data
   288            1          2.0      2.0     0.0%  std_dev = math.sqrt(variance)
   290            1          3.0      3.0     0.0%  return {
   291            1          1.0      1.0     0.0%  "count": len(transformed_data),
   292            1          0.0      0.0     0.0%  "sum": sum_data,
   293            1          0.0      0.0     0.0%  "average": avg_data,
   294            1          1.0      1.0     0.0%  "maximum": max_data,
   295            1          1.0      1.0     0.0%  "minimum": min_data,
   296            1          1.0      1.0     0.0%  "std_dev": std_dev,
```

**Performance Insights:**
- **Line 285**: 69.1% of function time (2 hits)
- **Line 270**: 10.5% of function time (1 hits)
- **Line 273**: 10.2% of function time (1 hits)

#### demonstrate_anti_patterns (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   302            1          4.0      4.0     0.0%  print("Running anti-pattern demonstrations...")
   305            1          8.0      8.0     0.0%  import numpy as np
   308            1         56.0     56.0     0.0%  bad_array = np.array(range(1000))  # Should use np
   311            1         24.0     24.0     0.0%  test_array = np.random.random(1000)
   312            1         39.0     39.0     0.0%  count_result = (test_array > 0.5).sum()  # Should 
   315            1         21.0     21.0     0.0%  matrix_a = np.random.random((100, 50))
   316            1         18.0     18.0     0.0%  matrix_b = np.random.random((50, 100))
   317            1        183.0    183.0     0.0%  result_tensordot = np.tensordot(matrix_a, matrix_b
   320            1          6.0      6.0     0.0%  original_array = np.random.random(1000)
   321            1          5.0      5.0     0.0%  copied_array = original_array.copy()  # Might be u
   324            1          8.0      8.0     0.0%  data = list(range(1000))
   327            1       3915.0   3915.0     0.0%  result = inefficient_nested_search(data, [500, 750
   330            1        502.0    502.0     0.0%  complex_result = overly_complex_function(10, 20, 3
   333            1         61.0     61.0     0.0%  string_data = [f"item_{i}" for i in range(100)]
   334            1        508.0    508.0     0.0%  processed_data = inefficient_data_operations(strin
   337            1   15501874.0 15501874.0   100.0%  fib_recursive_result = fibonacci_recursive(30)
   338            1        137.0    137.0     0.0%  fib_iterative_result = fibonacci_iterative(30)
   340            1          4.0      4.0     0.0%  return {
   341            1          1.0      1.0     0.0%  "nested_search": result,
   342            1          1.0      1.0     0.0%  "complex_function": complex_result,
   343            1          1.0      1.0     0.0%  "data_operations": len(processed_data),
   344            1          0.0      0.0     0.0%  "fibonacci_recursive": fib_recursive_result,
   345            1          0.0      0.0     0.0%  "fibonacci_iterative": fib_iterative_result,
   346            1          0.0      0.0     0.0%  "fibonacci_match": fib_recursive_result == fib_ite
   348            1          6.0      6.0     0.0%  "bad_array_size": len(bad_array),
   349            1          1.0      1.0     0.0%  "count_result": count_result,
   350            1          2.0      2.0     0.0%  "tensordot_shape": result_tensordot.shape,
   351            1          2.0      2.0     0.0%  "copied_array_size": len(copied_array),
```

**Performance Insights:**
- **Line 337**: 100.0% of function time (1 hits)
- **Line 327**: 0.0% of function time (1 hits)
- **Line 334**: 0.0% of function time (1 hits)

#### inefficient_nested_search (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    47            1          1.0      1.0     0.0%  found_items = []
    50            4          3.0      0.8     0.1%  for target in targets:
    51         2252       1666.0      0.7    50.4%  for item in data_list:
    52         2252       1631.0      0.7    49.3%  if item == target:
    53            3          4.0      1.3     0.1%  found_items.append(item)
    54            3          2.0      0.7     0.1%  break  # At least we break early
    56            1          1.0      1.0     0.0%  return found_items
```

**Performance Insights:**
- **Line 51**: 50.4% of function time (2,252 hits)
- **Line 52**: 49.3% of function time (2,252 hits)
- **Line 53**: 0.1% of function time (3 hits)

#### overly_complex_function (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    64            1          2.0      2.0     0.7%  result = 0
    67            1          1.0      1.0     0.3%  if param1 > 0:
    68            1          1.0      1.0     0.3%  if param2 > 0:
    69            1          0.0      0.0     0.0%  if param3 > 0:
    70            1          1.0      1.0     0.3%  if param4 > 0:
    71            1          1.0      1.0     0.3%  if param5 > 0:
    72            1          1.0      1.0     0.3%  if param6 > 0:
    73            1          1.0      1.0     0.3%  if param7 > 0:
    74            1          0.0      0.0     0.0%  result = (
    75            7          7.0      1.0     2.4%  param1
    76            1          1.0      1.0     0.3%  + param2
    77            1          1.0      1.0     0.3%  + param3
    78            1          1.0      1.0     0.3%  + param4
    79            1          1.0      1.0     0.3%  + param5
    80            1          0.0      0.0     0.0%  + param6
    81            1          1.0      1.0     0.3%  + param7
    99           11         14.0      1.3     4.8%  for i in range(10):
   100          110        132.0      1.2    45.2%  for j in range(10):
   101          100        116.0      1.2    39.7%  if i + j == result % 10:
   102           10          7.0      0.7     2.4%  result += 1
   104            1          3.0      3.0     1.0%  return result
```

**Performance Insights:**
- **Line 100**: 45.2% of function time (110 hits)
- **Line 101**: 39.7% of function time (100 hits)
- **Line 99**: 4.8% of function time (11 hits)

#### inefficient_data_operations (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   113            1          1.0      1.0     0.3%  processed_items = []
   114            1          0.0      0.0     0.0%  unique_items = []
   116          101         78.0      0.8    22.1%  for item in items:
   117          100        100.0      1.0    28.3%  if item not in processed_items:  # O(n) operation 
   118          100         78.0      0.8    22.1%  processed_items.append(item)
   119          100         95.0      0.9    26.9%  unique_items.append(f"processed_{item}")
   121            1          1.0      1.0     0.3%  return unique_items
```

**Performance Insights:**
- **Line 117**: 28.3% of function time (100 hits)
- **Line 119**: 26.9% of function time (100 hits)
- **Line 116**: 22.1% of function time (101 hits)

#### fibonacci_recursive (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    27      2935322    2360981.0      0.8    25.8%  if n <= 1:
    28      1467662    1719272.0      1.2    18.8%  return n
    29      1467660    5085541.0      3.5    55.5%  return fibonacci_recursive(n - 1) + fibonacci_recu
```

**Performance Insights:**
- **Line 29**: 55.5% of function time (1,467,660 hits)
- **Line 27**: 25.8% of function time (2,935,322 hits)
- **Line 28**: 18.8% of function time (1,467,662 hits)

#### fibonacci_iterative (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    34            2          2.0      1.0     2.2%  if n <= 1:
    36            2          3.0      1.5     3.2%  a, b = 0, 1
    37           55         43.0      0.8    46.2%  for _ in range(2, n + 1):
    38           53         41.0      0.8    44.1%  a, b = b, a + b
    39            2          4.0      2.0     4.3%  return b
```

**Performance Insights:**
- **Line 37**: 46.2% of function time (55 hits)
- **Line 38**: 44.1% of function time (53 hits)
- **Line 39**: 4.3% of function time (2 hits)

### Line Profiling Summary

- **Total Lines Profiled:** 284
- **Total Hits:** 22,697,860
- **Total Time:** 107.754491 seconds
- **Average Time per Hit:** 0.000004747 seconds
## Memory Profiler Analysis

### Memory Usage Statistics

| Metric | Value |
|--------|-------|
| Peak Memory Usage | 186.78 MB |
| Average Memory Usage | 157.23 MB |
| Memory Delta | +68.38 MB |
| Sample Count | 2245 |
| Initial Memory | 104.81 MB |
| Final Memory | 173.19 MB |

### Memory Timeline Analysis

- **Memory Growth Rate:** 2.0032 MB/second
- **Memory Spikes Detected:** 0 (>235.84 MB)

## Call Profiler Analysis

### 🕒 Top Functions by Execution Time

| Function | Total Time | Calls | Time/Call | Cumulative |
|----------|------------|-------|-----------|------------|
| `/Users/Adam/Pycroscope/docs/examples/usage_example.py:0(run_profiling_session)` | 33.9802s | 1 | 33.980151s | 33.9802s |
| `/Users/Adam/Pycroscope/docs/examples/usage_example.py:0(run_sample_workload)` | 33.9800s | 1 | 33.980021s | 33.9800s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(mixed_workload)` | 33.9645s | 1 | 33.964530s | 33.9645s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(demonstrate_anti_patterns)` | 15.5072s | 1 | 15.507181s | 15.5072s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(cpu_intensive_calculation)` | 15.1893s | 1 | 15.189342s | 15.1893s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(fibonacci_recursive)` | 4.2034s | 2935322 | 0.000001s | 4.2034s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(memory_intensive_operations)` | 1.4929s | 1 | 1.492858s | 1.4929s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(file_io_operations)` | 0.3479s | 1 | 0.347928s | 0.3479s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py:0(dump)` | 0.3362s | 10 | 0.033622s | 0.3362s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode)` | 0.2718s | 10180 | 0.000027s | 0.2718s |

### 📞 Most Called Functions

| Function | Calls | Total Time | Avg Time |
|----------|-------|------------|----------|
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(fibonacci_recursive)` | 2935322 | 4.2034s | 0.000001s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode_dict)` | 10240 | 0.2186s | 0.000021s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode)` | 10180 | 0.2718s | 0.000027s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode_list)` | 10020 | 0.1686s | 0.000017s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(floatstr)` | 10010 | 0.0444s | 0.000004s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(<genexpr>)` | 9014 | 0.0138s | 0.000002s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tokenize.py:0(_generate_tokens_from_c_tokenizer)` | 5639 | 0.0288s | 0.000005s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/line_profiler.py:0(tokeneater)` | 5607 | 0.0358s | 0.000006s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/inspect.py:0(tokeneater)` | 5607 | 0.0206s | 0.000004s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/collections/__init__.py:0(_make)` | 5607 | 0.0127s | 0.000002s |

**Total Functions Profiled:** 50

## Pattern Analysis Results

**Analysis Summary:** 40 patterns detected across 2 files

### Pattern Distribution

| Pattern Type | Count |
|--------------|-------|
| Scalar Array Operations | 18 |
| Inefficient Array Concatenation | 6 |
| Inefficient Broadcasting | 5 |
| Long Function | 4 |
| High Cyclomatic Complexity | 2 |
| Nested Loops | 1 |
| Too Many Parameters | 1 |
| Inefficient Array Operations | 1 |
| Missed Vectorization | 1 |
| Recursive Without Memoization | 1 |

### Severity Breakdown

| Severity | Count |
|----------|-------|
| [CRITICAL] Critical | 1 |
| [HIGH] High | 1 |
| [MEDIUM] Medium | 38 |

### Priority Issues

#### 1. [CRITICAL] High Cyclomatic Complexity

- **File:** `/Users/Adam/Pycroscope/docs/examples/usage_example.py`
- **Function:** `main`
- **Line:** 55
- **Severity:** Critical
- **Description:** Function 'main' has high cyclomatic complexity: 28
- **Suggestion:** Consider breaking down this function. Target complexity: <= 10

#### 2. [HIGH] Nested Loops

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** `cpu_intensive_calculation`
- **Line:** 132
- **Severity:** High
- **Description:** Function 'cpu_intensive_calculation' has 3 levels of nested loops
- **Suggestion:** Consider extracting inner loops into separate functions or using more efficient algorithms

#### 3. [WARNING] Long Function

- **File:** `/Users/Adam/Pycroscope/docs/examples/usage_example.py`
- **Function:** `main`
- **Line:** 55
- **Severity:** Medium
- **Description:** Function 'main' is too long: 221 lines
- **Suggestion:** Consider breaking down this function. Target: <= 50 lines

#### 4. [WARNING] Long Function

- **File:** `/Users/Adam/Pycroscope/docs/examples/usage_example.py`
- **Function:** `run_profiling_session`
- **Line:** 97
- **Severity:** Medium
- **Description:** Function 'run_profiling_session' is too long: 179 lines
- **Suggestion:** Consider breaking down this function. Target: <= 50 lines

#### 5. [WARNING] High Cyclomatic Complexity

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** `overly_complex_function`
- **Line:** 59
- **Severity:** Medium
- **Description:** Function 'overly_complex_function' has high cyclomatic complexity: 11
- **Suggestion:** Consider breaking down this function. Target complexity: <= 10

#### 6. [WARNING] Too Many Parameters

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** `overly_complex_function`
- **Line:** 59
- **Severity:** Medium
- **Description:** Function 'overly_complex_function' has too many parameters: 7
- **Suggestion:** Consider using data classes or reducing parameters. Target: <= 5

#### 7. [WARNING] Long Function

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** `demonstrate_anti_patterns`
- **Line:** 300
- **Severity:** Medium
- **Description:** Function 'demonstrate_anti_patterns' is too long: 55 lines
- **Suggestion:** Consider breaking down this function. Target: <= 50 lines

#### 8. [WARNING] Long Function

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** `mixed_workload`
- **Line:** 355
- **Severity:** Medium
- **Description:** Function 'mixed_workload' is too long: 87 lines
- **Suggestion:** Consider breaking down this function. Target: <= 50 lines

#### 9. [WARNING] Inefficient Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 308
- **Severity:** Medium
- **Description:** np.array(range(n)) is inefficient, use np.arange(n) instead
- **Suggestion:** Replace with np.arange() for better performance

#### 10. [WARNING] Missed Vectorization

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 202
- **Severity:** Medium
- **Description:** Loop over array elements could be vectorized with NumPy operations
- **Suggestion:** Replace explicit loop with vectorized NumPy operations

#### 11. [WARNING] Inefficient Array Concatenation

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 50
- **Severity:** Medium
- **Description:** Array concatenation in loop is inefficient
- **Suggestion:** Pre-allocate array and use indexing, or collect items and concatenate once

#### 12. [WARNING] Inefficient Array Concatenation

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 51
- **Severity:** Medium
- **Description:** Array concatenation in loop is inefficient
- **Suggestion:** Pre-allocate array and use indexing, or collect items and concatenate once

#### 13. [WARNING] Inefficient Array Concatenation

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 116
- **Severity:** Medium
- **Description:** Array concatenation in loop is inefficient
- **Suggestion:** Pre-allocate array and use indexing, or collect items and concatenate once

#### 14. [WARNING] Inefficient Array Concatenation

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 160
- **Severity:** Medium
- **Description:** Array concatenation in loop is inefficient
- **Suggestion:** Pre-allocate array and use indexing, or collect items and concatenate once

#### 15. [WARNING] Inefficient Array Concatenation

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 175
- **Severity:** Medium
- **Description:** Array concatenation in loop is inefficient
- **Suggestion:** Pre-allocate array and use indexing, or collect items and concatenate once

#### 16. [WARNING] Inefficient Array Concatenation

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 183
- **Severity:** Medium
- **Description:** Array concatenation in loop is inefficient
- **Suggestion:** Pre-allocate array and use indexing, or collect items and concatenate once

#### 17. [WARNING] Inefficient Broadcasting

- **File:** `/Users/Adam/Pycroscope/docs/examples/usage_example.py`
- **Function:** ``
- **Line:** 197
- **Severity:** Medium
- **Description:** Manual broadcasting detected - NumPy can handle this automatically
- **Suggestion:** Use NumPy's automatic broadcasting instead of explicit loops

#### 18. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/usage_example.py`
- **Function:** ``
- **Line:** 131
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 19. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/usage_example.py`
- **Function:** ``
- **Line:** 150
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 20. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/usage_example.py`
- **Function:** ``
- **Line:** 188
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 21. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/usage_example.py`
- **Function:** ``
- **Line:** 197
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 22. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/usage_example.py`
- **Function:** ``
- **Line:** 217
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 23. [WARNING] Recursive Without Memoization

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** `fibonacci_recursive`
- **Line:** 25
- **Severity:** Medium
- **Description:** Recursive function 'fibonacci_recursive' without memoization
- **Suggestion:** Consider adding memoization with @lru_cache or manual caching

#### 24. [WARNING] Inefficient Broadcasting

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 37
- **Severity:** Medium
- **Description:** Manual broadcasting detected - NumPy can handle this automatically
- **Suggestion:** Use NumPy's automatic broadcasting instead of explicit loops

#### 25. [WARNING] Inefficient Broadcasting

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 175
- **Severity:** Medium
- **Description:** Manual broadcasting detected - NumPy can handle this automatically
- **Suggestion:** Use NumPy's automatic broadcasting instead of explicit loops

#### 26. [WARNING] Inefficient Broadcasting

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 183
- **Severity:** Medium
- **Description:** Manual broadcasting detected - NumPy can handle this automatically
- **Suggestion:** Use NumPy's automatic broadcasting instead of explicit loops

#### 27. [WARNING] Inefficient Broadcasting

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 202
- **Severity:** Medium
- **Description:** Manual broadcasting detected - NumPy can handle this automatically
- **Suggestion:** Use NumPy's automatic broadcasting instead of explicit loops

#### 28. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 37
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 29. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 50
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 30. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 99
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 31. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 116
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 32. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 142
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 33. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 148
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 34. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 160
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 35. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 175
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 36. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 183
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 37. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 202
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 38. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 217
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 39. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 223
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

#### 40. [WARNING] Scalar Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 230
- **Severity:** Medium
- **Description:** Scalar operations in loop could be vectorized
- **Suggestion:** Use vectorized operations instead of scalar operations in loops

### Recommendations

1. Consider optimizing algorithmic complexity in functions with nested loops or high time complexity
2. Break down complex functions to improve readability and maintainability

## Performance Insights

- Long execution time detected (34.137s) - consider optimization
- High function call count (3,010,177) - potential optimization opportunity

## Technical Details

### Session Metadata

- **Start Time:** 2025-07-30 22:08:43.514154
- **End Time:** 2025-07-30 22:09:17.651118
- **Output Directory:** `profiling_results`
- **Session Name:** Default

