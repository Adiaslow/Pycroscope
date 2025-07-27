# Pycroscope Profiling Analysis Report

**Generated:** 2025-07-27 14:22:32
**Session ID:** `dc83b434-159c-4a92-b801-fbae283f8321`

## Executive Summary

- **Duration:** 37.171 seconds
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
   373            2        392.0    196.0    25.5%  prefix, suffix, dir, output_type = _sanitize_param
   375            2         66.0     33.0     4.3%  names = _get_candidate_names()
   376            2          2.0      1.0     0.1%  if output_type is bytes:
   379            2          3.0      1.5     0.2%  for seq in range(TMP_MAX):
   380            2        249.0    124.5    16.2%  name = next(names)
   381            2         40.0     20.0     2.6%  file = _os.path.join(dir, prefix + name + suffix)
   382            2          3.0      1.5     0.2%  _sys.audit("tempfile.mkdtemp", file)
   383            2          2.0      1.0     0.1%  try:
   384            2        734.0    367.0    47.8%  _os.mkdir(file, 0o700)
   395            2         44.0     22.0     2.9%  return _os.path.abspath(file)
```

**Performance Insights:**
- **Line 384**: 47.8% of function time (2 hits)
- **Line 373**: 25.5% of function time (2 hits)
- **Line 380**: 16.2% of function time (2 hits)

#### _sanitize_params (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   116            2        208.0    104.0    64.0%  output_type = _infer_return_type(prefix, suffix, d
   117            2          3.0      1.5     0.9%  if suffix is None:
   118            2          2.0      1.0     0.6%  suffix = output_type()
   119            2          1.0      0.5     0.3%  if prefix is None:
   120            1          1.0      1.0     0.3%  if output_type is str:
   121            1          2.0      2.0     0.6%  prefix = template
   124            2          1.0      0.5     0.3%  if dir is None:
   125            2          2.0      1.0     0.6%  if output_type is str:
   126            2        102.0     51.0    31.4%  dir = gettempdir()
   129            2          3.0      1.5     0.9%  return prefix, suffix, dir, output_type
```

**Performance Insights:**
- **Line 116**: 64.0% of function time (2 hits)
- **Line 126**: 31.4% of function time (2 hits)
- **Line 117**: 0.9% of function time (2 hits)

#### _infer_return_type (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    87            2          3.0      1.5     8.6%  return_type = None
    88            8          8.0      1.0    22.9%  for arg in args:
    89            6          5.0      0.8    14.3%  if arg is None:
    90            5          3.0      0.6     8.6%  continue
    92            1          4.0      4.0    11.4%  if isinstance(arg, _os.PathLike):
    95            1          1.0      1.0     2.9%  if isinstance(arg, bytes):
   101            1          1.0      1.0     2.9%  if return_type is bytes:
   104            1          1.0      1.0     2.9%  return_type = str
   105            2          2.0      1.0     5.7%  if return_type is None:
   106            1          3.0      3.0     8.6%  if tempdir is None or isinstance(tempdir, str):
   107            1          3.0      3.0     8.6%  return str  # tempfile APIs return a str by defaul
   111            1          1.0      1.0     2.9%  return return_type
```

**Performance Insights:**
- **Line 88**: 22.9% of function time (8 hits)
- **Line 89**: 14.3% of function time (6 hits)
- **Line 92**: 11.4% of function time (1 hits)

#### gettempdir (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   315            2         73.0     36.5   100.0%  return _os.fsdecode(_gettempdir())
```

**Performance Insights:**
- **Line 315**: 100.0% of function time (2 hits)

#### _gettempdir (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   304            2          1.0      0.5    33.3%  if tempdir is None:
   311            2          2.0      1.0    66.7%  return tempdir
```

**Performance Insights:**
- **Line 311**: 66.7% of function time (2 hits)
- **Line 304**: 33.3% of function time (2 hits)

#### _get_candidate_names (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   233            1          1.0      1.0    50.0%  if _name_sequence is None:
   240            1          1.0      1.0    50.0%  return _name_sequence
```

**Performance Insights:**
- **Line 233**: 50.0% of function time (1 hits)
- **Line 240**: 50.0% of function time (1 hits)

#### RLock (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/threading.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   133            1          1.0      1.0    50.0%  if _CRLock is None:
   135            1          1.0      1.0    50.0%  return _CRLock(*args, **kwargs)
```

**Performance Insights:**
- **Line 133**: 50.0% of function time (1 hits)
- **Line 135**: 50.0% of function time (1 hits)

#### _newname (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/threading.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   840            1          3.0      3.0   100.0%  return name_template % _counter()
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
    43            1       5124.0   5124.0     0.0%  from sample_workload import mixed_workload
    45            1          4.0      4.0     0.0%  print("Executing workload to be profiled...")
    46            1          2.0      2.0     0.0%  print("   (Replace this section with your own code
    47            1          2.0      2.0     0.0%  print()
    50            1   37005499.0 37005499.0   100.0%  results = mixed_workload()
    52            1          3.0      3.0     0.0%  return results
```

**Performance Insights:**
- **Line 50**: 100.0% of function time (1 hits)
- **Line 43**: 0.0% of function time (1 hits)
- **Line 45**: 0.0% of function time (1 hits)

#### _type_check (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/typing.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   187            3          4.0      1.3     4.4%  invalid_generic_forms = (Generic, Protocol)
   188            3          3.0      1.0     3.3%  if not allow_special_forms:
   189            3          3.0      1.0     3.3%  invalid_generic_forms += (ClassVar,)
   190            3          3.0      1.0     3.3%  if is_argument:
   191            3          3.0      1.0     3.3%  invalid_generic_forms += (Final,)
   193            3         53.0     17.7    58.9%  arg = _type_convert(arg, module=module, allow_spec
   194            3          4.0      1.3     4.4%  if (isinstance(arg, _GenericAlias) and
   197            3          3.0      1.0     3.3%  if arg in (Any, LiteralString, NoReturn, Never, Se
   199            3          3.0      1.0     3.3%  if allow_special_forms and arg in (ClassVar, Final
   201            3          4.0      1.3     4.4%  if isinstance(arg, _SpecialForm) or arg in (Generi
   203            3          2.0      0.7     2.2%  if type(arg) is tuple:
   205            3          5.0      1.7     5.6%  return arg
```

**Performance Insights:**
- **Line 193**: 58.9% of function time (3 hits)
- **Line 205**: 5.6% of function time (3 hits)
- **Line 187**: 4.4% of function time (3 hits)

#### _type_convert (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/typing.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   168            3          2.0      0.7    22.2%  if arg is None:
   170            3          3.0      1.0    33.3%  if isinstance(arg, str):
   172            3          4.0      1.3    44.4%  return arg
```

**Performance Insights:**
- **Line 172**: 44.4% of function time (3 hits)
- **Line 170**: 33.3% of function time (3 hits)
- **Line 168**: 22.2% of function time (3 hits)

#### _is_dunder (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/typing.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
  1162           12         19.0      1.6   100.0%  return attr.startswith('__') and attr.endswith('__
```

**Performance Insights:**
- **Line 1162**: 100.0% of function time (12 hits)

#### mixed_workload (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   357            1          4.0      4.0     0.0%  print("Running mixed workload demonstration...")
   359            1          0.0      0.0     0.0%  results = {}
   362            1          1.0      1.0     0.0%  start_time = time.time()
   363            1   15473649.0 15473649.0    41.8%  cpu_result = cpu_intensive_calculation()
   364            1          2.0      2.0     0.0%  results["cpu_time"] = time.time() - start_time
   365            1          1.0      1.0     0.0%  results["cpu_result_sample"] = cpu_result
   368            1          1.0      1.0     0.0%  start_time = time.time()
   369            1    1523823.0 1523823.0     4.1%  memory_data = memory_intensive_operations()
   370            1          2.0      2.0     0.0%  results["memory_time"] = time.time() - start_time
   371            1          1.0      1.0     0.0%  results["memory_objects_created"] = len(memory_dat
   374            1          1.0      1.0     0.0%  start_time = time.time()
   375            1     356486.0 356486.0     1.0%  io_result = file_io_operations()
   376            1          2.0      2.0     0.0%  results["io_time"] = time.time() - start_time
   377            1          1.0      1.0     0.0%  results["io_operations"] = io_result
   380            1          0.0      0.0     0.0%  start_time = time.time()
   381            1        148.0    148.0     0.0%  call_result = nested_function_calls()
   382            1          1.0      1.0     0.0%  results["call_time"] = time.time() - start_time
   383            1          1.0      1.0     0.0%  results["call_result"] = call_result
   386            1          1.0      1.0     0.0%  start_time = time.time()
   387            1      66015.0  66015.0     0.2%  processing_result = data_processing_pipeline()
   388            1          1.0      1.0     0.0%  results["processing_time"] = time.time() - start_t
   389            1          1.0      1.0     0.0%  results["processing_stats"] = processing_result
   392            1          1.0      1.0     0.0%  start_time = time.time()
   393            1   17979204.0 17979204.0    48.6%  antipattern_result = demonstrate_anti_patterns()
   394            1          3.0      3.0     0.0%  results["antipattern_time"] = time.time() - start_
   395            1          1.0      1.0     0.0%  results["antipattern_stats"] = antipattern_result
   398            1          1.0      1.0     0.0%  start_time = time.time()
   399            1    1594137.0 1594137.0     4.3%  fib_recursive = fibonacci_recursive(25)  # Small e
   400            1          3.0      3.0     0.0%  results["fib_recursive_time"] = time.time() - star
   402            1          1.0      1.0     0.0%  start_time = time.time()
   403            1         78.0     78.0     0.0%  fib_iterative = fibonacci_iterative(25)
   404            1          1.0      1.0     0.0%  results["fib_iterative_time"] = time.time() - star
   406            1          2.0      2.0     0.0%  results["fib_results_match"] = fib_recursive == fi
   408            1          2.0      2.0     0.0%  return results
```

**Performance Insights:**
- **Line 393**: 48.6% of function time (1 hits)
- **Line 363**: 41.8% of function time (1 hits)
- **Line 399**: 4.3% of function time (1 hits)

#### cpu_intensive_calculation (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   134            1          2.0      2.0     0.0%  print("Running CPU-intensive calculations...")
   137            1          0.0      0.0     0.0%  size = 200
   138            1      22098.0  22098.0     0.2%  matrix_a = [[random.random() for _ in range(size)]
   139            1      22947.0  22947.0     0.2%  matrix_b = [[random.random() for _ in range(size)]
   141            1          1.0      1.0     0.0%  result = 0.0
   142          201        183.0      0.9     0.0%  for i in range(size):
   143        40200      32958.0      0.8     0.2%  for j in range(size):
   144      8040000    6598693.0      0.8    49.5%  for k in range(size):
   145      8000000    6640867.0      0.8    49.8%  result += matrix_a[i][k] * matrix_b[k][j]
   148        10001       8403.0      0.8     0.1%  for i in range(10000):
   149        10000       9204.0      0.9     0.1%  result += math.sin(i) * math.cos(i) * math.sqrt(i 
   151            1          9.0      9.0     0.0%  return result
```

**Performance Insights:**
- **Line 145**: 49.8% of function time (8,000,000 hits)
- **Line 144**: 49.5% of function time (8,040,000 hits)
- **Line 143**: 0.2% of function time (40,200 hits)

#### memory_intensive_operations (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   156            1          5.0      5.0     0.0%  print("Running memory-intensive operations...")
   159            1          1.0      1.0     0.0%  large_list = []
   160        50001      41051.0      0.8     2.9%  for i in range(50000):
   161       100000      82144.0      0.8     5.7%  large_list.append(
   162        50000      48198.0      1.0     3.4%  {
   163        50000      38684.0      0.8     2.7%  "id": i,
   164        50000     678643.0     13.6    47.4%  "data": [random.random() for _ in range(20)],
   165        50000      68051.0      1.4     4.8%  "metadata": {
   166        50000      41811.0      0.8     2.9%  "timestamp": time.time(),
   167        50000      44608.0      0.9     3.1%  "category": f"category_{i % 10}",
   168        50000     156713.0      3.1    11.0%  "tags": [f"tag_{j}" for j in range(i % 5 + 1)],
   174            1          1.0      1.0     0.0%  lookup_dict = {}
   175        50001      42287.0      0.8     3.0%  for item in large_list:
   176        50000      40728.0      0.8     2.8%  category = item["metadata"]["category"]
   177        50000      42213.0      0.8     3.0%  if category not in lookup_dict:
   178           10          9.0      0.9     0.0%  lookup_dict[category] = []
   179        50000      42265.0      0.8     3.0%  lookup_dict[category].append(item)
   182            1          1.0      1.0     0.0%  temp_lists = []
   183          101        100.0      1.0     0.0%  for i in range(100):
   184          100      62197.0    622.0     4.3%  temp_list = [random.random() for _ in range(1000)]
   185          100         88.0      0.9     0.0%  temp_lists.append(temp_list)
   186          100        100.0      1.0     0.0%  if len(temp_lists) > 10:
   187           90        463.0      5.1     0.0%  temp_lists.pop(0)  # Remove oldest
   189            1          6.0      6.0     0.0%  return large_list
```

**Performance Insights:**
- **Line 164**: 47.4% of function time (50,000 hits)
- **Line 168**: 11.0% of function time (50,000 hits)
- **Line 161**: 5.7% of function time (100,000 hits)

#### file_io_operations (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   194            1          5.0      5.0     0.0%  print("Running file I/O operations...")
   197            1        764.0    764.0     0.2%  temp_dir = tempfile.mkdtemp()
   199            1          1.0      1.0     0.0%  try:
   201            1          0.0      0.0     0.0%  file_data = {}
   202           11         14.0      1.3     0.0%  for i in range(10):
   203           10        240.0     24.0     0.1%  filename = os.path.join(temp_dir, f"test_file_{i}.
   204           10         11.0      1.1     0.0%  data = {
   205           10         10.0      1.0     0.0%  "file_id": i,
   206           10       5605.0    560.5     1.6%  "content": [random.random() for _ in range(1000)],
   207           10         12.0      1.2     0.0%  "metadata": {"created": time.time()},
   210           20       1985.0     99.2     0.6%  with open(filename, "w") as f:
   211           10     343819.0  34381.9    96.6%  json.dump(data, f)
   213           10         12.0      1.2     0.0%  file_data[filename] = data
   216            1          1.0      1.0     0.0%  read_data = {}
   217           11         12.0      1.1     0.0%  for filename in file_data:
   218           20        271.0     13.6     0.1%  with open(filename, "r") as f:
   219           10       2486.0    248.6     0.7%  read_data[filename] = json.load(f)
   222            1          1.0      1.0     0.0%  total_values = 0
   223           11         10.0      0.9     0.0%  for data in read_data.values():
   224           10          8.0      0.8     0.0%  total_values += len(data["content"])
   226            1          4.0      4.0     0.0%  return {"files_processed": len(read_data), "total_
   230           11         39.0      3.5     0.0%  for filename in os.listdir(temp_dir):
   231           10        574.0     57.4     0.2%  os.remove(os.path.join(temp_dir, filename))
   232            1         52.0     52.0     0.0%  os.rmdir(temp_dir)
```

**Performance Insights:**
- **Line 211**: 96.6% of function time (10 hits)
- **Line 206**: 1.6% of function time (10 hits)
- **Line 219**: 0.7% of function time (10 hits)

#### dump (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   165           10         11.0      1.1     0.0%  if (not skipkeys and ensure_ascii and
   166           10         10.0      1.0     0.0%  check_circular and allow_nan and
   167           10         10.0      1.0     0.0%  cls is None and indent is None and separators is N
   168           10         10.0      1.0     0.0%  default is None and not sort_keys and not kw):
   169           10        407.0     40.7     0.1%  iterable = _default_encoder.iterencode(obj)
   179        10180     328897.0     32.3    96.9%  for chunk in iterable:
   180        10170      10238.0      1.0     3.0%  fp.write(chunk)
```

**Performance Insights:**
- **Line 179**: 96.9% of function time (10,180 hits)
- **Line 180**: 3.0% of function time (10,170 hits)
- **Line 169**: 0.1% of function time (10 hits)

#### _make_iterencode (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   275           10         10.0      1.0    15.9%  if _indent is not None and not isinstance(_indent,
   278           10         12.0      1.2    19.0%  def _iterencode_list(lst, _current_indent_level):
   334           10         12.0      1.2    19.0%  def _iterencode_dict(dct, _current_indent_level):
   414           10         13.0      1.3    20.6%  def _iterencode(o, _current_indent_level):
   443           10         16.0      1.6    25.4%  return _iterencode
```

**Performance Insights:**
- **Line 443**: 25.4% of function time (10 hits)
- **Line 414**: 20.6% of function time (10 hits)
- **Line 278**: 19.0% of function time (10 hits)

#### load (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   293           30       2338.0     77.9    98.8%  return loads(fp.read(),
   294           10          8.0      0.8     0.3%  cls=cls, object_hook=object_hook,
   295           10          7.0      0.7     0.3%  parse_float=parse_float, parse_int=parse_int,
   296           20         14.0      0.7     0.6%  parse_constant=parse_constant, object_pairs_hook=o
```

**Performance Insights:**
- **Line 293**: 98.8% of function time (30 hits)
- **Line 296**: 0.6% of function time (20 hits)
- **Line 294**: 0.3% of function time (10 hits)

#### loads (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   333           10         11.0      1.1     0.6%  if isinstance(s, str):
   334           10         11.0      1.1     0.6%  if s.startswith('\ufeff'):
   343           10          9.0      0.9     0.5%  if (cls is None and object_hook is None and
   344           10          8.0      0.8     0.4%  parse_int is None and parse_float is None and
   345           10          8.0      0.8     0.4%  parse_constant is None and object_pairs_hook is No
   346           10       1950.0    195.0    97.6%  return _default_decoder.decode(s)
```

**Performance Insights:**
- **Line 346**: 97.6% of function time (10 hits)
- **Line 333**: 0.6% of function time (10 hits)
- **Line 334**: 0.6% of function time (10 hits)

#### nested_function_calls (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   237            1          3.0      3.0     3.5%  print("Running nested function calls...")
   239            1          2.0      2.0     2.3%  def level_1(n: int) -> int:
   244            1          1.0      1.0     1.2%  def level_2(n: int) -> int:
   249            1          1.0      1.0     1.2%  def level_3(n: int) -> int:
   254            1          1.0      1.0     1.2%  def level_4(n: int) -> int:
   259            1          1.0      1.0     1.2%  def level_5(n: int) -> int:
   262            1         77.0     77.0    89.5%  return level_1(8)
```

**Performance Insights:**
- **Line 262**: 89.5% of function time (1 hits)
- **Line 237**: 3.5% of function time (1 hits)
- **Line 239**: 2.3% of function time (1 hits)

#### data_processing_pipeline (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   267            1          3.0      3.0     0.0%  print("Running data processing pipeline...")
   270            1       5935.0   5935.0     9.0%  raw_data = [random.random() * 1000 for _ in range(
   273            1       5562.0   5562.0     8.5%  filtered_data = [x for x in raw_data if x > 100]
   276            1       5396.0   5396.0     8.2%  transformed_data = [math.log(x) if x > 1 else 0 fo
   279            1         24.0     24.0     0.0%  sum_data = sum(transformed_data)
   280            1          2.0      2.0     0.0%  avg_data = sum_data / len(transformed_data) if tra
   281            1         61.0     61.0     0.1%  max_data = max(transformed_data) if transformed_da
   282            1         58.0     58.0     0.1%  min_data = min(transformed_data) if transformed_da
   285            2      48737.0  24368.5    74.1%  variance = sum((x - avg_data) ** 2 for x in transf
   286            1          1.0      1.0     0.0%  transformed_data
   288            1          2.0      2.0     0.0%  std_dev = math.sqrt(variance)
   290            1          3.0      3.0     0.0%  return {
   291            1          1.0      1.0     0.0%  "count": len(transformed_data),
   292            1          1.0      1.0     0.0%  "sum": sum_data,
   293            1          1.0      1.0     0.0%  "average": avg_data,
   294            1          1.0      1.0     0.0%  "maximum": max_data,
   295            1          0.0      0.0     0.0%  "minimum": min_data,
   296            1          1.0      1.0     0.0%  "std_dev": std_dev,
```

**Performance Insights:**
- **Line 285**: 74.1% of function time (2 hits)
- **Line 270**: 9.0% of function time (1 hits)
- **Line 273**: 8.5% of function time (1 hits)

#### demonstrate_anti_patterns (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   302            1          4.0      4.0     0.0%  print("Running anti-pattern demonstrations...")
   305            1          4.0      4.0     0.0%  import numpy as np
   308            1         65.0     65.0     0.0%  bad_array = np.array(range(1000))  # Should use np
   311            1       1124.0   1124.0     0.0%  test_array = np.random.random(1000)
   312            1        278.0    278.0     0.0%  count_result = (test_array > 0.5).sum()  # Should 
   315            1         21.0     21.0     0.0%  matrix_a = np.random.random((100, 50))
   316            1         15.0     15.0     0.0%  matrix_b = np.random.random((50, 100))
   317            1        183.0    183.0     0.0%  result_tensordot = np.tensordot(matrix_a, matrix_b
   320            1          6.0      6.0     0.0%  original_array = np.random.random(1000)
   321            1          3.0      3.0     0.0%  copied_array = original_array.copy()  # Might be u
   324            1          8.0      8.0     0.0%  data = list(range(1000))
   327            1       4030.0   4030.0     0.0%  result = inefficient_nested_search(data, [500, 750
   330            1        399.0    399.0     0.0%  complex_result = overly_complex_function(10, 20, 3
   333            1         60.0     60.0     0.0%  string_data = [f"item_{i}" for i in range(100)]
   334            1        433.0    433.0     0.0%  processed_data = inefficient_data_operations(strin
   337            1   17972176.0 17972176.0   100.0%  fib_recursive_result = fibonacci_recursive(30)
   338            1        137.0    137.0     0.0%  fib_iterative_result = fibonacci_iterative(30)
   340            1          4.0      4.0     0.0%  return {
   341            1          1.0      1.0     0.0%  "nested_search": result,
   342            1          1.0      1.0     0.0%  "complex_function": complex_result,
   343            1          2.0      2.0     0.0%  "data_operations": len(processed_data),
   344            1          1.0      1.0     0.0%  "fibonacci_recursive": fib_recursive_result,
   345            1          1.0      1.0     0.0%  "fibonacci_iterative": fib_iterative_result,
   346            1          1.0      1.0     0.0%  "fibonacci_match": fib_recursive_result == fib_ite
   348            1          2.0      2.0     0.0%  "bad_array_size": len(bad_array),
   349            1          1.0      1.0     0.0%  "count_result": count_result,
   350            1          1.0      1.0     0.0%  "tensordot_shape": result_tensordot.shape,
   351            1          1.0      1.0     0.0%  "copied_array_size": len(copied_array),
```

**Performance Insights:**
- **Line 337**: 100.0% of function time (1 hits)
- **Line 327**: 0.0% of function time (1 hits)
- **Line 311**: 0.0% of function time (1 hits)

#### inefficient_nested_search (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    47            1          1.0      1.0     0.0%  found_items = []
    50            4          4.0      1.0     0.1%  for target in targets:
    51         2252       1750.0      0.8    51.2%  for item in data_list:
    52         2252       1650.0      0.7    48.3%  if item == target:
    53            3          3.0      1.0     0.1%  found_items.append(item)
    54            3          3.0      1.0     0.1%  break  # At least we break early
    56            1          4.0      4.0     0.1%  return found_items
```

**Performance Insights:**
- **Line 51**: 51.2% of function time (2,252 hits)
- **Line 52**: 48.3% of function time (2,252 hits)
- **Line 50**: 0.1% of function time (4 hits)

#### overly_complex_function (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    64            1          3.0      3.0     1.4%  result = 0
    67            1          1.0      1.0     0.5%  if param1 > 0:
    68            1          1.0      1.0     0.5%  if param2 > 0:
    69            1          1.0      1.0     0.5%  if param3 > 0:
    70            1          1.0      1.0     0.5%  if param4 > 0:
    71            1          1.0      1.0     0.5%  if param5 > 0:
    72            1          1.0      1.0     0.5%  if param6 > 0:
    73            1          1.0      1.0     0.5%  if param7 > 0:
    74            1          1.0      1.0     0.5%  result = (
    75            7          7.0      1.0     3.2%  param1
    76            1          1.0      1.0     0.5%  + param2
    77            1          1.0      1.0     0.5%  + param3
    78            1          0.0      0.0     0.0%  + param4
    79            1          1.0      1.0     0.5%  + param5
    80            1          1.0      1.0     0.5%  + param6
    81            1          0.0      0.0     0.0%  + param7
    99           11         10.0      0.9     4.6%  for i in range(10):
   100          110         93.0      0.8    42.9%  for j in range(10):
   101          100         82.0      0.8    37.8%  if i + j == result % 10:
   102           10          8.0      0.8     3.7%  result += 1
   104            1          2.0      2.0     0.9%  return result
```

**Performance Insights:**
- **Line 100**: 42.9% of function time (110 hits)
- **Line 101**: 37.8% of function time (100 hits)
- **Line 99**: 4.6% of function time (11 hits)

#### inefficient_data_operations (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   113            1          0.0      0.0     0.0%  processed_items = []
   114            1          1.0      1.0     0.3%  unique_items = []
   116          101         75.0      0.7    21.7%  for item in items:
   117          100        121.0      1.2    35.1%  if item not in processed_items:  # O(n) operation 
   118          100         75.0      0.8    21.7%  processed_items.append(item)
   119          100         71.0      0.7    20.6%  unique_items.append(f"processed_{item}")
   121            1          2.0      2.0     0.6%  return unique_items
```

**Performance Insights:**
- **Line 117**: 35.1% of function time (100 hits)
- **Line 116**: 21.7% of function time (101 hits)
- **Line 118**: 21.7% of function time (100 hits)

#### fibonacci_recursive (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    27      2935322    2374387.0      0.8    22.6%  if n <= 1:
    28      1467662    1782469.0      1.2    17.0%  return n
    29      1467660    6344536.0      4.3    60.4%  return fibonacci_recursive(n - 1) + fibonacci_recu
```

**Performance Insights:**
- **Line 29**: 60.4% of function time (1,467,660 hits)
- **Line 27**: 22.6% of function time (2,935,322 hits)
- **Line 28**: 17.0% of function time (1,467,662 hits)

#### fibonacci_iterative (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    34            2          2.0      1.0     1.8%  if n <= 1:
    36            2          4.0      2.0     3.5%  a, b = 0, 1
    37           55         52.0      0.9    46.0%  for _ in range(2, n + 1):
    38           53         51.0      1.0    45.1%  a, b = b, a + b
    39            2          4.0      2.0     3.5%  return b
```

**Performance Insights:**
- **Line 37**: 46.0% of function time (55 hits)
- **Line 38**: 45.1% of function time (53 hits)
- **Line 36**: 3.5% of function time (2 hits)

### Line Profiling Summary

- **Total Lines Profiled:** 284
- **Total Hits:** 22,697,860
- **Total Time:** 118.022351 seconds
- **Average Time per Hit:** 0.000005200 seconds
## Memory Profiler Analysis

### Memory Usage Statistics

| Metric | Value |
|--------|-------|
| Peak Memory Usage | 190.06 MB |
| Average Memory Usage | 163.01 MB |
| Memory Delta | +71.66 MB |
| Sample Count | 2524 |
| Initial Memory | 107.58 MB |
| Final Memory | 179.23 MB |

### Memory Timeline Analysis

- **Memory Growth Rate:** 1.9280 MB/second
- **Memory Spikes Detected:** 0 (>244.52 MB)

## Call Profiler Analysis

### 🕒 Top Functions by Execution Time

| Function | Total Time | Calls | Time/Call | Cumulative |
|----------|------------|-------|-----------|------------|
| `/Users/Adam/Pycroscope/docs/examples/usage_example.py:0(run_profiling_session)` | 37.0106s | 1 | 37.010590s | 37.0106s |
| `/Users/Adam/Pycroscope/docs/examples/usage_example.py:0(run_sample_workload)` | 37.0105s | 1 | 37.010520s | 37.0105s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(mixed_workload)` | 36.9935s | 1 | 36.993481s | 36.9935s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(demonstrate_anti_patterns)` | 17.9789s | 1 | 17.978924s | 17.9789s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(cpu_intensive_calculation)` | 15.4729s | 1 | 15.472950s | 15.4729s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(fibonacci_recursive)` | 4.3153s | 2935322 | 0.000001s | 4.3153s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(memory_intensive_operations)` | 1.5231s | 1 | 1.523137s | 1.5231s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(file_io_operations)` | 0.3560s | 1 | 0.355971s | 0.3560s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py:0(dump)` | 0.3437s | 10 | 0.034367s | 0.3437s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode)` | 0.2779s | 10180 | 0.000027s | 0.2779s |

### 📞 Most Called Functions

| Function | Calls | Total Time | Avg Time |
|----------|-------|------------|----------|
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(fibonacci_recursive)` | 2935322 | 4.3153s | 0.000001s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode_dict)` | 10240 | 0.2236s | 0.000022s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode)` | 10180 | 0.2779s | 0.000027s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode_list)` | 10020 | 0.1727s | 0.000017s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(floatstr)` | 10010 | 0.0455s | 0.000005s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(<genexpr>)` | 8993 | 0.0161s | 0.000002s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tokenize.py:0(_generate_tokens_from_c_tokenizer)` | 5639 | 0.0294s | 0.000005s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/line_profiler.py:0(tokeneater)` | 5607 | 0.0360s | 0.000006s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/inspect.py:0(tokeneater)` | 5607 | 0.0204s | 0.000004s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/collections/__init__.py:0(_make)` | 5607 | 0.0129s | 0.000002s |

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

### Recommendations

1. Consider optimizing algorithmic complexity in functions with nested loops or high time complexity
2. Break down complex functions to improve readability and maintainability

## Performance Insights

- Long execution time detected (37.171s) - consider optimization
- High function call count (3,009,874) - potential optimization opportunity

## Technical Details

### Session Metadata

- **Start Time:** 2025-07-27 14:21:55.006558
- **End Time:** 2025-07-27 14:22:32.177952
- **Output Directory:** `profiling_results`
- **Session Name:** Default

