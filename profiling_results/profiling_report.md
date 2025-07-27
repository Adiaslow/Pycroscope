# 🔍 Pycroscope Profiling Analysis Report

**Generated:** 2025-07-27 12:54:08
**Session ID:** `016e4b76-94ca-4443-a85c-e404f859661f`

## 📊 Executive Summary

- **Duration:** 40.160 seconds
- **Status:** completed
- **Profilers Used:** line, memory, call
- **Total Results:** 3
- **Patterns Detected:** 38

## ⚙️ Configuration

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
| Pattern Complexity Threshold | `10` |
| Pattern Maintainability Threshold | `20.0` |
| Pattern Severity Threshold | `medium` |
| Pattern Confidence Threshold | `0.7` |
| Detect Nested Loops | `True` |
| Detect Dead Code | `True` |
| Detect Complexity Issues | `True` |
| Detect Maintainability Issues | `True` |
| Max Function Lines | `50` |
| Max Function Parameters | `5` |
| Correlate Patterns With Profiling | `True` |
| Prioritize Hotspot Patterns | `True` |
| Hotspot Correlation Threshold | `0.1` |
| Profiler Prefix | `pycroscope` |
| Use Thread Isolation | `True` |
| Cleanup On Exit | `True` |

## 📈 Line Profiler Analysis

## 📝 Line Profiler Analysis

### 🎯 Per-Function Line-by-Line Analysis

**Functions Profiled:** 30

#### mkdtemp (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   373            2        407.0    203.5    29.0%  prefix, suffix, dir, output_type = _sanitize_param
   375            2         68.0     34.0     4.8%  names = _get_candidate_names()
   376            2          2.0      1.0     0.1%  if output_type is bytes:
   379            2          3.0      1.5     0.2%  for seq in range(TMP_MAX):
   380            2        637.0    318.5    45.3%  name = next(names)
   381            2         46.0     23.0     3.3%  file = _os.path.join(dir, prefix + name + suffix)
   382            2          2.0      1.0     0.1%  _sys.audit("tempfile.mkdtemp", file)
   383            2          2.0      1.0     0.1%  try:
   384            2        204.0    102.0    14.5%  _os.mkdir(file, 0o700)
   395            2         34.0     17.0     2.4%  return _os.path.abspath(file)
```

**Performance Insights:**
- **Line 380**: 45.3% of function time (2 hits)
- **Line 373**: 29.0% of function time (2 hits)
- **Line 384**: 14.5% of function time (2 hits)

#### _sanitize_params (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   116            2        144.0     72.0    54.1%  output_type = _infer_return_type(prefix, suffix, d
   117            2          2.0      1.0     0.8%  if suffix is None:
   118            2          2.0      1.0     0.8%  suffix = output_type()
   119            2          2.0      1.0     0.8%  if prefix is None:
   120            1          1.0      1.0     0.4%  if output_type is str:
   121            1          1.0      1.0     0.4%  prefix = template
   124            2          2.0      1.0     0.8%  if dir is None:
   125            2          2.0      1.0     0.8%  if output_type is str:
   126            2        109.0     54.5    41.0%  dir = gettempdir()
   129            2          1.0      0.5     0.4%  return prefix, suffix, dir, output_type
```

**Performance Insights:**
- **Line 116**: 54.1% of function time (2 hits)
- **Line 126**: 41.0% of function time (2 hits)
- **Line 117**: 0.8% of function time (2 hits)

#### _infer_return_type (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    87            2          3.0      1.5     7.9%  return_type = None
    88            8          9.0      1.1    23.7%  for arg in args:
    89            6          6.0      1.0    15.8%  if arg is None:
    90            5          5.0      1.0    13.2%  continue
    92            1          6.0      6.0    15.8%  if isinstance(arg, _os.PathLike):
    95            1          1.0      1.0     2.6%  if isinstance(arg, bytes):
   101            1          1.0      1.0     2.6%  if return_type is bytes:
   104            1          1.0      1.0     2.6%  return_type = str
   105            2          2.0      1.0     5.3%  if return_type is None:
   106            1          1.0      1.0     2.6%  if tempdir is None or isinstance(tempdir, str):
   107            1          2.0      2.0     5.3%  return str  # tempfile APIs return a str by defaul
   111            1          1.0      1.0     2.6%  return return_type
```

**Performance Insights:**
- **Line 88**: 23.7% of function time (8 hits)
- **Line 89**: 15.8% of function time (6 hits)
- **Line 92**: 15.8% of function time (1 hits)

#### gettempdir (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   315            2         71.0     35.5   100.0%  return _os.fsdecode(_gettempdir())
```

**Performance Insights:**
- **Line 315**: 100.0% of function time (2 hits)

#### _gettempdir (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   304            2          2.0      1.0    33.3%  if tempdir is None:
   311            2          4.0      2.0    66.7%  return tempdir
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
  1361            1          1.0      1.0    33.3%  old_excepthook = excepthook
  1362            1          1.0      1.0    33.3%  old_sys_excepthook = _sys.excepthook
  1363            1          0.0      0.0     0.0%  if old_excepthook is None:
  1365            1          0.0      0.0     0.0%  if old_sys_excepthook is None:
  1368            1          1.0      1.0    33.3%  sys_exc_info = _sys.exc_info
  1369            1          0.0      0.0     0.0%  local_print = print
  1370            1          0.0      0.0     0.0%  local_sys = _sys
  1372            1          0.0      0.0     0.0%  def invoke_excepthook(thread):
  1404            1          0.0      0.0     0.0%  return invoke_excepthook
```

**Performance Insights:**
- **Line 1361**: 33.3% of function time (1 hits)
- **Line 1362**: 33.3% of function time (1 hits)
- **Line 1368**: 33.3% of function time (1 hits)

#### register_trace_function (/Users/Adam/Pycroscope/src/pycroscope/infrastructure/profilers/trace_multiplexer.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   124            1         58.0     58.0   100.0%  _trace_multiplexer.register_profiler(profiler_name
```

**Performance Insights:**
- **Line 124**: 100.0% of function time (1 hits)

#### run_sample_workload (/Users/Adam/Pycroscope/docs/examples/usage_example.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    43            1       2307.0   2307.0     0.0%  from sample_workload import mixed_workload
    45            1          9.0      9.0     0.0%  print("🎯 Executing workload to be profiled...")
    46            1          3.0      3.0     0.0%  print("   (Replace this section with your own code
    47            1          2.0      2.0     0.0%  print()
    50            1   39994301.0 39994301.0   100.0%  results = mixed_workload()
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
   187            3          4.0      1.3     4.2%  invalid_generic_forms = (Generic, Protocol)
   188            3          3.0      1.0     3.1%  if not allow_special_forms:
   189            3          4.0      1.3     4.2%  invalid_generic_forms += (ClassVar,)
   190            3          3.0      1.0     3.1%  if is_argument:
   191            3          4.0      1.3     4.2%  invalid_generic_forms += (Final,)
   193            3         57.0     19.0    59.4%  arg = _type_convert(arg, module=module, allow_spec
   194            3          3.0      1.0     3.1%  if (isinstance(arg, _GenericAlias) and
   197            3          3.0      1.0     3.1%  if arg in (Any, LiteralString, NoReturn, Never, Se
   199            3          3.0      1.0     3.1%  if allow_special_forms and arg in (ClassVar, Final
   201            3          4.0      1.3     4.2%  if isinstance(arg, _SpecialForm) or arg in (Generi
   203            3          3.0      1.0     3.1%  if type(arg) is tuple:
   205            3          5.0      1.7     5.2%  return arg
```

**Performance Insights:**
- **Line 193**: 59.4% of function time (3 hits)
- **Line 205**: 5.2% of function time (3 hits)
- **Line 187**: 4.2% of function time (3 hits)

#### _type_convert (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/typing.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   168            3          4.0      1.3    30.8%  if arg is None:
   170            3          3.0      1.0    23.1%  if isinstance(arg, str):
   172            3          6.0      2.0    46.2%  return arg
```

**Performance Insights:**
- **Line 172**: 46.2% of function time (3 hits)
- **Line 168**: 30.8% of function time (3 hits)
- **Line 170**: 23.1% of function time (3 hits)

#### _is_dunder (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/typing.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
  1162           12         30.0      2.5   100.0%  return attr.startswith('__') and attr.endswith('__
```

**Performance Insights:**
- **Line 1162**: 100.0% of function time (12 hits)

#### mixed_workload (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   357            1          3.0      3.0     0.0%  print("🚀 Running mixed workload demonstration...")
   359            1          1.0      1.0     0.0%  results = {}
   362            1          2.0      2.0     0.0%  start_time = time.time()
   363            1   17157808.0 17157808.0    42.9%  cpu_result = cpu_intensive_calculation()
   364            1          2.0      2.0     0.0%  results["cpu_time"] = time.time() - start_time
   365            1          1.0      1.0     0.0%  results["cpu_result_sample"] = cpu_result
   368            1          1.0      1.0     0.0%  start_time = time.time()
   369            1    1768626.0 1768626.0     4.4%  memory_data = memory_intensive_operations()
   370            1          2.0      2.0     0.0%  results["memory_time"] = time.time() - start_time
   371            1          1.0      1.0     0.0%  results["memory_objects_created"] = len(memory_dat
   374            1          1.0      1.0     0.0%  start_time = time.time()
   375            1     383707.0 383707.0     1.0%  io_result = file_io_operations()
   376            1          1.0      1.0     0.0%  results["io_time"] = time.time() - start_time
   377            1          1.0      1.0     0.0%  results["io_operations"] = io_result
   380            1          1.0      1.0     0.0%  start_time = time.time()
   381            1        200.0    200.0     0.0%  call_result = nested_function_calls()
   382            1          1.0      1.0     0.0%  results["call_time"] = time.time() - start_time
   383            1          1.0      1.0     0.0%  results["call_result"] = call_result
   386            1          1.0      1.0     0.0%  start_time = time.time()
   387            1      66107.0  66107.0     0.2%  processing_result = data_processing_pipeline()
   388            1          1.0      1.0     0.0%  results["processing_time"] = time.time() - start_t
   389            1          1.0      1.0     0.0%  results["processing_stats"] = processing_result
   392            1          1.0      1.0     0.0%  start_time = time.time()
   393            1   18937514.0 18937514.0    47.4%  antipattern_result = demonstrate_anti_patterns()
   394            1          2.0      2.0     0.0%  results["antipattern_time"] = time.time() - start_
   395            1          0.0      0.0     0.0%  results["antipattern_stats"] = antipattern_result
   398            1          1.0      1.0     0.0%  start_time = time.time()
   399            1    1668591.0 1668591.0     4.2%  fib_recursive = fibonacci_recursive(25)  # Small e
   400            1          2.0      2.0     0.0%  results["fib_recursive_time"] = time.time() - star
   402            1          1.0      1.0     0.0%  start_time = time.time()
   403            1         66.0     66.0     0.0%  fib_iterative = fibonacci_iterative(25)
   404            1          1.0      1.0     0.0%  results["fib_iterative_time"] = time.time() - star
   406            1          1.0      1.0     0.0%  results["fib_results_match"] = fib_recursive == fi
   408            1          2.0      2.0     0.0%  return results
```

**Performance Insights:**
- **Line 393**: 47.4% of function time (1 hits)
- **Line 363**: 42.9% of function time (1 hits)
- **Line 369**: 4.4% of function time (1 hits)

#### cpu_intensive_calculation (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   134            1          3.0      3.0     0.0%  print("🔥 Running CPU-intensive calculations...")
   137            1          1.0      1.0     0.0%  size = 200
   138            1      25189.0  25189.0     0.2%  matrix_a = [[random.random() for _ in range(size)]
   139            1      27589.0  27589.0     0.2%  matrix_b = [[random.random() for _ in range(size)]
   141            1          1.0      1.0     0.0%  result = 0.0
   142          201        192.0      1.0     0.0%  for i in range(size):
   143        40200      35791.0      0.9     0.2%  for j in range(size):
   144      8040000    7077442.0      0.9    48.3%  for k in range(size):
   145      8000000    7476321.0      0.9    51.0%  result += matrix_a[i][k] * matrix_b[k][j]
   148        10001       8876.0      0.9     0.1%  for i in range(10000):
   149        10000       9579.0      1.0     0.1%  result += math.sin(i) * math.cos(i) * math.sqrt(i 
   151            1          7.0      7.0     0.0%  return result
```

**Performance Insights:**
- **Line 145**: 51.0% of function time (8,000,000 hits)
- **Line 144**: 48.3% of function time (8,040,000 hits)
- **Line 143**: 0.2% of function time (40,200 hits)

#### memory_intensive_operations (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   156            1          5.0      5.0     0.0%  print("💾 Running memory-intensive operations...")
   159            1          1.0      1.0     0.0%  large_list = []
   160        50001      49535.0      1.0     3.0%  for i in range(50000):
   161       100000     100407.0      1.0     6.1%  large_list.append(
   162        50000      55330.0      1.1     3.4%  {
   163        50000      46677.0      0.9     2.8%  "id": i,
   164        50000     766708.0     15.3    46.6%  "data": [random.random() for _ in range(20)],
   165        50000      74671.0      1.5     4.5%  "metadata": {
   166        50000      51385.0      1.0     3.1%  "timestamp": time.time(),
   167        50000      54964.0      1.1     3.3%  "category": f"category_{i % 10}",
   168        50000     209344.0      4.2    12.7%  "tags": [f"tag_{j}" for j in range(i % 5 + 1)],
   174            1          1.0      1.0     0.0%  lookup_dict = {}
   175        50001      43597.0      0.9     2.6%  for item in large_list:
   176        50000      42466.0      0.8     2.6%  category = item["metadata"]["category"]
   177        50000      43201.0      0.9     2.6%  if category not in lookup_dict:
   178           10         13.0      1.3     0.0%  lookup_dict[category] = []
   179        50000      43449.0      0.9     2.6%  lookup_dict[category].append(item)
   182            1          1.0      1.0     0.0%  temp_lists = []
   183          101        100.0      1.0     0.0%  for i in range(100):
   184          100      63971.0    639.7     3.9%  temp_list = [random.random() for _ in range(1000)]
   185          100        104.0      1.0     0.0%  temp_lists.append(temp_list)
   186          100        104.0      1.0     0.0%  if len(temp_lists) > 10:
   187           90        558.0      6.2     0.0%  temp_lists.pop(0)  # Remove oldest
   189            1          6.0      6.0     0.0%  return large_list
```

**Performance Insights:**
- **Line 164**: 46.6% of function time (50,000 hits)
- **Line 168**: 12.7% of function time (50,000 hits)
- **Line 161**: 6.1% of function time (100,000 hits)

#### file_io_operations (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   194            1          4.0      4.0     0.0%  print("📁 Running file I/O operations...")
   197            1        730.0    730.0     0.2%  temp_dir = tempfile.mkdtemp()
   199            1          1.0      1.0     0.0%  try:
   201            1          1.0      1.0     0.0%  file_data = {}
   202           11         15.0      1.4     0.0%  for i in range(10):
   203           10        271.0     27.1     0.1%  filename = os.path.join(temp_dir, f"test_file_{i}.
   204           10         13.0      1.3     0.0%  data = {
   205           10         10.0      1.0     0.0%  "file_id": i,
   206           10       5772.0    577.2     1.5%  "content": [random.random() for _ in range(1000)],
   207           10         11.0      1.1     0.0%  "metadata": {"created": time.time()},
   210           20       2430.0    121.5     0.6%  with open(filename, "w") as f:
   211           10     370067.0  37006.7    96.6%  json.dump(data, f)
   213           10         14.0      1.4     0.0%  file_data[filename] = data
   216            1          1.0      1.0     0.0%  read_data = {}
   217           11         19.0      1.7     0.0%  for filename in file_data:
   218           20        353.0     17.6     0.1%  with open(filename, "r") as f:
   219           10       2704.0    270.4     0.7%  read_data[filename] = json.load(f)
   222            1          1.0      1.0     0.0%  total_values = 0
   223           11          5.0      0.5     0.0%  for data in read_data.values():
   224           10          6.0      0.6     0.0%  total_values += len(data["content"])
   226            1          1.0      1.0     0.0%  return {"files_processed": len(read_data), "total_
   230           11         44.0      4.0     0.0%  for filename in os.listdir(temp_dir):
   231           10        612.0     61.2     0.2%  os.remove(os.path.join(temp_dir, filename))
   232            1         47.0     47.0     0.0%  os.rmdir(temp_dir)
```

**Performance Insights:**
- **Line 211**: 96.6% of function time (10 hits)
- **Line 206**: 1.5% of function time (10 hits)
- **Line 219**: 0.7% of function time (10 hits)

#### dump (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   165           10         10.0      1.0     0.0%  if (not skipkeys and ensure_ascii and
   166           10          9.0      0.9     0.0%  check_circular and allow_nan and
   167           10          8.0      0.8     0.0%  cls is None and indent is None and separators is N
   168           10         11.0      1.1     0.0%  default is None and not sort_keys and not kw):
   169           10        423.0     42.3     0.1%  iterable = _default_encoder.iterencode(obj)
   179        10180     352833.0     34.7    96.7%  for chunk in iterable:
   180        10170      11470.0      1.1     3.1%  fp.write(chunk)
```

**Performance Insights:**
- **Line 179**: 96.7% of function time (10,180 hits)
- **Line 180**: 3.1% of function time (10,170 hits)
- **Line 169**: 0.1% of function time (10 hits)

#### _make_iterencode (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   275           10         10.0      1.0    15.9%  if _indent is not None and not isinstance(_indent,
   278           10         11.0      1.1    17.5%  def _iterencode_list(lst, _current_indent_level):
   334           10         13.0      1.3    20.6%  def _iterencode_dict(dct, _current_indent_level):
   414           10         12.0      1.2    19.0%  def _iterencode(o, _current_indent_level):
   443           10         17.0      1.7    27.0%  return _iterencode
```

**Performance Insights:**
- **Line 443**: 27.0% of function time (10 hits)
- **Line 334**: 20.6% of function time (10 hits)
- **Line 414**: 19.0% of function time (10 hits)

#### load (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   293           30       2536.0     84.5    98.5%  return loads(fp.read(),
   294           10         10.0      1.0     0.4%  cls=cls, object_hook=object_hook,
   295           10         10.0      1.0     0.4%  parse_float=parse_float, parse_int=parse_int,
   296           20         18.0      0.9     0.7%  parse_constant=parse_constant, object_pairs_hook=o
```

**Performance Insights:**
- **Line 293**: 98.5% of function time (30 hits)
- **Line 296**: 0.7% of function time (20 hits)
- **Line 294**: 0.4% of function time (10 hits)

#### loads (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   333           10         10.0      1.0     0.5%  if isinstance(s, str):
   334           10         10.0      1.0     0.5%  if s.startswith('\ufeff'):
   343           10          9.0      0.9     0.4%  if (cls is None and object_hook is None and
   344           10          9.0      0.9     0.4%  parse_int is None and parse_float is None and
   345           10          9.0      0.9     0.4%  parse_constant is None and object_pairs_hook is No
   346           10       2083.0    208.3    97.8%  return _default_decoder.decode(s)
```

**Performance Insights:**
- **Line 346**: 97.8% of function time (10 hits)
- **Line 333**: 0.5% of function time (10 hits)
- **Line 334**: 0.5% of function time (10 hits)

#### nested_function_calls (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   237            1          3.0      3.0     2.2%  print("🌳 Running nested function calls...")
   239            1          2.0      2.0     1.4%  def level_1(n: int) -> int:
   244            1          1.0      1.0     0.7%  def level_2(n: int) -> int:
   249            1          2.0      2.0     1.4%  def level_3(n: int) -> int:
   254            1          1.0      1.0     0.7%  def level_4(n: int) -> int:
   259            1          1.0      1.0     0.7%  def level_5(n: int) -> int:
   262            1        129.0    129.0    92.8%  return level_1(8)
```

**Performance Insights:**
- **Line 262**: 92.8% of function time (1 hits)
- **Line 237**: 2.2% of function time (1 hits)
- **Line 239**: 1.4% of function time (1 hits)

#### data_processing_pipeline (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   267            1          4.0      4.0     0.0%  print("⚙️  Running data processing pipeline...")
   270            1       6075.0   6075.0     9.2%  raw_data = [random.random() * 1000 for _ in range(
   273            1       5831.0   5831.0     8.9%  filtered_data = [x for x in raw_data if x > 100]
   276            1       5419.0   5419.0     8.2%  transformed_data = [math.log(x) if x > 1 else 0 fo
   279            1         23.0     23.0     0.0%  sum_data = sum(transformed_data)
   280            1          1.0      1.0     0.0%  avg_data = sum_data / len(transformed_data) if tra
   281            1         66.0     66.0     0.1%  max_data = max(transformed_data) if transformed_da
   282            1         65.0     65.0     0.1%  min_data = min(transformed_data) if transformed_da
   285            2      48325.0  24162.5    73.4%  variance = sum((x - avg_data) ** 2 for x in transf
   286            1          1.0      1.0     0.0%  transformed_data
   288            1          3.0      3.0     0.0%  std_dev = math.sqrt(variance)
   290            1          3.0      3.0     0.0%  return {
   291            1          1.0      1.0     0.0%  "count": len(transformed_data),
   292            1          1.0      1.0     0.0%  "sum": sum_data,
   293            1          0.0      0.0     0.0%  "average": avg_data,
   294            1          1.0      1.0     0.0%  "maximum": max_data,
   295            1          1.0      1.0     0.0%  "minimum": min_data,
   296            1          1.0      1.0     0.0%  "std_dev": std_dev,
```

**Performance Insights:**
- **Line 285**: 73.4% of function time (2 hits)
- **Line 270**: 9.2% of function time (1 hits)
- **Line 273**: 8.9% of function time (1 hits)

#### demonstrate_anti_patterns (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   302            1          5.0      5.0     0.0%  print("🎯 Running anti-pattern demonstrations...")
   305            1          4.0      4.0     0.0%  import numpy as np
   308            1         53.0     53.0     0.0%  bad_array = np.array(range(1000))  # Should use np
   311            1         29.0     29.0     0.0%  test_array = np.random.random(1000)
   312            1         47.0     47.0     0.0%  count_result = (test_array > 0.5).sum()  # Should 
   315            1         21.0     21.0     0.0%  matrix_a = np.random.random((100, 50))
   316            1         23.0     23.0     0.0%  matrix_b = np.random.random((50, 100))
   317            1        217.0    217.0     0.0%  result_tensordot = np.tensordot(matrix_a, matrix_b
   320            1          5.0      5.0     0.0%  original_array = np.random.random(1000)
   321            1          8.0      8.0     0.0%  copied_array = original_array.copy()  # Might be u
   324            1          6.0      6.0     0.0%  data = list(range(1000))
   327            1       4613.0   4613.0     0.0%  result = inefficient_nested_search(data, [500, 750
   330            1        451.0    451.0     0.0%  complex_result = overly_complex_function(10, 20, 3
   333            1         69.0     69.0     0.0%  string_data = [f"item_{i}" for i in range(100)]
   334            1        583.0    583.0     0.0%  processed_data = inefficient_data_operations(strin
   337            1   18930945.0 18930945.0   100.0%  fib_recursive_result = fibonacci_recursive(30)
   338            1        176.0    176.0     0.0%  fib_iterative_result = fibonacci_iterative(30)
   340            1          4.0      4.0     0.0%  return {
   341            1          1.0      1.0     0.0%  "nested_search": result,
   342            1          1.0      1.0     0.0%  "complex_function": complex_result,
   343            1          1.0      1.0     0.0%  "data_operations": len(processed_data),
   344            1          1.0      1.0     0.0%  "fibonacci_recursive": fib_recursive_result,
   345            1          0.0      0.0     0.0%  "fibonacci_iterative": fib_iterative_result,
   346            1          1.0      1.0     0.0%  "fibonacci_match": fib_recursive_result == fib_ite
   348            1          1.0      1.0     0.0%  "bad_array_size": len(bad_array),
   349            1          1.0      1.0     0.0%  "count_result": count_result,
   350            1          2.0      2.0     0.0%  "tensordot_shape": result_tensordot.shape,
   351            1          1.0      1.0     0.0%  "copied_array_size": len(copied_array),
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
    51         2252       1947.0      0.9    50.6%  for item in data_list:
    52         2252       1892.0      0.8    49.1%  if item == target:
    53            3          3.0      1.0     0.1%  found_items.append(item)
    54            3          3.0      1.0     0.1%  break  # At least we break early
    56            1          2.0      2.0     0.1%  return found_items
```

**Performance Insights:**
- **Line 51**: 50.6% of function time (2,252 hits)
- **Line 52**: 49.1% of function time (2,252 hits)
- **Line 50**: 0.1% of function time (4 hits)

#### overly_complex_function (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    64            1          1.0      1.0     0.4%  result = 0
    67            1          1.0      1.0     0.4%  if param1 > 0:
    68            1          1.0      1.0     0.4%  if param2 > 0:
    69            1          1.0      1.0     0.4%  if param3 > 0:
    70            1          1.0      1.0     0.4%  if param4 > 0:
    71            1          1.0      1.0     0.4%  if param5 > 0:
    72            1          1.0      1.0     0.4%  if param6 > 0:
    73            1          1.0      1.0     0.4%  if param7 > 0:
    74            1          1.0      1.0     0.4%  result = (
    75            7          9.0      1.3     3.5%  param1
    76            1          1.0      1.0     0.4%  + param2
    77            1          1.0      1.0     0.4%  + param3
    78            1          1.0      1.0     0.4%  + param4
    79            1          1.0      1.0     0.4%  + param5
    80            1          1.0      1.0     0.4%  + param6
    81            1          1.0      1.0     0.4%  + param7
    99           11          9.0      0.8     3.5%  for i in range(10):
   100          110        111.0      1.0    43.5%  for j in range(10):
   101          100         97.0      1.0    38.0%  if i + j == result % 10:
   102           10         10.0      1.0     3.9%  result += 1
   104            1          4.0      4.0     1.6%  return result
```

**Performance Insights:**
- **Line 100**: 43.5% of function time (110 hits)
- **Line 101**: 38.0% of function time (100 hits)
- **Line 102**: 3.9% of function time (10 hits)

#### inefficient_data_operations (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   113            1          1.0      1.0     0.2%  processed_items = []
   114            1          2.0      2.0     0.5%  unique_items = []
   116          101         88.0      0.9    21.7%  for item in items:
   117          100        130.0      1.3    32.1%  if item not in processed_items:  # O(n) operation 
   118          100         91.0      0.9    22.5%  processed_items.append(item)
   119          100         91.0      0.9    22.5%  unique_items.append(f"processed_{item}")
   121            1          2.0      2.0     0.5%  return unique_items
```

**Performance Insights:**
- **Line 117**: 32.1% of function time (100 hits)
- **Line 118**: 22.5% of function time (100 hits)
- **Line 119**: 22.5% of function time (100 hits)

#### fibonacci_recursive (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    27      2935322    2548452.0      0.9    23.2%  if n <= 1:
    28      1467662    1871044.0      1.3    17.0%  return n
    29      1467660    6573678.0      4.5    59.8%  return fibonacci_recursive(n - 1) + fibonacci_recu
```

**Performance Insights:**
- **Line 29**: 59.8% of function time (1,467,660 hits)
- **Line 27**: 23.2% of function time (2,935,322 hits)
- **Line 28**: 17.0% of function time (1,467,662 hits)

#### fibonacci_iterative (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    34            2         16.0      8.0    12.4%  if n <= 1:
    36            2          3.0      1.5     2.3%  a, b = 0, 1
    37           55         56.0      1.0    43.4%  for _ in range(2, n + 1):
    38           53         49.0      0.9    38.0%  a, b = b, a + b
    39            2          5.0      2.5     3.9%  return b
```

**Performance Insights:**
- **Line 37**: 43.4% of function time (55 hits)
- **Line 38**: 38.0% of function time (53 hits)
- **Line 34**: 12.4% of function time (2 hits)

### 📊 Line Profiling Summary

- **Total Lines Profiled:** 284
- **Total Hits:** 22,697,860
- **Total Time:** 127.042565 seconds
- **Average Time per Hit:** 0.000005597 seconds
## 📈 Memory Profiler Analysis

### 🧠 Memory Usage Statistics

| Metric | Value |
|--------|-------|
| Peak Memory Usage | 190.88 MB |
| Average Memory Usage | 162.50 MB |
| Memory Delta | +77.31 MB |
| Sample Count | 2744 |
| Initial Memory | 107.69 MB |
| Final Memory | 185.00 MB |

### 📊 Memory Timeline Analysis

- **Memory Growth Rate:** 1.9253 MB/second
- **Memory Spikes Detected:** 0 (>243.75 MB)

## 📈 Call Profiler Analysis

### 🕒 Top Functions by Execution Time

| Function | Total Time | Calls | Time/Call | Cumulative |
|----------|------------|-------|-----------|------------|
| `/Users/Adam/Pycroscope/docs/examples/usage_example.py:0(run_profiling_session)` | 39.9965s | 1 | 39.996543s | 39.9965s |
| `/Users/Adam/Pycroscope/docs/examples/usage_example.py:0(run_sample_workload)` | 39.9964s | 1 | 39.996381s | 39.9964s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(mixed_workload)` | 39.9824s | 1 | 39.982420s | 39.9824s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(demonstrate_anti_patterns)` | 18.9372s | 1 | 18.937164s | 18.9372s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(cpu_intensive_calculation)` | 17.1571s | 1 | 17.157103s | 17.1571s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(fibonacci_recursive)` | 4.5805s | 2935322 | 0.000002s | 4.5805s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(memory_intensive_operations)` | 1.7680s | 1 | 1.768000s | 1.7680s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(file_io_operations)` | 0.3832s | 1 | 0.383194s | 0.3832s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py:0(dump)` | 0.3699s | 10 | 0.036989s | 0.3699s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode)` | 0.2985s | 10180 | 0.000029s | 0.2985s |

### 📞 Most Called Functions

| Function | Calls | Total Time | Avg Time |
|----------|-------|------------|----------|
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(fibonacci_recursive)` | 2935322 | 4.5805s | 0.000002s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode_dict)` | 10240 | 0.2415s | 0.000024s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode)` | 10180 | 0.2985s | 0.000029s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode_list)` | 10020 | 0.1867s | 0.000019s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(floatstr)` | 10010 | 0.0488s | 0.000005s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(<genexpr>)` | 8969 | 0.0154s | 0.000002s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tokenize.py:0(_generate_tokens_from_c_tokenizer)` | 5639 | 0.0310s | 0.000005s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/site-packages/line_profiler/line_profiler.py:0(tokeneater)` | 5607 | 0.0379s | 0.000007s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/inspect.py:0(tokeneater)` | 5607 | 0.0216s | 0.000004s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/collections/__init__.py:0(_make)` | 5607 | 0.0135s | 0.000002s |

**Total Functions Profiled:** 50

## 🎯 Pattern Analysis Results

🔍 **Analysis Summary:** 38 patterns detected across 2 files

### 📊 Pattern Distribution

| Pattern Type | Count |
|--------------|-------|
| Scalar Array Operations | 18 |
| Inefficient Array Concatenation | 6 |
| Inefficient Broadcasting | 5 |
| Long Function | 4 |
| Nested Loops | 1 |
| Too Many Parameters | 1 |
| Inefficient Array Operations | 1 |
| Missed Vectorization | 1 |
| Recursive Without Memoization | 1 |

### 🚨 Severity Breakdown

| Severity | Count |
|----------|-------|
| 🚨 High | 1 |
| ⚠️ Medium | 37 |

### 🔥 Priority Issues

#### 1. 🚨 Nested Loops

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** `cpu_intensive_calculation`
- **Line:** 132
- **Severity:** High
- **Description:** Function 'cpu_intensive_calculation' has 3 levels of nested loops
- **Suggestion:** Consider extracting inner loops into separate functions or using more efficient algorithms

#### 2. ⚠️ Too Many Parameters

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** `overly_complex_function`
- **Line:** 59
- **Severity:** Medium
- **Description:** Function 'overly_complex_function' has too many parameters: 7
- **Suggestion:** Consider using data classes or reducing parameters. Target: <= 5

#### 3. ⚠️ Long Function

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** `demonstrate_anti_patterns`
- **Line:** 300
- **Severity:** Medium
- **Description:** Function 'demonstrate_anti_patterns' is too long: 55 lines
- **Suggestion:** Consider breaking down this function. Target: <= 50 lines

#### 4. ⚠️ Long Function

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** `mixed_workload`
- **Line:** 355
- **Severity:** Medium
- **Description:** Function 'mixed_workload' is too long: 87 lines
- **Suggestion:** Consider breaking down this function. Target: <= 50 lines

#### 5. ⚠️ Inefficient Array Operations

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 308
- **Severity:** Medium
- **Description:** np.array(range(n)) is inefficient, use np.arange(n) instead
- **Suggestion:** Replace with np.arange() for better performance

#### 6. ⚠️ Long Function

- **File:** `/Users/Adam/Pycroscope/docs/examples/usage_example.py`
- **Function:** `main`
- **Line:** 55
- **Severity:** Medium
- **Description:** Function 'main' is too long: 217 lines
- **Suggestion:** Consider breaking down this function. Target: <= 50 lines

#### 7. ⚠️ Long Function

- **File:** `/Users/Adam/Pycroscope/docs/examples/usage_example.py`
- **Function:** `run_profiling_session`
- **Line:** 97
- **Severity:** Medium
- **Description:** Function 'run_profiling_session' is too long: 175 lines
- **Suggestion:** Consider breaking down this function. Target: <= 50 lines

#### 8. ⚠️ Missed Vectorization

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 202
- **Severity:** Medium
- **Description:** Loop over array elements could be vectorized with NumPy operations
- **Suggestion:** Replace explicit loop with vectorized NumPy operations

#### 9. ⚠️ Inefficient Array Concatenation

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 50
- **Severity:** Medium
- **Description:** Array concatenation in loop is inefficient
- **Suggestion:** Pre-allocate array and use indexing, or collect items and concatenate once

#### 10. ⚠️ Inefficient Array Concatenation

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** ``
- **Line:** 51
- **Severity:** Medium
- **Description:** Array concatenation in loop is inefficient
- **Suggestion:** Pre-allocate array and use indexing, or collect items and concatenate once

### 💡 Recommendations

1. Consider optimizing algorithmic complexity in functions with nested loops or high time complexity

## 🎯 Performance Insights

- Long execution time detected (40.160s) - consider optimization
- High function call count (3,010,171) - potential optimization opportunity

## 🔧 Technical Details

### Session Metadata

- **Start Time:** 2025-07-27 12:53:27.922214
- **End Time:** 2025-07-27 12:54:08.082711
- **Output Directory:** `profiling_results`
- **Session Name:** Default

