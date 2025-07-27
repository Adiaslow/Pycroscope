# 🔍 Pycroscope Profiling Analysis Report

**Generated:** 2025-07-27 12:25:43
**Session ID:** `81576787-41a6-45f2-baac-1e284e99c13f`

## 📊 Executive Summary

- **Duration:** 21.636 seconds
- **Status:** completed
- **Profilers Used:** line, memory, call
- **Total Results:** 3
- **Patterns Detected:** 6

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
   373            2        355.0    177.5    20.1%  prefix, suffix, dir, output_type = _sanitize_param
   375            2         69.0     34.5     3.9%  names = _get_candidate_names()
   376            2          2.0      1.0     0.1%  if output_type is bytes:
   379            2          3.0      1.5     0.2%  for seq in range(TMP_MAX):
   380            2        724.0    362.0    40.9%  name = next(names)
   381            2         42.0     21.0     2.4%  file = _os.path.join(dir, prefix + name + suffix)
   382            2          2.0      1.0     0.1%  _sys.audit("tempfile.mkdtemp", file)
   383            2          2.0      1.0     0.1%  try:
   384            2        523.0    261.5    29.5%  _os.mkdir(file, 0o700)
   395            2         48.0     24.0     2.7%  return _os.path.abspath(file)
```

**Performance Insights:**
- **Line 380**: 40.9% of function time (2 hits)
- **Line 384**: 29.5% of function time (2 hits)
- **Line 373**: 20.1% of function time (2 hits)

#### _sanitize_params (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   116            2        123.0     61.5    52.8%  output_type = _infer_return_type(prefix, suffix, d
   117            2          2.0      1.0     0.9%  if suffix is None:
   118            2          2.0      1.0     0.9%  suffix = output_type()
   119            2          3.0      1.5     1.3%  if prefix is None:
   120            1          1.0      1.0     0.4%  if output_type is str:
   121            1          1.0      1.0     0.4%  prefix = template
   124            2          2.0      1.0     0.9%  if dir is None:
   125            2          2.0      1.0     0.9%  if output_type is str:
   126            2         94.0     47.0    40.3%  dir = gettempdir()
   129            2          3.0      1.5     1.3%  return prefix, suffix, dir, output_type
```

**Performance Insights:**
- **Line 116**: 52.8% of function time (2 hits)
- **Line 126**: 40.3% of function time (2 hits)
- **Line 119**: 1.3% of function time (2 hits)

#### _infer_return_type (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    87            2          6.0      3.0    14.6%  return_type = None
    88            8          9.0      1.1    22.0%  for arg in args:
    89            6          5.0      0.8    12.2%  if arg is None:
    90            5          5.0      1.0    12.2%  continue
    92            1          5.0      5.0    12.2%  if isinstance(arg, _os.PathLike):
    95            1          1.0      1.0     2.4%  if isinstance(arg, bytes):
   101            1          1.0      1.0     2.4%  if return_type is bytes:
   104            1          1.0      1.0     2.4%  return_type = str
   105            2          2.0      1.0     4.9%  if return_type is None:
   106            1          2.0      2.0     4.9%  if tempdir is None or isinstance(tempdir, str):
   107            1          3.0      3.0     7.3%  return str  # tempfile APIs return a str by defaul
   111            1          1.0      1.0     2.4%  return return_type
```

**Performance Insights:**
- **Line 88**: 22.0% of function time (8 hits)
- **Line 87**: 14.6% of function time (2 hits)
- **Line 89**: 12.2% of function time (6 hits)

#### gettempdir (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   315            2         66.0     33.0   100.0%  return _os.fsdecode(_gettempdir())
```

**Performance Insights:**
- **Line 315**: 100.0% of function time (2 hits)

#### _gettempdir (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   304            2          2.0      1.0    40.0%  if tempdir is None:
   311            2          3.0      1.5    60.0%  return tempdir
```

**Performance Insights:**
- **Line 311**: 60.0% of function time (2 hits)
- **Line 304**: 40.0% of function time (2 hits)

#### _get_candidate_names (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   233            1          2.0      2.0    66.7%  if _name_sequence is None:
   240            1          1.0      1.0    33.3%  return _name_sequence
```

**Performance Insights:**
- **Line 233**: 66.7% of function time (1 hits)
- **Line 240**: 33.3% of function time (1 hits)

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
   840            1          2.0      2.0   100.0%  return name_template % _counter()
```

**Performance Insights:**
- **Line 840**: 100.0% of function time (1 hits)

#### _make_invoke_excepthook (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/threading.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
  1361            1          2.0      2.0    25.0%  old_excepthook = excepthook
  1362            1          1.0      1.0    12.5%  old_sys_excepthook = _sys.excepthook
  1363            1          1.0      1.0    12.5%  if old_excepthook is None:
  1365            1          1.0      1.0    12.5%  if old_sys_excepthook is None:
  1368            1          1.0      1.0    12.5%  sys_exc_info = _sys.exc_info
  1369            1          0.0      0.0     0.0%  local_print = print
  1370            1          0.0      0.0     0.0%  local_sys = _sys
  1372            1          1.0      1.0    12.5%  def invoke_excepthook(thread):
  1404            1          1.0      1.0    12.5%  return invoke_excepthook
```

**Performance Insights:**
- **Line 1361**: 25.0% of function time (1 hits)
- **Line 1362**: 12.5% of function time (1 hits)
- **Line 1363**: 12.5% of function time (1 hits)

#### register_trace_function (/Users/Adam/Pycroscope/src/pycroscope/infrastructure/profilers/trace_multiplexer.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   124            1         45.0     45.0   100.0%  _trace_multiplexer.register_profiler(profiler_name
```

**Performance Insights:**
- **Line 124**: 100.0% of function time (1 hits)

#### run_sample_workload (/Users/Adam/Pycroscope/docs/examples/usage_example.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    43            1       3491.0   3491.0     0.0%  from sample_workload import mixed_workload
    45            1          9.0      9.0     0.0%  print("🎯 Executing workload to be profiled...")
    46            1          2.0      2.0     0.0%  print("   (Replace this section with your own code
    47            1          2.0      2.0     0.0%  print()
    50            1   21452093.0 21452093.0   100.0%  results = mixed_workload()
    52            1          8.0      8.0     0.0%  return results
```

**Performance Insights:**
- **Line 50**: 100.0% of function time (1 hits)
- **Line 43**: 0.0% of function time (1 hits)
- **Line 45**: 0.0% of function time (1 hits)

#### _type_check (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/typing.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   187            3          4.0      1.3     3.4%  invalid_generic_forms = (Generic, Protocol)
   188            3          3.0      1.0     2.5%  if not allow_special_forms:
   189            3          3.0      1.0     2.5%  invalid_generic_forms += (ClassVar,)
   190            3          3.0      1.0     2.5%  if is_argument:
   191            3          3.0      1.0     2.5%  invalid_generic_forms += (Final,)
   193            3         80.0     26.7    67.2%  arg = _type_convert(arg, module=module, allow_spec
   194            3          3.0      1.0     2.5%  if (isinstance(arg, _GenericAlias) and
   197            3          4.0      1.3     3.4%  if arg in (Any, LiteralString, NoReturn, Never, Se
   199            3          2.0      0.7     1.7%  if allow_special_forms and arg in (ClassVar, Final
   201            3          4.0      1.3     3.4%  if isinstance(arg, _SpecialForm) or arg in (Generi
   203            3          3.0      1.0     2.5%  if type(arg) is tuple:
   205            3          7.0      2.3     5.9%  return arg
```

**Performance Insights:**
- **Line 193**: 67.2% of function time (3 hits)
- **Line 205**: 5.9% of function time (3 hits)
- **Line 187**: 3.4% of function time (3 hits)

#### _type_convert (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/typing.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   168            3          9.0      3.0    45.0%  if arg is None:
   170            3          4.0      1.3    20.0%  if isinstance(arg, str):
   172            3          7.0      2.3    35.0%  return arg
```

**Performance Insights:**
- **Line 168**: 45.0% of function time (3 hits)
- **Line 172**: 35.0% of function time (3 hits)
- **Line 170**: 20.0% of function time (3 hits)

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
   339            1          3.0      3.0     0.0%  print("🚀 Running mixed workload demonstration...")
   341            1          1.0      1.0     0.0%  results = {}
   344            1          2.0      2.0     0.0%  start_time = time.time()
   345            1   17407133.0 17407133.0    81.2%  cpu_result = cpu_intensive_calculation()
   346            1          3.0      3.0     0.0%  results["cpu_time"] = time.time() - start_time
   347            1          1.0      1.0     0.0%  results["cpu_result_sample"] = cpu_result
   350            1          1.0      1.0     0.0%  start_time = time.time()
   351            1    1710393.0 1710393.0     8.0%  memory_data = memory_intensive_operations()
   352            1          2.0      2.0     0.0%  results["memory_time"] = time.time() - start_time
   353            1          1.0      1.0     0.0%  results["memory_objects_created"] = len(memory_dat
   356            1          2.0      2.0     0.0%  start_time = time.time()
   357            1     400954.0 400954.0     1.9%  io_result = file_io_operations()
   358            1          3.0      3.0     0.0%  results["io_time"] = time.time() - start_time
   359            1          2.0      2.0     0.0%  results["io_operations"] = io_result
   362            1          1.0      1.0     0.0%  start_time = time.time()
   363            1        156.0    156.0     0.0%  call_result = nested_function_calls()
   364            1          1.0      1.0     0.0%  results["call_time"] = time.time() - start_time
   365            1          1.0      1.0     0.0%  results["call_result"] = call_result
   368            1          1.0      1.0     0.0%  start_time = time.time()
   369            1      70197.0  70197.0     0.3%  processing_result = data_processing_pipeline()
   370            1          3.0      3.0     0.0%  results["processing_time"] = time.time() - start_t
   371            1          1.0      1.0     0.0%  results["processing_stats"] = processing_result
   374            1          1.0      1.0     0.0%  start_time = time.time()
   375            1       2455.0   2455.0     0.0%  antipattern_result = demonstrate_anti_patterns()
   376            1          2.0      2.0     0.0%  results["antipattern_time"] = time.time() - start_
   377            1          1.0      1.0     0.0%  results["antipattern_stats"] = antipattern_result
   380            1          1.0      1.0     0.0%  start_time = time.time()
   381            1    1847291.0 1847291.0     8.6%  fib_recursive = fibonacci_recursive(25)  # Small e
   382            1          6.0      6.0     0.0%  results["fib_recursive_time"] = time.time() - star
   384            1          1.0      1.0     0.0%  start_time = time.time()
   385            1        261.0    261.0     0.0%  fib_iterative = fibonacci_iterative(25)
   386            1          3.0      3.0     0.0%  results["fib_iterative_time"] = time.time() - star
   388            1          2.0      2.0     0.0%  results["fib_results_match"] = fib_recursive == fi
   390            1          4.0      4.0     0.0%  return results
```

**Performance Insights:**
- **Line 345**: 81.2% of function time (1 hits)
- **Line 381**: 8.6% of function time (1 hits)
- **Line 351**: 8.0% of function time (1 hits)

#### cpu_intensive_calculation (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   134            1          3.0      3.0     0.0%  print("🔥 Running CPU-intensive calculations...")
   137            1          1.0      1.0     0.0%  size = 200
   138            1      23409.0  23409.0     0.2%  matrix_a = [[random.random() for _ in range(size)]
   139            1      24450.0  24450.0     0.2%  matrix_b = [[random.random() for _ in range(size)]
   141            1          1.0      1.0     0.0%  result = 0.0
   142          201        207.0      1.0     0.0%  for i in range(size):
   143        40200      36169.0      0.9     0.2%  for j in range(size):
   144      8040000    7269199.0      0.9    49.0%  for k in range(size):
   145      8000000    7470750.0      0.9    50.3%  result += matrix_a[i][k] * matrix_b[k][j]
   148        10001       9516.0      1.0     0.1%  for i in range(10000):
   149        10000      10506.0      1.1     0.1%  result += math.sin(i) * math.cos(i) * math.sqrt(i 
   151            1         12.0     12.0     0.0%  return result
```

**Performance Insights:**
- **Line 145**: 50.3% of function time (8,000,000 hits)
- **Line 144**: 49.0% of function time (8,040,000 hits)
- **Line 143**: 0.2% of function time (40,200 hits)

#### memory_intensive_operations (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   156            1          6.0      6.0     0.0%  print("💾 Running memory-intensive operations...")
   159            1          1.0      1.0     0.0%  large_list = []
   160        50001      46229.0      0.9     2.9%  for i in range(50000):
   161       100000      92517.0      0.9     5.8%  large_list.append(
   162        50000      70092.0      1.4     4.4%  {
   163        50000      43087.0      0.9     2.7%  "id": i,
   164        50000     689497.0     13.8    43.3%  "data": [random.random() for _ in range(20)],
   165        50000      58571.0      1.2     3.7%  "metadata": {
   166        50000      49503.0      1.0     3.1%  "timestamp": time.time(),
   167        50000      51160.0      1.0     3.2%  "category": f"category_{i % 10}",
   168        50000     242194.0      4.8    15.2%  "tags": [f"tag_{j}" for j in range(i % 5 + 1)],
   174            1          1.0      1.0     0.0%  lookup_dict = {}
   175        50001      46195.0      0.9     2.9%  for item in large_list:
   176        50000      45062.0      0.9     2.8%  category = item["metadata"]["category"]
   177        50000      46196.0      0.9     2.9%  if category not in lookup_dict:
   178           10          8.0      0.8     0.0%  lookup_dict[category] = []
   179        50000      45679.0      0.9     2.9%  lookup_dict[category].append(item)
   182            1          2.0      2.0     0.0%  temp_lists = []
   183          101        113.0      1.1     0.0%  for i in range(100):
   184          100      65852.0    658.5     4.1%  temp_list = [random.random() for _ in range(1000)]
   185          100        107.0      1.1     0.0%  temp_lists.append(temp_list)
   186          100        111.0      1.1     0.0%  if len(temp_lists) > 10:
   187           90        594.0      6.6     0.0%  temp_lists.pop(0)  # Remove oldest
   189            1          8.0      8.0     0.0%  return large_list
```

**Performance Insights:**
- **Line 164**: 43.3% of function time (50,000 hits)
- **Line 168**: 15.2% of function time (50,000 hits)
- **Line 161**: 5.8% of function time (100,000 hits)

#### file_io_operations (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   194            1          4.0      4.0     0.0%  print("📁 Running file I/O operations...")
   197            1       1171.0   1171.0     0.3%  temp_dir = tempfile.mkdtemp()
   199            1          1.0      1.0     0.0%  try:
   201            1          1.0      1.0     0.0%  file_data = {}
   202           11         16.0      1.5     0.0%  for i in range(10):
   203           10        271.0     27.1     0.1%  filename = os.path.join(temp_dir, f"test_file_{i}.
   204           10         13.0      1.3     0.0%  data = {
   205           10         10.0      1.0     0.0%  "file_id": i,
   206           10       6014.0    601.4     1.5%  "content": [random.random() for _ in range(1000)],
   207           10         22.0      2.2     0.0%  "metadata": {"created": time.time()},
   210           20       3171.0    158.6     0.8%  with open(filename, "w") as f:
   211           10     385205.0  38520.5    96.2%  json.dump(data, f)
   213           10         19.0      1.9     0.0%  file_data[filename] = data
   216            1          1.0      1.0     0.0%  read_data = {}
   217           11         12.0      1.1     0.0%  for filename in file_data:
   218           20        387.0     19.4     0.1%  with open(filename, "r") as f:
   219           10       3221.0    322.1     0.8%  read_data[filename] = json.load(f)
   222            1          1.0      1.0     0.0%  total_values = 0
   223           11         12.0      1.1     0.0%  for data in read_data.values():
   224           10         12.0      1.2     0.0%  total_values += len(data["content"])
   226            1          1.0      1.0     0.0%  return {"files_processed": len(read_data), "total_
   230           11         39.0      3.5     0.0%  for filename in os.listdir(temp_dir):
   231           10        694.0     69.4     0.2%  os.remove(os.path.join(temp_dir, filename))
   232            1         46.0     46.0     0.0%  os.rmdir(temp_dir)
```

**Performance Insights:**
- **Line 211**: 96.2% of function time (10 hits)
- **Line 206**: 1.5% of function time (10 hits)
- **Line 219**: 0.8% of function time (10 hits)

#### dump (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   165           10         12.0      1.2     0.0%  if (not skipkeys and ensure_ascii and
   166           10         10.0      1.0     0.0%  check_circular and allow_nan and
   167           10          9.0      0.9     0.0%  cls is None and indent is None and separators is N
   168           10         10.0      1.0     0.0%  default is None and not sort_keys and not kw):
   169           10        503.0     50.3     0.1%  iterable = _default_encoder.iterencode(obj)
   179        10180     367224.0     36.1    96.7%  for chunk in iterable:
   180        10170      11923.0      1.2     3.1%  fp.write(chunk)
```

**Performance Insights:**
- **Line 179**: 96.7% of function time (10,180 hits)
- **Line 180**: 3.1% of function time (10,170 hits)
- **Line 169**: 0.1% of function time (10 hits)

#### _make_iterencode (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   275           10         10.0      1.0    14.5%  if _indent is not None and not isinstance(_indent,
   278           10         13.0      1.3    18.8%  def _iterencode_list(lst, _current_indent_level):
   334           10         17.0      1.7    24.6%  def _iterencode_dict(dct, _current_indent_level):
   414           10         11.0      1.1    15.9%  def _iterencode(o, _current_indent_level):
   443           10         18.0      1.8    26.1%  return _iterencode
```

**Performance Insights:**
- **Line 443**: 26.1% of function time (10 hits)
- **Line 334**: 24.6% of function time (10 hits)
- **Line 278**: 18.8% of function time (10 hits)

#### load (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   293           30       2897.0     96.6    98.1%  return loads(fp.read(),
   294           10         15.0      1.5     0.5%  cls=cls, object_hook=object_hook,
   295           10          9.0      0.9     0.3%  parse_float=parse_float, parse_int=parse_int,
   296           20         31.0      1.6     1.1%  parse_constant=parse_constant, object_pairs_hook=o
```

**Performance Insights:**
- **Line 293**: 98.1% of function time (30 hits)
- **Line 296**: 1.1% of function time (20 hits)
- **Line 294**: 0.5% of function time (10 hits)

#### loads (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   333           10         11.0      1.1     0.5%  if isinstance(s, str):
   334           10         13.0      1.3     0.6%  if s.startswith('\ufeff'):
   343           10         10.0      1.0     0.4%  if (cls is None and object_hook is None and
   344           10         10.0      1.0     0.4%  parse_int is None and parse_float is None and
   345           10         10.0      1.0     0.4%  parse_constant is None and object_pairs_hook is No
   346           10       2202.0    220.2    97.6%  return _default_decoder.decode(s)
```

**Performance Insights:**
- **Line 346**: 97.6% of function time (10 hits)
- **Line 334**: 0.6% of function time (10 hits)
- **Line 333**: 0.5% of function time (10 hits)

#### nested_function_calls (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   237            1          4.0      4.0     5.1%  print("🌳 Running nested function calls...")
   239            1          1.0      1.0     1.3%  def level_1(n: int) -> int:
   244            1          1.0      1.0     1.3%  def level_2(n: int) -> int:
   249            1          1.0      1.0     1.3%  def level_3(n: int) -> int:
   254            1          1.0      1.0     1.3%  def level_4(n: int) -> int:
   259            1          1.0      1.0     1.3%  def level_5(n: int) -> int:
   262            1         69.0     69.0    88.5%  return level_1(8)
```

**Performance Insights:**
- **Line 262**: 88.5% of function time (1 hits)
- **Line 237**: 5.1% of function time (1 hits)
- **Line 239**: 1.3% of function time (1 hits)

#### data_processing_pipeline (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   267            1          4.0      4.0     0.0%  print("⚙️  Running data processing pipeline...")
   270            1       6540.0   6540.0     9.4%  raw_data = [random.random() * 1000 for _ in range(
   273            1       6271.0   6271.0     9.0%  filtered_data = [x for x in raw_data if x > 100]
   276            1       5531.0   5531.0     7.9%  transformed_data = [math.log(x) if x > 1 else 0 fo
   279            1         30.0     30.0     0.0%  sum_data = sum(transformed_data)
   280            1          3.0      3.0     0.0%  avg_data = sum_data / len(transformed_data) if tra
   281            1         71.0     71.0     0.1%  max_data = max(transformed_data) if transformed_da
   282            1         66.0     66.0     0.1%  min_data = min(transformed_data) if transformed_da
   285            2      51401.0  25700.5    73.5%  variance = sum((x - avg_data) ** 2 for x in transf
   286            1          1.0      1.0     0.0%  transformed_data
   288            1          2.0      2.0     0.0%  std_dev = math.sqrt(variance)
   290            1          3.0      3.0     0.0%  return {
   291            1          1.0      1.0     0.0%  "count": len(transformed_data),
   292            1          1.0      1.0     0.0%  "sum": sum_data,
   293            1          1.0      1.0     0.0%  "average": avg_data,
   294            1          1.0      1.0     0.0%  "maximum": max_data,
   295            1          1.0      1.0     0.0%  "minimum": min_data,
   296            1          0.0      0.0     0.0%  "std_dev": std_dev,
```

**Performance Insights:**
- **Line 285**: 73.5% of function time (2 hits)
- **Line 270**: 9.4% of function time (1 hits)
- **Line 273**: 9.0% of function time (1 hits)

#### demonstrate_anti_patterns (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   305            1          4.0      4.0     0.2%  print("🎯 Running anti-pattern demonstrations...")
   308            1          4.0      4.0     0.2%  test_data = list(range(100))
   309            1          2.0      2.0     0.1%  search_targets = [10, 25, 50, 75, 90]
   310            1        178.0    178.0     7.8%  string_data = [f"item_{i}" for i in range(200)]
   313            1          2.0      2.0     0.1%  start_time = time.time()
   314            1        666.0    666.0    29.2%  nested_results = inefficient_nested_search(test_da
   315            1          2.0      2.0     0.1%  nested_time = time.time() - start_time
   318            1          1.0      1.0     0.0%  start_time = time.time()
   319            1        416.0    416.0    18.3%  complex_result = overly_complex_function(1, 2, 3, 
   320            1          1.0      1.0     0.0%  complex_time = time.time() - start_time
   323            1          1.0      1.0     0.0%  start_time = time.time()
   324            1        991.0    991.0    43.5%  inefficient_results = inefficient_data_operations(
   325            1          1.0      1.0     0.0%  inefficient_time = time.time() - start_time
   327            1          3.0      3.0     0.1%  return {
   328            1          2.0      2.0     0.1%  "nested_search_results": len(nested_results),
   329            1          1.0      1.0     0.0%  "nested_search_time": nested_time,
   330            1          1.0      1.0     0.0%  "complex_function_result": complex_result,
   331            1          1.0      1.0     0.0%  "complex_function_time": complex_time,
   332            1          1.0      1.0     0.0%  "inefficient_processing_results": len(inefficient_
   333            1          1.0      1.0     0.0%  "inefficient_processing_time": inefficient_time,
```

**Performance Insights:**
- **Line 324**: 43.5% of function time (1 hits)
- **Line 314**: 29.2% of function time (1 hits)
- **Line 319**: 18.3% of function time (1 hits)

#### inefficient_nested_search (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    47            1          1.0      1.0     0.2%  found_items = []
    50            6          5.0      0.8     1.0%  for target in targets:
    51          255        231.0      0.9    47.7%  for item in data_list:
    52          255        234.0      0.9    48.3%  if item == target:
    53            5          5.0      1.0     1.0%  found_items.append(item)
    54            5          5.0      1.0     1.0%  break  # At least we break early
    56            1          3.0      3.0     0.6%  return found_items
```

**Performance Insights:**
- **Line 52**: 48.3% of function time (255 hits)
- **Line 51**: 47.7% of function time (255 hits)
- **Line 50**: 1.0% of function time (6 hits)

#### overly_complex_function (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    64            1          1.0      1.0     0.4%  result = 0
    67            1          1.0      1.0     0.4%  if param1 > 0:
    68            1          0.0      0.0     0.0%  if param2 > 0:
    69            1          0.0      0.0     0.0%  if param3 > 0:
    70            1          1.0      1.0     0.4%  if param4 > 0:
    71            1          1.0      1.0     0.4%  if param5 > 0:
    72            1          1.0      1.0     0.4%  if param6 > 0:
    73            1          1.0      1.0     0.4%  if param7 > 0:
    74            1          1.0      1.0     0.4%  result = (
    75            7          7.0      1.0     3.1%  param1
    76            1          1.0      1.0     0.4%  + param2
    77            1          1.0      1.0     0.4%  + param3
    78            1          1.0      1.0     0.4%  + param4
    79            1          1.0      1.0     0.4%  + param5
    80            1          1.0      1.0     0.4%  + param6
    81            1          1.0      1.0     0.4%  + param7
    99           11          8.0      0.7     3.6%  for i in range(10):
   100          110        104.0      0.9    46.2%  for j in range(10):
   101          100         89.0      0.9    39.6%  if i + j == result % 10:
   102            2          2.0      1.0     0.9%  result += 1
   104            1          2.0      2.0     0.9%  return result
```

**Performance Insights:**
- **Line 100**: 46.2% of function time (110 hits)
- **Line 101**: 39.6% of function time (100 hits)
- **Line 99**: 3.6% of function time (11 hits)

#### inefficient_data_operations (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   113            1          1.0      1.0     0.1%  processed_items = []
   114            1          1.0      1.0     0.1%  unique_items = []
   116          201        166.0      0.8    20.1%  for item in items:
   117          200        311.0      1.6    37.7%  if item not in processed_items:  # O(n) operation 
   118          200        164.0      0.8    19.9%  processed_items.append(item)
   119          200        179.0      0.9    21.7%  unique_items.append(f"processed_{item}")
   121            1          2.0      2.0     0.2%  return unique_items
```

**Performance Insights:**
- **Line 117**: 37.7% of function time (200 hits)
- **Line 119**: 21.7% of function time (200 hits)
- **Line 116**: 20.1% of function time (201 hits)

#### fibonacci_recursive (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    27       242785     223865.0      0.9    22.7%  if n <= 1:
    28       121393     165787.0      1.4    16.8%  return n
    29       121392     595860.0      4.9    60.5%  return fibonacci_recursive(n - 1) + fibonacci_recu
```

**Performance Insights:**
- **Line 29**: 60.5% of function time (121,392 hits)
- **Line 27**: 22.7% of function time (242,785 hits)
- **Line 28**: 16.8% of function time (121,393 hits)

#### fibonacci_iterative (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    34            1          2.0      2.0     3.7%  if n <= 1:
    36            1          2.0      2.0     3.7%  a, b = 0, 1
    37           25         25.0      1.0    46.3%  for _ in range(2, n + 1):
    38           24         23.0      1.0    42.6%  a, b = b, a + b
    39            1          2.0      2.0     3.7%  return b
```

**Performance Insights:**
- **Line 37**: 46.3% of function time (25 hits)
- **Line 38**: 42.6% of function time (24 hits)
- **Line 34**: 3.7% of function time (1 hits)

### 📊 Line Profiling Summary

- **Total Lines Profiled:** 276
- **Total Hits:** 17,309,120
- **Total Time:** 61.178532 seconds
- **Average Time per Hit:** 0.000003534 seconds
## 📈 Memory Profiler Analysis

### 🧠 Memory Usage Statistics

| Metric | Value |
|--------|-------|
| Peak Memory Usage | 191.83 MB |
| Average Memory Usage | 129.32 MB |
| Memory Delta | +79.17 MB |
| Sample Count | 1241 |
| Initial Memory | 109.83 MB |
| Final Memory | 189.00 MB |

### 📊 Memory Timeline Analysis

- **Memory Growth Rate:** 3.6599 MB/second
- **Memory Spikes Detected:** 0 (>193.98 MB)

## 📈 Call Profiler Analysis

### 🕒 Top Functions by Execution Time

| Function | Total Time | Calls | Time/Call | Cumulative |
|----------|------------|-------|-----------|------------|
| `/Users/Adam/Pycroscope/docs/examples/usage_example.py:0(run_profiling_session)` | 21.4556s | 1 | 21.455576s | 21.4556s |
| `/Users/Adam/Pycroscope/docs/examples/usage_example.py:0(run_sample_workload)` | 21.4555s | 1 | 21.455497s | 21.4555s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(mixed_workload)` | 21.4388s | 1 | 21.438789s | 21.4388s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(cpu_intensive_calculation)` | 17.4064s | 1 | 17.406442s | 17.4064s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(memory_intensive_operations)` | 1.7097s | 1 | 1.709666s | 1.7097s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(fibonacci_recursive)` | 0.4033s | 242785 | 0.000002s | 0.4033s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(file_io_operations)` | 0.4004s | 1 | 0.400411s | 0.4004s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py:0(dump)` | 0.3849s | 10 | 0.038494s | 0.3849s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode)` | 0.3097s | 10180 | 0.000030s | 0.3097s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode_dict)` | 0.2495s | 10240 | 0.000024s | 0.2495s |

### 📞 Most Called Functions

| Function | Calls | Total Time | Avg Time |
|----------|-------|------------|----------|
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(fibonacci_recursive)` | 242785 | 0.4033s | 0.000002s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode_dict)` | 10240 | 0.2495s | 0.000024s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode)` | 10180 | 0.3097s | 0.000030s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode_list)` | 10020 | 0.1925s | 0.000019s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(floatstr)` | 10010 | 0.0500s | 0.000005s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(<genexpr>)` | 8997 | 0.0160s | 0.000002s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tokenize.py:0(_generate_tokens_from_c_tokenizer)` | 5535 | 0.0323s | 0.000006s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/site-packages/line_profiler/line_profiler.py:0(tokeneater)` | 5503 | 0.0391s | 0.000007s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/inspect.py:0(tokeneater)` | 5503 | 0.0222s | 0.000004s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/collections/__init__.py:0(_make)` | 5503 | 0.0141s | 0.000003s |

**Total Functions Profiled:** 50

## 🎯 Pattern Analysis Results

🔍 **Analysis Summary:** 6 patterns detected across 2 files

### 📊 Pattern Distribution

| Pattern Type | Count |
|--------------|-------|
| Long Function | 3 |
| Nested Loops | 1 |
| Too Many Parameters | 1 |
| Recursive Without Memoization | 1 |

### 🚨 Severity Breakdown

| Severity | Count |
|----------|-------|
| 🚨 High | 1 |
| ⚠️ Medium | 5 |

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
- **Function:** `mixed_workload`
- **Line:** 337
- **Severity:** Medium
- **Description:** Function 'mixed_workload' is too long: 87 lines
- **Suggestion:** Consider breaking down this function. Target: <= 50 lines

#### 4. ⚠️ Long Function

- **File:** `/Users/Adam/Pycroscope/docs/examples/usage_example.py`
- **Function:** `main`
- **Line:** 55
- **Severity:** Medium
- **Description:** Function 'main' is too long: 217 lines
- **Suggestion:** Consider breaking down this function. Target: <= 50 lines

#### 5. ⚠️ Long Function

- **File:** `/Users/Adam/Pycroscope/docs/examples/usage_example.py`
- **Function:** `run_profiling_session`
- **Line:** 97
- **Severity:** Medium
- **Description:** Function 'run_profiling_session' is too long: 175 lines
- **Suggestion:** Consider breaking down this function. Target: <= 50 lines

#### 6. ⚠️ Recursive Without Memoization

- **File:** `/Users/Adam/Pycroscope/docs/examples/sample_workload.py`
- **Function:** `fibonacci_recursive`
- **Line:** 25
- **Severity:** Medium
- **Description:** Recursive function 'fibonacci_recursive' without memoization
- **Suggestion:** Consider adding memoization with @lru_cache or manual caching

### 💡 Recommendations

1. Consider optimizing algorithmic complexity in functions with nested loops or high time complexity

## 🎯 Performance Insights

- Long execution time detected (21.636s) - consider optimization
- High function call count (317,202) - potential optimization opportunity

## 🔧 Technical Details

### Session Metadata

- **Start Time:** 2025-07-27 12:25:21.363611
- **End Time:** 2025-07-27 12:25:42.999940
- **Output Directory:** `profiling_results`
- **Session Name:** Default

