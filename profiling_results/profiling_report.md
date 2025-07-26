# 🔍 Pycroscope Profiling Analysis Report

**Generated:** 2025-07-26 15:02:32
**Session ID:** `a3e9d417-7707-4c65-b482-5047a98de4bc`

## 📊 Executive Summary

- **Duration:** 17.946 seconds
- **Status:** completed
- **Profilers Used:** line, memory, call
- **Total Results:** 3

## ⚙️ Configuration

| Setting | Value |
|---------|-------|
| Line Profiling | `True` |
| Memory Profiling | `True` |
| Call Profiling | `True` |
| Sampling Profiling | `False` |
| Output Dir | `profiling_results` |
| Session Name | `None` |
| Save Raw Data | `True` |
| Sampling Interval | `0.01` |
| Memory Precision | `3` |
| Max Call Depth | `50` |
| Generate Reports | `True` |
| Create Visualizations | `True` |
| Analyze Patterns | `True` |
| Profiler Prefix | `pycroscope` |
| Use Thread Isolation | `True` |
| Cleanup On Exit | `True` |

## 📈 Line Profiler Analysis

## 📝 Line Profiler Analysis

### 🎯 Per-Function Line-by-Line Analysis

**Functions Profiled:** 26

#### mkdtemp (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   373            2        385.0    192.5    31.3%  prefix, suffix, dir, output_type = _sanitize_param
   375            2         76.0     38.0     6.2%  names = _get_candidate_names()
   376            2          2.0      1.0     0.2%  if output_type is bytes:
   379            2          2.0      1.0     0.2%  for seq in range(TMP_MAX):
   380            2        263.0    131.5    21.4%  name = next(names)
   381            2         42.0     21.0     3.4%  file = _os.path.join(dir, prefix + name + suffix)
   382            2          2.0      1.0     0.2%  _sys.audit("tempfile.mkdtemp", file)
   383            2          2.0      1.0     0.2%  try:
   384            2        421.0    210.5    34.2%  _os.mkdir(file, 0o700)
   395            2         35.0     17.5     2.8%  return _os.path.abspath(file)
```

**Performance Insights:**
- **Line 384**: 34.2% of function time (2 hits)
- **Line 373**: 31.3% of function time (2 hits)
- **Line 380**: 21.4% of function time (2 hits)

#### _sanitize_params (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   116            2        131.0     65.5    42.4%  output_type = _infer_return_type(prefix, suffix, d
   117            2          1.0      0.5     0.3%  if suffix is None:
   118            2          2.0      1.0     0.6%  suffix = output_type()
   119            2          2.0      1.0     0.6%  if prefix is None:
   120            1          1.0      1.0     0.3%  if output_type is str:
   121            1          1.0      1.0     0.3%  prefix = template
   124            2          1.0      0.5     0.3%  if dir is None:
   125            2          2.0      1.0     0.6%  if output_type is str:
   126            2        166.0     83.0    53.7%  dir = gettempdir()
   129            2          2.0      1.0     0.6%  return prefix, suffix, dir, output_type
```

**Performance Insights:**
- **Line 126**: 53.7% of function time (2 hits)
- **Line 116**: 42.4% of function time (2 hits)
- **Line 118**: 0.6% of function time (2 hits)

#### _infer_return_type (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    87            2          3.0      1.5     8.3%  return_type = None
    88            8         10.0      1.2    27.8%  for arg in args:
    89            6          5.0      0.8    13.9%  if arg is None:
    90            5          2.0      0.4     5.6%  continue
    92            1          6.0      6.0    16.7%  if isinstance(arg, _os.PathLike):
    95            1          1.0      1.0     2.8%  if isinstance(arg, bytes):
   101            1          1.0      1.0     2.8%  if return_type is bytes:
   104            1          1.0      1.0     2.8%  return_type = str
   105            2          2.0      1.0     5.6%  if return_type is None:
   106            1          2.0      2.0     5.6%  if tempdir is None or isinstance(tempdir, str):
   107            1          2.0      2.0     5.6%  return str  # tempfile APIs return a str by defaul
   111            1          1.0      1.0     2.8%  return return_type
```

**Performance Insights:**
- **Line 88**: 27.8% of function time (8 hits)
- **Line 92**: 16.7% of function time (1 hits)
- **Line 89**: 13.9% of function time (6 hits)

#### gettempdir (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tempfile.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   315            2         70.0     35.0   100.0%  return _os.fsdecode(_gettempdir())
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
  1361            1          0.0      0.0     0.0%  old_excepthook = excepthook
  1362            1          0.0      0.0     0.0%  old_sys_excepthook = _sys.excepthook
  1363            1          0.0      0.0     0.0%  if old_excepthook is None:
  1365            1          0.0      0.0     0.0%  if old_sys_excepthook is None:
  1368            1          0.0      0.0     0.0%  sys_exc_info = _sys.exc_info
  1369            1          0.0      0.0     0.0%  local_print = print
  1370            1          1.0      1.0    33.3%  local_sys = _sys
  1372            1          1.0      1.0    33.3%  def invoke_excepthook(thread):
  1404            1          1.0      1.0    33.3%  return invoke_excepthook
```

**Performance Insights:**
- **Line 1370**: 33.3% of function time (1 hits)
- **Line 1372**: 33.3% of function time (1 hits)
- **Line 1404**: 33.3% of function time (1 hits)

#### register_trace_function (/Users/Adam/Pycroscope/src/pycroscope/infrastructure/profilers/trace_multiplexer.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   124            1         40.0     40.0   100.0%  _trace_multiplexer.register_profiler(profiler_name
```

**Performance Insights:**
- **Line 124**: 100.0% of function time (1 hits)

#### run_sample_workload (/Users/Adam/Pycroscope/docs/examples/usage_example.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    41            1       1995.0   1995.0     0.0%  from sample_workload import mixed_workload
    43            1          5.0      5.0     0.0%  print("🎯 Executing workload to be profiled...")
    44            1          2.0      2.0     0.0%  print("   (Replace this section with your own code
    45            1          2.0      2.0     0.0%  print()
    48            1   17810692.0 17810692.0   100.0%  results = mixed_workload()
    50            1          3.0      3.0     0.0%  return results
```

**Performance Insights:**
- **Line 48**: 100.0% of function time (1 hits)
- **Line 41**: 0.0% of function time (1 hits)
- **Line 43**: 0.0% of function time (1 hits)

#### _type_check (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/typing.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   187            2          2.0      1.0     2.7%  invalid_generic_forms = (Generic, Protocol)
   188            2          2.0      1.0     2.7%  if not allow_special_forms:
   189            2          2.0      1.0     2.7%  invalid_generic_forms += (ClassVar,)
   190            2          2.0      1.0     2.7%  if is_argument:
   191            2          2.0      1.0     2.7%  invalid_generic_forms += (Final,)
   193            2         48.0     24.0    65.8%  arg = _type_convert(arg, module=module, allow_spec
   194            2          2.0      1.0     2.7%  if (isinstance(arg, _GenericAlias) and
   197            2          2.0      1.0     2.7%  if arg in (Any, LiteralString, NoReturn, Never, Se
   199            2          2.0      1.0     2.7%  if allow_special_forms and arg in (ClassVar, Final
   201            2          3.0      1.5     4.1%  if isinstance(arg, _SpecialForm) or arg in (Generi
   203            2          2.0      1.0     2.7%  if type(arg) is tuple:
   205            2          4.0      2.0     5.5%  return arg
```

**Performance Insights:**
- **Line 193**: 65.8% of function time (2 hits)
- **Line 205**: 5.5% of function time (2 hits)
- **Line 201**: 4.1% of function time (2 hits)

#### _type_convert (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/typing.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   168            2          2.0      1.0    22.2%  if arg is None:
   170            2          3.0      1.5    33.3%  if isinstance(arg, str):
   172            2          4.0      2.0    44.4%  return arg
```

**Performance Insights:**
- **Line 172**: 44.4% of function time (2 hits)
- **Line 170**: 33.3% of function time (2 hits)
- **Line 168**: 22.2% of function time (2 hits)

#### _is_dunder (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/typing.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
  1162            6         10.0      1.7   100.0%  return attr.startswith('__') and attr.endswith('__
```

**Performance Insights:**
- **Line 1162**: 100.0% of function time (6 hits)

#### mixed_workload (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   212            1          3.0      3.0     0.0%  print("🚀 Running mixed workload demonstration...")
   214            1          1.0      1.0     0.0%  results = {}
   217            1          1.0      1.0     0.0%  start_time = time.time()
   218            1   14366304.0 14366304.0    80.7%  cpu_result = cpu_intensive_calculation()
   219            1          3.0      3.0     0.0%  results["cpu_time"] = time.time() - start_time
   220            1          1.0      1.0     0.0%  results["cpu_result_sample"] = cpu_result
   223            1          1.0      1.0     0.0%  start_time = time.time()
   224            1    1604090.0 1604090.0     9.0%  memory_data = memory_intensive_operations()
   225            1          2.0      2.0     0.0%  results["memory_time"] = time.time() - start_time
   226            1          1.0      1.0     0.0%  results["memory_objects_created"] = len(memory_dat
   229            1          1.0      1.0     0.0%  start_time = time.time()
   230            1     342408.0 342408.0     1.9%  io_result = file_io_operations()
   231            1          3.0      3.0     0.0%  results["io_time"] = time.time() - start_time
   232            1          1.0      1.0     0.0%  results["io_operations"] = io_result
   235            1          1.0      1.0     0.0%  start_time = time.time()
   236            1        131.0    131.0     0.0%  call_result = nested_function_calls()
   237            1          2.0      2.0     0.0%  results["call_time"] = time.time() - start_time
   238            1          1.0      1.0     0.0%  results["call_result"] = call_result
   241            1          1.0      1.0     0.0%  start_time = time.time()
   242            1      58795.0  58795.0     0.3%  processing_result = data_processing_pipeline()
   243            1          2.0      2.0     0.0%  results["processing_time"] = time.time() - start_t
   244            1          1.0      1.0     0.0%  results["processing_stats"] = processing_result
   247            1          1.0      1.0     0.0%  start_time = time.time()
   248            1    1427694.0 1427694.0     8.0%  fib_recursive = fibonacci_recursive(25)  # Small e
   249            1          2.0      2.0     0.0%  results["fib_recursive_time"] = time.time() - star
   251            1          1.0      1.0     0.0%  start_time = time.time()
   252            1        117.0    117.0     0.0%  fib_iterative = fibonacci_iterative(25)
   253            1          1.0      1.0     0.0%  results["fib_iterative_time"] = time.time() - star
   255            1          2.0      2.0     0.0%  results["fib_results_match"] = fib_recursive == fi
   257            1          4.0      4.0     0.0%  return results
```

**Performance Insights:**
- **Line 218**: 80.7% of function time (1 hits)
- **Line 224**: 9.0% of function time (1 hits)
- **Line 248**: 8.0% of function time (1 hits)

#### cpu_intensive_calculation (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    44            1          2.0      2.0     0.0%  print("🔥 Running CPU-intensive calculations...")
    47            1          1.0      1.0     0.0%  size = 200
    48            1      21824.0  21824.0     0.2%  matrix_a = [[random.random() for _ in range(size)]
    49            1      22870.0  22870.0     0.2%  matrix_b = [[random.random() for _ in range(size)]
    51            1          1.0      1.0     0.0%  result = 0.0
    52          201        175.0      0.9     0.0%  for i in range(size):
    53        40200      31369.0      0.8     0.3%  for j in range(size):
    54      8040000    6172073.0      0.8    49.8%  for k in range(size):
    55      8000000    6134855.0      0.8    49.5%  result += matrix_a[i][k] * matrix_b[k][j]
    58        10001       8029.0      0.8     0.1%  for i in range(10000):
    59        10000       9012.0      0.9     0.1%  result += math.sin(i) * math.cos(i) * math.sqrt(i 
    61            1         11.0     11.0     0.0%  return result
```

**Performance Insights:**
- **Line 54**: 49.8% of function time (8,040,000 hits)
- **Line 55**: 49.5% of function time (8,000,000 hits)
- **Line 53**: 0.3% of function time (40,200 hits)

#### memory_intensive_operations (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    66            1          8.0      8.0     0.0%  print("💾 Running memory-intensive operations...")
    69            1          1.0      1.0     0.0%  large_list = []
    70        50001      39087.0      0.8     2.6%  for i in range(50000):
    71       100000      77968.0      0.8     5.2%  large_list.append(
    72        50000      45199.0      0.9     3.0%  {
    73        50000      36780.0      0.7     2.5%  "id": i,
    74        50000     640160.0     12.8    42.8%  "data": [random.random() for _ in range(20)],
    75        50000      59834.0      1.2     4.0%  "metadata": {
    76        50000      40132.0      0.8     2.7%  "timestamp": time.time(),
    77        50000      42566.0      0.9     2.8%  "category": f"category_{i % 10}",
    78        50000     169902.0      3.4    11.3%  "tags": [f"tag_{j}" for j in range(i % 5 + 1)],
    84            1          1.0      1.0     0.0%  lookup_dict = {}
    85        50001      71867.0      1.4     4.8%  for item in large_list:
    86        50000      71141.0      1.4     4.8%  category = item["metadata"]["category"]
    87        50000      72027.0      1.4     4.8%  if category not in lookup_dict:
    88           10          6.0      0.6     0.0%  lookup_dict[category] = []
    89        50000      69358.0      1.4     4.6%  lookup_dict[category].append(item)
    92            1          1.0      1.0     0.0%  temp_lists = []
    93          101         97.0      1.0     0.0%  for i in range(100):
    94          100      60523.0    605.2     4.0%  temp_list = [random.random() for _ in range(1000)]
    95          100         88.0      0.9     0.0%  temp_lists.append(temp_list)
    96          100         88.0      0.9     0.0%  if len(temp_lists) > 10:
    97           90        438.0      4.9     0.0%  temp_lists.pop(0)  # Remove oldest
    99            1          8.0      8.0     0.0%  return large_list
```

**Performance Insights:**
- **Line 74**: 42.8% of function time (50,000 hits)
- **Line 78**: 11.3% of function time (50,000 hits)
- **Line 71**: 5.2% of function time (100,000 hits)

#### file_io_operations (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   104            1          5.0      5.0     0.0%  print("📁 Running file I/O operations...")
   107            1        642.0    642.0     0.2%  temp_dir = tempfile.mkdtemp()
   109            1          1.0      1.0     0.0%  try:
   111            1          1.0      1.0     0.0%  file_data = {}
   112           11         12.0      1.1     0.0%  for i in range(10):
   113           10        226.0     22.6     0.1%  filename = os.path.join(temp_dir, f"test_file_{i}.
   114           10          8.0      0.8     0.0%  data = {
   115           10         10.0      1.0     0.0%  "file_id": i,
   116           10       5380.0    538.0     1.6%  "content": [random.random() for _ in range(1000)],
   117           10         10.0      1.0     0.0%  "metadata": {"created": time.time()},
   120           20       1848.0     92.4     0.5%  with open(filename, "w") as f:
   121           10     330351.0  33035.1    96.6%  json.dump(data, f)
   123           10         14.0      1.4     0.0%  file_data[filename] = data
   126            1          1.0      1.0     0.0%  read_data = {}
   127           11         14.0      1.3     0.0%  for filename in file_data:
   128           20        257.0     12.8     0.1%  with open(filename, "r") as f:
   129           10       2461.0    246.1     0.7%  read_data[filename] = json.load(f)
   132            1          1.0      1.0     0.0%  total_values = 0
   133           11          9.0      0.8     0.0%  for data in read_data.values():
   134           10          9.0      0.9     0.0%  total_values += len(data["content"])
   136            1          1.0      1.0     0.0%  return {"files_processed": len(read_data), "total_
   140           11         39.0      3.5     0.0%  for filename in os.listdir(temp_dir):
   141           10        544.0     54.4     0.2%  os.remove(os.path.join(temp_dir, filename))
   142            1         45.0     45.0     0.0%  os.rmdir(temp_dir)
```

**Performance Insights:**
- **Line 121**: 96.6% of function time (10 hits)
- **Line 116**: 1.6% of function time (10 hits)
- **Line 129**: 0.7% of function time (10 hits)

#### dump (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   165           10         10.0      1.0     0.0%  if (not skipkeys and ensure_ascii and
   166           10         10.0      1.0     0.0%  check_circular and allow_nan and
   167           10         10.0      1.0     0.0%  cls is None and indent is None and separators is N
   168           10          7.0      0.7     0.0%  default is None and not sort_keys and not kw):
   169           10        421.0     42.1     0.1%  iterable = _default_encoder.iterencode(obj)
   179        10180     316288.0     31.1    96.9%  for chunk in iterable:
   180        10170       9677.0      1.0     3.0%  fp.write(chunk)
```

**Performance Insights:**
- **Line 179**: 96.9% of function time (10,180 hits)
- **Line 180**: 3.0% of function time (10,170 hits)
- **Line 169**: 0.1% of function time (10 hits)

#### _make_iterencode (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   275           10         13.0      1.3    22.0%  if _indent is not None and not isinstance(_indent,
   278           10         10.0      1.0    16.9%  def _iterencode_list(lst, _current_indent_level):
   334           10         11.0      1.1    18.6%  def _iterencode_dict(dct, _current_indent_level):
   414           10         11.0      1.1    18.6%  def _iterencode(o, _current_indent_level):
   443           10         14.0      1.4    23.7%  return _iterencode
```

**Performance Insights:**
- **Line 443**: 23.7% of function time (10 hits)
- **Line 275**: 22.0% of function time (10 hits)
- **Line 334**: 18.6% of function time (10 hits)

#### load (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   293           30       2317.0     77.2    98.3%  return loads(fp.read(),
   294           10         10.0      1.0     0.4%  cls=cls, object_hook=object_hook,
   295           10         10.0      1.0     0.4%  parse_float=parse_float, parse_int=parse_int,
   296           20         19.0      0.9     0.8%  parse_constant=parse_constant, object_pairs_hook=o
```

**Performance Insights:**
- **Line 293**: 98.3% of function time (30 hits)
- **Line 296**: 0.8% of function time (20 hits)
- **Line 294**: 0.4% of function time (10 hits)

#### loads (/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   333           10          9.0      0.9     0.5%  if isinstance(s, str):
   334           10          8.0      0.8     0.4%  if s.startswith('\ufeff'):
   343           10          8.0      0.8     0.4%  if (cls is None and object_hook is None and
   344           10          7.0      0.7     0.4%  parse_int is None and parse_float is None and
   345           10          6.0      0.6     0.3%  parse_constant is None and object_pairs_hook is No
   346           10       1924.0    192.4    98.1%  return _default_decoder.decode(s)
```

**Performance Insights:**
- **Line 346**: 98.1% of function time (10 hits)
- **Line 333**: 0.5% of function time (10 hits)
- **Line 334**: 0.4% of function time (10 hits)

#### nested_function_calls (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   147            1          3.0      3.0     4.5%  print("🌳 Running nested function calls...")
   149            1          1.0      1.0     1.5%  def level_1(n: int) -> int:
   154            1          1.0      1.0     1.5%  def level_2(n: int) -> int:
   159            1          1.0      1.0     1.5%  def level_3(n: int) -> int:
   164            1          1.0      1.0     1.5%  def level_4(n: int) -> int:
   169            1          1.0      1.0     1.5%  def level_5(n: int) -> int:
   172            1         59.0     59.0    88.1%  return level_1(8)
```

**Performance Insights:**
- **Line 172**: 88.1% of function time (1 hits)
- **Line 147**: 4.5% of function time (1 hits)
- **Line 149**: 1.5% of function time (1 hits)

#### data_processing_pipeline (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
   177            1          3.0      3.0     0.0%  print("⚙️  Running data processing pipeline...")
   180            1       5865.0   5865.0    10.0%  raw_data = [random.random() * 1000 for _ in range(
   183            1       5577.0   5577.0     9.5%  filtered_data = [x for x in raw_data if x > 100]
   186            1       5401.0   5401.0     9.2%  transformed_data = [math.log(x) if x > 1 else 0 fo
   189            1         26.0     26.0     0.0%  sum_data = sum(transformed_data)
   190            1          1.0      1.0     0.0%  avg_data = sum_data / len(transformed_data) if tra
   191            1         69.0     69.0     0.1%  max_data = max(transformed_data) if transformed_da
   192            1         62.0     62.0     0.1%  min_data = min(transformed_data) if transformed_da
   195            2      41534.0  20767.0    70.9%  variance = sum((x - avg_data) ** 2 for x in transf
   196            1          1.0      1.0     0.0%  transformed_data
   198            1          2.0      2.0     0.0%  std_dev = math.sqrt(variance)
   200            1          2.0      2.0     0.0%  return {
   201            1          1.0      1.0     0.0%  "count": len(transformed_data),
   202            1          0.0      0.0     0.0%  "sum": sum_data,
   203            1          0.0      0.0     0.0%  "average": avg_data,
   204            1          0.0      0.0     0.0%  "maximum": max_data,
   205            1          0.0      0.0     0.0%  "minimum": min_data,
   206            1          0.0      0.0     0.0%  "std_dev": std_dev,
```

**Performance Insights:**
- **Line 195**: 70.9% of function time (2 hits)
- **Line 180**: 10.0% of function time (1 hits)
- **Line 183**: 9.5% of function time (1 hits)

#### fibonacci_recursive (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    27       242785     186491.0      0.8    24.2%  if n <= 1:
    28       121393     139553.0      1.1    18.1%  return n
    29       121392     444928.0      3.7    57.7%  return fibonacci_recursive(n - 1) + fibonacci_recu
```

**Performance Insights:**
- **Line 29**: 57.7% of function time (121,392 hits)
- **Line 27**: 24.2% of function time (242,785 hits)
- **Line 28**: 18.1% of function time (121,393 hits)

#### fibonacci_iterative (/Users/Adam/Pycroscope/docs/examples/sample_workload.py)

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================================
    34            1          1.0      1.0     2.3%  if n <= 1:
    36            1          2.0      2.0     4.7%  a, b = 0, 1
    37           25         21.0      0.8    48.8%  for _ in range(2, n + 1):
    38           24         18.0      0.8    41.9%  a, b = b, a + b
    39            1          1.0      1.0     2.3%  return b
```

**Performance Insights:**
- **Line 37**: 48.8% of function time (25 hits)
- **Line 38**: 41.9% of function time (24 hits)
- **Line 36**: 4.7% of function time (1 hits)

### 📊 Line Profiling Summary

- **Total Lines Profiled:** 217
- **Total Hits:** 17,307,497
- **Total Time:** 51.013884 seconds
- **Average Time per Hit:** 0.000002948 seconds
## 📈 Memory Profiler Analysis

### 🧠 Memory Usage Statistics

| Metric | Value |
|--------|-------|
| Peak Memory Usage | 187.97 MB |
| Average Memory Usage | 125.33 MB |
| Memory Delta | +69.86 MB |
| Sample Count | 1006 |
| Initial Memory | 105.95 MB |
| Final Memory | 175.81 MB |

### 📊 Memory Timeline Analysis

- **Memory Growth Rate:** 3.8937 MB/second
- **Memory Spikes Detected:** 0 (>187.99 MB)

## 📈 Call Profiler Analysis

### 🕒 Top Functions by Execution Time

| Function | Total Time | Calls | Time/Call | Cumulative |
|----------|------------|-------|-----------|------------|
| `/Users/Adam/Pycroscope/docs/examples/usage_example.py:0(run_sample_workload)` | 17.8128s | 1 | 17.812818s | 17.8128s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(mixed_workload)` | 17.7997s | 1 | 17.799707s | 17.7997s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(cpu_intensive_calculation)` | 14.3658s | 1 | 14.365796s | 14.3658s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(memory_intensive_operations)` | 1.6034s | 1 | 1.603447s | 1.6034s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(file_io_operations)` | 0.3419s | 1 | 0.341932s | 0.3419s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(fibonacci_recursive)` | 0.3361s | 242785 | 0.000001s | 0.3361s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/__init__.py:0(dump)` | 0.3302s | 10 | 0.033022s | 0.3302s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode)` | 0.2679s | 10180 | 0.000026s | 0.2679s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode_dict)` | 0.2154s | 10240 | 0.000021s | 0.2154s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode_list)` | 0.1670s | 10020 | 0.000017s | 0.1670s |

### 📞 Most Called Functions

| Function | Calls | Total Time | Avg Time |
|----------|-------|------------|----------|
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(fibonacci_recursive)` | 242785 | 0.3361s | 0.000001s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode_dict)` | 10240 | 0.2154s | 0.000021s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode)` | 10180 | 0.2679s | 0.000026s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(_iterencode_list)` | 10020 | 0.1670s | 0.000017s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/json/encoder.py:0(floatstr)` | 10010 | 0.0440s | 0.000004s |
| `/Users/Adam/Pycroscope/docs/examples/sample_workload.py:0(<genexpr>)` | 9024 | 0.0141s | 0.000002s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/tokenize.py:0(_generate_tokens_from_c_tokenizer)` | 4885 | 0.0242s | 0.000005s |
| `/Users/Adam/Pycroscope/venv/lib/python3.12/site-packages/line_profiler/line_profiler.py:0(tokeneater)` | 4857 | 0.0296s | 0.000006s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/inspect.py:0(tokeneater)` | 4857 | 0.0169s | 0.000003s |
| `/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/collections/__init__.py:0(_make)` | 4857 | 0.0107s | 0.000002s |

**Total Functions Profiled:** 50

## 🎯 Performance Insights

- Long execution time detected (17.946s) - consider optimization
- High function call count (314,728) - potential optimization opportunity

## 🔧 Technical Details

### Session Metadata

- **Start Time:** 2025-07-26 15:02:14.227348
- **End Time:** 2025-07-26 15:02:32.173031
- **Output Directory:** `profiling_results`
- **Session Name:** Default

