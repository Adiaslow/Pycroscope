"""
Interfaces for performance anti-pattern analysis.

Defines protocols and abstract base classes for pattern detection and analysis
following SOLID principles and integrating with Pycroscope's architecture.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Protocol, Union
from dataclasses import dataclass
from pathlib import Path


class PatternType(Enum):
    """Types of patterns that can be detected."""

    # Performance Anti-Patterns
    NESTED_LOOPS = "nested_loops"
    INEFFICIENT_DATA_STRUCTURE = "inefficient_data_structure"
    UNNECESSARY_COMPUTATION = "unnecessary_computation"
    MEMORY_LEAK_PATTERN = "memory_leak_pattern"

    # Algorithmic Complexity Issues
    QUADRATIC_COMPLEXITY = "quadratic_complexity"
    EXPONENTIAL_COMPLEXITY = "exponential_complexity"
    RECURSIVE_WITHOUT_MEMOIZATION = "recursive_without_memoization"

    # Code Quality Issues
    DEAD_CODE = "dead_code"
    UNUSED_IMPORTS = "unused_imports"
    DUPLICATE_CODE = "duplicate_code"
    HIGH_CYCLOMATIC_COMPLEXITY = "high_cyclomatic_complexity"

    # Maintainability Issues
    LOW_MAINTAINABILITY_INDEX = "low_maintainability_index"
    LONG_FUNCTION = "long_function"
    TOO_MANY_PARAMETERS = "too_many_parameters"

    # Scientific Computing Anti-Patterns (Sequential CPU Focus)
    MISSED_VECTORIZATION = "missed_vectorization"
    INEFFICIENT_ARRAY_OPERATIONS = "inefficient_array_operations"
    SUBOPTIMAL_MATRIX_OPERATIONS = "suboptimal_matrix_operations"
    NON_CONTIGUOUS_MEMORY_ACCESS = "non_contiguous_memory_access"
    UNNECESSARY_ARRAY_COPY = "unnecessary_array_copy"
    INEFFICIENT_BROADCASTING = "inefficient_broadcasting"
    SCALAR_ARRAY_OPERATIONS = "scalar_array_operations"
    WRONG_DTYPE_USAGE = "wrong_dtype_usage"
    INEFFICIENT_ARRAY_CONCATENATION = "inefficient_array_concatenation"
    SUBOPTIMAL_LINEAR_ALGEBRA = "suboptimal_linear_algebra"
    INEFFICIENT_MEMORY_LAYOUT = "inefficient_memory_layout"
    SUBOPTIMAL_ALGORITHM_CHOICE = "suboptimal_algorithm_choice"

    # General Performance Anti-Patterns (from perflint and others)
    UNNECESSARY_LIST_CAST = "unnecessary_list_cast"
    INCORRECT_DICTIONARY_ITERATOR = "incorrect_dictionary_iterator"
    LOOP_INVARIANT_STATEMENT = "loop_invariant_statement"
    GLOBAL_NAME_IN_LOOP = "global_name_in_loop"
    TRY_EXCEPT_IN_LOOP = "try_except_in_loop"
    INEFFICIENT_BYTES_SLICING = "inefficient_bytes_slicing"
    DOTTED_IMPORT_IN_LOOP = "dotted_import_in_loop"
    USE_TUPLE_OVER_LIST = "use_tuple_over_list"
    USE_LIST_COMPREHENSION = "use_list_comprehension"
    USE_LIST_COPY = "use_list_copy"
    USE_DICT_COMPREHENSION = "use_dict_comprehension"

    # Data Structure & Memory Patterns
    PYTHON_LIST_VS_NUMPY_ARRAY = "python_list_vs_numpy_array"
    INEFFICIENT_STRING_OPERATIONS = "inefficient_string_operations"
    CACHE_INEFFICIENT_ACCESS = "cache_inefficient_access"
    WRONG_AXIS_SPECIFICATION = "wrong_axis_specification"
    INEFFICIENT_INDEXING_PATTERNS = "inefficient_indexing_patterns"
    SUBOPTIMAL_AGGREGATION = "suboptimal_aggregation"


@dataclass
class AnalysisResult:
    """Result of pattern analysis."""

    pattern_type: PatternType
    severity: str  # "low", "medium", "high", "critical"
    confidence: float  # 0.0 to 1.0
    description: str
    location: Dict[str, Any]  # file, line, function info
    suggestion: str
    metrics: Dict[str, Any]  # relevant metrics (complexity, time, etc.)
    profiling_correlation: Optional[Dict[str, Any]] = (
        None  # correlation with profiling data
    )


class PatternAnalyzer(Protocol):
    """Protocol for pattern analyzers."""

    @property
    def analyzer_type(self) -> str:
        """Get the type of analyzer."""
        ...

    @property
    def supported_patterns(self) -> List[PatternType]:
        """Get list of patterns this analyzer can detect."""
        ...

    def analyze(self, code: str, file_path: Path, **kwargs) -> List[AnalysisResult]:
        """Analyze code for patterns."""
        ...

    def analyze_with_profiling_data(
        self, code: str, file_path: Path, profiling_data: Dict[str, Any], **kwargs
    ) -> List[AnalysisResult]:
        """Analyze code with correlation to profiling data."""
        ...


class ComplexityAnalyzer(Protocol):
    """Protocol for algorithmic complexity analysis."""

    def analyze_time_complexity(self, code: str, file_path: Path) -> Dict[str, Any]:
        """Analyze time complexity of functions in code."""
        ...

    def analyze_space_complexity(self, code: str, file_path: Path) -> Dict[str, Any]:
        """Analyze space complexity of functions in code."""
        ...

    def get_complexity_metrics(self, code: str, file_path: Path) -> Dict[str, Any]:
        """Get detailed complexity metrics."""
        ...


class AntiPatternDetector(Protocol):
    """Protocol for anti-pattern detection."""

    def detect_performance_antipatterns(
        self, code: str, file_path: Path
    ) -> List[AnalysisResult]:
        """Detect performance anti-patterns in code."""
        ...

    def detect_maintainability_issues(
        self, code: str, file_path: Path
    ) -> List[AnalysisResult]:
        """Detect maintainability issues in code."""
        ...

    def correlate_with_hotspots(
        self, patterns: List[AnalysisResult], profiling_data: Dict[str, Any]
    ) -> List[AnalysisResult]:
        """Correlate detected patterns with profiling hotspots."""
        ...


class AnalysisOrchestrator(ABC):
    """Abstract orchestrator for running multiple analyzers."""

    @abstractmethod
    def register_analyzer(self, analyzer: PatternAnalyzer) -> None:
        """Register a pattern analyzer."""
        pass

    @abstractmethod
    def run_analysis(
        self, code_files: List[Path], profiling_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[AnalysisResult]]:
        """Run analysis on multiple files."""
        pass

    @abstractmethod
    def generate_report(
        self, results: Dict[str, List[AnalysisResult]]
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        pass

    @abstractmethod
    def prioritize_findings(
        self, results: List[AnalysisResult]
    ) -> List[AnalysisResult]:
        """Prioritize findings by severity and impact."""
        pass


class AnalysisConfig(Protocol):
    """Protocol for analysis configuration."""

    @property
    def enabled_patterns(self) -> List[PatternType]:
        """Get list of enabled pattern types."""
        ...

    @property
    def severity_threshold(self) -> str:
        """Get minimum severity threshold for reporting."""
        ...

    @property
    def confidence_threshold(self) -> float:
        """Get minimum confidence threshold for reporting."""
        ...

    @property
    def max_results_per_file(self) -> int:
        """Get maximum number of results per file."""
        ...

    @property
    def correlate_with_profiling(self) -> bool:
        """Whether to correlate patterns with profiling data."""
        ...
