"""
Optimization recommendations engine for Pycroscope.

Generates concrete, actionable optimization recommendations based on
detected patterns from all analyzers and collectors.
"""

import re
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .base_analyzer import BaseAnalyzer
from ..core.models import (
    ProfileSession,
    AnalysisResult,
    DetectedPattern,
    OptimizationRecommendation,
    SourceLocation,
    StaticAnalysisResult,
    DynamicAnalysisResult,
)


class OptimizationCategory(Enum):
    """Categories of optimization recommendations."""

    ALGORITHM = "algorithm"
    MEMORY = "memory"
    IO = "io"
    CPU = "cpu"
    CONCURRENCY = "concurrency"
    CACHING = "caching"
    DATA_STRUCTURE = "data_structure"
    EXCEPTION_HANDLING = "exception_handling"
    IMPORT_OPTIMIZATION = "import_optimization"
    GC_TUNING = "gc_tuning"
    ARCHITECTURE = "architecture"


class ImplementationEffort(Enum):
    """Implementation effort levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OptimizationStrategy:
    """Complete optimization strategy for a specific issue."""

    primary_recommendation: OptimizationRecommendation
    supporting_recommendations: List[OptimizationRecommendation]
    implementation_order: List[str]
    expected_total_improvement: float
    risk_assessment: str


class OptimizationRecommendationEngine(BaseAnalyzer):
    """
    Optimization recommendations engine.

    Analyzes detected patterns and generates comprehensive, prioritized
    optimization recommendations with implementation guidance.
    """

    def __init__(self, config=None):
        """
        Initialize the optimization engine.

        Args:
            config: Optional analysis configuration
        """
        super().__init__(config)

        # Recommendation templates and strategies
        self.recommendation_templates = self._initialize_recommendation_templates()
        self.optimization_strategies = self._initialize_optimization_strategies()

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer type."""
        return "optimization"

    @property
    def dependencies(self) -> List[str]:
        """List of collector names required by this analyzer."""
        return []  # Depends on patterns from other analyzers, not collectors directly

    def _perform_analysis(self, profile_data: ProfileSession) -> AnalysisResult:
        """
        Generate optimization recommendations.

        Args:
            profile_data: Profiling session data

        Returns:
            Analysis results with optimization recommendations
        """
        # Extract patterns from previous analysis results
        patterns = (
            profile_data.analysis_result.detected_patterns
            if profile_data.analysis_result
            else []
        )

        # Generate recommendations based on patterns
        recommendations = self._generate_comprehensive_recommendations(
            patterns, profile_data
        )

        # Create optimization strategies
        strategies = self._create_optimization_strategies(recommendations)

        # Prioritize recommendations
        prioritized_recommendations = self._prioritize_recommendations(
            recommendations, profile_data
        )

        # Generate implementation roadmap
        implementation_roadmap = self._generate_implementation_roadmap(strategies)

        # Calculate optimization potential score
        optimization_score = self._calculate_optimization_potential(recommendations)

        return AnalysisResult(
            session_id=profile_data.session_id,
            analysis_timestamp=datetime.now(),
            static_analysis=StaticAnalysisResult(),
            dynamic_analysis=DynamicAnalysisResult(),
            detected_patterns=[],  # This analyzer doesn't detect new patterns
            recommendations=prioritized_recommendations,
            overall_score=optimization_score,
            performance_grade=self._grade_from_score(optimization_score),
        )

    def _initialize_recommendation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize recommendation templates for different pattern types."""
        return {
            # Algorithm optimization templates
            "nested_loops": {
                "category": OptimizationCategory.ALGORITHM,
                "title": "Optimize Nested Loop Performance",
                "description_template": "Nested loops detected with O(n²) complexity - consider algorithmic improvements",
                "actions": [
                    "Replace nested iteration with hash-based lookups",
                    "Use set operations for intersection/union operations",
                    "Consider sorting + two-pointer technique for ordered data",
                    "Implement early termination conditions where possible",
                ],
                "code_examples": [
                    "# Instead of nested loops:\n# for x in list1:\n#     for y in list2:\n#         if condition(x, y): ...\n\n# Use hash map:\nlookup = {item: index for index, item in enumerate(list2)}\nfor x in list1:\n    if x in lookup: ...",
                    "# For set operations:\nresult = set(list1) & set(list2)  # Intersection\nresult = set(list1) | set(list2)  # Union",
                ],
                "effort": ImplementationEffort.MEDIUM,
                "improvement_range": (0.5, 0.9),
            },
            "inefficient_search": {
                "category": OptimizationCategory.DATA_STRUCTURE,
                "title": "Replace Linear Search with Efficient Lookup",
                "description_template": "Linear search patterns detected - use appropriate data structures",
                "actions": [
                    "Replace list searches with set/dict lookups",
                    "Use binary search for sorted data",
                    "Implement indexing for frequently searched data",
                    "Consider using databases for large datasets",
                ],
                "code_examples": [
                    "# Instead of: if item in large_list\n# Use: lookup_set = set(large_list)\n#      if item in lookup_set",
                    "# For sorted data:\nimport bisect\nindex = bisect.bisect_left(sorted_list, target)",
                ],
                "effort": ImplementationEffort.LOW,
                "improvement_range": (0.6, 0.95),
            },
            # Memory optimization templates
            "memory_leak": {
                "category": OptimizationCategory.MEMORY,
                "title": "Fix Memory Leak",
                "description_template": "Memory leak detected - implement proper resource management",
                "actions": [
                    "Identify and fix circular references",
                    "Implement proper cleanup in __del__ methods",
                    "Use weak references where appropriate",
                    "Add explicit resource disposal",
                    "Use context managers for resource management",
                ],
                "code_examples": [
                    "# Use weak references to break cycles:\nimport weakref\nself.parent = weakref.ref(parent)",
                    "# Context managers for cleanup:\nwith resource_manager() as resource:\n    # Use resource\n    pass  # Automatic cleanup",
                ],
                "effort": ImplementationEffort.HIGH,
                "improvement_range": (0.7, 1.0),
            },
            "excessive_allocations": {
                "category": OptimizationCategory.MEMORY,
                "title": "Reduce Memory Allocations",
                "description_template": "Excessive memory allocations detected - implement object pooling or reuse",
                "actions": [
                    "Implement object pooling for frequently created objects",
                    "Reuse containers instead of creating new ones",
                    "Use generators instead of lists where possible",
                    "Pre-allocate large data structures",
                    "Use __slots__ to reduce object memory overhead",
                ],
                "code_examples": [
                    "# Object pooling:\nclass ObjectPool:\n    def __init__(self):\n        self._pool = []\n    \n    def get(self):\n        return self._pool.pop() if self._pool else ExpensiveObject()\n    \n    def return_object(self, obj):\n        obj.reset()\n        self._pool.append(obj)",
                    "# Use __slots__ for memory efficiency:\nclass Point:\n    __slots__ = ['x', 'y']\n    def __init__(self, x, y):\n        self.x, self.y = x, y",
                ],
                "effort": ImplementationEffort.MEDIUM,
                "improvement_range": (0.4, 0.8),
            },
            # I/O optimization templates
            "io_bottleneck": {
                "category": OptimizationCategory.IO,
                "title": "Optimize I/O Operations",
                "description_template": "I/O bottleneck detected - implement asynchronous or batched operations",
                "actions": [
                    "Use asynchronous I/O for concurrent operations",
                    "Implement read/write batching",
                    "Add connection pooling for network operations",
                    "Use memory mapping for large files",
                    "Cache frequently accessed data",
                ],
                "code_examples": [
                    "# Async I/O:\nimport asyncio\nimport aiofiles\n\nasync def read_files(filenames):\n    tasks = [read_file_async(f) for f in filenames]\n    return await asyncio.gather(*tasks)",
                    "# Batched writes:\ndef write_batch(data_items, batch_size=1000):\n    for i in range(0, len(data_items), batch_size):\n        batch = data_items[i:i+batch_size]\n        write_to_file(batch)",
                ],
                "effort": ImplementationEffort.HIGH,
                "improvement_range": (0.6, 0.9),
            },
            "small_io_operations": {
                "category": OptimizationCategory.IO,
                "title": "Batch Small I/O Operations",
                "description_template": "Many small I/O operations detected - implement batching",
                "actions": [
                    "Batch multiple small operations together",
                    "Use buffered I/O streams",
                    "Implement write-behind caching",
                    "Aggregate operations before executing",
                ],
                "code_examples": [
                    "# Buffered writing:\nwith open(file, 'w', buffering=8192) as f:\n    for item in data:\n        f.write(f'{item}\\n')",
                    "# Operation batching:\nbatch = []\nfor operation in operations:\n    batch.append(operation)\n    if len(batch) >= BATCH_SIZE:\n        execute_batch(batch)\n        batch.clear()",
                ],
                "effort": ImplementationEffort.LOW,
                "improvement_range": (0.3, 0.7),
            },
            # Exception handling templates
            "exception_hotspot": {
                "category": OptimizationCategory.EXCEPTION_HANDLING,
                "title": "Optimize Exception-Heavy Code",
                "description_template": "Exception hotspot detected - reduce exception-based control flow",
                "actions": [
                    "Replace exception-based logic with explicit checks",
                    "Use EAFP (Easier to Ask for Forgiveness than Permission) judiciously",
                    "Implement proper validation before operations",
                    "Cache validation results where possible",
                ],
                "code_examples": [
                    "# Instead of:\n# try:\n#     value = dictionary[key]\n# except KeyError:\n#     value = default\n\n# Use:\nvalue = dictionary.get(key, default)",
                    "# For file operations:\nif os.path.exists(filename):\n    with open(filename) as f:\n        # Process file\nelse:\n    # Handle missing file",
                ],
                "effort": ImplementationEffort.LOW,
                "improvement_range": (0.2, 0.6),
            },
            # CPU optimization templates
            "cpu_hotspot": {
                "category": OptimizationCategory.CPU,
                "title": "Optimize CPU-Intensive Function",
                "description_template": "CPU hotspot detected - optimize computational efficiency",
                "actions": [
                    "Profile function internals for specific bottlenecks",
                    "Use more efficient algorithms",
                    "Implement caching for expensive computations",
                    "Consider using NumPy for numerical operations",
                    "Parallelize independent computations",
                ],
                "code_examples": [
                    "# Caching expensive computations:\nfrom functools import lru_cache\n\n@lru_cache(maxsize=1000)\ndef expensive_computation(input_data):\n    # Expensive operation here\n    return result",
                    "# NumPy for numerical operations:\nimport numpy as np\n# Instead of Python loops:\n# result = [x**2 + 2*x + 1 for x in data]\n# Use NumPy:\ndata = np.array(data)\nresult = data**2 + 2*data + 1",
                ],
                "effort": ImplementationEffort.MEDIUM,
                "improvement_range": (0.4, 0.8),
            },
            # Import optimization templates
            "slow_imports": {
                "category": OptimizationCategory.IMPORT_OPTIMIZATION,
                "title": "Optimize Slow Module Imports",
                "description_template": "Slow import operations detected - implement lazy loading",
                "actions": [
                    "Use lazy imports for heavy modules",
                    "Move imports closer to usage points",
                    "Use importlib for conditional imports",
                    "Consider module restructuring to reduce dependencies",
                ],
                "code_examples": [
                    "# Lazy import:\ndef get_heavy_module():\n    global _heavy_module\n    if '_heavy_module' not in globals():\n        import heavy_module\n        _heavy_module = heavy_module\n    return _heavy_module",
                    "# Conditional import:\ndef process_data(data, use_advanced=False):\n    if use_advanced:\n        import advanced_processor\n        return advanced_processor.process(data)\n    else:\n        return simple_process(data)",
                ],
                "effort": ImplementationEffort.LOW,
                "improvement_range": (0.3, 0.7),
            },
            # GC optimization templates
            "frequent_gc": {
                "category": OptimizationCategory.GC_TUNING,
                "title": "Optimize Garbage Collection Frequency",
                "description_template": "Frequent garbage collection detected - tune memory allocation patterns",
                "actions": [
                    "Increase GC thresholds for generation 0",
                    "Reduce object creation rate",
                    "Use object pooling for frequently created objects",
                    "Optimize data structures to create fewer intermediate objects",
                ],
                "code_examples": [
                    "# Tune GC thresholds:\nimport gc\n# Default: gc.set_threshold(700, 10, 10)\n# More lenient: \ngc.set_threshold(1000, 15, 15)",
                    "# Reduce object creation:\n# Instead of: result = sum([x*2 for x in data])\n# Use generator: result = sum(x*2 for x in data)",
                ],
                "effort": ImplementationEffort.LOW,
                "improvement_range": (0.2, 0.5),
            },
        }

    def _initialize_optimization_strategies(self) -> Dict[str, List[str]]:
        """Initialize optimization strategies for pattern combinations."""
        return {
            "memory_pressure": ["memory_leak", "excessive_allocations", "frequent_gc"],
            "algorithmic_inefficiency": [
                "nested_loops",
                "inefficient_search",
                "cpu_hotspot",
            ],
            "io_performance": [
                "io_bottleneck",
                "small_io_operations",
                "synchronous_io_in_loop",
            ],
            "exception_performance": [
                "exception_hotspot",
                "exception_memory_correlation",
            ],
        }

    def _generate_comprehensive_recommendations(
        self, patterns: List[DetectedPattern], profile_data: ProfileSession
    ) -> List[OptimizationRecommendation]:
        """Generate comprehensive optimization recommendations."""
        recommendations = []

        # Group patterns by type for strategic recommendations
        patterns_by_type = defaultdict(list)
        for pattern in patterns:
            patterns_by_type[pattern.pattern_type].append(pattern)

        # Generate recommendations for each pattern type
        for pattern_type, pattern_list in patterns_by_type.items():
            if pattern_type in self.recommendation_templates:
                recommendations.extend(
                    self._generate_pattern_recommendations(
                        pattern_type, pattern_list, profile_data
                    )
                )

        # Generate strategic recommendations for pattern combinations
        strategic_recommendations = self._generate_strategic_recommendations(
            patterns_by_type, profile_data
        )
        recommendations.extend(strategic_recommendations)

        # Generate architecture-level recommendations
        architecture_recommendations = self._generate_architecture_recommendations(
            patterns, profile_data
        )
        recommendations.extend(architecture_recommendations)

        return recommendations

    def _generate_pattern_recommendations(
        self,
        pattern_type: str,
        patterns: List[DetectedPattern],
        profile_data: ProfileSession,
    ) -> List[OptimizationRecommendation]:
        """Generate recommendations for a specific pattern type."""
        template = self.recommendation_templates[pattern_type]
        recommendations = []

        # Aggregate impact from all patterns of this type
        total_impact = sum(p.impact_estimate for p in patterns)
        avg_confidence = sum(self._extract_confidence(p) for p in patterns) / len(
            patterns
        )

        # Customize recommendation based on aggregated data
        customized_actions = self._customize_actions(
            template["actions"], patterns, profile_data
        )
        customized_examples = self._customize_code_examples(
            template["code_examples"], patterns
        )

        # Calculate estimated improvement
        min_improvement, max_improvement = template["improvement_range"]
        estimated_improvement = (
            min_improvement + (max_improvement - min_improvement) * total_impact
        )

        recommendation = OptimizationRecommendation(
            category=template["category"].value,
            title=template["title"],
            description=self._customize_description(
                template["description_template"], patterns
            ),
            estimated_improvement=estimated_improvement,
            confidence=avg_confidence,
            implementation_effort=template["effort"].value,
            suggested_actions=customized_actions,
            code_examples=customized_examples,
        )

        recommendations.append(recommendation)
        return recommendations

    def _generate_strategic_recommendations(
        self,
        patterns_by_type: Dict[str, List[DetectedPattern]],
        profile_data: ProfileSession,
    ) -> List[OptimizationRecommendation]:
        """Generate strategic recommendations for pattern combinations."""
        recommendations = []

        # Check for memory pressure strategy
        memory_patterns = set(patterns_by_type.keys()) & {
            "memory_leak",
            "excessive_allocations",
            "frequent_gc",
        }
        if len(memory_patterns) >= 2:
            recommendations.append(
                self._create_memory_strategy_recommendation(patterns_by_type)
            )

        # Check for algorithmic strategy
        algo_patterns = set(patterns_by_type.keys()) & {
            "nested_loops",
            "inefficient_search",
            "cpu_hotspot",
        }
        if len(algo_patterns) >= 2:
            recommendations.append(
                self._create_algorithmic_strategy_recommendation(patterns_by_type)
            )

        # Check for I/O strategy
        io_patterns = set(patterns_by_type.keys()) & {
            "io_bottleneck",
            "small_io_operations",
            "synchronous_io_in_loop",
        }
        if len(io_patterns) >= 2:
            recommendations.append(
                self._create_io_strategy_recommendation(patterns_by_type)
            )

        return recommendations

    def _generate_architecture_recommendations(
        self, patterns: List[DetectedPattern], profile_data: ProfileSession
    ) -> List[OptimizationRecommendation]:
        """Generate architecture-level optimization recommendations."""
        recommendations = []

        # Count high-impact patterns
        high_impact_patterns = [p for p in patterns if p.impact_estimate > 0.7]

        if len(high_impact_patterns) > 5:
            recommendations.append(
                OptimizationRecommendation(
                    category=OptimizationCategory.ARCHITECTURE.value,
                    title="Consider Architecture Refactoring",
                    description="Multiple high-impact performance issues suggest architectural improvements needed",
                    estimated_improvement=0.8,
                    confidence=0.7,
                    implementation_effort=ImplementationEffort.HIGH.value,
                    suggested_actions=[
                        "Conduct comprehensive architecture review",
                        "Consider microservices for independent scaling",
                        "Implement caching layers",
                        "Design for horizontal scaling",
                        "Review data flow and processing pipelines",
                    ],
                    code_examples=[
                        "# Implement caching layer:\nfrom functools import lru_cache\nimport redis\n\n# Memory cache for fast access\n@lru_cache(maxsize=1000)\ndef get_frequent_data(key):\n    # Fetch from database\n    pass\n\n# Redis for shared cache\nredis_client = redis.Redis()\ndef get_cached_data(key):\n    cached = redis_client.get(key)\n    if cached:\n        return json.loads(cached)\n    data = fetch_from_db(key)\n    redis_client.setex(key, 3600, json.dumps(data))\n    return data"
                    ],
                )
            )

        return recommendations

    def _create_memory_strategy_recommendation(
        self, patterns_by_type: Dict[str, List[DetectedPattern]]
    ) -> OptimizationRecommendation:
        """Create comprehensive memory optimization strategy."""
        return OptimizationRecommendation(
            category=OptimizationCategory.MEMORY.value,
            title="Comprehensive Memory Optimization Strategy",
            description="Multiple memory-related issues detected - implement holistic memory management",
            estimated_improvement=0.8,
            confidence=0.9,
            implementation_effort=ImplementationEffort.HIGH.value,
            suggested_actions=[
                "Implement application-wide object pooling",
                "Add comprehensive memory monitoring",
                "Design memory-efficient data structures",
                "Implement proper resource lifecycle management",
                "Tune garbage collection parameters",
                "Add memory usage alerts and limits",
            ],
            code_examples=[
                "# Comprehensive memory manager:\nclass MemoryManager:\n    def __init__(self):\n        self.pools = {}\n        self.monitors = []\n    \n    def get_pool(self, object_type):\n        if object_type not in self.pools:\n            self.pools[object_type] = ObjectPool(object_type)\n        return self.pools[object_type]\n    \n    def monitor_usage(self):\n        import psutil\n        memory_percent = psutil.virtual_memory().percent\n        if memory_percent > 80:\n            self.trigger_cleanup()\n            \n    def trigger_cleanup(self):\n        import gc\n        gc.collect()\n        for pool in self.pools.values():\n            pool.cleanup_unused()"
            ],
        )

    def _create_algorithmic_strategy_recommendation(
        self, patterns_by_type: Dict[str, List[DetectedPattern]]
    ) -> OptimizationRecommendation:
        """Create comprehensive algorithmic optimization strategy."""
        return OptimizationRecommendation(
            category=OptimizationCategory.ALGORITHM.value,
            title="Algorithmic Performance Optimization Strategy",
            description="Multiple algorithmic inefficiencies detected - systematic algorithm review needed",
            estimated_improvement=0.9,
            confidence=0.8,
            implementation_effort=ImplementationEffort.HIGH.value,
            suggested_actions=[
                "Conduct algorithm complexity audit",
                "Replace O(n²) algorithms with O(n log n) alternatives",
                "Implement comprehensive caching strategy",
                "Use appropriate data structures for access patterns",
                "Add algorithmic performance monitoring",
            ],
            code_examples=[
                "# Algorithm optimization framework:\nclass AlgorithmOptimizer:\n    def __init__(self):\n        self.cache = {}\n        self.complexity_monitor = ComplexityMonitor()\n    \n    def optimize_search(self, data, target):\n        # Use appropriate search based on data characteristics\n        if len(data) < 100:\n            return linear_search(data, target)\n        elif self.is_sorted(data):\n            return binary_search(data, target)\n        else:\n            # Build hash index for repeated searches\n            index = self.get_or_build_index(data)\n            return index.get(target)\n    \n    @lru_cache(maxsize=100)\n    def expensive_computation(self, inputs):\n        # Cache expensive operations\n        return compute_result(inputs)"
            ],
        )

    def _create_io_strategy_recommendation(
        self, patterns_by_type: Dict[str, List[DetectedPattern]]
    ) -> OptimizationRecommendation:
        """Create comprehensive I/O optimization strategy."""
        return OptimizationRecommendation(
            category=OptimizationCategory.IO.value,
            title="I/O Performance Optimization Strategy",
            description="Multiple I/O performance issues detected - implement asynchronous I/O architecture",
            estimated_improvement=0.85,
            confidence=0.8,
            implementation_effort=ImplementationEffort.HIGH.value,
            suggested_actions=[
                "Migrate to asynchronous I/O architecture",
                "Implement comprehensive I/O batching",
                "Add connection pooling and resource management",
                "Design I/O caching and buffering strategy",
                "Monitor I/O performance metrics",
            ],
            code_examples=[
                "# Async I/O framework:\nimport asyncio\nimport aiofiles\nfrom aiohttp import ClientSession\n\nclass AsyncIOManager:\n    def __init__(self):\n        self.session_pool = SessionPool()\n        self.file_cache = FileCache()\n    \n    async def batch_file_operations(self, operations):\n        # Batch multiple file operations\n        tasks = []\n        for op in operations:\n            if op.type == 'read':\n                tasks.append(self.read_file_async(op.path))\n            elif op.type == 'write':\n                tasks.append(self.write_file_async(op.path, op.data))\n        \n        return await asyncio.gather(*tasks, return_exceptions=True)\n    \n    async def read_file_async(self, path):\n        # Check cache first\n        if path in self.file_cache:\n            return self.file_cache[path]\n        \n        async with aiofiles.open(path, 'r') as f:\n            content = await f.read()\n            self.file_cache[path] = content\n            return content"
            ],
        )

    def _customize_actions(
        self,
        base_actions: List[str],
        patterns: List[DetectedPattern],
        profile_data: ProfileSession,
    ) -> List[str]:
        """Customize action recommendations based on specific patterns."""
        customized = base_actions.copy()

        # Add pattern-specific actions
        for pattern in patterns:
            if pattern.evidence:
                # Add specific actions based on evidence
                if (
                    "memory_growth_mb" in pattern.evidence
                    and pattern.evidence["memory_growth_mb"] > 100
                ):
                    customized.append(
                        "URGENT: Memory growth >100MB detected - implement immediate memory monitoring"
                    )

                if (
                    "cpu_percentage" in pattern.evidence
                    and pattern.evidence["cpu_percentage"] > 80
                ):
                    customized.append(
                        "HIGH PRIORITY: CPU usage >80% - consider load balancing or optimization"
                    )

                if (
                    "exception_count" in pattern.evidence
                    and pattern.evidence["exception_count"] > 100
                ):
                    customized.append(
                        "Review exception handling logic - high exception frequency detected"
                    )

        return customized

    def _customize_code_examples(
        self, base_examples: List[str], patterns: List[DetectedPattern]
    ) -> List[str]:
        """Customize code examples based on specific pattern characteristics."""
        customized = base_examples.copy()

        # Add pattern-specific examples
        for pattern in patterns:
            if pattern.source_location and pattern.source_location.function_name:
                function_name = pattern.source_location.function_name
                customized.append(
                    f"# Optimize function: {function_name}\n# Review implementation for performance improvements"
                )

        return customized

    def _customize_description(
        self, template: str, patterns: List[DetectedPattern]
    ) -> str:
        """Customize description based on pattern details."""
        description = template

        if patterns:
            # Add specific details from patterns
            total_impact = sum(p.impact_estimate for p in patterns)
            pattern_count = len(patterns)

            description += f" ({pattern_count} instances found, estimated {total_impact:.1f} total impact)"

        return description

    def _prioritize_recommendations(
        self,
        recommendations: List[OptimizationRecommendation],
        profile_data: ProfileSession,
    ) -> List[OptimizationRecommendation]:
        """Prioritize recommendations by impact, confidence, and effort."""

        def priority_score(rec):
            # Calculate priority score: (impact * confidence) / effort_weight
            effort_weights = {
                "low": 1.0,
                "medium": 2.0,
                "high": 4.0,
                "critical": 1.0,  # Critical issues get high priority despite effort
            }

            effort_weight = effort_weights.get(rec.implementation_effort, 2.0)
            return (rec.estimated_improvement * rec.confidence) / effort_weight

        return sorted(recommendations, key=priority_score, reverse=True)

    def _create_optimization_strategies(
        self, recommendations: List[OptimizationRecommendation]
    ) -> List[OptimizationStrategy]:
        """Create comprehensive optimization strategies."""
        strategies = []

        # Group recommendations by category
        by_category = defaultdict(list)
        for rec in recommendations:
            by_category[rec.category].append(rec)

        # Create strategy for each category with multiple recommendations
        for category, recs in by_category.items():
            if len(recs) > 1:
                strategy = self._create_category_strategy(category, recs)
                if strategy:
                    strategies.append(strategy)

        return strategies

    def _create_category_strategy(
        self, category: str, recommendations: List[OptimizationRecommendation]
    ) -> Optional[OptimizationStrategy]:
        """Create optimization strategy for a specific category."""
        if not recommendations:
            return None

        # Sort by priority
        sorted_recs = sorted(
            recommendations,
            key=lambda r: r.estimated_improvement * r.confidence,
            reverse=True,
        )

        primary = sorted_recs[0]
        supporting = sorted_recs[1:]

        # Create implementation order
        implementation_order = self._determine_implementation_order(sorted_recs)

        # Calculate total expected improvement
        total_improvement = min(1.0, sum(r.estimated_improvement for r in sorted_recs))

        # Assess implementation risk
        risk_assessment = self._assess_implementation_risk(sorted_recs)

        return OptimizationStrategy(
            primary_recommendation=primary,
            supporting_recommendations=supporting,
            implementation_order=implementation_order,
            expected_total_improvement=total_improvement,
            risk_assessment=risk_assessment,
        )

    def _determine_implementation_order(
        self, recommendations: List[OptimizationRecommendation]
    ) -> List[str]:
        """Determine optimal implementation order for recommendations."""

        # Sort by effort (low effort first) then by impact
        def order_key(rec):
            effort_order = {
                "low": 0,
                "medium": 1,
                "high": 2,
                "critical": -1,
            }  # Critical first
            return (
                effort_order.get(rec.implementation_effort, 1),
                -rec.estimated_improvement,
            )

        ordered_recs = sorted(recommendations, key=order_key)
        return [rec.title for rec in ordered_recs]

    def _assess_implementation_risk(
        self, recommendations: List[OptimizationRecommendation]
    ) -> str:
        """Assess the implementation risk for a set of recommendations."""
        high_effort_count = sum(
            1
            for r in recommendations
            if r.implementation_effort in ["high", "critical"]
        )
        low_confidence_count = sum(1 for r in recommendations if r.confidence < 0.7)

        if high_effort_count > 2 or low_confidence_count > 1:
            return "high"
        elif high_effort_count > 0 or low_confidence_count > 0:
            return "medium"
        else:
            return "low"

    def _generate_implementation_roadmap(
        self, strategies: List[OptimizationStrategy]
    ) -> Dict[str, Any]:
        """Generate implementation roadmap for optimization strategies."""
        roadmap = {
            "quick_wins": [],
            "medium_term": [],
            "long_term": [],
            "high_impact": [],
        }

        for strategy in strategies:
            primary = strategy.primary_recommendation

            # Categorize based on effort and impact
            if (
                primary.implementation_effort == "low"
                and primary.estimated_improvement > 0.3
            ):
                roadmap["quick_wins"].append(primary.title)
            elif primary.implementation_effort == "medium":
                roadmap["medium_term"].append(primary.title)
            elif primary.implementation_effort in ["high", "critical"]:
                roadmap["long_term"].append(primary.title)

            if primary.estimated_improvement > 0.7:
                roadmap["high_impact"].append(primary.title)

        return roadmap

    def _extract_confidence(self, pattern: DetectedPattern) -> float:
        """Extract confidence score from pattern."""
        if hasattr(pattern, "confidence"):
            return pattern.confidence
        elif pattern.evidence and "confidence" in pattern.evidence:
            return pattern.evidence["confidence"]
        else:
            # Default confidence based on impact and severity
            severity_confidence = {
                "low": 0.6,
                "medium": 0.7,
                "high": 0.8,
                "critical": 0.9,
            }
            return severity_confidence.get(pattern.severity, 0.7)

    def _calculate_optimization_potential(
        self, recommendations: List[OptimizationRecommendation]
    ) -> float:
        """Calculate overall optimization potential score."""
        if not recommendations:
            return 0.8  # Neutral score

        # Calculate weighted average of estimated improvements
        total_weighted_improvement = sum(
            rec.estimated_improvement * rec.confidence for rec in recommendations
        )
        total_weight = sum(rec.confidence for rec in recommendations)

        if total_weight == 0:
            return 0.8

        avg_improvement = total_weighted_improvement / total_weight

        # Score represents how much optimization potential exists
        # Higher potential = lower current score
        return max(0.1, 1.0 - avg_improvement)
