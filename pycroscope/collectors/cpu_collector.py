"""
CPU profiler for Pycroscope.

Monitors CPU usage and instruction-level execution patterns to identify
computational bottlenecks and optimization opportunities.
"""

import dis
import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

try:
    import psutil
except ImportError:
    psutil = None

from ..core.models import EventType, ExecutionEvent, FrameInfo, SourceLocation
from .base import BaseCollector


class CPUCollector(BaseCollector):
    """
    Collector for CPU profiling and instruction-level analysis.

    Tracks CPU usage patterns, instruction execution, and computational
    hotspots to identify performance optimization opportunities.
    """

    def __init__(self, config=None):
        """
        Initialize the CPU collector.

        Args:
            config: Optional collector configuration
        """
        super().__init__(config)

        # CPU monitoring
        self._cpu_stats = {
            "total_cpu_time": 0,
            "cpu_samples": [],
            "instruction_counts": defaultdict(int),
            "function_cpu_usage": defaultdict(list),
            "hot_paths": defaultdict(int),
            "expensive_operations": [],
        }

        # Instruction tracking
        self._instruction_cache = {}
        self._bytecode_cache = {}
        self._hot_functions = set()

        # CPU sampling
        self._sampling_interval = 0.001  # 1ms sampling interval
        self._cpu_sampler_thread = None
        self._stop_sampling = threading.Event()

        # Performance thresholds
        self._high_cpu_threshold = 80.0  # 80% CPU usage
        self._instruction_hotspot_threshold = 1000  # Min instruction count for hotspot

        # Thread safety
        self._cpu_lock = threading.RLock()

        # Current execution context
        self._current_frame_stack = []
        self._frame_start_times = {}

        # System monitoring
        self._process = psutil.Process() if psutil else None
        self._initial_cpu_time = None

    @property
    def name(self) -> str:
        """Unique identifier for this collector type."""
        return "cpu"

    def _install_hooks(self) -> None:
        """Install CPU monitoring hooks."""
        # Save initial CPU time
        if self._process:
            self._initial_cpu_time = self._process.cpu_times()

        # Install trace function for instruction-level profiling
        self._original_trace = sys.gettrace()
        sys.settrace(self._cpu_trace_function)

        # Start CPU sampling thread
        self._start_cpu_sampling()

    def _uninstall_hooks(self) -> None:
        """Remove CPU monitoring hooks."""
        # Stop CPU sampling
        self._stop_cpu_sampling()

        # Restore original trace function
        if self._original_trace:
            sys.settrace(self._original_trace)
        else:
            sys.settrace(None)

    def _collect_events(self) -> Iterator[Dict[str, Any]]:
        """Collect CPU events from buffer."""
        # Return buffered events - BaseCollector handles the actual buffering
        yield from []

    def _start_cpu_sampling(self):
        """Start the CPU sampling thread."""
        self._stop_sampling.clear()
        self._cpu_sampler_thread = threading.Thread(
            target=self._cpu_sampling_loop, daemon=True
        )
        self._cpu_sampler_thread.start()

    def _stop_cpu_sampling(self):
        """Stop the CPU sampling thread."""
        if self._cpu_sampler_thread:
            self._stop_sampling.set()
            self._cpu_sampler_thread.join(timeout=1.0)

    def _cpu_sampling_loop(self):
        """Main CPU sampling loop."""
        while not self._stop_sampling.wait(self._sampling_interval):
            try:
                self._sample_cpu_usage()
            except Exception:
                # Don't let sampling errors break the collector
                pass

    def _sample_cpu_usage(self):
        """Sample current CPU usage."""
        if not self._process:
            return

        try:
            # Get CPU usage
            cpu_percent = self._process.cpu_percent()
            cpu_times = self._process.cpu_times()

            # Get memory info for context
            memory_info = self._process.memory_info()

            # Create CPU sample
            sample = {
                "timestamp": time.perf_counter_ns(),
                "cpu_percent": cpu_percent,
                "user_time": cpu_times.user,
                "system_time": cpu_times.system,
                "memory_rss": memory_info.rss,
                "memory_vms": memory_info.vms,
                "thread_count": self._process.num_threads(),
            }

            with self._cpu_lock:
                self._cpu_stats["cpu_samples"].append(sample)

                # Keep only recent samples (last 1000)
                if len(self._cpu_stats["cpu_samples"]) > 1000:
                    self._cpu_stats["cpu_samples"] = self._cpu_stats["cpu_samples"][
                        -1000:
                    ]

                # Check for high CPU usage
                if cpu_percent > self._high_cpu_threshold:
                    self._record_high_cpu_event(cpu_percent, sample)

        except Exception:
            # psutil might fail in some environments
            pass

    def _cpu_trace_function(self, frame, event, arg):
        """Main trace function for CPU profiling."""
        current_time = time.perf_counter_ns()

        try:
            if event == "call":
                self._handle_function_call(frame, current_time)
            elif event == "return":
                self._handle_function_return(frame, current_time)
            elif event == "line":
                self._handle_line_execution(frame, current_time)
            elif event == "exception":
                self._handle_exception(frame, arg, current_time)
        except Exception:
            # Don't let our profiling break execution
            pass

        # Continue with original trace function if it exists
        if self._original_trace:
            return self._original_trace(frame, event, arg)

        return self._cpu_trace_function

    def _handle_function_call(self, frame, timestamp):
        """Handle function call events."""
        frame_id = id(frame)

        # Track frame start time
        self._frame_start_times[frame_id] = timestamp

        # Add to frame stack
        self._current_frame_stack.append(frame_id)

        # Analyze bytecode if this is a hot function
        function_name = frame.f_code.co_name
        filename = frame.f_code.co_filename

        if self._should_analyze_bytecode(function_name, filename):
            self._analyze_function_bytecode(frame.f_code)

    def _handle_function_return(self, frame, timestamp):
        """Handle function return events."""
        frame_id = id(frame)

        # Calculate function execution time
        if frame_id in self._frame_start_times:
            start_time = self._frame_start_times[frame_id]
            execution_time = timestamp - start_time

            # Record function CPU usage
            self._record_function_cpu_usage(frame, execution_time, timestamp)

            # Clean up
            del self._frame_start_times[frame_id]

        # Remove from frame stack
        if self._current_frame_stack and self._current_frame_stack[-1] == frame_id:
            self._current_frame_stack.pop()

    def _handle_line_execution(self, frame, timestamp):
        """Handle line execution events."""
        # Sample line execution for hot path analysis
        if self._should_sample_line():
            self._record_line_execution(frame, timestamp)

    def _handle_exception(self, frame, exc_info, timestamp):
        """Handle exception events for CPU impact analysis."""
        # Exception handling can be CPU intensive
        self._record_exception_cpu_impact(frame, exc_info, timestamp)

    def _should_analyze_bytecode(self, function_name, filename):
        """Determine if function bytecode should be analyzed."""
        # Analyze bytecode for functions that appear frequently
        function_key = f"{filename}:{function_name}"
        return function_key in self._hot_functions or len(self._hot_functions) < 100

    def _analyze_function_bytecode(self, code_obj):
        """Analyze function bytecode for optimization opportunities."""
        code_key = (code_obj.co_filename, code_obj.co_name)

        if code_key in self._bytecode_cache:
            return self._bytecode_cache[code_key]

        try:
            # Disassemble bytecode
            instructions = list(dis.get_instructions(code_obj))

            # Analyze instruction patterns
            analysis = {
                "total_instructions": len(instructions),
                "instruction_types": defaultdict(int),
                "loop_indicators": 0,
                "call_count": 0,
                "load_count": 0,
                "store_count": 0,
                "complexity_score": 0,
            }

            for instruction in instructions:
                opname = instruction.opname
                analysis["instruction_types"][opname] += 1

                # Count specific instruction types
                if "JUMP" in opname:
                    analysis["loop_indicators"] += 1
                elif "CALL" in opname:
                    analysis["call_count"] += 1
                elif "LOAD" in opname:
                    analysis["load_count"] += 1
                elif "STORE" in opname:
                    analysis["store_count"] += 1

            # Calculate complexity score
            analysis["complexity_score"] = (
                analysis["loop_indicators"] * 3
                + analysis["call_count"] * 2
                + (analysis["load_count"] + analysis["store_count"]) * 0.1
            )

            # Cache the analysis
            self._bytecode_cache[code_key] = analysis

            # Update instruction counts
            with self._cpu_lock:
                for opname, count in analysis["instruction_types"].items():
                    self._cpu_stats["instruction_counts"][opname] += count

            return analysis

        except Exception:
            return None

    def _record_function_cpu_usage(self, frame, execution_time, timestamp):
        """Record CPU usage for a function."""
        source_location = SourceLocation(
            filename=frame.f_code.co_filename,
            line_number=frame.f_lineno,
            function_name=frame.f_code.co_name,
        )

        # Create frame info
        frame_info = FrameInfo(
            source_location=source_location,
            local_variables={},  # Don't capture locals for performance
        )

        # Create execution event
        event = ExecutionEvent(
            timestamp=timestamp,
            event_type=EventType.CALL,
            thread_id=threading.get_ident(),
            frame_info=frame_info,
            execution_time=execution_time,
            event_data={
                "operation": "function_cpu",
                "execution_time_ms": execution_time / 1_000_000,
                "call_depth": len(self._current_frame_stack),
                "is_expensive": execution_time > 10_000_000,  # 10ms
                "bytecode_analysis": self._bytecode_cache.get(
                    (frame.f_code.co_filename, frame.f_code.co_name)
                ),
            },
        )

        # Buffer the event
        self._add_to_buffer(event.__dict__)

        # Update function CPU usage stats
        function_key = f"{source_location.filename}:{source_location.function_name}"
        with self._cpu_lock:
            self._cpu_stats["function_cpu_usage"][function_key].append(
                {
                    "execution_time": execution_time,
                    "timestamp": timestamp,
                    "call_depth": len(self._current_frame_stack),
                }
            )

            # Mark as hot function if frequently called
            if len(self._cpu_stats["function_cpu_usage"][function_key]) > 10:
                self._hot_functions.add(function_key)

            # Track expensive operations
            if execution_time > 10_000_000:  # 10ms
                self._cpu_stats["expensive_operations"].append(
                    {
                        "function": function_key,
                        "execution_time_ms": execution_time / 1_000_000,
                        "timestamp": timestamp,
                        "call_depth": len(self._current_frame_stack),
                    }
                )

    def _should_sample_line(self):
        """Determine if line execution should be sampled."""
        # Sample based on configuration sampling rate
        return self._should_sample()

    def _record_line_execution(self, frame, timestamp):
        """Record line execution for hot path analysis."""
        source_location = SourceLocation(
            filename=frame.f_code.co_filename,
            line_number=frame.f_lineno,
            function_name=frame.f_code.co_name,
        )

        # Track hot paths
        path_key = f"{source_location.filename}:{source_location.line_number}"
        with self._cpu_lock:
            self._cpu_stats["hot_paths"][path_key] += 1

    def _record_exception_cpu_impact(self, frame, exc_info, timestamp):
        """Record CPU impact of exception handling."""
        source_location = SourceLocation(
            filename=frame.f_code.co_filename,
            line_number=frame.f_lineno,
            function_name=frame.f_code.co_name,
        )

        # Exception handling overhead
        frame_info = FrameInfo(source_location=source_location, local_variables={})

        event = ExecutionEvent(
            timestamp=timestamp,
            event_type=EventType.EXCEPTION,
            thread_id=threading.get_ident(),
            frame_info=frame_info,
            event_data={
                "operation": "exception_cpu_impact",
                "exception_type": exc_info[0].__name__ if exc_info[0] else "unknown",
                "call_depth": len(self._current_frame_stack),
            },
        )

        self._add_to_buffer(event.__dict__)

    def _record_high_cpu_event(self, cpu_percent, sample):
        """Record high CPU usage event."""
        # Get current frame if available
        current_frame = sys._getframe(1)
        source_location = SourceLocation(
            filename=current_frame.f_code.co_filename,
            line_number=current_frame.f_lineno,
            function_name=current_frame.f_code.co_name,
        )

        frame_info = FrameInfo(source_location=source_location, local_variables={})

        event = ExecutionEvent(
            timestamp=sample["timestamp"],
            event_type=EventType.CALL,
            thread_id=threading.get_ident(),
            frame_info=frame_info,
            event_data={
                "operation": "high_cpu_usage",
                "cpu_percent": cpu_percent,
                "user_time": sample["user_time"],
                "system_time": sample["system_time"],
                "memory_rss": sample["memory_rss"],
                "thread_count": sample["thread_count"],
            },
        )

        self._add_to_buffer(event.__dict__)

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics including CPU analysis."""
        base_stats = super().get_stats()

        with self._cpu_lock:
            # Analyze CPU usage patterns
            cpu_analysis = self._analyze_cpu_patterns()

            # Get hot functions
            hot_functions = self._get_hot_functions()

            # Get instruction analysis
            instruction_analysis = self._analyze_instructions()

            # Get hot paths
            hot_paths = self._get_hot_paths()

            # Calculate CPU efficiency
            cpu_efficiency = self._calculate_cpu_efficiency()

            cpu_stats = {
                "cpu_usage_analysis": cpu_analysis,
                "hot_functions": hot_functions,
                "instruction_analysis": instruction_analysis,
                "hot_paths": hot_paths,
                "cpu_efficiency_score": cpu_efficiency,
                "total_samples": len(self._cpu_stats["cpu_samples"]),
                "expensive_operations": self._cpu_stats["expensive_operations"][
                    -20:
                ],  # Last 20
                "bytecode_cache_size": len(self._bytecode_cache),
                "hot_functions_count": len(self._hot_functions),
                "optimization_recommendations": self._generate_cpu_recommendations(),
            }

            base_stats["cpu_analysis"] = cpu_stats

        return base_stats

    def _analyze_cpu_patterns(self) -> Dict[str, Any]:
        """Analyze CPU usage patterns from samples."""
        samples = self._cpu_stats["cpu_samples"]
        if not samples:
            return {}

        cpu_percentages = [s["cpu_percent"] for s in samples]
        user_times = [s["user_time"] for s in samples]
        system_times = [s["system_time"] for s in samples]

        return {
            "average_cpu_percent": sum(cpu_percentages) / len(cpu_percentages),
            "max_cpu_percent": max(cpu_percentages),
            "min_cpu_percent": min(cpu_percentages),
            "high_cpu_events": len(
                [p for p in cpu_percentages if p > self._high_cpu_threshold]
            ),
            "average_user_time": sum(user_times) / len(user_times),
            "average_system_time": sum(system_times) / len(system_times),
            "cpu_utilization_trend": self._calculate_cpu_trend(cpu_percentages),
        }

    def _get_hot_functions(self) -> List[Dict[str, Any]]:
        """Get functions with highest CPU usage."""
        function_stats = []

        for function_key, usage_list in self._cpu_stats["function_cpu_usage"].items():
            total_time = sum(usage["execution_time"] for usage in usage_list)
            call_count = len(usage_list)
            avg_time = total_time / call_count if call_count > 0 else 0

            function_stats.append(
                {
                    "function": function_key,
                    "total_cpu_time_ms": total_time / 1_000_000,
                    "call_count": call_count,
                    "average_time_ms": avg_time / 1_000_000,
                    "max_time_ms": max(usage["execution_time"] for usage in usage_list)
                    / 1_000_000,
                }
            )

        return sorted(
            function_stats, key=lambda x: x["total_cpu_time_ms"], reverse=True
        )[:20]

    def _analyze_instructions(self) -> Dict[str, Any]:
        """Analyze instruction execution patterns."""
        if not self._cpu_stats["instruction_counts"]:
            return {}

        total_instructions = sum(self._cpu_stats["instruction_counts"].values())
        sorted_instructions = sorted(
            self._cpu_stats["instruction_counts"].items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "total_instructions_executed": total_instructions,
            "most_common_instructions": sorted_instructions[:20],
            "instruction_diversity": len(self._cpu_stats["instruction_counts"]),
            "loop_heavy_ratio": self._cpu_stats["instruction_counts"].get(
                "JUMP_BACKWARD", 0
            )
            / total_instructions,
            "call_ratio": (
                self._cpu_stats["instruction_counts"].get("CALL_FUNCTION", 0)
                + self._cpu_stats["instruction_counts"].get("CALL_METHOD", 0)
            )
            / total_instructions,
        }

    def _get_hot_paths(self) -> List[Dict[str, Any]]:
        """Get most frequently executed code paths."""
        sorted_paths = sorted(
            self._cpu_stats["hot_paths"].items(), key=lambda x: x[1], reverse=True
        )

        return [
            {"path": path, "execution_count": count}
            for path, count in sorted_paths[:20]
        ]

    def _calculate_cpu_efficiency(self) -> float:
        """Calculate overall CPU efficiency score."""
        samples = self._cpu_stats["cpu_samples"]
        if not samples:
            return 0.0

        # Efficiency based on consistent moderate CPU usage vs spikes
        cpu_percentages = [s["cpu_percent"] for s in samples]
        if not cpu_percentages:
            return 0.0

        avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
        cpu_variance = sum((p - avg_cpu) ** 2 for p in cpu_percentages) / len(
            cpu_percentages
        )

        # Lower variance with reasonable CPU usage indicates efficiency
        if avg_cpu > 0:
            return max(0, 100 - (cpu_variance / avg_cpu))
        else:
            return 100.0

    def _calculate_cpu_trend(self, cpu_percentages) -> str:
        """Calculate CPU usage trend."""
        if len(cpu_percentages) < 10:
            return "insufficient_data"

        # Simple trend calculation
        first_half = cpu_percentages[: len(cpu_percentages) // 2]
        second_half = cpu_percentages[len(cpu_percentages) // 2 :]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        diff = second_avg - first_avg

        if abs(diff) < 5:
            return "stable"
        elif diff > 5:
            return "increasing"
        else:
            return "decreasing"

    def _generate_cpu_recommendations(self) -> List[str]:
        """Generate CPU optimization recommendations."""
        recommendations = []

        # Check for hot functions
        hot_functions = self._get_hot_functions()
        if hot_functions and hot_functions[0]["total_cpu_time_ms"] > 1000:  # 1 second
            recommendations.append(
                f"Optimize hot function: {hot_functions[0]['function']}"
            )

        # Check for high CPU variance
        samples = self._cpu_stats["cpu_samples"]
        if samples:
            cpu_percentages = [s["cpu_percent"] for s in samples]
            if cpu_percentages:
                avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
                max_cpu = max(cpu_percentages)

                if max_cpu - avg_cpu > 50:
                    recommendations.append(
                        "High CPU variance detected - consider load balancing"
                    )

        # Check instruction patterns
        instruction_counts = self._cpu_stats["instruction_counts"]
        total_instructions = sum(instruction_counts.values())

        if total_instructions > 0:
            loop_ratio = instruction_counts.get("JUMP_BACKWARD", 0) / total_instructions
            if loop_ratio > 0.3:
                recommendations.append(
                    "High loop ratio detected - consider algorithm optimization"
                )

        # Check for excessive expensive operations
        if len(self._cpu_stats["expensive_operations"]) > 50:
            recommendations.append(
                "Many expensive operations detected - consider async processing"
            )

        return recommendations
