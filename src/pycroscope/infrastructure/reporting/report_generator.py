"""
Report generation for Pycroscope profiling sessions.

Generates comprehensive markdown reports from profiling data with fail-fast behavior
and strict validation following clean architecture principles.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from ...core.session import ProfileSession
from ...core.exceptions import ConfigurationError, ValidationError


class ReportGenerator:
    """
    Generates comprehensive markdown reports from profiling sessions.

    Handles report creation with strict validation and fail-fast behavior,
    producing detailed, actionable analysis reports.
    """

    def __init__(self, session: ProfileSession):
        self.session = session
        self._validate_session()

    def _validate_session(self) -> None:
        """Validate session before report generation."""
        if not self.session.is_complete:
            raise ValidationError("Cannot generate report for incomplete session")

        if not self.session.results:
            raise ValidationError("Cannot generate report for session with no results")

    def generate_comprehensive_report(self, output_dir: Optional[Path] = None) -> Path:
        """
        Generate comprehensive markdown report from session data.

        Args:
            output_dir: Directory for output files (must be configured)

        Returns:
            Path to generated markdown report
        """
        if output_dir is None:
            if self.session.config.output_dir is None:
                raise ConfigurationError(
                    "Output directory must be configured for report generation",
                    config_key="output_dir",
                )
            output_dir = self.session.config.output_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use clean filename without session ID
        report_path = output_dir / "profiling_report.md"

        self._write_comprehensive_markdown_report(report_path)
        return report_path

    def _write_comprehensive_markdown_report(self, report_path: Path) -> None:
        """Write comprehensive markdown report to file."""
        with open(report_path, "w", encoding="utf-8") as file_handle:
            # Header
            file_handle.write("# 🔍 Pycroscope Profiling Analysis Report\n\n")
            file_handle.write(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            file_handle.write(f"**Session ID:** `{self.session.session_id}`\n\n")

            # Executive Summary
            file_handle.write("## 📊 Executive Summary\n\n")

            duration = self.session.duration or 0
            status = self.session.status.value
            completed_profilers = list(self.session.results.keys())
            total_results = len(self.session.results)

            file_handle.write(f"- **Duration:** {duration:.3f} seconds\n")
            file_handle.write(f"- **Status:** {status}\n")
            file_handle.write(
                f"- **Profilers Used:** {', '.join(completed_profilers)}\n"
            )
            file_handle.write(f"- **Total Results:** {total_results}\n\n")

            # Configuration section
            file_handle.write("## ⚙️ Configuration\n\n")
            config_data = self.session.config.model_dump()
            file_handle.write("| Setting | Value |\n")
            file_handle.write("|---------|-------|\n")
            for key, value in config_data.items():
                if key not in ["extra_config"]:
                    file_handle.write(
                        f"| {key.replace('_', ' ').title()} | `{value}` |\n"
                    )
            file_handle.write("\n")

            # Detailed analysis for each profiler
            for profiler_name in completed_profilers:
                result = self.session.get_result(profiler_name)
                if result and result.data:
                    self._write_profiler_analysis(
                        file_handle, profiler_name, result.data
                    )

            # Performance insights
            file_handle.write("## 🎯 Performance Insights\n\n")
            insights = self._generate_performance_insights()
            for insight in insights:
                file_handle.write(f"- {insight}\n")
            file_handle.write("\n")

            # Technical details
            file_handle.write("## 🔧 Technical Details\n\n")
            file_handle.write("### Session Metadata\n\n")
            file_handle.write(f"- **Start Time:** {self.session.start_time}\n")
            file_handle.write(f"- **End Time:** {self.session.end_time}\n")
            file_handle.write(
                f"- **Output Directory:** `{self.session.config.output_dir}`\n"
            )
            file_handle.write(
                f"- **Session Name:** {self.session.config.session_name or 'Default'}\n\n"
            )

    def _write_profiler_analysis(
        self, file_handle, profiler_name: str, data: Dict[str, Any]
    ) -> None:
        """Write detailed analysis for specific profiler."""
        file_handle.write(f"## 📈 {profiler_name.title()} Profiler Analysis\n\n")

        if profiler_name == "call":
            self._write_call_profiler_analysis(file_handle, data)
        elif profiler_name == "memory":
            self._write_memory_profiler_analysis(file_handle, data)
        elif profiler_name == "line":
            # Get the result object for line profiler
            line_result = None
            for result in self.session.results.values():
                if hasattr(result, "data") and result.data == data:
                    line_result = result
                    break

            if line_result:
                line_analysis = self._write_line_profiler_analysis(line_result)
                file_handle.write(line_analysis)
            else:
                file_handle.write("Line profiling data not available.\n\n")

        else:
            file_handle.write(f"**Data Points Collected:** {len(data)}\n\n")

    def _write_call_profiler_analysis(self, file_handle, data: Dict[str, Any]) -> None:
        """Write call profiler analysis."""
        if "stats" not in data:
            raise RuntimeError(
                "FAIL FAST: Call profiler data missing required 'stats' field"
            )

        stats = data["stats"]
        if not stats:
            raise RuntimeError(
                "FAIL FAST: Call profiler generated no function statistics"
            )

        # Top functions by time
        file_handle.write("### 🕒 Top Functions by Execution Time\n\n")
        file_handle.write(
            "| Function | Total Time | Calls | Time/Call | Cumulative |\n"
        )
        file_handle.write(
            "|----------|------------|-------|-----------|------------|\n"
        )

        sorted_by_time = sorted(
            stats.items(), key=lambda x: x[1].get("tottime", 0), reverse=True
        )[:10]

        for func_name, func_stats in sorted_by_time:
            total_time = func_stats.get("tottime", 0)
            calls = func_stats.get("ncalls", 0)
            cumtime = func_stats.get("cumtime", 0)
            time_per_call = total_time / calls if calls > 0 else 0

            file_handle.write(
                f"| `{func_name}` | {total_time:.4f}s | {calls} | {time_per_call:.6f}s | {cumtime:.4f}s |\n"
            )

        file_handle.write("\n### 📞 Most Called Functions\n\n")
        file_handle.write("| Function | Calls | Total Time | Avg Time |\n")
        file_handle.write("|----------|-------|------------|----------|\n")

        sorted_by_calls = sorted(
            stats.items(), key=lambda x: x[1].get("ncalls", 0), reverse=True
        )[:10]

        for func_name, func_stats in sorted_by_calls:
            calls = func_stats.get("ncalls", 0)
            total_time = func_stats.get("tottime", 0)
            avg_time = total_time / calls if calls > 0 else 0

            file_handle.write(
                f"| `{func_name}` | {calls} | {total_time:.4f}s | {avg_time:.6f}s |\n"
            )

        file_handle.write(f"\n**Total Functions Profiled:** {len(stats)}\n\n")

    def _write_memory_profiler_analysis(
        self, file_handle, data: Dict[str, Any]
    ) -> None:
        """Write memory profiler analysis."""
        if "samples" not in data:
            file_handle.write("No memory samples available.\n\n")
            return

        samples = data.get("samples", [])
        peak_memory = data.get("peak_memory_mb", 0)
        avg_memory = data.get("avg_memory_mb", 0)
        memory_delta = data.get("memory_delta_mb", 0)

        file_handle.write("### 🧠 Memory Usage Statistics\n\n")
        file_handle.write("| Metric | Value |\n")
        file_handle.write("|--------|-------|\n")
        file_handle.write(f"| Peak Memory Usage | {peak_memory:.2f} MB |\n")
        file_handle.write(f"| Average Memory Usage | {avg_memory:.2f} MB |\n")
        file_handle.write(f"| Memory Delta | {memory_delta:+.2f} MB |\n")
        file_handle.write(f"| Sample Count | {len(samples)} |\n")

        if samples:
            initial_memory = samples[0].get("rss_mb", 0)
            final_memory = samples[-1].get("rss_mb", 0)
            file_handle.write(f"| Initial Memory | {initial_memory:.2f} MB |\n")
            file_handle.write(f"| Final Memory | {final_memory:.2f} MB |\n")

        file_handle.write("\n### 📊 Memory Timeline Analysis\n\n")
        if len(samples) >= 2:
            # Calculate memory growth rate
            duration = samples[-1]["timestamp"] - samples[0]["timestamp"]
            growth_rate = memory_delta / duration if duration > 0 else 0
            file_handle.write(
                f"- **Memory Growth Rate:** {growth_rate:.4f} MB/second\n"
            )

            # Identify memory spikes
            spikes = 0
            threshold = avg_memory * 1.5  # 50% above average
            for sample in samples:
                if sample.get("rss_mb", 0) > threshold:
                    spikes += 1

            file_handle.write(
                f"- **Memory Spikes Detected:** {spikes} (>{threshold:.2f} MB)\n"
            )

        file_handle.write("\n")

    def _write_line_profiler_analysis(self, line_result: Any) -> str:
        """Write line profiler analysis section."""
        line_data = line_result.data
        content = []

        content.append("## 📝 Line Profiler Analysis")
        content.append("")

        if "function_profiles" in line_data and line_data["function_profiles"]:
            function_profiles = line_data["function_profiles"]
            content.append(f"### 🎯 Per-Function Line-by-Line Analysis")
            content.append("")
            content.append(f"**Functions Profiled:** {len(function_profiles)}")
            content.append("")

            # Add detailed breakdowns for each function
            for func_key, profile in function_profiles.items():
                content.append(f"#### {profile['function']} ({profile['filename']})")
                content.append("")

                # Generate formatted output from line_details
                line_details = profile.get("line_details", [])
                if line_details:
                    content.append("```")
                    content.append(
                        f"Line #      Hits         Time  Per Hit   % Time  Line Contents"
                    )
                    content.append("=" * 80)

                    total_time = profile.get("total_time", 0)
                    for line in line_details:
                        line_num = line["line_number"]
                        hits = line["hits"]
                        time_us = line["time_us"]
                        per_hit_us = line["per_hit_us"]
                        percent_time = (
                            (time_us / 1000000 / total_time * 100)
                            if total_time > 0
                            else 0
                        )
                        line_content = line["line_content"][:50]  # Truncate long lines

                        content.append(
                            f"{line_num:6d} {hits:12d} {time_us:12.1f} {per_hit_us:8.1f} {percent_time:7.1f}%  {line_content}"
                        )
                    content.append("```")
                    content.append("")

                    # Add performance insights for this function
                    hottest_lines = sorted(
                        line_details, key=lambda x: x["time_us"], reverse=True
                    )[:3]
                    content.append("**Performance Insights:**")
                    for i, line in enumerate(hottest_lines, 1):
                        percent_time = (
                            (line["time_us"] / 1000000 / total_time * 100)
                            if total_time > 0
                            else 0
                        )
                        content.append(
                            f"- **Line {line['line_number']}**: {percent_time:.1f}% of function time ({line['hits']:,} hits)"
                        )
                    content.append("")

        elif "line_stats" in line_data and line_data["line_stats"]:
            line_stats = line_data["line_stats"]
            content.append(f"### 🔥 Hottest Lines")
            content.append("")

            # Get top 20 lines by total time
            sorted_lines = sorted(
                line_stats.items(), key=lambda x: x[1].get("time", 0), reverse=True
            )[:20]

            if sorted_lines:
                content.append(
                    "| Line | File | Hits | Total Time | Time/Hit | Function |"
                )
                content.append(
                    "|------|------|------|------------|----------|----------|"
                )

                for line_key, stats in sorted_lines:
                    filename = stats.get("filename", "").split("/")[-1]  # Just filename
                    line_no = stats.get("line_number", 0)
                    hits = stats.get("hits", 0)
                    time_val = stats.get("time", 0)
                    time_per_hit = stats.get("time_per_hit", 0)
                    function = stats.get("function", "")

                    content.append(
                        f"| {line_no} | {filename} | {hits:,} | {time_val:.6f}s | {time_per_hit:.6f}s | {function} |"
                    )

                content.append("")

        # Add summary statistics
        total_lines = line_data.get("total_lines", 0)
        total_hits = line_data.get("total_hits", 0)
        total_time = line_data.get("total_time", 0)

        if total_lines > 0:
            content.append("### 📊 Line Profiling Summary")
            content.append("")
            content.append(f"- **Total Lines Profiled:** {total_lines:,}")
            content.append(f"- **Total Hits:** {total_hits:,}")
            content.append(f"- **Total Time:** {total_time:.6f} seconds")
            if total_hits > 0:
                content.append(
                    f"- **Average Time per Hit:** {(total_time / total_hits):.9f} seconds"
                )
            content.append("")

        return "\n".join(content)

    def _generate_performance_insights(self) -> List[str]:
        """Generate performance insights from profiling data."""
        insights = []

        # Duration analysis
        if self.session.duration:
            if self.session.duration > 1.0:
                insights.append(
                    f"Long execution time detected ({self.session.duration:.3f}s) - consider optimization"
                )
            elif self.session.duration < 0.001:
                insights.append(
                    "Very fast execution - consider longer test runs for better profiling accuracy"
                )

        # Memory analysis
        memory_result = self.session.get_result("memory")
        if memory_result and memory_result.data:
            memory_delta = memory_result.data.get("memory_delta_mb", 0)
            if memory_delta > 100:
                insights.append(
                    f"High memory usage detected ({memory_delta:.1f}MB growth)"
                )
            elif memory_delta < 0:
                insights.append(
                    "Memory was freed during execution - good memory management"
                )

        # Call analysis
        call_result = self.session.get_result("call")
        if call_result and call_result.data and "stats" in call_result.data:
            stats = call_result.data["stats"]
            total_calls = sum(s.get("ncalls", 0) for s in stats.values())
            if total_calls > 10000:
                insights.append(
                    f"High function call count ({total_calls:,}) - potential optimization opportunity"
                )

        if not insights:
            insights.append("No significant performance issues detected")

        return insights
