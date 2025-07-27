"""
Report Formatters for Pycroscope

Formats profiling reports in different output formats (text, markdown).
"""

from typing import Dict, Any
from datetime import datetime


class BaseFormatter:
    """Base class for report formatters."""

    def format_report(self, report_data: Dict[str, Any]) -> str:
        """Format a report from structured data."""
        raise NotImplementedError("Subclasses must implement format_report")

    def _format_session_info(self, session_info: Dict[str, Any]) -> str:
        """Format session information section."""
        raise NotImplementedError("Subclasses must implement _format_session_info")

    def _format_profiling_summary(self, summary: Dict[str, Any]) -> str:
        """Format profiling summary section."""
        raise NotImplementedError("Subclasses must implement _format_profiling_summary")


class TextFormatter(BaseFormatter):
    """Formats reports as plain text."""

    def format_report(self, report_data: Dict[str, Any]) -> str:
        """Format a comprehensive text report."""
        sections = []

        # Header
        sections.append("=" * 80)
        sections.append("PYCROSCOPE PROFILING REPORT")
        sections.append("=" * 80)
        sections.append("")

        # Session Info
        sections.append(self._format_session_info(report_data.get("session_info", {})))
        sections.append("")

        # Profiling Summary
        sections.append(
            self._format_profiling_summary(report_data.get("profiling_summary", {}))
        )
        sections.append("")

        # Profiler Results
        sections.append(
            self._format_profiler_results(report_data.get("profiler_results", {}))
        )
        sections.append("")

        # Performance Insights
        sections.append(
            self._format_performance_insights(
                report_data.get("performance_insights", [])
            )
        )
        sections.append("")

        # Recommendations
        sections.append(
            self._format_recommendations(report_data.get("recommendations", []))
        )
        sections.append("")

        # Raw Data References
        sections.append(
            self._format_raw_data_references(report_data.get("raw_data_references", {}))
        )
        sections.append("")

        # Footer
        sections.append("=" * 80)
        sections.append(f"Report generated: {datetime.now().isoformat()}")
        sections.append("=" * 80)

        return "\n".join(sections)

    def _format_session_info(self, session_info: Dict[str, Any]) -> str:
        """Format session information as text."""
        lines = ["SESSION INFORMATION", "-" * 40]

        lines.append(f"Session ID: {session_info.get('session_id', 'Unknown')}")
        lines.append(f"Status: {session_info.get('status', 'Unknown')}")
        lines.append(f"Duration: {session_info.get('duration', 0):.3f} seconds")

        if session_info.get("start_time"):
            lines.append(f"Start Time: {session_info['start_time']}")
        if session_info.get("end_time"):
            lines.append(f"End Time: {session_info['end_time']}")

        # Configuration
        config = session_info.get("config", {})
        lines.append("\nConfiguration:")
        lines.append(
            f"  Call Profiling: {'✓' if config.get('call_profiling') else '✗'}"
        )
        lines.append(
            f"  Line Profiling: {'✓' if config.get('line_profiling') else '✗'}"
        )
        lines.append(
            f"  Memory Profiling: {'✓' if config.get('memory_profiling') else '✗'}"
        )

        if config.get("session_name"):
            lines.append(f"  Session Name: {config['session_name']}")

        return "\n".join(lines)

    def _format_profiling_summary(self, summary: Dict[str, Any]) -> str:
        """Format profiling summary as text."""
        lines = ["PROFILING SUMMARY", "-" * 40]

        lines.append(f"Total Profilers Used: {summary.get('total_profilers', 0)}")
        lines.append(f"Profilers: {', '.join(summary.get('profilers_used', []))}")
        lines.append(
            f"Data Points Collected: {summary.get('data_points_collected', 0):,}"
        )
        lines.append(
            f"Total Execution Time: {summary.get('total_execution_time', 0):.3f}s"
        )

        return "\n".join(lines)

    def _format_profiler_results(self, profiler_results: Dict[str, Any]) -> str:
        """Format profiler results as text."""
        lines = ["PROFILER RESULTS", "-" * 40]

        for profiler_name, result in profiler_results.items():
            lines.append(f"\n{profiler_name.upper()} PROFILER:")
            lines.append(f"  Status: {result.get('status', 'Unknown')}")
            lines.append(f"  Duration: {result.get('duration', 0):.3f}s")

            # Data summary
            data_summary = result.get("data_summary", {})
            if data_summary:
                lines.append("  Data Summary:")
                for key, value in data_summary.items():
                    if isinstance(value, list) and len(value) > 3:
                        lines.append(f"    {key}: {len(value)} items")
                    else:
                        lines.append(f"    {key}: {value}")

            # Key metrics
            key_metrics = result.get("key_metrics", {})
            if key_metrics:
                lines.append("  Key Metrics:")
                for key, value in key_metrics.items():
                    if isinstance(value, list) and len(value) > 3:
                        lines.append(f"    {key}: {len(value)} items")
                    else:
                        lines.append(f"    {key}: {value}")

        return "\n".join(lines)

    def _format_performance_insights(self, insights: list) -> str:
        """Format performance insights as text."""
        lines = ["PERFORMANCE INSIGHTS", "-" * 40]

        if not insights:
            lines.append("No specific performance insights generated.")
        else:
            for i, insight in enumerate(insights, 1):
                lines.append(f"{i}. {insight}")

        return "\n".join(lines)

    def _format_recommendations(self, recommendations: list) -> str:
        """Format recommendations as text."""
        lines = ["OPTIMIZATION RECOMMENDATIONS", "-" * 40]

        if not recommendations:
            lines.append("No specific recommendations generated.")
        else:
            for i, recommendation in enumerate(recommendations, 1):
                lines.append(f"{i}. {recommendation}")

        return "\n".join(lines)

    def _format_raw_data_references(self, references: Dict[str, str]) -> str:
        """Format raw data references as text."""
        lines = ["RAW DATA REFERENCES", "-" * 40]

        if not references:
            lines.append("No raw data files available.")
        else:
            for name, path in references.items():
                lines.append(f"{name}: {path}")

        return "\n".join(lines)


class MarkdownFormatter(BaseFormatter):
    """Formats reports as Markdown."""

    def format_report(self, report_data: Dict[str, Any]) -> str:
        """Format a comprehensive Markdown report."""
        sections = []

        # Header
        sections.append("# Pycroscope Profiling Report")
        sections.append("")

        # Session Info
        sections.append(self._format_session_info(report_data.get("session_info", {})))
        sections.append("")

        # Profiling Summary
        sections.append(
            self._format_profiling_summary(report_data.get("profiling_summary", {}))
        )
        sections.append("")

        # Profiler Results
        sections.append(
            self._format_profiler_results(report_data.get("profiler_results", {}))
        )
        sections.append("")

        # Performance Insights
        sections.append(
            self._format_performance_insights(
                report_data.get("performance_insights", [])
            )
        )
        sections.append("")

        # Recommendations
        sections.append(
            self._format_recommendations(report_data.get("recommendations", []))
        )
        sections.append("")

        # Raw Data References
        sections.append(
            self._format_raw_data_references(report_data.get("raw_data_references", {}))
        )
        sections.append("")

        # Footer
        sections.append("---")
        sections.append(f"*Report generated: {datetime.now().isoformat()}*")

        return "\n".join(sections)

    def _format_session_info(self, session_info: Dict[str, Any]) -> str:
        """Format session information as Markdown."""
        lines = ["## Session Information"]

        lines.append(f"- **Session ID:** `{session_info.get('session_id', 'Unknown')}`")
        lines.append(f"- **Status:** {session_info.get('status', 'Unknown')}")
        lines.append(f"- **Duration:** {session_info.get('duration', 0):.3f} seconds")

        if session_info.get("start_time"):
            lines.append(f"- **Start Time:** {session_info['start_time']}")
        if session_info.get("end_time"):
            lines.append(f"- **End Time:** {session_info['end_time']}")

        # Configuration
        config = session_info.get("config", {})
        lines.append("")
        lines.append("### Configuration")
        lines.append(
            f"- **Call Profiling:** {'✅' if config.get('call_profiling') else '❌'}"
        )
        lines.append(
            f"- **Line Profiling:** {'✅' if config.get('line_profiling') else '❌'}"
        )
        lines.append(
            f"- **Memory Profiling:** {'✅' if config.get('memory_profiling') else '❌'}"
        )

        if config.get("session_name"):
            lines.append(f"- **Session Name:** {config['session_name']}")

        return "\n".join(lines)

    def _format_profiling_summary(self, summary: Dict[str, Any]) -> str:
        """Format profiling summary as Markdown."""
        lines = ["## Profiling Summary"]

        lines.append(f"- **Total Profilers Used:** {summary.get('total_profilers', 0)}")
        lines.append(f"- **Profilers:** {', '.join(summary.get('profilers_used', []))}")
        lines.append(
            f"- **Data Points Collected:** {summary.get('data_points_collected', 0):,}"
        )
        lines.append(
            f"- **Total Execution Time:** {summary.get('total_execution_time', 0):.3f}s"
        )

        return "\n".join(lines)

    def _format_profiler_results(self, profiler_results: Dict[str, Any]) -> str:
        """Format profiler results as Markdown."""
        lines = ["## Profiler Results"]

        for profiler_name, result in profiler_results.items():
            lines.append(f"\n### {profiler_name.title()} Profiler")
            lines.append(f"- **Status:** {result.get('status', 'Unknown')}")
            lines.append(f"- **Duration:** {result.get('duration', 0):.3f}s")

            # Data summary
            data_summary = result.get("data_summary", {})
            if data_summary:
                lines.append("\n#### Data Summary")
                for key, value in data_summary.items():
                    if isinstance(value, list) and len(value) > 3:
                        lines.append(
                            f"- **{key.replace('_', ' ').title()}:** {len(value)} items"
                        )
                    else:
                        lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")

            # Key metrics
            key_metrics = result.get("key_metrics", {})
            if key_metrics:
                lines.append("\n#### Key Metrics")
                for key, value in key_metrics.items():
                    if isinstance(value, list) and len(value) > 3:
                        lines.append(
                            f"- **{key.replace('_', ' ').title()}:** {len(value)} items"
                        )
                    else:
                        lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")

        return "\n".join(lines)

    def _format_performance_insights(self, insights: list) -> str:
        """Format performance insights as Markdown."""
        lines = ["## Performance Insights"]

        if not insights:
            lines.append("No specific performance insights generated.")
        else:
            for insight in insights:
                lines.append(f"- {insight}")

        return "\n".join(lines)

    def _format_recommendations(self, recommendations: list) -> str:
        """Format recommendations as Markdown."""
        lines = ["## Optimization Recommendations"]

        if not recommendations:
            lines.append("No specific recommendations generated.")
        else:
            for recommendation in recommendations:
                lines.append(f"- {recommendation}")

        return "\n".join(lines)

    def _format_raw_data_references(self, references: Dict[str, str]) -> str:
        """Format raw data references as Markdown."""
        lines = ["## Raw Data References"]

        if not references:
            lines.append("No raw data files available.")
        else:
            for name, path in references.items():
                lines.append(f"- **{name.replace('_', ' ').title()}:** `{path}`")

        return "\n".join(lines)
