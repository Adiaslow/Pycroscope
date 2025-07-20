"""
Output formatters for CLI results.

Provides various output formats for profiling results including
tables, JSON, and YAML for different consumption needs.
"""

import json
import sys
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

import yaml


class TableFormatter:
    """
    Format data as ASCII tables.

    Provides clean tabular output for terminal display.
    """

    def __init__(self, max_width: int = 120):
        """
        Initialize table formatter.

        Args:
            max_width: Maximum table width
        """
        self.max_width = max_width

    def format_table(
        self, headers: List[str], rows: List[List[str]], title: Optional[str] = None
    ) -> str:
        """
        Format data as a table.

        Args:
            headers: Column headers
            rows: Data rows
            title: Optional table title

        Returns:
            Formatted table string
        """
        if not rows:
            return "No data to display"

        # Calculate column widths
        col_widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        # Ensure we don't exceed max width
        total_width = sum(col_widths) + len(headers) * 3 + 1
        if total_width > self.max_width:
            # Proportionally reduce column widths
            scale = (self.max_width - len(headers) * 3 - 1) / sum(col_widths)
            col_widths = [max(8, int(width * scale)) for width in col_widths]

        # Build table
        lines = []

        # Title
        if title:
            lines.append(f"\n{title}")
            lines.append("=" * len(title))

        # Header separator
        separator = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"
        lines.append(separator)

        # Headers
        header_line = "|"
        for header, width in zip(headers, col_widths):
            header_line += f" {header:<{width}} |"
        lines.append(header_line)
        lines.append(separator)

        # Data rows
        for row in rows:
            row_line = "|"
            for cell, width in zip(row, col_widths):
                cell_str = str(cell)
                if len(cell_str) > width:
                    cell_str = cell_str[: width - 3] + "..."
                row_line += f" {cell_str:<{width}} |"
            lines.append(row_line)

        lines.append(separator)

        return "\n".join(lines)

    def format_key_value(
        self, data: Dict[str, Any], title: Optional[str] = None
    ) -> str:
        """
        Format data as key-value pairs.

        Args:
            data: Data to format
            title: Optional title

        Returns:
            Formatted string
        """
        lines = []

        if title:
            lines.append(f"\n{title}")
            lines.append("=" * len(title))

        max_key_width = max(len(str(key)) for key in data.keys()) if data else 0

        for key, value in data.items():
            lines.append(f"{str(key):<{max_key_width}} : {value}")

        return "\n".join(lines)


class ResultFormatter:
    """
    Main result formatter with multiple output formats.

    Supports table, JSON, and YAML output formats.
    """

    def __init__(self, format_type: str = "table"):
        """
        Initialize result formatter.

        Args:
            format_type: Output format ('table', 'json', 'yaml')
        """
        self.format_type = format_type
        self.table_formatter = TableFormatter()

    def format_session_list(self, sessions: List[Dict[str, Any]]) -> str:
        """
        Format a list of profiling sessions.

        Args:
            sessions: List of session metadata

        Returns:
            Formatted output
        """
        if self.format_type == "json":
            return json.dumps(sessions, indent=2, default=self._json_serializer)

        elif self.format_type == "yaml":
            return yaml.dump(sessions, default_flow_style=False)

        else:  # table
            if not sessions:
                return "No profiling sessions found"

            headers = ["Session ID", "Timestamp", "Package", "Events", "Memory (MB)"]
            rows = []

            for session in sessions:
                rows.append(
                    [
                        session.get("session_id", "")[:12] + "...",
                        self._format_timestamp(session.get("timestamp")),
                        session.get("target_package", ""),
                        str(session.get("total_events", 0)),
                        f"{session.get('peak_memory', 0) / (1024*1024):.1f}",
                    ]
                )

            return self.table_formatter.format_table(
                headers, rows, "Profiling Sessions"
            )

    def format_session_comparison(self, comparison: Dict[str, Any]) -> str:
        """
        Format session comparison results.

        Args:
            comparison: Comparison results

        Returns:
            Formatted output
        """
        if self.format_type == "json":
            return json.dumps(comparison, indent=2, default=self._json_serializer)

        elif self.format_type == "yaml":
            return yaml.dump(comparison, default_flow_style=False)

        else:  # table
            lines = []

            # Overall assessment
            overall = comparison.get("overall_assessment", "unknown")
            confidence = comparison.get("confidence_level", 0) * 100

            lines.append(f"\nSession Comparison Results")
            lines.append("=" * 26)
            lines.append(f"Overall Assessment: {overall.upper()}")
            lines.append(f"Confidence Level:   {confidence:.1f}%")

            # Metric comparisons
            metrics = comparison.get("metric_comparisons", [])
            if metrics:
                headers = ["Metric", "Baseline", "Current", "Change", "Impact"]
                rows = []

                for metric in metrics:
                    change_pct = metric.get("percentage_change", 0)
                    change_str = f"{change_pct:+.1f}%"
                    if metric.get("comparison_type") == "improvement":
                        change_str += " ↗"
                    elif metric.get("comparison_type") == "regression":
                        change_str += " ↘"

                    rows.append(
                        [
                            metric.get("metric_type", "").replace("_", " ").title(),
                            f"{metric.get('baseline_value', 0):.2f}",
                            f"{metric.get('comparison_value', 0):.2f}",
                            change_str,
                            metric.get("significance", "unknown"),
                        ]
                    )

                lines.append(
                    self.table_formatter.format_table(headers, rows, "Metric Changes")
                )

            # Recommendations
            recommendations = comparison.get("improvement_highlights", [])
            if recommendations:
                lines.append("\nImprovement Highlights:")
                for i, rec in enumerate(recommendations, 1):
                    lines.append(f"  {i}. {rec}")

            # Regressions
            regressions = comparison.get("regression_risks", [])
            if regressions:
                lines.append("\nRegression Risks:")
                for i, risk in enumerate(regressions, 1):
                    lines.append(f"  {i}. {risk}")

            return "\n".join(lines)

    def format_analysis_result(self, analysis: Dict[str, Any]) -> str:
        """
        Format analysis results.

        Args:
            analysis: Analysis results

        Returns:
            Formatted output
        """
        if self.format_type == "json":
            return json.dumps(analysis, indent=2, default=self._json_serializer)

        elif self.format_type == "yaml":
            return yaml.dump(analysis, default_flow_style=False)

        else:  # table
            lines = []

            # Overall results
            session_id = analysis.get("session_id", "unknown")[:16]
            overall_score = analysis.get("overall_score", 0)
            performance_grade = analysis.get("performance_grade", "unknown")

            lines.append(f"\nAnalysis Results - {session_id}")
            lines.append("=" * (18 + len(session_id)))
            lines.append(f"Overall Score:      {overall_score:.1f}/100")
            lines.append(f"Performance Grade:  {performance_grade.upper()}")

            # Detected patterns
            patterns = analysis.get("detected_patterns", [])
            if patterns:
                headers = ["Pattern", "Severity", "Location", "Description"]
                rows = []

                for pattern in patterns[:10]:  # Show top 10
                    location = pattern.get("source_location", {})
                    filename = Path(location.get("filename", "")).name
                    line_num = location.get("line_number", "")
                    loc_str = f"{filename}:{line_num}" if line_num else filename

                    rows.append(
                        [
                            pattern.get("pattern_type", "").replace("_", " ").title(),
                            pattern.get("severity", "").upper(),
                            loc_str,
                            pattern.get("description", "")[:50] + "...",
                        ]
                    )

                lines.append(
                    self.table_formatter.format_table(
                        headers, rows, f"Detected Patterns ({len(patterns)} total)"
                    )
                )

            # Recommendations
            recommendations = analysis.get("recommendations", [])
            if recommendations:
                lines.append(
                    f"\nOptimization Recommendations ({len(recommendations)} total):"
                )
                for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                    improvement = rec.get("estimated_improvement", 0)
                    confidence = rec.get("confidence", 0) * 100
                    lines.append(f"  {i}. {rec.get('title', '')}")
                    lines.append(
                        f"     Impact: {improvement:.1f}x, Confidence: {confidence:.0f}%"
                    )
                    lines.append(f"     {rec.get('description', '')}")
                    lines.append("")

            return "\n".join(lines)

    def format_status(self, status: Dict[str, Any]) -> str:
        """
        Format system status information.

        Args:
            status: Status information

        Returns:
            Formatted output
        """
        if self.format_type == "json":
            return json.dumps(status, indent=2, default=self._json_serializer)

        elif self.format_type == "yaml":
            return yaml.dump(status, default_flow_style=False)

        else:  # table
            return self.table_formatter.format_key_value(
                status, "Pycroscope System Status"
            )

    def format_error(self, error_message: str) -> str:
        """
        Format error message.

        Args:
            error_message: Error message

        Returns:
            Formatted error output
        """
        if self.format_type in ["json", "yaml"]:
            error_data = {
                "error": True,
                "message": error_message,
                "timestamp": datetime.now().isoformat(),
            }

            if self.format_type == "json":
                return json.dumps(error_data, indent=2)
            else:
                return yaml.dump(error_data)

        else:  # table
            return f"Error: {error_message}"

    def _format_timestamp(self, timestamp: Optional[str]) -> str:
        """Format timestamp for display."""
        if not timestamp:
            return ""

        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, AttributeError):
            return str(timestamp)[:16]

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for special types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return str(obj)
