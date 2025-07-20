"""
CLI command implementations for Pycroscope.

Provides command classes for all CLI operations including profiling,
session management, comparison, and data export.
"""

import os
import sys
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from argparse import ArgumentParser, Namespace
from datetime import datetime

from .formatters import ResultFormatter
from ..core.config import ProfileConfig, AnalysisType
from ..core.profiler_suite import ProfilerSuite
from ..core.registry import ComponentRegistry
from ..storage.file_store import FileDataStore
from ..storage.memory_store import MemoryDataStore
from ..storage.session_comparer import SessionComparer, quick_compare_sessions
from ..analysis.engine import AnalysisEngine
from ..analysis.static_analyzer import StaticAnalyzer
from ..analysis.dynamic_analyzer import DynamicAnalyzer


class BaseCommand(ABC):
    """Abstract base class for all CLI commands."""

    def __init__(self, config: ProfileConfig, formatter: ResultFormatter):
        """
        Initialize command.

        Args:
            config: Profile configuration
            formatter: Output formatter
        """
        self.config = config
        self.formatter = formatter

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """Add command-specific arguments to parser."""
        pass

    @abstractmethod
    def execute(self, args: Namespace) -> int:
        """
        Execute the command.

        Args:
            args: Parsed command arguments

        Returns:
            Exit code (0 for success)
        """
        pass


class ProfileCommand(BaseCommand):
    """Command to profile a Python script or module."""

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """Add profile command arguments."""
        parser.add_argument("target", help="Python script or module to profile")

        parser.add_argument(
            "--args", nargs="*", help="Arguments to pass to the target script"
        )

        parser.add_argument(
            "--session-id", help="Custom session ID (auto-generated if not provided)"
        )

        parser.add_argument(
            "--timeout",
            type=int,
            default=300,
            help="Maximum profiling time in seconds (default: 300)",
        )

        parser.add_argument("--output-dir", help="Directory to save profiling results")

        parser.add_argument(
            "--analyze", action="store_true", help="Run analysis after profiling"
        )

        parser.add_argument(
            "--no-analysis", action="store_true", help="Skip automatic analysis"
        )

    def execute(self, args: Namespace) -> int:
        """Execute profiling command."""
        try:
            # Validate target
            target_path = Path(args.target)
            if not target_path.exists():
                print(f"Error: Target file not found: {args.target}")
                return 1

            # Setup profiler
            print(f"Starting profiling session for: {args.target}")

            # Create profiler suite
            profiler = ProfilerSuite(self.config)

            # Setup storage
            if args.output_dir:
                output_path = Path(args.output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                storage_config = self.config.storage
                storage_config.storage_path = output_path

            # Start profiling
            with profiler:
                try:
                    # Create target command
                    cmd = [sys.executable, str(target_path)]
                    if args.args:
                        cmd.extend(args.args)

                    # Run target with profiling
                    process = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )

                    # Wait for completion with timeout
                    try:
                        stdout, stderr = process.communicate(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        print(
                            f"Error: Profiling timed out after {args.timeout} seconds"
                        )
                        return 1

                    if process.returncode != 0:
                        print(
                            f"Warning: Target script exited with code {process.returncode}"
                        )
                        if stderr:
                            print(f"Script stderr: {stderr}")

                except Exception as e:
                    print(f"Error running target script: {e}")
                    return 1

                # Get profiling session
                session = profiler.current_session
                if not session:
                    print("Error: No profiling session created")
                    return 1

                print(f"Profiling completed. Session ID: {session.session_id}")
                print(f"Total events collected: {session.total_events}")
                print(f"Peak memory usage: {session.peak_memory / (1024*1024):.1f} MB")

                # Run analysis if requested
                if args.analyze and not args.no_analysis:
                    print("\nRunning analysis...")

                    # Setup analysis engine
                    analysis_engine = AnalysisEngine(self.config.analysis)
                    analysis_engine.register_analyzer(
                        AnalysisType.STATIC, StaticAnalyzer()
                    )
                    analysis_engine.register_analyzer(
                        AnalysisType.DYNAMIC, DynamicAnalyzer()
                    )

                    # Run analysis
                    analysis_result = analysis_engine.analyze(session)

                    # Display results
                    result_data = {
                        "session_id": analysis_result.session_id,
                        "overall_score": analysis_result.overall_score,
                        "performance_grade": analysis_result.performance_grade,
                        "detected_patterns": [
                            {
                                "pattern_type": p.pattern_type,
                                "severity": p.severity,
                                "description": p.description,
                                "source_location": {
                                    "filename": p.source_location.filename,
                                    "line_number": p.source_location.line_number,
                                    "function_name": p.source_location.function_name,
                                },
                            }
                            for p in analysis_result.detected_patterns
                        ],
                        "recommendations": [
                            {
                                "title": r.title,
                                "description": r.description,
                                "estimated_improvement": r.estimated_improvement,
                                "confidence": r.confidence,
                                "complexity": r.complexity,
                            }
                            for r in analysis_result.recommendations
                        ],
                    }

                    print(self.formatter.format_analysis_result(result_data))

            return 0

        except Exception as e:
            print(f"Error during profiling: {e}")
            return 1


class ListCommand(BaseCommand):
    """Command to list profiling sessions."""

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """Add list command arguments."""
        parser.add_argument("--package", help="Filter by target package name")

        parser.add_argument(
            "--limit",
            type=int,
            default=20,
            help="Maximum number of sessions to show (default: 20)",
        )

        parser.add_argument("--storage-path", help="Path to storage directory")

    def execute(self, args: Namespace) -> int:
        """Execute list command."""
        try:
            # Setup storage
            storage_config = self.config.storage
            if args.storage_path:
                storage_config.storage_path = args.storage_path

            store = FileDataStore(storage_config)

            # Get session list
            session_ids = store.list_sessions(
                package_name=args.package, limit=args.limit
            )

            if not session_ids:
                print(self.formatter.format_session_list([]))
                return 0

            # Get metadata for each session
            sessions = []
            for session_id in session_ids:
                metadata = store.get_session_metadata(session_id)
                if metadata:
                    sessions.append(metadata)

            # Display results
            print(self.formatter.format_session_list(sessions))

            return 0

        except Exception as e:
            print(f"Error listing sessions: {e}")
            return 1


class CompareCommand(BaseCommand):
    """Command to compare two profiling sessions."""

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """Add compare command arguments."""
        parser.add_argument("baseline", help="Baseline session ID")

        parser.add_argument("comparison", help="Comparison session ID")

        parser.add_argument("--storage-path", help="Path to storage directory")

        parser.add_argument(
            "--detailed", action="store_true", help="Show detailed comparison results"
        )

    def execute(self, args: Namespace) -> int:
        """Execute compare command."""
        try:
            # Setup storage
            storage_config = self.config.storage
            if args.storage_path:
                storage_config.storage_path = args.storage_path

            store = FileDataStore(storage_config)

            # Load sessions
            baseline_session = store.load_session(args.baseline)
            if not baseline_session:
                print(f"Error: Baseline session not found: {args.baseline}")
                return 1

            comparison_session = store.load_session(args.comparison)
            if not comparison_session:
                print(f"Error: Comparison session not found: {args.comparison}")
                return 1

            # Perform comparison
            comparison_result = quick_compare_sessions(
                baseline_session, comparison_session
            )

            # Convert to display format
            comparison_data = {
                "baseline_session_id": comparison_result.baseline_session_id,
                "comparison_session_id": comparison_result.comparison_session_id,
                "overall_assessment": comparison_result.overall_assessment.value,
                "confidence_level": comparison_result.confidence_level,
                "metric_comparisons": [
                    {
                        "metric_type": mc.metric_type.value,
                        "baseline_value": mc.baseline_value,
                        "comparison_value": mc.comparison_value,
                        "percentage_change": mc.percentage_change,
                        "comparison_type": mc.comparison_type.value,
                        "significance": mc.significance,
                    }
                    for mc in comparison_result.metric_comparisons
                ],
                "improvement_highlights": comparison_result.improvement_highlights,
                "regression_risks": comparison_result.regression_risks,
            }

            # Display results
            print(self.formatter.format_session_comparison(comparison_data))

            return 0

        except Exception as e:
            print(f"Error comparing sessions: {e}")
            return 1


class StatusCommand(BaseCommand):
    """Command to show system status."""

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """Add status command arguments."""
        parser.add_argument("--storage-path", help="Path to storage directory")

    def execute(self, args: Namespace) -> int:
        """Execute status command."""
        try:
            # Gather system information
            status = {
                "pycroscope_version": "1.0.0",
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform,
                "current_time": datetime.now().isoformat(),
            }

            # Storage information
            storage_config = self.config.storage
            if args.storage_path:
                storage_config.storage_path = args.storage_path

            try:
                store = FileDataStore(storage_config)
                storage_stats = store.get_storage_stats()
                status.update(
                    {
                        "storage_path": storage_stats["storage_path"],
                        "total_sessions": storage_stats["total_sessions"],
                        "storage_size_mb": f"{storage_stats['total_size_mb']:.1f}",
                        "compression_enabled": storage_stats["compression_enabled"],
                    }
                )
            except Exception as e:
                status["storage_error"] = str(e)

            # Configuration status
            status.update(
                {
                    "enabled_collectors": [
                        c.value for c in self.config.get_enabled_collectors()
                    ],
                    "enabled_analyzers": [
                        a.value for a in self.config.analysis.enabled_analyzers
                    ],
                    "debug_mode": self.config.debug_mode,
                    "max_sessions": self.config.storage.max_sessions,
                    "retention_days": self.config.storage.retention_days,
                }
            )

            # Display results
            print(self.formatter.format_status(status))

            return 0

        except Exception as e:
            print(f"Error getting status: {e}")
            return 1


class ExportCommand(BaseCommand):
    """Command to export profiling session data."""

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """Add export command arguments."""
        parser.add_argument("session_id", help="Session ID to export")

        parser.add_argument("--output", help="Output file path (default: stdout)")

        parser.add_argument(
            "--format",
            choices=["json", "yaml"],
            default="json",
            help="Export format (default: json)",
        )

        parser.add_argument(
            "--include-analysis",
            action="store_true",
            help="Include analysis results in export",
        )

        parser.add_argument("--storage-path", help="Path to storage directory")

    def execute(self, args: Namespace) -> int:
        """Execute export command."""
        try:
            # Setup storage
            storage_config = self.config.storage
            if args.storage_path:
                storage_config.storage_path = args.storage_path

            store = FileDataStore(storage_config)

            # Load session
            session = store.load_session(args.session_id)
            if not session:
                print(f"Error: Session not found: {args.session_id}")
                return 1

            # Prepare export data
            export_data = {
                "session_id": session.session_id,
                "timestamp": session.timestamp.isoformat(),
                "target_package": session.target_package,
                "total_events": session.total_events,
                "peak_memory": session.peak_memory,
                "execution_events": [
                    {
                        "event_type": event.event_type,
                        "timestamp": event.timestamp.isoformat(),
                        "execution_time": event.execution_time,
                        "frame_info": {
                            "filename": event.frame_info.source_location.filename,
                            "line_number": event.frame_info.source_location.line_number,
                            "function_name": event.frame_info.source_location.function_name,
                        },
                    }
                    for event in session.execution_events[
                        :1000
                    ]  # Limit for export size
                ],
                "memory_snapshots": [
                    {
                        "timestamp": snapshot.timestamp.isoformat(),
                        "total_memory": snapshot.total_memory,
                        "process_memory": snapshot.process_memory,
                    }
                    for snapshot in session.memory_snapshots
                ],
            }

            # Include analysis if requested
            if args.include_analysis and session.analysis_result:
                export_data["analysis_result"] = {
                    "overall_score": session.analysis_result.overall_score,
                    "performance_grade": session.analysis_result.performance_grade,
                    "detected_patterns": len(session.analysis_result.detected_patterns),
                    "recommendations": len(session.analysis_result.recommendations),
                }

            # Format output
            if args.format == "json":
                import json

                output = json.dumps(export_data, indent=2)
            else:  # yaml
                import yaml

                output = yaml.dump(export_data, default_flow_style=False)

            # Write output
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    f.write(output)
                print(f"Session exported to: {output_path}")
            else:
                print(output)

            return 0

        except Exception as e:
            print(f"Error exporting session: {e}")
            return 1


class AnalyzeCommand(BaseCommand):
    """Command to analyze a stored profiling session."""

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """Add analyze command arguments."""
        parser.add_argument("session_id", help="Session ID to analyze")

        parser.add_argument("--storage-path", help="Path to storage directory")

        parser.add_argument(
            "--analyzers",
            nargs="*",
            choices=[
                "static",
                "dynamic",
                "pattern",
                "correlation",
                "complexity",
                "optimization",
            ],
            help="Specific analyzers to run (default: all)",
        )

        parser.add_argument("--output-file", help="Save analysis results to file")

        parser.add_argument(
            "--detailed", action="store_true", help="Include detailed analysis data"
        )

    def execute(self, args: Namespace) -> int:
        """Execute analyze command."""
        try:
            # Setup storage
            storage_config = self.config.storage
            if args.storage_path:
                storage_config.storage_path = args.storage_path

            store = FileDataStore(storage_config)

            # Load session
            session = store.load_session(args.session_id)
            if not session:
                print(f"Error: Session not found: {args.session_id}")
                return 1

            print(f"Running analysis on session: {args.session_id}")

            # Setup analysis engine with selected analyzers
            from ..analysis import (
                AnalysisEngine,
                StaticAnalyzer,
                DynamicAnalyzer,
                AdvancedPatternDetector,
                CrossCorrelationAnalyzer,
                AlgorithmComplexityDetector,
                OptimizationRecommendationEngine,
            )
            from ..core.config import AnalysisConfig, AnalysisType

            analysis_config = AnalysisConfig()

            # Configure enabled analyzers
            if args.analyzers:
                analysis_config.enabled_analyzers = set()
                analyzer_map = {
                    "static": AnalysisType.STATIC,
                    "dynamic": AnalysisType.DYNAMIC,
                    "pattern": AnalysisType.PATTERN,
                    "correlation": AnalysisType.CORRELATION,
                    "complexity": AnalysisType.COMPLEXITY,
                    "optimization": AnalysisType.OPTIMIZATION,
                }
                for analyzer_name in args.analyzers:
                    analysis_config.enabled_analyzers.add(analyzer_map[analyzer_name])
            else:
                # Enable all analyzers
                analysis_config.enabled_analyzers = {
                    AnalysisType.STATIC,
                    AnalysisType.DYNAMIC,
                    AnalysisType.PATTERN,
                    AnalysisType.CORRELATION,
                    AnalysisType.COMPLEXITY,
                    AnalysisType.OPTIMIZATION,
                }

            # Create analysis engine
            engine = AnalysisEngine(analysis_config)

            # Register analyzers
            engine.register_analyzer(AnalysisType.STATIC, StaticAnalyzer())
            engine.register_analyzer(AnalysisType.DYNAMIC, DynamicAnalyzer())
            engine.register_analyzer(AnalysisType.PATTERN, AdvancedPatternDetector())
            engine.register_analyzer(
                AnalysisType.CORRELATION, CrossCorrelationAnalyzer()
            )
            engine.register_analyzer(
                AnalysisType.COMPLEXITY, AlgorithmComplexityDetector()
            )
            engine.register_analyzer(
                AnalysisType.OPTIMIZATION, OptimizationRecommendationEngine()
            )

            # Run analysis
            analysis_result = engine.analyze(session)

            # Update session with analysis result
            session.analysis_result = analysis_result

            # Save updated session
            store.store_session(session)

            # Format and display results
            result_data = {
                "session_id": analysis_result.session_id,
                "analysis_timestamp": analysis_result.analysis_timestamp.isoformat(),
                "overall_score": analysis_result.overall_score,
                "performance_grade": analysis_result.performance_grade,
                "detected_patterns": [
                    {
                        "pattern_type": p.pattern_type,
                        "severity": p.severity,
                        "description": p.description,
                        "impact_estimate": p.impact_estimate,
                        "source_location": (
                            {
                                "filename": p.source_location.filename,
                                "line_number": p.source_location.line_number,
                                "function_name": p.source_location.function_name,
                            }
                            if p.source_location
                            else None
                        ),
                        "evidence": p.evidence if args.detailed else {},
                    }
                    for p in analysis_result.detected_patterns
                ],
                "recommendations": [
                    {
                        "title": getattr(r, "title", ""),
                        "description": getattr(r, "description", ""),
                        "category": getattr(r, "category", ""),
                        "estimated_improvement": getattr(
                            r, "estimated_improvement", 0.0
                        ),
                        "confidence": getattr(r, "confidence", 0.0),
                        "implementation_effort": getattr(
                            r, "implementation_effort", "medium"
                        ),
                        "suggested_actions": (
                            getattr(r, "suggested_actions", []) if args.detailed else []
                        ),
                        "code_examples": (
                            getattr(r, "code_examples", []) if args.detailed else []
                        ),
                    }
                    for r in analysis_result.recommendations
                ],
            }

            # Output results
            output = self.formatter.format_analysis_result(result_data)

            if args.output_file:
                with open(args.output_file, "w") as f:
                    f.write(output)
                print(f"Analysis results saved to: {args.output_file}")
            else:
                print(output)

            print(f"\nAnalysis completed. Results saved to session.")

            return 0

        except Exception as e:
            print(f"Error during analysis: {e}")
            return 1


class DeleteCommand(BaseCommand):
    """Command to delete profiling sessions."""

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """Add delete command arguments."""
        parser.add_argument("session_ids", nargs="+", help="Session IDs to delete")

        parser.add_argument("--storage-path", help="Path to storage directory")

        parser.add_argument(
            "--force", action="store_true", help="Delete without confirmation"
        )

        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be deleted without actually deleting",
        )

    def execute(self, args: Namespace) -> int:
        """Execute delete command."""
        try:
            # Setup storage
            storage_config = self.config.storage
            if args.storage_path:
                storage_config.storage_path = args.storage_path

            store = FileDataStore(storage_config)

            # Validate session IDs exist
            existing_sessions = []
            missing_sessions = []

            for session_id in args.session_ids:
                metadata = store.get_session_metadata(session_id)
                if metadata:
                    existing_sessions.append(session_id)
                else:
                    missing_sessions.append(session_id)

            if missing_sessions:
                print(f"Warning: Sessions not found: {', '.join(missing_sessions)}")

            if not existing_sessions:
                print("No valid sessions to delete.")
                return 1

            # Show what will be deleted
            print(f"Sessions to delete: {len(existing_sessions)}")
            for session_id in existing_sessions:
                metadata = store.get_session_metadata(session_id)
                if metadata:
                    timestamp = metadata.get("timestamp", "unknown")
                    package = metadata.get("target_package", "unknown")
                    print(f"  - {session_id} ({package}, {timestamp})")

            if args.dry_run:
                print("Dry run mode - no sessions were deleted.")
                return 0

            # Confirm deletion
            if not args.force:
                response = input(
                    f"\nDelete {len(existing_sessions)} session(s)? [y/N]: "
                )
                if response.lower() not in ["y", "yes"]:
                    print("Deletion cancelled.")
                    return 0

            # Delete sessions
            deleted_count = 0
            for session_id in existing_sessions:
                if store.delete_session(session_id):
                    deleted_count += 1
                    print(f"Deleted: {session_id}")
                else:
                    print(f"Failed to delete: {session_id}")

            print(f"\nDeleted {deleted_count} session(s).")

            return 0

        except Exception as e:
            print(f"Error deleting sessions: {e}")
            return 1


class CleanupCommand(BaseCommand):
    """Command to cleanup old profiling sessions."""

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """Add cleanup command arguments."""
        parser.add_argument(
            "--max-age-days",
            type=int,
            help="Maximum age in days (default: from config)",
        )

        parser.add_argument(
            "--max-sessions",
            type=int,
            help="Maximum number of sessions to keep (default: from config)",
        )

        parser.add_argument("--storage-path", help="Path to storage directory")

        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be cleaned without actually cleaning",
        )

        parser.add_argument(
            "--force", action="store_true", help="Cleanup without confirmation"
        )

    def execute(self, args: Namespace) -> int:
        """Execute cleanup command."""
        try:
            # Setup storage
            storage_config = self.config.storage
            if args.storage_path:
                storage_config.storage_path = args.storage_path

            store = FileDataStore(storage_config)

            # Get current stats
            stats = store.get_storage_stats()
            print(f"Current storage stats:")
            print(f"  Total sessions: {stats['total_sessions']}")
            print(f"  Storage size: {stats['total_size_mb']:.1f} MB")
            print(f"  Storage path: {stats['storage_path']}")

            cleanup_count = 0

            # Cleanup by age
            if args.max_age_days or (
                not args.max_sessions and store._config.retention_days > 0
            ):
                max_age = args.max_age_days or store._config.retention_days
                print(f"\nCleaning up sessions older than {max_age} days...")

                if args.dry_run:
                    # Simulate cleanup to count
                    from datetime import datetime, timedelta

                    cutoff_date = datetime.now() - timedelta(days=max_age)

                    old_sessions = []
                    for session_id in store.list_sessions():
                        metadata = store.get_session_metadata(session_id)
                        if metadata:
                            timestamp_str = metadata.get("timestamp", "")
                            try:
                                session_date = datetime.fromisoformat(
                                    timestamp_str.replace("Z", "+00:00")
                                )
                                if session_date < cutoff_date:
                                    old_sessions.append(session_id)
                            except ValueError:
                                old_sessions.append(session_id)  # Invalid timestamp

                    print(f"Would delete {len(old_sessions)} old sessions:")
                    for session_id in old_sessions[:10]:  # Show first 10
                        print(f"  - {session_id}")
                    if len(old_sessions) > 10:
                        print(f"  ... and {len(old_sessions) - 10} more")
                    cleanup_count += len(old_sessions)
                else:
                    cleanup_count += store.cleanup_old_sessions(max_age)

            # Cleanup by count
            if args.max_sessions:
                print(f"\nLimiting to {args.max_sessions} most recent sessions...")

                session_list = store.list_sessions()
                if len(session_list) > args.max_sessions:
                    excess_sessions = session_list[args.max_sessions :]

                    if args.dry_run:
                        print(f"Would delete {len(excess_sessions)} excess sessions:")
                        for session_id in excess_sessions[:10]:
                            print(f"  - {session_id}")
                        if len(excess_sessions) > 10:
                            print(f"  ... and {len(excess_sessions) - 10} more")
                        cleanup_count += len(excess_sessions)
                    else:
                        if not args.force:
                            response = input(
                                f"Delete {len(excess_sessions)} excess sessions? [y/N]: "
                            )
                            if response.lower() not in ["y", "yes"]:
                                print("Cleanup cancelled.")
                                return 0

                        for session_id in excess_sessions:
                            if store.delete_session(session_id):
                                cleanup_count += 1

            if args.dry_run:
                print(f"\nDry run mode - would clean up {cleanup_count} session(s).")
            else:
                print(f"\nCleaned up {cleanup_count} session(s).")

            # Show updated stats
            if not args.dry_run and cleanup_count > 0:
                new_stats = store.get_storage_stats()
                print(f"\nUpdated storage stats:")
                print(f"  Total sessions: {new_stats['total_sessions']}")
                print(f"  Storage size: {new_stats['total_size_mb']:.1f} MB")
                saved_mb = stats["total_size_mb"] - new_stats["total_size_mb"]
                print(f"  Space saved: {saved_mb:.1f} MB")

            return 0

        except Exception as e:
            print(f"Error during cleanup: {e}")
            return 1


class ConfigCommand(BaseCommand):
    """Command to manage configuration."""

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """Add config command arguments."""
        subparsers = parser.add_subparsers(
            dest="config_action", help="Configuration actions"
        )

        # Show config
        show_parser = subparsers.add_parser("show", help="Show current configuration")
        show_parser.add_argument("--section", help="Show specific config section")

        # Set config
        set_parser = subparsers.add_parser("set", help="Set configuration value")
        set_parser.add_argument(
            "key", help="Configuration key (e.g., storage.max_sessions)"
        )
        set_parser.add_argument("value", help="Configuration value")

        # Reset config
        reset_parser = subparsers.add_parser(
            "reset", help="Reset configuration to defaults"
        )
        reset_parser.add_argument(
            "--force", action="store_true", help="Reset without confirmation"
        )

        # Validate config
        validate_parser = subparsers.add_parser(
            "validate", help="Validate configuration"
        )

    def execute(self, args: Namespace) -> int:
        """Execute config command."""
        try:
            if args.config_action == "show":
                return self._show_config(args)
            elif args.config_action == "set":
                return self._set_config(args)
            elif args.config_action == "reset":
                return self._reset_config(args)
            elif args.config_action == "validate":
                return self._validate_config(args)
            else:
                print("Error: No config action specified. Use --help for options.")
                return 1

        except Exception as e:
            print(f"Error managing configuration: {e}")
            return 1

    def _show_config(self, args: Namespace) -> int:
        """Show current configuration."""
        config_data = {
            "general": {
                "target_package": self.config.target_package,
                "working_directory": str(self.config.working_directory),
                "debug_mode": self.config.debug_mode,
                "verbose": self.config.verbose,
                "parallel_collection": self.config.parallel_collection,
                "max_threads": self.config.max_threads,
                "timeout_seconds": self.config.timeout_seconds,
            },
            "collectors": {
                collector_type.value: {
                    "enabled": collector_config.enabled,
                    "sampling_rate": collector_config.sampling_rate,
                    "buffer_size": collector_config.buffer_size,
                }
                for collector_type, collector_config in self.config.collectors.items()
            },
            "analysis": {
                "enabled_analyzers": [
                    a.value for a in self.config.analysis.enabled_analyzers
                ],
                "confidence_threshold": self.config.analysis.confidence_threshold,
                "impact_threshold": self.config.analysis.impact_threshold,
            },
            "storage": {
                "storage_type": self.config.storage.storage_type.value,
                "storage_path": (
                    str(self.config.storage.storage_path)
                    if self.config.storage.storage_path
                    else None
                ),
                "compression": self.config.storage.compression,
                "max_sessions": self.config.storage.max_sessions,
                "retention_days": self.config.storage.retention_days,
                "auto_cleanup": self.config.storage.auto_cleanup,
            },
        }

        if args.section:
            if args.section in config_data:
                # Use format_status for single section
                section_data = {
                    f"{args.section}_{k}": v
                    for k, v in config_data[args.section].items()
                }
                output = self.formatter.format_status(section_data)
            else:
                print(f"Error: Unknown config section: {args.section}")
                print(f"Available sections: {', '.join(config_data.keys())}")
                return 1
        else:
            # Flatten config data for display
            flat_config = {}
            for section, values in config_data.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        flat_config[f"{section}.{key}"] = value
                else:
                    flat_config[section] = values
            output = self.formatter.format_status(flat_config)

        print(output)
        return 0

    def _set_config(self, args: Namespace) -> int:
        """Set configuration value."""
        print(f"Setting {args.key} = {args.value}")

        # This would implement config setting logic
        # For now, just show what would be set
        print("Configuration setting not implemented yet - would update config file")

        return 0

    def _reset_config(self, args: Namespace) -> int:
        """Reset configuration to defaults."""
        if not args.force:
            response = input("Reset configuration to defaults? [y/N]: ")
            if response.lower() not in ["y", "yes"]:
                print("Reset cancelled.")
                return 0

        print("Configuration reset not implemented yet - would restore default config")

        return 0

    def _validate_config(self, args: Namespace) -> int:
        """Validate current configuration."""
        print("Validating configuration...")

        issues = []

        # Check storage path
        if self.config.storage.storage_path:
            if not self.config.storage.storage_path.exists():
                issues.append(
                    f"Storage path does not exist: {self.config.storage.storage_path}"
                )

        # Check thread count
        if self.config.max_threads < 1:
            issues.append("max_threads must be at least 1")

        # Check thresholds
        if not 0.0 <= self.config.analysis.confidence_threshold <= 1.0:
            issues.append("analysis.confidence_threshold must be between 0.0 and 1.0")

        if not 0.0 <= self.config.analysis.impact_threshold <= 1.0:
            issues.append("analysis.impact_threshold must be between 0.0 and 1.0")

        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        else:
            print("Configuration is valid.")
            return 0
