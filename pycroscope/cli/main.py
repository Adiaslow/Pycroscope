"""
Main CLI entry point for Pycroscope.

Provides command-line interface for profiling operations with comprehensive
argument parsing and subcommand dispatch.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from ..core.config import ConfigLoader, ProfileConfig
from .commands import (
    AnalyzeCommand,
    CleanupCommand,
    CompareCommand,
    ConfigCommand,
    DeleteCommand,
    ExportCommand,
    ListCommand,
    ProfileCommand,
    StatusCommand,
)
from .formatters import ResultFormatter


def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser with all subcommands.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="pycroscope",
        description="Advanced Python profiling and optimization framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile a Python script
  pycroscope profile my_script.py
  
  # Profile with custom configuration
  pycroscope profile --config myconfig.yaml my_script.py
  
  # List all profiling sessions
  pycroscope list
  
  # Analyze a stored session
  pycroscope analyze session_id
  
  # Compare two sessions
  pycroscope compare session1 session2
  
  # Delete old sessions
  pycroscope delete session1 session2
  
  # Cleanup old sessions
  pycroscope cleanup --max-age-days 30
  
  # Show system status
  pycroscope status
  
  # Manage configuration
  pycroscope config show
  
  # Export analysis results
  pycroscope export session_id --format json

Advanced Usage:
  # Profile with specific collectors
  pycroscope profile --collectors line memory call my_script.py
  
  # Analyze with specific analyzers
  pycroscope analyze session_id --analyzers pattern correlation
  
  # Compare sessions with detailed output
  pycroscope compare session1 session2 --detailed
  
  # Export with analysis results
  pycroscope export session_id --include-analysis --format yaml
        """,
    )

    # Global options
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    parser.add_argument("--config", type=str, help="Path to configuration file")

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (use -vv for debug level)",
    )

    parser.add_argument(
        "--output-format",
        choices=["table", "json", "yaml"],
        default="table",
        help="Output format for results",
    )

    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
        help="Command to execute",
    )

    # Profile command
    profile_parser = subparsers.add_parser(
        "profile",
        help="Profile a Python script or module",
        description="Start profiling a Python script or module with comprehensive data collection",
    )
    ProfileCommand.add_arguments(profile_parser)

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a stored profiling session",
        description="Run advanced analysis on a stored profiling session",
    )
    AnalyzeCommand.add_arguments(analyze_parser)

    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List profiling sessions",
        description="List available profiling sessions with filtering options",
    )
    ListCommand.add_arguments(list_parser)

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two profiling sessions",
        description="Compare performance between two sessions with detailed analysis",
    )
    CompareCommand.add_arguments(compare_parser)

    # Delete command
    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete profiling sessions",
        description="Delete one or more profiling sessions",
    )
    DeleteCommand.add_arguments(delete_parser)

    # Cleanup command
    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Cleanup old profiling sessions",
        description="Remove old profiling sessions based on age or count limits",
    )
    CleanupCommand.add_arguments(cleanup_parser)

    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show system status",
        description="Display Pycroscope system status and configuration",
    )
    StatusCommand.add_arguments(status_parser)

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Manage configuration",
        description="View and manage Pycroscope configuration settings",
    )
    ConfigCommand.add_arguments(config_parser)

    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export analysis results",
        description="Export profiling session data and analysis results in various formats",
    )
    ExportCommand.add_arguments(export_parser)

    return parser


def load_config(config_path: Optional[str] = None) -> ProfileConfig:
    """
    Load configuration from file or use defaults.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Loaded configuration
    """
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            print(
                f"Error: Configuration file not found: {config_path}", file=sys.stderr
            )
            sys.exit(1)

        try:
            return ConfigLoader.from_file(config_file)
        except Exception as e:
            print(f"Error loading configuration: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        return ProfileConfig()


def setup_logging(verbosity: int) -> None:
    """
    Setup logging based on verbosity level.

    Args:
        verbosity: Verbosity level (0=normal, 1=verbose, 2=debug)
    """
    import logging

    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set specific loggers for our components
    if verbosity >= 2:
        logging.getLogger("pycroscope").setLevel(logging.DEBUG)
    elif verbosity >= 1:
        logging.getLogger("pycroscope").setLevel(logging.INFO)


def print_welcome_message():
    """Print welcome message for first-time users."""
    print("🔬 Pycroscope - Advanced Python Performance Profiler")
    print("=" * 60)
    print("Complete profiling and optimization framework")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.

    Args:
        argv: Optional command line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()

    # Parse arguments
    if argv is None:
        argv = sys.argv[1:]

    # Handle no arguments case
    if not argv:
        print_welcome_message()
        parser.print_help()
        return 1

    args = parser.parse_args(argv)

    # Show help if no command specified
    if not hasattr(args, "command") or not args.command:
        print_welcome_message()
        parser.print_help()
        return 1

    # Setup logging
    setup_logging(args.verbose)

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Create result formatter with color support
    formatter = ResultFormatter(format_type=args.output_format)

    # Dispatch to appropriate command
    try:
        if args.command == "profile":
            command = ProfileCommand(config, formatter)
            return command.execute(args)

        elif args.command == "analyze":
            command = AnalyzeCommand(config, formatter)
            return command.execute(args)

        elif args.command == "list":
            command = ListCommand(config, formatter)
            return command.execute(args)

        elif args.command == "compare":
            command = CompareCommand(config, formatter)
            return command.execute(args)

        elif args.command == "delete":
            command = DeleteCommand(config, formatter)
            return command.execute(args)

        elif args.command == "cleanup":
            command = CleanupCommand(config, formatter)
            return command.execute(args)

        elif args.command == "status":
            command = StatusCommand(config, formatter)
            return command.execute(args)

        elif args.command == "config":
            command = ConfigCommand(config, formatter)
            return command.execute(args)

        elif args.command == "export":
            command = ExportCommand(config, formatter)
            return command.execute(args)

        else:
            print(f"Error: Unknown command: {args.command}", file=sys.stderr)
            return 1

    except KeyboardInterrupt:
        print("\n⚡ Operation cancelled by user", file=sys.stderr)
        return 130

    except Exception as e:
        if args.verbose >= 2:
            import traceback

            traceback.print_exc()
        else:
            print(f"❌ Error: {e}", file=sys.stderr)
        return 1


def cli_entry_point():
    """Entry point for console script."""
    sys.exit(main())


if __name__ == "__main__":
    cli_entry_point()
