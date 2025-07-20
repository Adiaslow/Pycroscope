"""
Command-line interface for Pycroscope.

Provides a comprehensive CLI for profiling operations, session management,
analysis execution, and optimization insights visualization.

Complete CLI Commands:
- profile: Profile Python scripts with comprehensive data collection
- analyze: Run advanced analysis on stored sessions
- list: List and filter profiling sessions
- compare: Compare sessions with detailed performance analysis
- delete: Delete profiling sessions with confirmation
- cleanup: Clean up old sessions by age or count
- status: Show system status and configuration
- config: Manage configuration settings
- export: Export data and analysis results in multiple formats

Advanced Features:
- Multiple output formats (table, JSON, YAML)
- Configurable analyzers and collectors
- Interactive confirmations and dry-run modes
- Comprehensive error handling and logging
- Session management with metadata
- Storage cleanup and maintenance
"""

from .main import main, cli_entry_point
from .commands import (
    ProfileCommand,
    ListCommand,
    CompareCommand,
    StatusCommand,
    ExportCommand,
    AnalyzeCommand,
    DeleteCommand,
    CleanupCommand,
    ConfigCommand,
)
from .formatters import ResultFormatter, TableFormatter

__all__ = [
    "main",
    "cli_entry_point",
    "ProfileCommand",
    "AnalyzeCommand",
    "ListCommand",
    "CompareCommand",
    "DeleteCommand",
    "CleanupCommand",
    "StatusCommand",
    "ConfigCommand",
    "ExportCommand",
    "ResultFormatter",
    "TableFormatter",
]
