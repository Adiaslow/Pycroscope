"""
Command-line interface for Pycroscope.

Provides simple CLI access to profiling capabilities
for quick testing and basic usage.
"""

import sys
import click
from pathlib import Path

from .core.config import ProfileConfig
from .core.session import ProfileSession
from .infrastructure.profilers.orchestra import ProfilerOrchestra


@click.group()
@click.version_option(version="1.0.0")
def main():
    """Pycroscope: Python performance analysis using established profiling tools."""
    pass


@main.command()
@click.argument("script_path", type=click.Path(exists=True))
@click.option("--line/--no-line", default=True, help="Enable line profiling")
@click.option("--memory/--no-memory", default=True, help="Enable memory profiling")
@click.option("--call/--no-call", default=True, help="Enable call profiling")
@click.option("--output-dir", type=click.Path(), help="Output directory for results")
@click.option("--minimal", is_flag=True, help="Use minimal overhead configuration")
def profile(script_path, line, memory, call, output_dir, minimal):
    """Profile a Python script with all profilers enabled by default."""

    # Create configuration
    config = ProfileConfig(
        line_profiling=line,
        memory_profiling=memory,
        call_profiling=call,
        output_dir=output_dir or Path("./profiling_results"),
    )

    # Use minimal overhead if requested (disable some features)
    if minimal:
        config.line_profiling = False
        config.create_visualizations = False
        config.analyze_patterns = False

    # Create session and orchestrator
    session = ProfileSession.create(config)
    orchestrator = ProfilerOrchestra(session)

    # Run profiling
    click.echo(f"Profiling {script_path}...")

    started_profilers = orchestrator.start_profiling()
    if started_profilers:
        # Execute the script in the profiled environment
        script_path = Path(script_path)

        # Add script directory to Python path
        sys.path.insert(0, str(script_path.parent))

        # Execute the script
        exec(script_path.read_text(), {"__file__": str(script_path)})

        # Stop profiling and get results
        results = orchestrator.stop_profiling()
        session.complete()

        click.echo(f"Profiling complete! Session ID: {session.session_id}")
        click.echo(f"Duration: {session.duration:.3f}s")
        click.echo(f"Profilers used: {', '.join(results.keys())}")

        if config.save_raw_data:
            saved_path = session.save()
            click.echo(f"Results saved to: {saved_path}")


@main.command()
@click.option(
    "--sessions-dir",
    type=click.Path(exists=True),
    help="Directory containing saved sessions",
)
def list_sessions(sessions_dir):
    """List saved profiling sessions."""
    if sessions_dir is None:
        click.echo("Please specify --sessions-dir or set output directory")
        return

    sessions_dir = Path(sessions_dir)
    session_files = list(sessions_dir.glob("profiling_data.json"))

    if not session_files:
        click.echo(f"No sessions found in {sessions_dir}")
        return

    click.echo(f"Found {len(session_files)} sessions in {sessions_dir}:")

    for session_file in sorted(session_files):
        try:
            import json

            with open(session_file, "r") as f:
                session_data = json.load(f)

            click.echo(f"  {session_file.name}")
            click.echo(f"    Status: {session_data.get('status', 'unknown')}")
            duration = session_data.get("duration", 0)
            if duration:
                click.echo(f"    Duration: {duration:.3f}s")
            profilers = list(session_data.get("results", {}).keys())
            if profilers:
                click.echo(f"    Profilers: {', '.join(profilers)}")
        except Exception as e:
            click.echo(f"  {session_file.name} - Error reading: {e}")


if __name__ == "__main__":
    main()
