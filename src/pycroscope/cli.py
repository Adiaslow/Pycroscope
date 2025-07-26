"""
Command-line interface for Pycroscope.

Provides simple CLI access to profiling capabilities
for quick testing and basic usage.
"""

import sys
import click
from pathlib import Path

from .core.config import ProfileConfig
from .infrastructure.profilers.orchestra import ProfilerOrchestra
from .core.session import ProfileSession


@click.group()
@click.version_option(version="2.0.0")
def main():
    """Pycroscope: Python performance analysis using established profiling tools."""
    pass


@main.command()
@click.argument("script_path", type=click.Path(exists=True))
@click.option("--line/--no-line", default=True, help="Enable line profiling")
@click.option("--memory/--no-memory", default=True, help="Enable memory profiling")
@click.option("--call/--no-call", default=True, help="Enable call profiling")
@click.option(
    "--sampling/--no-sampling", default=False, help="Enable sampling profiling"
)
@click.option("--output-dir", type=click.Path(), help="Output directory for results")
@click.option("--minimal", is_flag=True, help="Use minimal overhead configuration")
def profile(script_path, line, memory, call, sampling, output_dir, minimal):
    """Profile a Python script."""

    # Create configuration
    config = ProfileConfig(
        line_profiling=line,
        memory_profiling=memory,
        call_profiling=call,
        sampling_profiling=sampling,
        output_dir=output_dir,
    )

    # Use minimal overhead if requested
    if minimal:
        config = config.with_minimal_overhead()

    # Set up profiler orchestrator
    orchestrator = ProfilerOrchestra(config)

    # Run profiling
    click.echo(f"Profiling {script_path}...")

    if orchestrator.start_profiling():
        # Execute the script in the profiled environment
        script_path = Path(script_path)

        # Add script directory to Python path
        sys.path.insert(0, str(script_path.parent))

        # Execute the script
        exec(script_path.read_text(), {"__file__": str(script_path)})

        # Stop profiling and get results
        session = orchestrator.stop_profiling()

        click.echo(f"Profiling complete! Session ID: {session.session_id}")
        click.echo(f"Duration: {session.duration:.3f}s")
        click.echo(f"Profilers used: {', '.join(session.get_completed_profilers())}")

        if config.save_raw_data:
            click.echo(f"Results saved to: {config.output_dir}")


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
    session_files = list(sessions_dir.glob("session_*.json"))

    if not session_files:
        click.echo(f"No sessions found in {sessions_dir}")
        return

    click.echo(f"Found {len(session_files)} sessions in {sessions_dir}:")

    for session_file in sorted(session_files):
        session = ProfileSession.load(session_file)
        summary = session.summary()

        click.echo(f"  {session_file.name}")
        click.echo(f"    Status: {summary['status']}")
        click.echo(f"    Duration: {summary['duration']:.3f}s")
        click.echo(f"    Profilers: {', '.join(summary['completed_profilers'])}")


@main.command()
@click.option(
    "--output-dir", type=click.Path(), help="Output directory for demo results"
)
def demo(output_dir):
    """Run a demo profiling session (Pycroscope profiling itself)."""

    # Create configuration for demo
    config = ProfileConfig(
        line_profiling=True,
        memory_profiling=True,
        call_profiling=True,
        sampling_profiling=False,  # Keep it simple for demo
        output_dir=output_dir,
        session_name="pycroscope_demo",
    )

    orchestrator = ProfilerOrchestra(config)

    click.echo("Running Pycroscope demo (profiling itself)...")

    with orchestrator.profile() as session:
        # Demo workload: Pycroscope analyzing its own operations
        click.echo("  Creating sample data structures...")
        data = []
        for i in range(1000):
            data.append({"index": i, "value": i * 2, "text": f"item_{i}"})

        click.echo("  Processing data...")
        processed = []
        for item in data:
            if item["value"] % 10 == 0:
                processed.append(
                    {
                        "original_index": item["index"],
                        "computed_value": item["value"] ** 2,
                        "label": item["text"].upper(),
                    }
                )

        click.echo("  Performing calculations...")
        total_value = sum(item["computed_value"] for item in processed)
        average_value = total_value / len(processed) if processed else 0

        click.echo(f"  Demo complete: {len(processed)} items processed")
        click.echo(f"  Total value: {total_value}, Average: {average_value:.2f}")

    click.echo(f"Demo profiling complete!")
    click.echo(f"Session ID: {session.session_id}")
    click.echo(f"Duration: {session.duration:.3f}s")
    click.echo(f"Profilers used: {', '.join(session.get_completed_profilers())}")

    if config.save_raw_data:
        click.echo(f"Demo results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
