"""
Analysis orchestrator for performance anti-pattern detection.

Integrates pattern analysis with Pycroscope's existing profiling infrastructure
following the framework's architectural patterns and providing seamless integration.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import concurrent.futures
import json

from .interfaces import (
    AnalysisOrchestrator,
    PatternAnalyzer,
    AnalysisResult,
    PatternType,
)
from .config import AnalysisConfig
from .detectors import AntiPatternDetector
from ..core.session import ProfileSession
from ..core.exceptions import PycroscopeError


class PerformanceAnalysisOrchestrator(AnalysisOrchestrator):
    """
    Orchestrates performance anti-pattern analysis across the codebase.

    Integrates with Pycroscope's existing profiling session system and
    provides comprehensive pattern analysis with correlation to profiling data.
    """

    def __init__(
        self, config: AnalysisConfig, session: Optional[ProfileSession] = None
    ):
        self.config = config
        self.session = session
        self._analyzers: Dict[str, PatternAnalyzer] = {}
        self._anti_pattern_detector = AntiPatternDetector(config)

    def register_analyzer(self, analyzer: PatternAnalyzer) -> None:
        """Register a pattern analyzer."""
        self._analyzers[analyzer.analyzer_type] = analyzer

    def run_analysis(
        self, code_files: List[Path], profiling_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[AnalysisResult]]:
        """Run comprehensive analysis on multiple files."""
        if not self.config.enabled:
            return {}

        all_results = {}

        # Use profiling data from session if not provided
        if profiling_data is None and self.session:
            profiling_data = self._extract_profiling_data()

        # Analyze files in parallel for better performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(
                    self._analyze_file, file_path, profiling_data
                ): file_path
                for file_path in code_files
                if file_path.suffix == ".py"
            }

            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                results = future.result()
                if results:
                    all_results[str(file_path)] = results

        # Post-process results
        all_results = self._post_process_results(all_results, profiling_data)

        return all_results

    def generate_report(
        self, results: Dict[str, List[AnalysisResult]]
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        if not results:
            return {"summary": "No analysis results available"}

        # Calculate summary statistics
        total_patterns = sum(len(file_results) for file_results in results.values())
        pattern_counts = self._count_patterns_by_type(results)
        severity_counts = self._count_patterns_by_severity(results)

        # Find top issues
        top_issues = self._get_top_issues(results)

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        report = {
            "summary": {
                "total_files_analyzed": len(results),
                "total_patterns_detected": total_patterns,
                "pattern_distribution": pattern_counts,
                "severity_distribution": severity_counts,
            },
            "top_issues": top_issues,
            "recommendations": recommendations,
            "detailed_results": self._format_detailed_results(results),
            "configuration": {
                "enabled_patterns": [p.value for p in self.config.enabled_patterns],
                "severity_threshold": self.config.severity_threshold,
                "confidence_threshold": self.config.confidence_threshold,
            },
        }

        if self.config.save_intermediate_results:
            self._save_intermediate_results(report)

        return report

    def prioritize_findings(
        self, results: List[AnalysisResult]
    ) -> List[AnalysisResult]:
        """Prioritize findings by severity, confidence, and performance correlation."""

        def priority_score(result: AnalysisResult) -> float:
            # Base score from severity
            severity_weights = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            base_score = severity_weights.get(result.severity, 0)

            # Boost for confidence
            confidence_boost = result.confidence

            # Major boost for performance correlation
            performance_boost = 0
            if result.profiling_correlation:
                impact = result.profiling_correlation.get("performance_impact", 0)
                strength = result.profiling_correlation.get(
                    "correlation_strength", "low"
                )
                strength_multiplier = {"low": 1.2, "medium": 1.5, "high": 2.0}.get(
                    strength, 1.0
                )
                performance_boost = min(
                    impact * strength_multiplier, 3.0
                )  # Cap the boost

            return base_score * confidence_boost + performance_boost

        return sorted(results, key=priority_score, reverse=True)

    def _analyze_file(
        self, file_path: Path, profiling_data: Optional[Dict[str, Any]]
    ) -> List[AnalysisResult]:
        """Analyze a single file for patterns."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return []

        results = []

        # Run performance anti-pattern detection
        if self.config.is_pattern_enabled(
            PatternType.NESTED_LOOPS
        ) or self.config.is_pattern_enabled(PatternType.QUADRATIC_COMPLEXITY):
            performance_results = (
                self._anti_pattern_detector.detect_performance_antipatterns(
                    code, file_path
                )
            )
            results.extend(performance_results)

        # Run maintainability analysis
        if any(
            self.config.is_pattern_enabled(p)
            for p in [
                PatternType.DEAD_CODE,
                PatternType.UNUSED_IMPORTS,
                PatternType.LOW_MAINTAINABILITY_INDEX,
            ]
        ):
            maintainability_results = (
                self._anti_pattern_detector.detect_maintainability_issues(
                    code, file_path
                )
            )
            results.extend(maintainability_results)

        # Run registered analyzers
        for analyzer in self._analyzers.values():
            if profiling_data and hasattr(analyzer, "analyze_with_profiling_data"):
                analyzer_results = analyzer.analyze_with_profiling_data(
                    code, file_path, profiling_data
                )
            else:
                analyzer_results = analyzer.analyze(code, file_path)
            results.extend(analyzer_results)

        # Filter results based on configuration
        results = [
            r
            for r in results
            if self.config.should_report_result(r.severity, r.confidence)
        ]

        # Don't limit results per file - show all patterns
        # Sort by priority but don't truncate
        if results:
            results = self.prioritize_findings(results)

        return results

    def _extract_profiling_data(self) -> Dict[str, Any]:
        """Extract profiling data from the current session."""
        if not self.session or not self.session.results:
            return {}

        profiling_data = {}
        for profiler_type, result in self.session.results.items():
            if result.success and result.data:
                profiling_data[profiler_type] = result.data

        return profiling_data

    def _post_process_results(
        self,
        results: Dict[str, List[AnalysisResult]],
        profiling_data: Optional[Dict[str, Any]],
    ) -> Dict[str, List[AnalysisResult]]:
        """Post-process results with profiling correlation."""
        if not profiling_data or not self.config.correlate_with_profiling:
            return results

        processed_results = {}
        for file_path, file_results in results.items():
            # Correlate with hotspots
            correlated_results = self._anti_pattern_detector.correlate_with_hotspots(
                file_results, profiling_data
            )
            # Prioritize results
            processed_results[file_path] = self.prioritize_findings(correlated_results)

        return processed_results

    def _count_patterns_by_type(
        self, results: Dict[str, List[AnalysisResult]]
    ) -> Dict[str, int]:
        """Count patterns by type across all files."""
        counts = {}
        for file_results in results.values():
            for result in file_results:
                pattern_name = result.pattern_type.value
                counts[pattern_name] = counts.get(pattern_name, 0) + 1
        return counts

    def _count_patterns_by_severity(
        self, results: Dict[str, List[AnalysisResult]]
    ) -> Dict[str, int]:
        """Count patterns by severity across all files."""
        counts = {}
        for file_results in results.values():
            for result in file_results:
                counts[result.severity] = counts.get(result.severity, 0) + 1
        return counts

    def _get_top_issues(
        self, results: Dict[str, List[AnalysisResult]], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top issues across all files."""
        all_results = []
        for file_path, file_results in results.items():
            for result in file_results:
                all_results.append(
                    {
                        "file": file_path,
                        "pattern_type": result.pattern_type.value,
                        "severity": result.severity,
                        "confidence": result.confidence,
                        "description": result.description,
                        "location": result.location,
                        "suggestion": result.suggestion,
                        "performance_correlation": result.profiling_correlation
                        is not None,
                    }
                )

        # Prioritize and return top issues
        prioritized = self.prioritize_findings(
            [
                AnalysisResult(
                    pattern_type=PatternType(issue["pattern_type"]),
                    severity=issue["severity"],
                    confidence=issue["confidence"],
                    description=issue["description"],
                    location=issue["location"],
                    suggestion=issue["suggestion"],
                    metrics={},
                    profiling_correlation=issue.get("profiling_correlation"),
                )
                for issue in all_results
            ]
        )

        return [
            {
                "pattern_type": result.pattern_type.value,
                "severity": result.severity,
                "description": result.description,
                "file": result.location.get("file", ""),
                "line": result.location.get("line", 0),
                "function": result.location.get("function", ""),
                "suggestion": result.suggestion,
                "performance_correlated": result.profiling_correlation is not None,
            }
            for result in prioritized[:limit]
        ]

    def _generate_recommendations(
        self, results: Dict[str, List[AnalysisResult]]
    ) -> List[str]:
        """Generate high-level recommendations based on analysis results."""
        recommendations = []
        pattern_counts = self._count_patterns_by_type(results)

        # Algorithmic complexity recommendations
        if (
            pattern_counts.get("nested_loops", 0) > 0
            or pattern_counts.get("quadratic_complexity", 0) > 0
        ):
            recommendations.append(
                "Consider optimizing algorithmic complexity in functions with nested loops or high time complexity"
            )

        # Code quality recommendations
        if pattern_counts.get("high_cyclomatic_complexity", 0) > 0:
            recommendations.append(
                "Break down complex functions to improve readability and maintainability"
            )

        # Dead code recommendations
        if (
            pattern_counts.get("dead_code", 0) > 0
            or pattern_counts.get("unused_imports", 0) > 0
        ):
            recommendations.append(
                "Remove unused code and imports to improve code cleanliness"
            )

        # Maintainability recommendations
        if pattern_counts.get("low_maintainability_index", 0) > 0:
            recommendations.append(
                "Improve code maintainability by reducing complexity and adding documentation"
            )

        # Performance correlation recommendations
        any_correlated = any(
            any(result.profiling_correlation for result in file_results)
            for file_results in results.values()
        )
        if any_correlated:
            recommendations.append(
                "Prioritize fixing patterns detected in performance hotspots for maximum impact"
            )

        if not recommendations:
            recommendations.append(
                "Code quality looks good! Consider enabling more pattern detection types."
            )

        return recommendations

    def _format_detailed_results(
        self, results: Dict[str, List[AnalysisResult]]
    ) -> Dict[str, Any]:
        """Format detailed results for the report."""
        formatted = {}
        for file_path, file_results in results.items():
            formatted[file_path] = [
                {
                    "pattern_type": result.pattern_type.value,
                    "severity": result.severity,
                    "confidence": result.confidence,
                    "description": result.description,
                    "location": result.location,
                    "suggestion": (
                        result.suggestion if self.config.include_suggestions else None
                    ),
                    "metrics": result.metrics,
                    "profiling_correlation": result.profiling_correlation,
                }
                for result in file_results
            ]
        return formatted

    def _save_intermediate_results(self, report: Dict[str, Any]) -> None:
        """Save intermediate results for debugging."""
        if self.session and self.session.config.output_dir:
            output_path = (
                self.session.config.output_dir / "analysis_intermediate_results.json"
            )
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"Intermediate analysis results saved to: {output_path}")
            except Exception as e:
                print(f"Warning: Could not save intermediate results: {e}")


def create_analysis_orchestrator(
    session: ProfileSession, config: Optional[AnalysisConfig] = None
) -> PerformanceAnalysisOrchestrator:
    """
    Factory function to create analysis orchestrator integrated with profiling session.

    Args:
        session: Active profiling session
        config: Analysis configuration (uses defaults if None)

    Returns:
        Configured analysis orchestrator
    """
    if config is None:
        config = AnalysisConfig(enabled=True)

    orchestrator = PerformanceAnalysisOrchestrator(config, session)

    # Register scientific computing detector for numerical code analysis
    from .detectors import ScientificComputingDetector

    scientific_detector = ScientificComputingDetector(config)
    orchestrator.register_analyzer(scientific_detector)

    return orchestrator
