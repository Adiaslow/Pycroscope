"""
Main analysis engine for the Pycroscope system.

Orchestrates multiple analysis passes through different analyzers to provide
comprehensive insights into profiling data.
"""

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.config import AnalysisConfig, AnalysisType
from ..core.interfaces import Analyzer
from ..core.models import (
    AnalysisResult,
    DetectedPattern,
    DynamicAnalysisResult,
    OptimizationRecommendation,
    ProfileSession,
    StaticAnalysisResult,
)
from .base_analyzer import BaseAnalyzer


class AnalysisEngine:
    """
    Multi-pass analysis engine for Pycroscope.

    Coordinates multiple analyzers to provide comprehensive analysis
    of profiling data through sequential passes.
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the analysis engine.

        Args:
            config: Optional analysis configuration
        """
        self._config = config or AnalysisConfig()
        self._analyzers: Dict[AnalysisType, Analyzer] = {}
        self._analysis_order: List[AnalysisType] = []
        self._lock = threading.RLock()

        # Analysis statistics
        self._total_analyses = 0
        self._successful_analyses = 0
        self._failed_analyses = 0

        # Initialize with configured analyzers
        self._initialize_analyzers()

    def _initialize_analyzers(self) -> None:
        """Initialize analyzers based on configuration."""
        # Define analysis order for multi-pass processing
        self._analysis_order = [
            AnalysisType.STATIC,
            AnalysisType.DYNAMIC,
            AnalysisType.PATTERN,
            AnalysisType.OPTIMIZATION,
        ]

        # Filter by enabled analyzers
        self._analysis_order = [
            analysis_type
            for analysis_type in self._analysis_order
            if analysis_type in self._config.enabled_analyzers
        ]

    def register_analyzer(
        self, analysis_type: AnalysisType, analyzer: Analyzer
    ) -> None:
        """
        Register an analyzer for a specific analysis type.

        Args:
            analysis_type: Type of analysis this analyzer performs
            analyzer: Analyzer implementation
        """
        with self._lock:
            self._analyzers[analysis_type] = analyzer

            # Update analysis order if needed
            if (
                analysis_type not in self._analysis_order
                and analysis_type in self._config.enabled_analyzers
            ):
                self._analysis_order.append(analysis_type)

    def unregister_analyzer(self, analysis_type: AnalysisType) -> None:
        """
        Unregister an analyzer.

        Args:
            analysis_type: Type of analysis to unregister
        """
        with self._lock:
            if analysis_type in self._analyzers:
                del self._analyzers[analysis_type]

            if analysis_type in self._analysis_order:
                self._analysis_order.remove(analysis_type)

    def analyze(self, profile_session: ProfileSession) -> AnalysisResult:
        """
        Perform comprehensive multi-pass analysis.

        Args:
            profile_session: Complete profiling session data

        Returns:
            Comprehensive analysis results
        """
        with self._lock:
            self._total_analyses += 1

        try:
            # Initialize analysis result
            analysis_result = self._create_initial_result(profile_session)

            # Run analysis passes in order
            for analysis_type in self._analysis_order:
                if analysis_type in self._analyzers:
                    analyzer = self._analyzers[analysis_type]

                    try:
                        # Run specific analysis
                        pass_result = analyzer.analyze(profile_session)

                        # Integrate results
                        analysis_result = self._integrate_analysis_result(
                            analysis_result, pass_result, analysis_type
                        )

                    except Exception as e:
                        # Log error but continue with other analyses
                        continue

            # Finalize analysis
            analysis_result = self._finalize_analysis(analysis_result, profile_session)

            with self._lock:
                self._successful_analyses += 1

            return analysis_result

        except Exception as e:
            with self._lock:
                self._failed_analyses += 1

            # Return error result
            return self._create_error_result(profile_session, str(e))

    def _create_initial_result(self, profile_session: ProfileSession) -> AnalysisResult:
        """
        Create initial analysis result structure.

        Args:
            profile_session: Profiling session data

        Returns:
            Initial analysis result
        """
        return AnalysisResult(
            session_id=profile_session.session_id,
            analysis_timestamp=datetime.now(),
            static_analysis=StaticAnalysisResult(),
            dynamic_analysis=DynamicAnalysisResult(),
            detected_patterns=[],
            recommendations=[],
            overall_score=0.0,
            performance_grade="analyzing",
        )

    def _integrate_analysis_result(
        self,
        current_result: AnalysisResult,
        pass_result: AnalysisResult,
        analysis_type: AnalysisType,
    ) -> AnalysisResult:
        """
        Integrate results from a specific analysis pass.

        Args:
            current_result: Current accumulated results
            pass_result: Results from specific analysis pass
            analysis_type: Type of analysis that was performed

        Returns:
            Updated analysis result
        """
        # Update static analysis results
        if analysis_type == AnalysisType.STATIC:
            static_analysis = pass_result.static_analysis
        else:
            static_analysis = current_result.static_analysis

        # Update dynamic analysis results
        if analysis_type == AnalysisType.DYNAMIC:
            dynamic_analysis = pass_result.dynamic_analysis
        else:
            dynamic_analysis = current_result.dynamic_analysis

        # Merge patterns and recommendations
        all_patterns = list(current_result.detected_patterns)
        all_patterns.extend(pass_result.detected_patterns)

        all_recommendations = list(current_result.recommendations)
        all_recommendations.extend(pass_result.recommendations)

        # Create integrated result
        return AnalysisResult(
            session_id=current_result.session_id,
            analysis_timestamp=current_result.analysis_timestamp,
            static_analysis=static_analysis,
            dynamic_analysis=dynamic_analysis,
            detected_patterns=all_patterns,
            recommendations=all_recommendations,
            overall_score=current_result.overall_score,
            performance_grade=current_result.performance_grade,
        )

    def _finalize_analysis(
        self, analysis_result: AnalysisResult, profile_session: ProfileSession
    ) -> AnalysisResult:
        """
        Finalize analysis by calculating overall scores and grades.

        Args:
            analysis_result: Analysis results to finalize
            profile_session: Original profile session

        Returns:
            Finalized analysis result
        """
        # Calculate overall score based on patterns and recommendations
        overall_score = self._calculate_overall_score(analysis_result)

        # Determine performance grade
        performance_grade = self._determine_performance_grade(
            overall_score, analysis_result
        )

        # Filter patterns and recommendations by thresholds
        filtered_patterns = self._filter_patterns(analysis_result.detected_patterns)
        filtered_recommendations = self._filter_recommendations(
            analysis_result.recommendations
        )

        return AnalysisResult(
            session_id=analysis_result.session_id,
            analysis_timestamp=analysis_result.analysis_timestamp,
            static_analysis=analysis_result.static_analysis,
            dynamic_analysis=analysis_result.dynamic_analysis,
            detected_patterns=filtered_patterns,
            recommendations=filtered_recommendations,
            overall_score=overall_score,
            performance_grade=performance_grade,
        )

    def _calculate_overall_score(self, analysis_result: AnalysisResult) -> float:
        """
        Calculate overall performance score (0.0 to 100.0).

        Args:
            analysis_result: Analysis results

        Returns:
            Overall score
        """
        # Base score starts at 50 (neutral)
        base_score = 50.0

        # Deduct points for critical patterns
        critical_patterns = [
            p for p in analysis_result.detected_patterns if p.severity == "critical"
        ]
        base_score -= len(critical_patterns) * 15.0

        # Deduct points for high severity patterns
        high_patterns = [
            p for p in analysis_result.detected_patterns if p.severity == "high"
        ]
        base_score -= len(high_patterns) * 8.0

        # Deduct points for medium severity patterns
        medium_patterns = [
            p for p in analysis_result.detected_patterns if p.severity == "medium"
        ]
        base_score -= len(medium_patterns) * 3.0

        # Add points for high-impact recommendations
        high_impact_recs = [
            r for r in analysis_result.recommendations if r.estimated_improvement >= 2.0
        ]
        base_score += len(high_impact_recs) * 5.0

        # Ensure score is within bounds
        return max(0.0, min(100.0, base_score))

    def _determine_performance_grade(
        self, score: float, analysis_result: AnalysisResult
    ) -> str:
        """
        Determine performance grade based on score and analysis.

        Args:
            score: Overall performance score
            analysis_result: Analysis results

        Returns:
            Performance grade string
        """
        if score >= 90:
            return "excellent"
        elif score >= 80:
            return "good"
        elif score >= 70:
            return "fair"
        elif score >= 60:
            return "poor"
        else:
            return "critical"

    def _filter_patterns(
        self, patterns: List[DetectedPattern]
    ) -> List[DetectedPattern]:
        """
        Filter patterns based on configuration thresholds.

        Args:
            patterns: All detected patterns

        Returns:
            Filtered patterns
        """
        return [
            pattern
            for pattern in patterns
            if pattern.impact_estimate >= self._config.impact_threshold
        ]

    def _filter_recommendations(
        self, recommendations: List[OptimizationRecommendation]
    ) -> List[OptimizationRecommendation]:
        """
        Filter recommendations based on configuration thresholds.

        Args:
            recommendations: All recommendations

        Returns:
            Filtered recommendations
        """
        return [
            rec
            for rec in recommendations
            if (
                rec.estimated_improvement >= self._config.impact_threshold
                and rec.confidence >= self._config.confidence_threshold
            )
        ]

    def _create_error_result(
        self, profile_session: ProfileSession, error_message: str
    ) -> AnalysisResult:
        """
        Create error analysis result.

        Args:
            profile_session: Original profile session
            error_message: Error description

        Returns:
            Error analysis result
        """
        return AnalysisResult(
            session_id=profile_session.session_id,
            analysis_timestamp=datetime.now(),
            static_analysis=StaticAnalysisResult(),
            dynamic_analysis=DynamicAnalysisResult(),
            detected_patterns=[],
            recommendations=[],
            overall_score=0.0,
            performance_grade=f"error: {error_message}",
        )

    def get_available_analyzers(self) -> Dict[AnalysisType, str]:
        """
        Get list of available analyzers.

        Returns:
            Dictionary mapping analysis types to analyzer names
        """
        with self._lock:
            return {
                analysis_type: analyzer.name
                for analysis_type, analyzer in self._analyzers.items()
            }

    def get_analysis_stats(self) -> Dict[str, Any]:
        """
        Get analysis engine statistics.

        Returns:
            Dictionary with engine statistics
        """
        with self._lock:
            success_rate = (
                self._successful_analyses / self._total_analyses * 100
                if self._total_analyses > 0
                else 0.0
            )

            return {
                "total_analyses": self._total_analyses,
                "successful_analyses": self._successful_analyses,
                "failed_analyses": self._failed_analyses,
                "success_rate_percent": success_rate,
                "registered_analyzers": len(self._analyzers),
                "analysis_order": [t.value for t in self._analysis_order],
                "enabled_analyzers": [t.value for t in self._config.enabled_analyzers],
            }

    def configure(self, config: AnalysisConfig) -> None:
        """
        Update engine configuration.

        Args:
            config: New analysis configuration
        """
        with self._lock:
            self._config = config
            self._initialize_analyzers()

    def __len__(self) -> int:
        """Return number of registered analyzers."""
        return len(self._analyzers)

    def __contains__(self, analysis_type: AnalysisType) -> bool:
        """Check if an analyzer type is registered."""
        return analysis_type in self._analyzers

    def __getitem__(self, analysis_type: AnalysisType) -> Analyzer:
        """Get analyzer by type."""
        return self._analyzers[analysis_type]
