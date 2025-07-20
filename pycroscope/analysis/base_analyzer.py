"""
Base analyzer implementation providing common functionality.

All concrete analyzers inherit from BaseAnalyzer to ensure consistent
behavior and provide shared utilities for analysis operations.
"""

import threading
import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set

from ..core.config import AnalysisConfig
from ..core.interfaces import Analyzer, Configurable, Lifecycle
from ..core.models import (
    AnalysisResult,
    DetectedPattern,
    OptimizationRecommendation,
    ProfileSession,
)


class BaseAnalyzer(Analyzer, Configurable, Lifecycle):
    """
    Abstract base implementation for all analysis engines.

    Provides common functionality including configuration management,
    lifecycle control, and analysis result formatting.
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the base analyzer.

        Args:
            config: Optional analysis configuration
        """
        self._config = config or AnalysisConfig()
        self._is_running = False
        self._analysis_count = 0
        self._total_analysis_time = 0.0
        self._lock = threading.Lock()

        # Patterns and recommendations cache
        self._pattern_cache: Dict[str, List[DetectedPattern]] = {}
        self._recommendation_cache: Dict[str, List[OptimizationRecommendation]] = {}

        # Apply configuration
        self.configure(self._config.__dict__)

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this analyzer type."""
        pass

    @property
    @abstractmethod
    def dependencies(self) -> List[str]:
        """List of collector names required by this analyzer."""
        pass

    @property
    def is_running(self) -> bool:
        """Whether analyzer is actively running."""
        return self._is_running

    @property
    def configuration(self) -> AnalysisConfig:
        """Current analyzer configuration."""
        return self._config

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Apply configuration settings to this analyzer.

        Args:
            config: Configuration dictionary
        """
        # Update configuration from dictionary
        for key, value in config.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    def start(self) -> None:
        """Initialize analyzer for active use."""
        if self._is_running:
            return

        self._is_running = True
        self._analysis_count = 0
        self._total_analysis_time = 0.0

    def stop(self) -> None:
        """Shutdown analyzer and release resources."""
        if not self._is_running:
            return

        self._is_running = False

        # Clear caches
        self._pattern_cache.clear()
        self._recommendation_cache.clear()

    def analyze(self, profile_data: ProfileSession) -> AnalysisResult:
        """
        Process profiling data and return analysis results.

        Args:
            profile_data: Complete profiling session data

        Returns:
            Structured analysis results with insights and recommendations
        """
        if not self._is_running:
            self.start()

        start_time = time.perf_counter()

        try:
            # Validate input data
            self._validate_input_data(profile_data)

            # Check if dependencies are met
            missing_deps = self._check_dependencies(profile_data)
            if missing_deps:
                raise ValueError(f"Missing required data: {missing_deps}")

            # Perform analysis
            result = self._perform_analysis(profile_data)

            # Update statistics
            with self._lock:
                self._analysis_count += 1
                analysis_time = time.perf_counter() - start_time
                self._total_analysis_time += analysis_time

            return result

        except Exception as e:
            # Log error and return empty result
            return self._create_error_result(profile_data, str(e))

    @abstractmethod
    def _perform_analysis(self, profile_data: ProfileSession) -> AnalysisResult:
        """
        Perform the actual analysis. Implemented by subclasses.

        Args:
            profile_data: Profiling session data

        Returns:
            Analysis results
        """
        pass

    def _validate_input_data(self, profile_data: ProfileSession) -> None:
        """
        Validate input data before analysis.

        Args:
            profile_data: Profiling session data to validate

        Raises:
            ValueError: If data is invalid
        """
        if not profile_data:
            raise ValueError("Profile data cannot be None")

        if not profile_data.session_id:
            raise ValueError("Profile session must have a valid session ID")

        # Check if we have any data to analyze
        has_events = len(profile_data.execution_events) > 0
        has_memory = len(profile_data.memory_snapshots) > 0
        has_call_tree = profile_data.call_tree is not None

        if not (has_events or has_memory or has_call_tree):
            raise ValueError("Profile session contains no analyzable data")

    def _check_dependencies(self, profile_data: ProfileSession) -> List[str]:
        """
        Check if required dependencies are available in the profile data.

        Args:
            profile_data: Profiling session data

        Returns:
            List of missing dependencies
        """
        missing = []

        for dependency in self.dependencies:
            if dependency == "line" and not profile_data.execution_events:
                missing.append("line execution events")
            elif dependency == "memory" and not profile_data.memory_snapshots:
                missing.append("memory snapshots")
            elif dependency == "call" and not profile_data.call_tree:
                missing.append("call tree data")

        return missing

    def _create_error_result(
        self, profile_data: ProfileSession, error_message: str
    ) -> AnalysisResult:
        """
        Create an error analysis result.

        Args:
            profile_data: Original profile data
            error_message: Error description

        Returns:
            Analysis result indicating error
        """
        from datetime import datetime

        from ..core.models import DynamicAnalysisResult, StaticAnalysisResult

        return AnalysisResult(
            session_id=profile_data.session_id,
            analysis_timestamp=datetime.now(),
            static_analysis=StaticAnalysisResult(),
            dynamic_analysis=DynamicAnalysisResult(),
            detected_patterns=[],
            recommendations=[],
            overall_score=0.0,
            performance_grade=f"error: {error_message}",
        )

    def _should_analyze_pattern(self, pattern_type: str, confidence: float) -> bool:
        """
        Determine if a pattern should be included based on configuration thresholds.

        Args:
            pattern_type: Type of pattern detected
            confidence: Confidence level of the pattern

        Returns:
            True if pattern should be included
        """
        return confidence >= self._config.confidence_threshold

    def _should_generate_recommendation(self, impact: float, confidence: float) -> bool:
        """
        Determine if a recommendation should be generated.

        Args:
            impact: Estimated impact of the recommendation
            confidence: Confidence in the recommendation

        Returns:
            True if recommendation should be generated
        """
        return (
            impact >= self._config.impact_threshold
            and confidence >= self._config.confidence_threshold
        )

    def _cache_patterns(self, session_id: str, patterns: List[DetectedPattern]) -> None:
        """Cache detected patterns for potential reuse."""
        with self._lock:
            self._pattern_cache[session_id] = patterns.copy()

    def _cache_recommendations(
        self, session_id: str, recommendations: List[OptimizationRecommendation]
    ) -> None:
        """Cache recommendations for potential reuse."""
        with self._lock:
            self._recommendation_cache[session_id] = recommendations.copy()

    def _get_cached_patterns(self, session_id: str) -> Optional[List[DetectedPattern]]:
        """Retrieve cached patterns if available."""
        with self._lock:
            return self._pattern_cache.get(session_id)

    def _get_cached_recommendations(
        self, session_id: str
    ) -> Optional[List[OptimizationRecommendation]]:
        """Retrieve cached recommendations if available."""
        with self._lock:
            return self._recommendation_cache.get(session_id)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get analyzer statistics.

        Returns:
            Dictionary with analyzer runtime statistics
        """
        with self._lock:
            avg_analysis_time = (
                self._total_analysis_time / self._analysis_count
                if self._analysis_count > 0
                else 0.0
            )

            return {
                "name": self.name,
                "is_running": self._is_running,
                "analysis_count": self._analysis_count,
                "total_analysis_time": self._total_analysis_time,
                "average_analysis_time": avg_analysis_time,
                "dependencies": self.dependencies,
                "cached_patterns": len(self._pattern_cache),
                "cached_recommendations": len(self._recommendation_cache),
                "confidence_threshold": self._config.confidence_threshold,
                "impact_threshold": self._config.impact_threshold,
            }

    def clear_cache(self) -> None:
        """Clear all cached analysis results."""
        with self._lock:
            self._pattern_cache.clear()
            self._recommendation_cache.clear()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def __repr__(self) -> str:
        """String representation."""
        status = "running" if self._is_running else "stopped"
        return f"{self.__class__.__name__}(name='{self.name}', status='{status}')"
