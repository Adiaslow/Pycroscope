# Contributing to Pycroscope

Thank you for your interest in contributing to Pycroscope! This document provides guidelines and information for contributors.

## 🎯 Design Philosophy

Before contributing, please familiarize yourself with Pycroscope's core design principles:

- **"One Way, Many Options"**: Single canonical interface with rich configuration options
- **Geometric Beauty**: Clean proportions and harmonious component relationships
- **SOLID Principles**: Separation of concerns and dependency injection
- **No Technical Debt**: Avoid workarounds, shortcuts, or temporary fixes

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**

   ```bash
   git clone https://github.com/your-username/pycroscope.git
   cd pycroscope
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**

   ```bash
   pip install -e ".[dev,test]"
   ```

4. **Install Pre-commit Hooks**

   ```bash
   pre-commit install
   ```

5. **Verify Setup**
   ```bash
   python -m pytest tests/
   python -c "import pycroscope; print('Setup successful!')"
   ```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=pycroscope --cov-report=html

# Run specific test file
python -m pytest tests/core/test_models.py

# Run with verbose output
python -m pytest -v
```

### Test Requirements

- **100% Coverage**: All new code must be covered by tests
- **Multiple Python Versions**: Test on Python 3.8-3.12
- **Cross-Platform**: Consider Windows, macOS, and Linux
- **Edge Cases**: Include tests for error conditions

### Writing Tests

```python
# Example test structure
def test_feature_functionality():
    """Test that feature works correctly."""
    # Arrange
    config = ProfileConfig()

    # Act
    result = some_function(config)

    # Assert
    assert result is not None
    assert isinstance(result, ExpectedType)
```

## 📝 Code Style

### Formatting

- **Black**: Code formatting (`black pycroscope/ tests/`)
- **isort**: Import sorting (`isort pycroscope/ tests/`)
- **Line Length**: 88 characters maximum

### Type Hints

```python
from typing import Dict, List, Optional

def process_data(
    events: List[ExecutionEvent],
    config: Optional[ProfileConfig] = None
) -> Dict[str, Any]:
    """Process execution events with optional configuration."""
    pass
```

### Docstrings

```python
def analyze_performance(session: ProfileSession) -> AnalysisResult:
    """
    Analyze performance data from a profiling session.

    Args:
        session: Complete profiling session data

    Returns:
        Analysis results with insights and recommendations

    Raises:
        ValueError: If session data is invalid
    """
    pass
```

## 🏗️ Architecture Guidelines

### Component Structure

```
pycroscope/
├── core/           # Core infrastructure
├── collectors/     # Data collection components
├── analysis/       # Analysis engines
├── storage/        # Data persistence
├── cli/           # Command-line interface
└── visualization/ # Interactive dashboards (future)
```

### Adding New Collectors

1. **Inherit from BaseCollector**

   ```python
   from .base import BaseCollector

   class MyCollector(BaseCollector):
       @property
       def name(self) -> str:
           return "my_collector"
   ```

2. **Implement Required Methods**

   - `_install_hooks()`
   - `_uninstall_hooks()`
   - `_collect_events()`

3. **Register in ProfilerSuite**
   ```python
   # In _register_default_components()
   self._registry.register_collector(CollectorType.MY_TYPE, MyCollector)
   ```

### Adding New Analyzers

1. **Inherit from BaseAnalyzer**

   ```python
   from .base_analyzer import BaseAnalyzer

   class MyAnalyzer(BaseAnalyzer):
       @property
       def name(self) -> str:
           return "my_analyzer"
   ```

2. **Implement Analysis Logic**
   ```python
   def _perform_analysis(self, profile_data: ProfileSession) -> AnalysisResult:
       # Your analysis logic here
       pass
   ```

## 📋 Contribution Process

### 1. Planning

- **Check Existing Issues**: Look for related discussions
- **Create Issue**: Describe your proposed changes
- **Discuss Approach**: Get feedback before implementing

### 2. Implementation

- **Create Branch**: `git checkout -b feature/your-feature-name`
- **Write Code**: Follow architecture and style guidelines
- **Add Tests**: Ensure 100% coverage for new code
- **Update Documentation**: Include docstrings and README updates

### 3. Quality Checks

```bash
# Run all quality checks
black pycroscope/ tests/
isort pycroscope/ tests/
mypy pycroscope/
python -m pytest --cov=pycroscope --cov-fail-under=95
```

### 4. Pull Request

- **Create PR**: Use the provided template
- **Link Issues**: Reference related issue numbers
- **Request Review**: Tag maintainers for review

## 🐛 Bug Reports

When reporting bugs:

1. **Search First**: Check existing issues
2. **Use Template**: Fill out the bug report template completely
3. **Minimal Example**: Provide reproducible code
4. **Environment Info**: Include Python version, OS, etc.

## 💡 Feature Requests

For new features:

1. **Check Roadmap**: See if it's already planned
2. **Use Template**: Fill out the feature request template
3. **Explain Use Case**: Describe why it's valuable
4. **Consider API**: Think about how it would be used

## 🔍 Code Review

### What We Look For

- **Correctness**: Does the code work as intended?
- **Tests**: Are there comprehensive tests?
- **Documentation**: Are docstrings and comments clear?
- **Architecture**: Does it follow our design principles?
- **Performance**: Are there any performance implications?

### Review Process

1. **Automated Checks**: CI must pass
2. **Maintainer Review**: Core team reviews code
3. **Feedback**: Address any requested changes
4. **Approval**: Get approval from maintainer
5. **Merge**: Code is merged to main branch

## 📊 Performance Considerations

### Profiling Overhead

- Keep profiling overhead minimal
- Use sampling where appropriate
- Avoid synchronous operations in hot paths

### Memory Usage

- Be conscious of memory allocations
- Use generators where possible
- Clean up resources properly

### Thread Safety

- All collectors must be thread-safe
- Use appropriate locking mechanisms
- Consider concurrent access patterns

## 📚 Documentation

### What to Document

- **Public APIs**: All public functions and classes
- **Configuration**: New configuration options
- **Examples**: Usage examples for new features
- **Architecture**: Significant architectural changes

### Documentation Style

- **Clear and Concise**: Explain what, why, and how
- **Examples**: Include code examples
- **Type Information**: Use proper type hints
- **Cross-References**: Link to related components

## 🚦 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Performance benchmarks run

## 🤝 Community

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Code Reviews**: Technical discussions

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

## 📞 Getting Help

If you need help:

1. **Check Documentation**: README and docstrings
2. **Search Issues**: Look for similar questions
3. **Create Discussion**: Start a GitHub discussion
4. **Ask Maintainers**: Tag us in issues or PRs

Thank you for contributing to Pycroscope! Your contributions help make Python performance analysis better for everyone. 🎉
