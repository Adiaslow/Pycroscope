# GitHub Actions Workflow Badges

This file contains the correct badge URLs for adding to your README.md file.

## Available Badges

### Main Test Suite Status

```markdown
![Tests](https://github.com/Adiaslow/pycroscope/workflows/Tests%20and%20Coverage/badge.svg)
```

### Coverage Badge (Codecov)

```markdown
[![codecov](https://codecov.io/gh/Adiaslow/pycroscope/branch/main/graph/badge.svg)](https://codecov.io/gh/Adiaslow/pycroscope)
```

### Python Version Support

```markdown
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
```

### Coverage Percentage (Dynamic)

```markdown
[![codecov](https://codecov.io/gh/Adiaslow/pycroscope/branch/main/graph/badge.svg?token=YOUR_CODECOV_TOKEN)](https://codecov.io/gh/Adiaslow/pycroscope)
```

### License

```markdown
![License](https://img.shields.io/badge/license-MIT-green)
```

### PyPI Version (when published)

```markdown
[![PyPI version](https://badge.fury.io/py/pycroscope.svg)](https://badge.fury.io/py/pycroscope)
```

## Complete Badge Section for README

Add this section near the top of your README.md:

```markdown
[![Tests](https://github.com/Adiaslow/pycroscope/workflows/Tests%20and%20Coverage/badge.svg)](https://github.com/Adiaslow/pycroscope/actions)
[![codecov](https://codecov.io/gh/Adiaslow/pycroscope/branch/main/graph/badge.svg)](https://codecov.io/gh/Adiaslow/pycroscope)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
```

## Usage Instructions

1. ✅ **Already updated** with GitHub username: `Adiaslow`
2. Repository name is correct: `pycroscope`
3. Add your Codecov token for private repositories if needed
4. ✅ **Already added** these badges to your README.md file

## Workflow Status Checks

The current workflow includes these jobs:

- **test**: Runs on Python 3.9-3.12 across Ubuntu, Windows, macOS
- **architecture-validation**: Validates SOLID principles and design patterns
- **security**: Runs safety and bandit security scans
- **code-quality**: Checks formatting, imports, and type hints
- **performance-check**: Basic performance validation
- **release-check**: Package build and installation validation

All jobs must pass for the main badge to show "passing" status.
