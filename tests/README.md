# Tests for Multimodal Semantic Search Framework

This directory contains tests for the semantic search framework.

## Test Files

- **test_framework.py**: Comprehensive unit tests for all components
- **validate_syntax.py**: Python syntax validation (runs without dependencies)

## Running Tests

### Syntax Validation (No Dependencies Required)

```bash
python tests/validate_syntax.py
```

This validates Python syntax for all framework files without requiring PyTorch or other dependencies.

### Full Unit Tests (Requires Dependencies)

First, install dependencies:

```bash
pip install -r requirements.txt
```

Then run the full test suite:

```bash
python tests/test_framework.py
```

Or using unittest:

```bash
python -m unittest tests.test_framework
```

### Running Specific Tests

```bash
# Run only import tests
python -m unittest tests.test_framework.TestImports

# Run only configuration tests
python -m unittest tests.test_framework.TestConfiguration

# Run only encoder tests
python -m unittest tests.test_framework.TestEncoders
```

## Test Coverage

The test suite covers:

- ✓ Module imports
- ✓ Configuration system
- ✓ Encoder initialization
- ✓ Model initialization
- ✓ Utility functions
- ✓ Python syntax validation

## CI/CD Integration

For continuous integration, use the syntax validation as a first step:

```yaml
# Example GitHub Actions workflow
- name: Validate Python Syntax
  run: python tests/validate_syntax.py

- name: Install Dependencies
  run: pip install -r requirements.txt

- name: Run Unit Tests
  run: python tests/test_framework.py
```

## Test Results

Current status: **All syntax validation passes ✓**

When dependencies are installed, all unit tests should pass.
