# Contributing to NeuroSC

Thank you for your interest in contributing to NeuroSC! This document provides guidelines and instructions for contributing.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and considerate in all interactions.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected vs actual behavior
- Your environment (OS, Python version, package versions)
- Code snippets or error messages

### Suggesting Features

We welcome feature suggestions! Please open an issue with:
- A clear description of the feature
- The use case and motivation
- Possible implementation approach (if you have ideas)

### Contributing Code

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/NeuroSC.git
   cd NeuroSC
   ```

2. **Create a development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes**
   - Write clear, documented code
   - Follow PEP 8 style guidelines
   - Add docstrings to functions and classes
   - Include type hints where appropriate

5. **Add tests**
   ```bash
   # Add tests in tests/
   pytest tests/
   ```

6. **Format your code**
   ```bash
   black neurosc/
   flake8 neurosc/
   ```

7. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

8. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://github.com/psf/black) for formatting (line length: 100)
- Use meaningful variable and function names
- Add comments for complex logic

### Documentation

- Add docstrings to all public functions, classes, and modules
- Use NumPy-style docstrings
- Include examples in docstrings when helpful
- Update README.md if adding new features

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for good test coverage
- Use pytest for testing

### Commit Messages

Use clear, descriptive commit messages:
- `Add: new feature X`
- `Fix: bug in function Y`
- `Update: documentation for Z`
- `Refactor: improve code structure in module A`

## Project Structure

```
NeuroSC/
├── neurosc/          # Main package
│   ├── data/         # Data preprocessing
│   ├── models/       # Model implementations
│   ├── training/     # Training utilities
│   ├── inference/    # Inference utilities
│   ├── tl/           # Scanpy-compatible API
│   ├── tools/        # High-level tools
│   └── utils/        # Utility functions
├── examples/         # Example scripts
├── tests/            # Test suite
└── docs/             # Documentation
```

## Adding New Features

### Adding a New Model

1. Create a wrapper in `neurosc/models/`
2. Inherit from `BaseFoundationModel`
3. Implement required methods: `forward`, `embed`, `predict`, `from_pretrained`
4. Register the model in `model_registry.py`
5. Add tests
6. Update documentation

### Adding New Tools

1. Add function to appropriate module (`tl/`, `tools/`, or `utils/`)
2. Make it scanpy-compatible if applicable
3. Add comprehensive docstring with examples
4. Export in `__init__.py`
5. Add to README

## Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion
- Contact the maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to NeuroSC! 🧠🧬

