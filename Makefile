.PHONY: install dev clean test format lint run-load run-query run-index

# Install production dependencies
install:
	pip install -e .

# Install development dependencies
dev:
	pip install -e ".[dev]"

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run tests
test:
	pytest tests/

# Format code
format:
	black txtinspect/ tests/

# Lint code
lint:
	flake8 txtinspect/ tests/
	mypy txtinspect/
