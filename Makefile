.PHONY: help install run test coverage lint format clean

help:
	@echo "  make help		Show this help message"
	@echo "  make install		Install dependencies"
	@echo "  make run		Run the track generation demo script"
	@echo "  make test		Run pytest for the project"
	@echo "  make coverage		Run pytest with coverage report"
	@echo "  make lint		Run linters"
	@echo "  make format		Format the codebase"
	@echo "  make clean		Clean up build artifacts and cache"

install:
	uv sync --dev -U

run:
	uv run generate_tracks.py

test:
	uv run pytest tests/

lint:
	uv run ruff check src/cam_track_gen/ tests/

format:
	uv run ruff format .
	uv run ruff check --select "I" --fix

coverage:
	uv run pytest --cov=cam_track_gen --cov-report=term-missing --cov-report=html tests/

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf output/*
	rm -f .coverage
	rm -f lcov.info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
