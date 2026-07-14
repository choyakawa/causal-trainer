"""Stable module and console-script entry point."""

from .training.runner import main

__all__ = ("main",)


if __name__ == "__main__":
    main()
