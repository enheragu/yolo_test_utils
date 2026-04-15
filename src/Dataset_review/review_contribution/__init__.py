"""Review contribution package for RGB/LWIR latent contribution analysis."""


def main() -> None:
	"""Lazy package entrypoint to avoid side effects on module execution."""
	from .pipeline import main as _main

	_main()


__all__ = ["main"]
