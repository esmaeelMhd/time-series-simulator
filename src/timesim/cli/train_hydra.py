"""Hydra-first training CLI entrypoint."""

from __future__ import annotations

from scripts.train_hydra import main as hydra_train_main


def main() -> None:
    hydra_train_main()


if __name__ == "__main__":
    main()
