"""Entry point for the integrated system using new features."""
from __future__ import annotations

import config
from integrated_system import IntegratedSystem


def main() -> None:
    system = IntegratedSystem()
    system.run("Generate population", config.POPULATION_SIZE)


if __name__ == "__main__":
    main()
