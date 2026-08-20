"""Audacia LLM Evaluation.

Attaches a `NullHandler` and never configures levels or handlers itself, so
importing this package cannot change a consumer's logging. Use
`configure_logging()` to opt in to output from notebooks and scripts.
"""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())


def configure_logging(level: int = logging.INFO) -> None:
    """Send this package's log records to stderr at `level`."""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
