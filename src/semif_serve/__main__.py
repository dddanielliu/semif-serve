"""Console entry point: load the model, then serve until interrupted."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import ConfigError, Settings
from .engine import SemIfEngine, StubEngine
from .server import build_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="semif-serve", description=__doc__)
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Serve deterministic scores with no model, for wire-format checks without a GPU.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger = logging.getLogger("semif_serve")

    try:
        settings = Settings.from_env()
    except ConfigError as error:
        parser.error(str(error))

    if args.stub:
        engine = StubEngine(settings)
        logger.warning("Serving the stub engine: answers are deterministic noise, not decisions.")
    else:
        engine = SemIfEngine(settings)
        logger.info("Loading %s at %s", settings.model, settings.revision[:12])
        metadata = engine.load()
        logger.info("Loaded %s (torch %s)", metadata.get("source"), metadata.get("torch_version"))
        logger.info("Warmed kernels in %.1fs", engine.warmup())

    server = build_server(settings, engine)
    logger.info("Listening on http://%s:%s/v1/systemone", settings.host, settings.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
