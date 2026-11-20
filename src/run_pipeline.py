"""End-to-end training, evaluation, TreeSHAP, and Y-randomization orchestration pipeline."""
import argparse
import logging
import subprocess
import sys
import time

from src.config import LOG_LEVEL, OUTPUTS_DIR, RUN_ID, RUN_TIMESTAMP, Y_RAND_ITERATIONS
from src.evaluator import save_with_run_id

logger = logging.getLogger(__name__)


def run_step(cmd: list[str], description: str, allow_failure: bool = False) -> int:
    """Run a pipeline step as a subprocess."""
    logger.info("STEP: %s | CMD: python %s", description, " ".join(cmd))
    start = time.time()
    result = subprocess.run([sys.executable] + cmd, capture_output=False, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        status = "FAILED" if not allow_failure else "FAILED (non-critical)"
        logger.error("[%s] %s (exit code %d) [%.1fs]", status, description, result.returncode, elapsed)
        if not allow_failure:
            sys.exit(1)
    else:
        logger.info("[SUCCESS] %s [%.1fs]", description, elapsed)
    return result.returncode


def main() -> None:
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Adenosine Receptor Profiler Production Pipeline")
    parser.add_argument("--skip-retrain", action="store_true", help="Skip production conformal training")
    parser.add_argument(
        "--y-rand-iterations",
        type=int,
        default=Y_RAND_ITERATIONS,
        help="Y-randomization permutation iterations",
    )
    args = parser.parse_args()

    pipeline_metadata = {
        "run_id": RUN_ID,
        "timestamp": RUN_TIMESTAMP,
        "skip_retrain": args.skip_retrain,
        "y_rand_iterations": args.y_rand_iterations,
    }

    try:
        if not args.skip_retrain:
            run_step(
                ["-m", "src.retrain_production"],
                "Production 3-Model Training (XGBoost + LightGBM + RandomForest)",
            )

        run_step(
            ["-m", "src.evaluator"],
            "Holdout Scaffold Evaluation & Dataset Diagnostics",
        )

        run_step(
            ["-m", "src.shap_analysis", "--all"],
            "Global TreeSHAP Attribution Summary (All Subtypes)",
        )

        run_step(
            ["-m", "src.y_randomization", "--all", "--iterations", str(args.y_rand_iterations)],
            f"Y-Randomization Permutation Validation (All Subtypes, n={args.y_rand_iterations})",
        )
    finally:
        save_with_run_id(pipeline_metadata, OUTPUTS_DIR, "pipeline_run", RUN_ID)

    elapsed = time.time() - start_time
    logger.info("Pipeline completed in %.1fs [run_id=%s]", elapsed, RUN_ID)


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
