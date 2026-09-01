import logging
import sys
import time
from pathlib import Path

from src.config import LOG_LEVEL, GNN_EPOCHS, Y_RAND_ITERATIONS

logger = logging.getLogger(__name__)


def run_step(cmd: list[str], description: str, allow_failure: bool = False):
    """Run a pipeline step as a subprocess."""
    import subprocess

    logger.info("=" * 70)
    logger.info("STEP: %s", description)
    logger.info("CMD:  python %s", " ".join(cmd))
    logger.info("=" * 70)

    start = time.time()
    result = subprocess.run(
        [sys.executable] + cmd,
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        status = "FAILED" if not allow_failure else "FAILED (non-critical)"
        logger.error("[%s] %s (exit code %d) [%.1fs]", status, description, result.returncode, elapsed)
        if not allow_failure:
            logger.error("Pipeline halted due to critical failure.")
            sys.exit(1)
    else:
        logger.info("[SUCCESS] %s [%.1fs]", description, elapsed)

    return result.returncode


def main():
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("ADENOSINE SELECTIVITY MODEL — FULL PUBLICATION PIPELINE")
    logger.info("=" * 70)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-retrain", action="store_true", help="Skip production retraining")
    parser.add_argument("--no-skip-gnn", action="store_true", help="Include GNN training (legacy)")
    parser.add_argument("--skip-nested-cv", action="store_true", help="Skip nested CV (uses default params)")
    parser.add_argument("--y-rand-iterations", type=int, default=Y_RAND_ITERATIONS, help="Y-randomization iterations")
    parser.add_argument("--gnn-epochs", type=int, default=GNN_EPOCHS, help="GNN training epochs")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow experiment tracking")
    args = parser.parse_args()

    from src.config import RUN_ID, RUN_TIMESTAMP, OUTPUTS_DIR
    from src.run_id import register_run, save_with_run_id

    pipeline_metadata = {
        "run_id": RUN_ID,
        "timestamp": RUN_TIMESTAMP,
        "skip_retrain": args.skip_retrain,
        "skip_gnn": not args.no_skip_gnn,
        "y_rand_iterations": args.y_rand_iterations,
        "gnn_epochs": args.gnn_epochs,
    }
    register_run(RUN_ID, pipeline_metadata)
    logger.info("Pipeline run registered: %s", RUN_ID)

    if args.mlflow:
        from src.mlflow_tracking import mlflow
        mlflow.set_experiment("adenosine-selectivity-pipeline")
        mlflow.start_run(run_name=f"pipeline_{RUN_ID}")
        mlflow.log_params(pipeline_metadata)

    try:
        if not args.skip_retrain:
            run_step(["-m", "src.retrain_production"], "Production Model Training & Conformal Prediction (MAPIE)")

        if args.no_skip_gnn:
            run_step(
                ["-m", "src.gnn_model", "--all", "--epochs", str(args.gnn_epochs)],
                f"GNN Training — All Subtypes ({args.gnn_epochs} epochs)",
                allow_failure=True,
            )

        run_step(["-m", "src.selectivity_models"], "Pairwise Affinity Difference Selectivity Models")

        run_step(
            ["-m", "src.y_randomization", "--all", "--iterations", str(args.y_rand_iterations)],
            f"Y-Randomization (ALL Subtypes, n={args.y_rand_iterations})",
        )

        run_step(["-m", "src.shap_analysis", "--all"], "SHAP Tree Explainability (ALL Subtypes)")

        run_step(["-m", "src.evaluator"], "Conformal Model Metrics Evaluation (Full + Actives-Only)")

        run_step(["-m", "src.external_validation"], "External Validation (GPCRdb Blind Test)", allow_failure=True)

        run_step(["-m", "src.literature_benchmark"], "Literature Benchmark Comparison", allow_failure=True)
    finally:
        save_with_run_id(pipeline_metadata, OUTPUTS_DIR, "pipeline_run", RUN_ID)
        if args.mlflow:
            mlflow.end_run()

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("Total Runtime: %dh %dm %ds", hours, minutes, seconds)
    logger.info("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
