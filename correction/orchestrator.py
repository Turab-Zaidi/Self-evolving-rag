"""Orchestrator: runs all 3 self-evolution stages in sequence."""

import subprocess
import sys
import os


def run_stage(script_path, stage_name):
    print(f"\n{'='*60}")
    print(f"  STARTING {stage_name}")
    print(f"{'='*60}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run([sys.executable, script_path], capture_output=False, env=env)
    if result.returncode != 0:
        print(f"  {stage_name} FAILED! Halting.")
        exit(1)
    print(f"  {stage_name} COMPLETED SUCCESSFULLY.")


def run_self_evolving_loop():
    print("Initializing Autonomous Self-Evolving Loop...\n")
    run_stage("correction/optimize_planner.py", "STAGE 1: Query Planner Optimizer (DSPy)")
    run_stage("correction/optimize_hyperparams.py", "STAGE 2: Hyperparameter Optimizer (Optuna)")
    run_stage("correction/optimize_generator.py", "STAGE 3: Generator Optimizer (DSPy)")
    print(f"\n{'='*60}")
    print("  ALL OPTIMIZATION STAGES COMPLETE.")
    print("  Run 'python evaluation/run_eval.py --cycle 1' to verify improvements.")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_self_evolving_loop()
