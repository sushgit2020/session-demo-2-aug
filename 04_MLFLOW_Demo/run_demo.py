#!/usr/bin/env python3
"""
Usage examples:
    python run_demo.py                          # Run full demo
    python run_demo.py --dataset iris           # Run on iris only
    python run_demo.py --quick                  # Quick test with fewer combinations
    python run_demo.py --start-mlflow           # Start MLflow UI only
    mlflow ui
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import logging

# Import our parameter sweep module
from train import MLflowParameterSweep, run_parameter_sweep_demo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def start_mlflow_ui(port=5000):
    """Start MLflow UI server."""
    outputs_dir = Path("outputs")
    mlruns_dir = outputs_dir / "mlruns"
    
    if not mlruns_dir.exists():
        logger.info("🔄 No MLflow experiments found. Creating sample run first...")
        run_quick_demo()
    
    logger.info(f"🚀 Starting MLflow UI on http://localhost:{port}")
    logger.info("📊 Press Ctrl+C to stop the server")
    
    try:
        cmd = [
            "mlflow", "ui", 
            "--backend-store-uri", f"file://{mlruns_dir.absolute()}",
            "--host", "127.0.0.1",
            "--port", str(port)
        ]
        subprocess.run(cmd, cwd=outputs_dir.parent)
    except KeyboardInterrupt:
        logger.info("🛑 MLflow UI stopped")
    except FileNotFoundError:
        logger.error("❌ MLflow not found! Please install with: pip install mlflow")
        sys.exit(1)


def run_single_dataset(dataset_name, max_combinations=15):
    """Run parameter sweep on a single dataset."""
    logger.info(f"🎯 Running parameter sweep on {dataset_name} dataset")
    
    sweep = MLflowParameterSweep(experiment_name=f"parameter-sweep-{dataset_name}")
    
    try:
        # Load dataset
        sweep.load_dataset(dataset_name)
        
        # Run parameter sweep
        results = sweep.run_parameter_sweep(max_combinations_per_algo=max_combinations)
        
        # Analyze results
        if results:
            df = sweep.analyze_results()
            logger.info(f"✅ Completed! Tested {len(results)} combinations")
            return df
        else:
            logger.warning("⚠️ No results obtained")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error running parameter sweep: {e}")
        return None


def run_quick_demo():
    """Run a quick demo with fewer parameter combinations."""
    logger.info("⚡ Running quick parameter sweep demo")
    
    sweep = MLflowParameterSweep(experiment_name="quick-parameter-sweep")
    
    # Test only iris dataset with fewer combinations
    sweep.load_dataset("iris")
    results = sweep.run_parameter_sweep(max_combinations_per_algo=5)
    
    if results:
        df = sweep.analyze_results()
        logger.info(f"✅ Quick demo completed! Tested {len(results)} combinations")
        return df
    else:
        logger.warning("⚠️ Quick demo produced no results")
        return None


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['mlflow', 'sklearn', 'pandas', 'numpy', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing required packages: {missing_packages}")
        logger.info("📦 Install them with: pip install -r requirements.txt")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="MLflow Parameter Sweep Demo Runner")
    parser.add_argument(
        "--dataset", 
        choices=["iris", "wine", "synthetic"], 
        help="Run parameter sweep on specific dataset"
    )
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run quick demo with fewer parameter combinations"
    )
    parser.add_argument(
        "--start-mlflow", 
        action="store_true", 
        help="Start MLflow UI server"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=5000, 
        help="Port for MLflow UI (default: 5000)"
    )
    parser.add_argument(
        "--max-combinations", 
        type=int, 
        default=15, 
        help="Maximum parameter combinations per algorithm (default: 15)"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create outputs directory
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    (outputs_dir / "plots").mkdir(exist_ok=True)
    (outputs_dir / "mlruns").mkdir(exist_ok=True)
    
    try:
        if args.start_mlflow:
            start_mlflow_ui(port=args.port)
        
        elif args.quick:
            run_quick_demo()
            logger.info(f"🔗 View results in MLflow UI: http://localhost:{args.port}")
            logger.info("💡 Run with --start-mlflow to launch the UI")
        
        elif args.dataset:
            run_single_dataset(args.dataset, max_combinations=args.max_combinations)
            logger.info(f"🔗 View results in MLflow UI: http://localhost:{args.port}")
            logger.info("💡 Run with --start-mlflow to launch the UI")
        
        else:
            # Run full demo
            run_parameter_sweep_demo()
            logger.info(f"🔗 View results in MLflow UI: http://localhost:{args.port}")
            logger.info("💡 Run with --start-mlflow to launch the UI")
    
    except KeyboardInterrupt:
        logger.info("🛑 Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error running demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()