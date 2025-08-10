#!/usr/bin/env python3
"""
Simple MLflow Parameter Sweep Demo with Scikit-Learn
===================================================

This demo shows how to use MLflow to track hyperparameter optimization
experiments with scikit-learn models. It demonstrates:

1. Parameter sweeps with multiple algorithms
2. MLflow experiment tracking
3. Model comparison and selection
4. Visualization of results

Perfect for learning MLOps parameter optimization!
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime
import warnings
from itertools import product

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Scikit-learn imports
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLflowParameterSweep:
    """
    A simple parameter sweep class using MLflow for experiment tracking.
    
    This class demonstrates how to systematically explore hyperparameters
    while tracking all experiments with MLflow.
    """
    
    def __init__(self, experiment_name="parameter-sweep-demo", output_dir="outputs"):
        """
        Initialize the parameter sweep.
        
        Args:
            experiment_name: Name of the MLflow experiment
            output_dir: Directory to save outputs
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        
        # Set up MLflow
        self._setup_mlflow()
        
        # Results storage
        self.results = []
        
        logger.info(f"🚀 MLflow Parameter Sweep initialized")
        logger.info(f"📊 Experiment: {self.experiment_name}")
        logger.info(f"📂 Output directory: {self.output_dir}")
    
    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        try:
            # Set MLflow tracking URI (cross-platform compatible)
            if os.environ.get('MLFLOW_TRACKING_URI'):
                mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI')
            else:
                mlruns_path = self.output_dir.absolute() / "mlruns"
                mlruns_path.mkdir(parents=True, exist_ok=True)
                
                mlflow_uri = mlruns_path.as_uri()
                
                if os.name == 'nt' and not mlflow_uri.startswith('file:///'):
                    mlflow_uri = mlflow_uri.replace('file://', 'file:///')
            
            mlflow.set_tracking_uri(mlflow_uri)
            logger.info(f"📍 MLflow tracking URI: {mlflow_uri}")
            logger.info(f"📂 MLruns directory: {self.output_dir.absolute() / 'mlruns'}")
            
            # Set or create experiment
            try:
                # Try to get existing experiment
                try:
                    experiment = mlflow.get_experiment_by_name(self.experiment_name)
                except Exception:
                    experiment = None
                
                if experiment is None or experiment.lifecycle_stage == "deleted":
                    # Create cross-platform artifact location
                    artifact_path = self.output_dir.absolute() / "mlflow-artifacts"
                    artifact_path.mkdir(parents=True, exist_ok=True)
                    artifact_uri = artifact_path.as_uri()
                    
                    # Fix for Windows file URIs
                    if os.name == 'nt' and not artifact_uri.startswith('file:///'):
                        artifact_uri = artifact_uri.replace('file://', 'file:///')
                    
                    experiment_id = mlflow.create_experiment(
                        self.experiment_name, 
                        artifact_location=artifact_uri
                    )
                    logger.info(f"✅ Created MLflow experiment: {self.experiment_name} (ID: {experiment_id})")
                else:
                    experiment_id = experiment.experiment_id
                    logger.info(f"✅ Using existing MLflow experiment: {self.experiment_name} (ID: {experiment_id})")
                
                # Set the experiment
                mlflow.set_experiment(self.experiment_name)
                
                # Verify the experiment is active
                current_exp = mlflow.get_experiment_by_name(self.experiment_name)
                if current_exp:
                    logger.info(f"🔄 Active experiment: {current_exp.name} (ID: {current_exp.experiment_id})")
                    self.mlflow_enabled = True
                else:
                    logger.warning("Failed to verify experiment setup")
                    self.mlflow_enabled = False
                
            except Exception as e:
                logger.warning(f"MLflow experiment setup warning: {e}")
                self.mlflow_enabled = False
                
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")
            self.mlflow_enabled = False
    
    def load_dataset(self, dataset_name="iris"):
        """
        Load a dataset for experimentation.
        
        Args:
            dataset_name: Name of dataset to load (iris, wine, synthetic)
        """
        logger.info(f"📂 Loading dataset: {dataset_name}")
        
        if dataset_name == "iris":
            data = load_iris()
            description = "Iris flower classification (3 classes, 4 features)"
        elif dataset_name == "wine":
            data = load_wine()
            description = "Wine quality classification (3 classes, 13 features)"
        elif dataset_name == "synthetic":
            X, y = make_classification(
                n_samples=1000, n_features=20, n_informative=15,
                n_redundant=5, n_classes=3, random_state=42
            )
            data = type('obj', (), {'data': X, 'target': y, 'feature_names': [f'feature_{i}' for i in range(20)]})
            description = "Synthetic classification (3 classes, 20 features)"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.X, self.y = data.data, data.target
        self.dataset_name = dataset_name
        self.feature_names = getattr(data, 'feature_names', [f'feature_{i}' for i in range(self.X.shape[1])])
        
        logger.info(f"✅ Dataset loaded: {description}")
        logger.info(f"   📊 Samples: {len(self.X)}, Features: {self.X.shape[1]}, Classes: {len(np.unique(self.y))}")
        
        return self.X, self.y
    
    def define_parameter_grid(self):
        """
        Define parameter grids for different algorithms.
        
        Returns:
            Dictionary of algorithm names and their parameter grids
        """
        parameter_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'LogisticRegression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [100, 500, 1000]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        }
        
        logger.info("📋 Parameter grids defined:")
        for algo, params in parameter_grids.items():
            total_combinations = np.prod([len(v) for v in params.values()])
            logger.info(f"   {algo}: {total_combinations} combinations")
        
        return parameter_grids
    
    def run_parameter_sweep(self, max_combinations_per_algo=20):
        """
        Run parameter sweep across multiple algorithms.
        
        Args:
            max_combinations_per_algo: Maximum parameter combinations to try per algorithm
        """
        logger.info("🔄 Starting parameter sweep...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features (will be used for applicable algorithms)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        parameter_grids = self.define_parameter_grid()
        
        # Start parent MLflow run
        parent_run_name = f"parameter_sweep_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=parent_run_name) if self.mlflow_enabled else self._dummy_context():
            if self.mlflow_enabled:
                # Log dataset information
                mlflow.log_param("dataset_name", self.dataset_name)
                mlflow.log_param("n_samples", len(self.X))
                mlflow.log_param("n_features", self.X.shape[1])
                mlflow.log_param("n_classes", len(np.unique(self.y)))
                mlflow.log_param("test_size", 0.2)
                mlflow.set_tag("run_type", "parameter_sweep")
                mlflow.set_tag("dataset", self.dataset_name)
            
            for algo_name, param_grid in parameter_grids.items():
                logger.info(f"\n🎯 Testing algorithm: {algo_name}")
                
                # Generate parameter combinations
                param_names = list(param_grid.keys())
                param_values = list(param_grid.values())
                all_combinations = list(product(*param_values))
                
                # Limit combinations if too many
                if len(all_combinations) > max_combinations_per_algo:
                    selected_combinations = np.random.choice(
                        len(all_combinations), max_combinations_per_algo, replace=False
                    )
                    combinations_to_test = [all_combinations[i] for i in selected_combinations]
                    logger.info(f"   📝 Testing {len(combinations_to_test)} random combinations out of {len(all_combinations)}")
                else:
                    combinations_to_test = all_combinations
                    logger.info(f"   📝 Testing all {len(combinations_to_test)} combinations")
                
                # Test each combination
                for i, param_combo in enumerate(combinations_to_test):
                    params = dict(zip(param_names, param_combo))
                    
                    try:
                        result = self._test_parameter_combination(
                            algo_name, params, X_train, X_test, y_train, y_test,
                            X_train_scaled, X_test_scaled, i + 1
                        )
                        
                        if result:
                            self.results.append(result)
                            
                    except Exception as e:
                        logger.warning(f"   ⚠️  Skipping combination {i+1}: {str(e)}")
                        continue
            
            # Log summary metrics to parent run
            if self.mlflow_enabled and self.results:
                best_result = max(self.results, key=lambda x: x['accuracy'])
                mlflow.log_metric("best_accuracy", best_result['accuracy'])
                mlflow.log_metric("total_experiments", len(self.results))
                mlflow.log_param("best_algorithm", best_result['algorithm'])
        
        logger.info(f"\n✅ Parameter sweep completed! Tested {len(self.results)} combinations")
        return self.results
    
    def _test_parameter_combination(self, algo_name, params, X_train, X_test, y_train, y_test,
                                   X_train_scaled, X_test_scaled, combination_number):
        """Test a single parameter combination."""
        
        # Create model
        if algo_name == 'RandomForest':
            model = RandomForestClassifier(random_state=42, **params)
            X_train_use, X_test_use = X_train, X_test
        elif algo_name == 'LogisticRegression':
            model = LogisticRegression(random_state=42, **params)
            X_train_use, X_test_use = X_train_scaled, X_test_scaled
        elif algo_name == 'SVM':
            model = SVC(random_state=42, **params)
            X_train_use, X_test_use = X_train_scaled, X_test_scaled
        elif algo_name == 'KNN':
            model = KNeighborsClassifier(**params)
            X_train_use, X_test_use = X_train_scaled, X_test_scaled
        else:
            return None
        
        # Start MLflow run for this combination
        run_name = f"{algo_name}_{combination_number:03d}"
        
        with mlflow.start_run(run_name=run_name, nested=True) if self.mlflow_enabled else self._dummy_context():
            # Train model
            model.fit(X_train_use, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_use)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_use, y_train, cv=3, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Log to MLflow
            if self.mlflow_enabled:
                try:
                    # Log parameters
                    mlflow.log_param("algorithm", algo_name)
                    mlflow.log_param("dataset", self.dataset_name)
                    for param_name, param_value in params.items():
                        mlflow.log_param(param_name, param_value)
                    
                    # Log metrics
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("f1_score", f1)
                    mlflow.log_metric("cv_mean", cv_mean)
                    mlflow.log_metric("cv_std", cv_std)
                    
                    # Log model
                    mlflow.sklearn.log_model(model, f"model_{algo_name.lower()}")
                    
                    # Add tags for better organization
                    mlflow.set_tag("algorithm_family", "classification")
                    mlflow.set_tag("dataset", self.dataset_name)
                    mlflow.set_tag("model_type", algo_name)
                    
                except Exception as e:
                    logger.warning(f"MLflow logging warning: {e}")
            
            # Store result
            result = {
                'algorithm': algo_name,
                'parameters': params,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'combination_number': combination_number
            }
            
            logger.info(f"   ✅ Combination {combination_number}: Accuracy={accuracy:.4f}, F1={f1:.4f}, CV={cv_mean:.4f}(±{cv_std:.4f})")
            
            return result
    
    def analyze_results(self):
        """Analyze and visualize the parameter sweep results."""
        if not self.results:
            logger.warning("No results to analyze!")
            return
        
        logger.info("📊 Analyzing results...")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Find best results
        best_overall = df.loc[df['accuracy'].idxmax()]
        logger.info(f"\n🏆 Best Overall Result:")
        logger.info(f"   Algorithm: {best_overall['algorithm']}")
        logger.info(f"   Accuracy: {best_overall['accuracy']:.4f}")
        logger.info(f"   Parameters: {best_overall['parameters']}")
        
        # Best per algorithm
        logger.info(f"\n🏆 Best Results by Algorithm:")
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            best_algo = algo_df.loc[algo_df['accuracy'].idxmax()]
            logger.info(f"   {algo}: {best_algo['accuracy']:.4f}")
        
        # Create visualizations
        self._create_visualizations(df)
        
        # Save results
        results_file = self.output_dir / "parameter_sweep_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"📄 Results saved to: {results_file}")
        
        return df
    
    def _create_visualizations(self, df):
        """Create visualizations of the parameter sweep results."""
        # Set up the plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Parameter Sweep Results - {self.dataset_name.title()} Dataset', fontsize=16)
        
        # 1. Accuracy by Algorithm (Box Plot)
        df.boxplot(column='accuracy', by='algorithm', ax=axes[0, 0])
        axes[0, 0].set_title('Accuracy Distribution by Algorithm')
        axes[0, 0].set_xlabel('Algorithm')
        axes[0, 0].set_ylabel('Accuracy')
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Best Accuracy per Algorithm (Bar Plot)
        best_per_algo = df.groupby('algorithm')['accuracy'].max().sort_values(ascending=False)
        axes[0, 1].bar(best_per_algo.index, best_per_algo.values)
        axes[0, 1].set_title('Best Accuracy by Algorithm')
        axes[0, 1].set_xlabel('Algorithm')
        axes[0, 1].set_ylabel('Best Accuracy')
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(best_per_algo.values):
            axes[0, 1].text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
        
        # 3. Accuracy vs F1 Score Scatter
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            axes[1, 0].scatter(algo_df['accuracy'], algo_df['f1_score'], label=algo, alpha=0.7)
        axes[1, 0].set_xlabel('Accuracy')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Accuracy vs F1 Score')
        axes[1, 0].legend()
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Diagonal line
        
        # 4. CV Mean vs Test Accuracy
        axes[1, 1].scatter(df['cv_mean'], df['accuracy'], c=df['algorithm'].astype('category').cat.codes, alpha=0.7)
        axes[1, 1].set_xlabel('Cross-Validation Mean Accuracy')
        axes[1, 1].set_ylabel('Test Accuracy')
        axes[1, 1].set_title('CV Accuracy vs Test Accuracy')
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Diagonal line
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "plots" / f"parameter_sweep_analysis_{self.dataset_name}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"📈 Visualization saved: {plot_file}")
        plt.show()
    
    def _dummy_context(self):
        """Dummy context manager for when MLflow is disabled."""
        from contextlib import nullcontext
        return nullcontext()


def run_parameter_sweep_demo():
    """
    Run the complete parameter sweep demo.
    """
    logger.info("🚀 Starting MLflow Parameter Sweep Demo")
    logger.info("=" * 60)
    
    # Initialize parameter sweep
    sweep = MLflowParameterSweep(experiment_name="scikit-learn-parameter-sweep")
    
    # Test different datasets
    datasets = ['iris', 'wine', 'synthetic']
    
    for dataset in datasets:
        logger.info(f"\n🎯 Running parameter sweep on {dataset} dataset")
        logger.info("-" * 50)
        
        # Load dataset
        sweep.load_dataset(dataset)
        
        # Run parameter sweep
        results = sweep.run_parameter_sweep(max_combinations_per_algo=15)
        
        # Analyze results
        if results:
            df = sweep.analyze_results()
            
            # Print summary
            logger.info(f"\n📊 Summary for {dataset}:")
            logger.info(f"   Total combinations tested: {len(results)}")
            logger.info(f"   Algorithms tested: {df['algorithm'].nunique()}")
            logger.info(f"   Best accuracy: {df['accuracy'].max():.4f}")
            logger.info(f"   Best algorithm: {df.loc[df['accuracy'].idxmax(), 'algorithm']}")
        
        logger.info(f"✅ Completed parameter sweep for {dataset} dataset")
    
    logger.info(f"\n🎉 Parameter sweep demo completed!")
    logger.info(f"📊 Check MLflow UI for detailed experiment tracking")
    if sweep.mlflow_enabled:
        logger.info(f"🔗 MLflow tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    """
    Run the parameter sweep demo.
    
    Usage:
        python mlflow_parameter_sweep.py
    """
    run_parameter_sweep_demo()