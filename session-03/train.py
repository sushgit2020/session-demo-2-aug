"""
Simple Scikit-Learn ML Training Demo for Beginners
=================================================

This is a beginner-friendly machine learning script using scikit-learn.
It demonstrates multiple ML algorithms and shows complete MLOps workflows.

The script covers:
- Data loading and preprocessing
- Multiple ML algorithms (Classification & Regression)
- Model training, evaluation, and comparison
- Model persistence and deployment
- Visualization of results

Perfect for beginners learning ML and MLOps concepts!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import joblib
from datetime import datetime
import warnings

# Scikit-learn imports
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, 
    load_diabetes, make_classification, fetch_california_housing
)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleMLTrainer:
    """
    A comprehensive ML trainer class for beginners using scikit-learn.
    
    This class demonstrates multiple ML algorithms and complete workflows
    including data loading, preprocessing, training, evaluation, and model saving.
    """
    
    def __init__(self, output_dir="outputs"):
        """
        Initialize the ML trainer.
        
        Args:
            output_dir: Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.datasets = {}
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        logger.info(f"ML Trainer initialized with output directory: {output_dir}")
    
    def load_sample_datasets(self):
        """
        Load various sample datasets for demonstration.
        This shows different types of ML problems (classification & regression).
        """
        logger.info("🔄 Loading sample datasets...")
        
        # Classification datasets
        self.datasets['iris'] = {
            'data': load_iris(),
            'type': 'classification',
            'description': 'Iris flower species classification (3 classes, 4 features)'
        }
        
        self.datasets['wine'] = {
            'data': load_wine(),
            'type': 'classification', 
            'description': 'Wine quality classification (3 classes, 13 features)'
        }
        
        self.datasets['breast_cancer'] = {
            'data': load_breast_cancer(),
            'type': 'classification',
            'description': 'Breast cancer diagnosis (2 classes, 30 features)'
        }
        
        # Regression datasets
        self.datasets['diabetes'] = {
            'data': load_diabetes(),
            'type': 'regression',
            'description': 'Diabetes progression prediction (1 target, 10 features)'
        }
        
        self.datasets['california_housing'] = {
            'data': fetch_california_housing(),
            'type': 'regression',
            'description': 'California housing prices prediction (1 target, 8 features)'
        }
        
        # Create a synthetic dataset for demonstration
        X_synthetic, y_synthetic = make_classification(
            n_samples=1000, n_features=10, n_informative=5,
            n_redundant=2, n_classes=2, random_state=42
        )
        
        synthetic_data = type('obj', (object,), {
            'data': X_synthetic,
            'target': y_synthetic,
            'feature_names': [f'feature_{i}' for i in range(10)],
            'target_names': ['class_0', 'class_1']
        })
        
        self.datasets['synthetic'] = {
            'data': synthetic_data,
            'type': 'classification',
            'description': 'Synthetic binary classification (2 classes, 10 features)'
        }
        
        logger.info(f"✅ Loaded {len(self.datasets)} datasets:")
        for name, info in self.datasets.items():
            logger.info(f"   - {name}: {info['description']}")
    
    def explore_dataset(self, dataset_name):
        """
        Explore and visualize a dataset.
        
        Args:
            dataset_name: Name of the dataset to explore
        """
        if dataset_name not in self.datasets:
            logger.error(f"Dataset '{dataset_name}' not found!")
            return
        
        logger.info(f"🔍 Exploring dataset: {dataset_name}")
        
        dataset_info = self.datasets[dataset_name]
        data = dataset_info['data']
        
        # Basic information
        X, y = data.data, data.target
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        logger.info(f"   📊 Dataset Info:")
        logger.info(f"      - Samples: {n_samples}")
        logger.info(f"      - Features: {n_features}")
        logger.info(f"      - Classes/Targets: {n_classes}")
        logger.info(f"      - Type: {dataset_info['type']}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Dataset Exploration: {dataset_name.title()}', fontsize=16)
        
        # Feature distribution (first few features)
        axes[0, 0].hist(X[:, 0], bins=30, alpha=0.7)
        axes[0, 0].set_title(f'Feature 0 Distribution')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # Target distribution
        unique, counts = np.unique(y, return_counts=True)
        axes[0, 1].bar(unique, counts)
        axes[0, 1].set_title('Target Distribution')
        axes[0, 1].set_xlabel('Class/Target')
        axes[0, 1].set_ylabel('Count')
        
        # Correlation heatmap (first 5 features if available)
        n_viz_features = min(5, n_features)
        if n_viz_features > 1:
            corr_matrix = np.corrcoef(X[:, :n_viz_features].T)
            im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
            axes[1, 0].set_title('Feature Correlation (First 5)')
            plt.colorbar(im, ax=axes[1, 0])
        
        # Feature vs Target (for first feature)
        if dataset_info['type'] == 'classification':
            for class_label in unique:
                mask = y == class_label
                axes[1, 1].scatter(X[mask, 0], X[mask, 1] if n_features > 1 else np.random.normal(0, 0.1, sum(mask)),
                                 label=f'Class {class_label}', alpha=0.6)
            axes[1, 1].legend()
            axes[1, 1].set_title('Feature Scatter (First 2 Features)')
        else:
            axes[1, 1].scatter(X[:, 0], y, alpha=0.6)
            axes[1, 1].set_title('Feature vs Target')
            axes[1, 1].set_xlabel('Feature 0')
            axes[1, 1].set_ylabel('Target')
        
        plt.tight_layout()
        plot_path = self.output_dir / "plots" / f"{dataset_name}_exploration.png"
        plt.savefig(plot_path)
        plt.show()
        
        logger.info(f"   📈 Exploration plot saved: {plot_path}")
        
        return {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'type': dataset_info['type']
        }
    
    def train_classification_models(self, dataset_name):
        """
        Train multiple classification models and compare their performance.
        
        Args:
            dataset_name: Name of the classification dataset
        """
        if dataset_name not in self.datasets:
            logger.error(f"Dataset '{dataset_name}' not found!")
            return
        
        if self.datasets[dataset_name]['type'] != 'classification':
            logger.error(f"Dataset '{dataset_name}' is not a classification dataset!")
            return
        
        logger.info(f"🎯 Training classification models on: {dataset_name}")
        
        # Get data
        data = self.datasets[dataset_name]['data']
        X, y = data.data, data.target
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models to train
        models_to_train = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'SVM': SVC(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier()
        }
        
        results = {}
        trained_models = {}
        
        for model_name, model in models_to_train.items():
            logger.info(f"   🔄 Training {model_name}...")
            
            # Train the model
            if model_name in ['Logistic Regression', 'SVM']:
                # These models benefit from scaled features
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled if model_name in ['Logistic Regression', 'SVM'] else X_train, 
                                      y_train, cv=5, scoring='accuracy')
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
            
            trained_models[model_name] = model
            
            logger.info(f"      ✅ Accuracy: {accuracy:.4f}, F1: {f1:.4f}, CV: {cv_scores.mean():.4f}(±{cv_scores.std():.4f})")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        logger.info(f"   🏆 Best model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
        
        # Save results
        self.results[f"{dataset_name}_classification"] = results
        self.models[f"{dataset_name}_classification"] = trained_models
        
        # Create comparison visualization
        self._plot_classification_results(dataset_name, results, y_test, trained_models['Random Forest'].predict(X_test))
        
        # Save models
        self._save_models(f"{dataset_name}_classification", trained_models, scaler)
        
        return results
    
    def train_regression_models(self, dataset_name):
        """
        Train multiple regression models and compare their performance.
        
        Args:
            dataset_name: Name of the regression dataset
        """
        if dataset_name not in self.datasets:
            logger.error(f"Dataset '{dataset_name}' not found!")
            return
        
        if self.datasets[dataset_name]['type'] != 'regression':
            logger.error(f"Dataset '{dataset_name}' is not a regression dataset!")
            return
        
        logger.info(f"📈 Training regression models on: {dataset_name}")
        
        # Get data
        data = self.datasets[dataset_name]['data']
        X, y = data.data, data.target
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models to train
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR()
        }
        
        results = {}
        trained_models = {}
        
        for model_name, model in models_to_train.items():
            logger.info(f"   🔄 Training {model_name}...")
            
            # Train the model
            if model_name in ['Linear Regression', 'SVR']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled if model_name in ['Linear Regression', 'SVR'] else X_train,
                                      y_train, cv=5, scoring='r2')
            
            results[model_name] = {
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred.tolist()
            }
            
            trained_models[model_name] = model
            
            logger.info(f"      ✅ RMSE: {rmse:.4f}, R²: {r2:.4f}, CV: {cv_scores.mean():.4f}(±{cv_scores.std():.4f})")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['r2_score'])
        logger.info(f"   🏆 Best model: {best_model_name} (R²: {results[best_model_name]['r2_score']:.4f})")
        
        # Save results
        self.results[f"{dataset_name}_regression"] = results
        self.models[f"{dataset_name}_regression"] = trained_models
        
        # Create comparison visualization
        self._plot_regression_results(dataset_name, results, y_test, trained_models['Random Forest'].predict(X_test))
        
        # Save models
        self._save_models(f"{dataset_name}_regression", trained_models, scaler)
        
        return results
    
    def hyperparameter_tuning_demo(self, dataset_name='iris'):
        """
        Demonstrate hyperparameter tuning with GridSearchCV.
        
        Args:
            dataset_name: Dataset to use for tuning demo
        """
        logger.info(f"🔧 Hyperparameter tuning demo on: {dataset_name}")
        
        if dataset_name not in self.datasets:
            logger.error(f"Dataset '{dataset_name}' not found!")
            return
        
        # Get data
        data = self.datasets[dataset_name]['data']
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
        
        # Perform grid search
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        
        logger.info("   🔄 Performing grid search...")
        grid_search.fit(X_train, y_train)
        
        # Get results
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        
        logger.info(f"   ✅ Best CV score: {best_score:.4f}")
        logger.info(f"   🏆 Best parameters: {best_params}")
        
        # Test the best model
        best_model = grid_search.best_estimator_
        test_score = best_model.score(X_test, y_test)
        logger.info(f"   📊 Test accuracy: {test_score:.4f}")
        
        return {
            'best_score': best_score,
            'best_params': best_params,
            'test_score': test_score
        }
    
    def _plot_classification_results(self, dataset_name, results, y_test, y_pred):
        """Plot classification results comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Classification Results: {dataset_name.title()}', fontsize=16)
        
        # Metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        models = list(results.keys())
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            axes[i//2, i%2].bar(models, values)
            axes[i//2, i%2].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i//2, i%2].set_ylabel(metric.replace("_", " ").title())
            axes[i//2, i%2].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i//2, i%2].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = self.output_dir / "plots" / f"{dataset_name}_classification_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Confusion Matrix for best model
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        y_pred_best = results[best_model_name]['predictions']
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred_best)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = self.output_dir / "plots" / f"{dataset_name}_confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"   📈 Plots saved: {plot_path}, {cm_path}")
    
    def _plot_regression_results(self, dataset_name, results, y_test, y_pred):
        """Plot regression results comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Regression Results: {dataset_name.title()}', fontsize=16)
        
        # Metrics comparison
        models = list(results.keys())
        
        # RMSE comparison
        rmse_values = [results[model]['rmse'] for model in models]
        axes[0, 0].bar(models, rmse_values)
        axes[0, 0].set_title('RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # R² comparison
        r2_values = [results[model]['r2_score'] for model in models]
        axes[0, 1].bar(models, r2_values)
        axes[0, 1].set_title('R² Score Comparison')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Actual vs Predicted scatter plot
        axes[1, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].set_title('Actual vs Predicted')
        
        # Residuals plot
        residuals = y_test - y_pred
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Plot')
        
        plt.tight_layout()
        plot_path = self.output_dir / "plots" / f"{dataset_name}_regression_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"   📈 Plot saved: {plot_path}")
    
    def _save_models(self, experiment_name, models, scaler):
        """Save trained models and scaler."""
        save_dir = self.output_dir / "models" / experiment_name
        save_dir.mkdir(exist_ok=True)
        
        # Save each model
        for model_name, model in models.items():
            model_path = save_dir / f"{model_name.lower().replace(' ', '_')}.joblib"
            joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = save_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'models': list(models.keys()),
            'scaler_used': True
        }
        
        metadata_path = save_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"   💾 Models saved to: {save_dir}")
    
    def create_summary_report(self):
        """Create a comprehensive summary report of all experiments."""
        logger.info("📋 Creating summary report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'datasets_used': list(self.datasets.keys()),
            'experiments': self.results,
            'total_models_trained': sum(len(models) for models in self.models.values()),
        }
        
        # Find best models across all experiments
        best_models = {}
        for exp_name, exp_results in self.results.items():
            if 'classification' in exp_name:
                best_model = max(exp_results, key=lambda x: exp_results[x]['accuracy'])
                best_models[exp_name] = {
                    'model': best_model,
                    'accuracy': exp_results[best_model]['accuracy']
                }
            elif 'regression' in exp_name:
                best_model = max(exp_results, key=lambda x: exp_results[x]['r2_score'])
                best_models[exp_name] = {
                    'model': best_model,
                    'r2_score': exp_results[best_model]['r2_score']
                }
        
        report['best_models'] = best_models
        
        # Save report
        report_path = self.output_dir / "summary_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"   📊 Summary report saved: {report_path}")
        
        # Print summary
        logger.info("=" * 50)
        logger.info("🎉 EXPERIMENT SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Datasets explored: {len(self.datasets)}")
        logger.info(f"Total models trained: {report['total_models_trained']}")
        logger.info(f"Experiments completed: {len(self.results)}")
        logger.info("\n🏆 Best Models:")
        for exp_name, best_info in best_models.items():
            metric = 'accuracy' if 'classification' in exp_name else 'r2_score'
            logger.info(f"   {exp_name}: {best_info['model']} ({metric}: {best_info.get(metric, 'N/A'):.4f})")
        
        return report


def run_comprehensive_ml_demo():
    """
    Run the complete ML demo showcasing various algorithms and workflows.
    """
    logger.info("🚀 Starting Comprehensive Scikit-Learn ML Demo")
    logger.info("=" * 60)
    
    # Initialize trainer
    trainer = SimpleMLTrainer()
    
    # Step 1: Load datasets
    trainer.load_sample_datasets()
    
    # Step 2: Explore datasets
    logger.info("\n🔍 Dataset Exploration Phase")
    for dataset_name in ['iris', 'wine', 'diabetes']:
        trainer.explore_dataset(dataset_name)
    
    # Step 3: Train classification models
    logger.info("\n🎯 Classification Training Phase")
    for dataset_name in ['iris', 'wine', 'breast_cancer', 'synthetic']:
        trainer.train_classification_models(dataset_name)
    
    # Step 4: Train regression models
    logger.info("\n📈 Regression Training Phase")
    trainer.train_regression_models('diabetes')
    trainer.train_regression_models('california_housing')
    
    # Step 5: Hyperparameter tuning demo
    logger.info("\n🔧 Hyperparameter Tuning Demo")
    trainer.hyperparameter_tuning_demo('iris')
    
    # Step 6: Create summary report
    trainer.create_summary_report()
    
    logger.info("\n✅ Demo completed successfully!")
    logger.info(f"Check the outputs directory for saved models, plots, and reports.")
    
    return trainer


if __name__ == "__main__":
    """
    Run this script to see the complete scikit-learn ML demo.
    
    Usage:
        python simple_ml_demo.py
    """
    trainer = run_comprehensive_ml_demo()