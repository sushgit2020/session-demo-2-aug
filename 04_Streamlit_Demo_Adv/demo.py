import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Streamlit Fundamentals Demo",
    page_icon="🌺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🌺 Streamlit Fundamentals Demo</h1>', unsafe_allow_html=True)
st.markdown("**Interactive Machine Learning with Iris Dataset**")
st.markdown("---")

# Sidebar navigation
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    ["🏠 Home", "📊 Data Exploration", "🎯 Model Training", "📈 Model Evaluation", "🔮 Predictions", "📁 Custom Data"]
)

# Initialize session state
if 'iris_data' not in st.session_state:
    iris = load_iris()
    st.session_state.iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    st.session_state.iris_target = iris.target
    st.session_state.target_names = iris.target_names

if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}

if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# Cache data loading for performance
@st.cache_data
def load_iris_dataset():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    return df, iris.target, iris.target_names

# Helper functions
def create_correlation_heatmap(data):
    """Create an interactive correlation heatmap"""
    corr_matrix = data.corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",
        title="Feature Correlation Heatmap"
    )
    return fig

def create_pairplot_plotly(data, target):
    """Create interactive pairplot using plotly"""
    feature_cols = data.columns
    n_features = len(feature_cols)
    
    fig = make_subplots(
        rows=n_features, cols=n_features,
        subplot_titles=[f"{col1} vs {col2}" for col1 in feature_cols for col2 in feature_cols]
    )
    
    target_names = st.session_state.target_names
    colors = ['red', 'green', 'blue']
    
    for i, col1 in enumerate(feature_cols):
        for j, col2 in enumerate(feature_cols):
            if i == j:
                # Diagonal: histogram
                for k, class_name in enumerate(target_names):
                    mask = target == k
                    fig.add_trace(
                        go.Histogram(
                            x=data.loc[mask, col1],
                            name=class_name,
                            marker_color=colors[k],
                            opacity=0.7,
                            showlegend=(i == 0 and j == 0)
                        ),
                        row=i+1, col=j+1
                    )
            else:
                # Off-diagonal: scatter plot
                for k, class_name in enumerate(target_names):
                    mask = target == k
                    fig.add_trace(
                        go.Scatter(
                            x=data.loc[mask, col2],
                            y=data.loc[mask, col1],
                            mode='markers',
                            name=class_name,
                            marker=dict(color=colors[k]),
                            showlegend=(i == 0 and j == 1)
                        ),
                        row=i+1, col=j+1
                    )
    
    fig.update_layout(height=800, title_text="Pairplot of Iris Features")
    return fig

# HOME PAGE
if page == "🏠 Home":
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/logo.png", width=300)
    
    st.markdown("## Welcome to the Streamlit Fundamentals Demo!")
    
    st.markdown("""
    This interactive application demonstrates key Streamlit features using the famous Iris dataset.
    
    ### 🎯 What you'll learn:
    - **Data Visualization**: Interactive charts and plots
    - **Model Training**: Train ML models with real-time parameter tuning
    - **State Management**: Persistent data across user interactions
    - **UI Components**: Various Streamlit widgets and layouts
    - **Performance**: Caching and optimization techniques
    """)
    
    # Dataset overview
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Samples", len(st.session_state.iris_data))
    with col2:
        st.metric("Features", len(st.session_state.iris_data.columns))
    with col3:
        st.metric("Classes", len(st.session_state.target_names))
    with col4:
        st.metric("Missing Values", st.session_state.iris_data.isnull().sum().sum())
    
    # Quick preview
    st.markdown("### 👀 Data Preview")
    preview_data = st.session_state.iris_data.copy()
    preview_data['species'] = [st.session_state.target_names[i] for i in st.session_state.iris_target]
    st.dataframe(preview_data.head(10), use_container_width=True)

# DATA EXPLORATION PAGE
elif page == "📊 Data Exploration":
    st.header("📊 Data Exploration")
    
    data = st.session_state.iris_data
    target = st.session_state.iris_target
    
    # Dataset statistics
    st.subheader("📈 Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)
    
    # Distribution plots
    st.subheader("📊 Feature Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_feature = st.selectbox("Select Feature:", data.columns)
        
    with col2:
        plot_type = st.radio("Plot Type:", ["Histogram", "Box Plot", "Violin Plot"])
    
    # Create the selected plot
    fig = None
    if plot_type == "Histogram":
        fig = px.histogram(
            data,
            x=selected_feature,
            color=[st.session_state.target_names[i] for i in target],
            marginal="box",
            title=f"Distribution of {selected_feature}",
            barmode="overlay",
            opacity=0.7
        )
    elif plot_type == "Box Plot":
        plot_data = data.copy()
        plot_data['species'] = [st.session_state.target_names[i] for i in target]
        fig = px.box(
            plot_data,
            x='species',
            y=selected_feature,
            title=f"Box Plot of {selected_feature}",
            color='species'
        )
    elif plot_type == "Violin Plot":
        plot_data = data.copy()
        plot_data['species'] = [st.session_state.target_names[i] for i in target]
        fig = px.violin(
            plot_data,
            x='species',
            y=selected_feature,
            title=f"Violin Plot of {selected_feature}",
            color='species',
            box=True
        )
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Feature Correlations")
    fig_corr = create_correlation_heatmap(data)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Scatter plot matrix
    if st.checkbox("Show Pairplot (may take a moment to load)"):
        st.subheader("🎯 Feature Relationships")
        
        # scatter matrix using plotly express
        plot_data = data.copy()
        plot_data['species'] = [st.session_state.target_names[i] for i in target]
        
        fig = px.scatter_matrix(
            plot_data,
            dimensions=data.columns,
            color='species',
            title="Iris Feature Scatter Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)

# MODEL TRAINING PAGE
elif page == "🎯 Model Training":
    st.header("🎯 Model Training")
    
    data = st.session_state.iris_data
    target = st.session_state.iris_target
    
    # Model selection
    st.subheader("🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Model Type:",
            ["Random Forest", "Logistic Regression", "SVM", "K-Nearest Neighbors"]
        )
        
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random State", value=42, min_value=0, max_value=1000)
    
    with col2:
        # Model-specific parameters
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
            max_depth = st.slider("Max Depth", 1, 20, 5, 1)
            model_params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": random_state}
            
        elif model_type == "Logistic Regression":
            C = st.slider("Regularization (C)", 0.01, 100.0, 1.0, 0.01)
            max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)
            model_params = {"C": C, "max_iter": max_iter, "random_state": random_state}
            
        elif model_type == "SVM":
            C = st.slider("Regularization (C)", 0.01, 100.0, 1.0, 0.01)
            kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
            model_params = {"C": C, "kernel": kernel, "random_state": random_state}
            
        elif model_type == "K-Nearest Neighbors":
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, 1)
            weights = st.selectbox("Weights", ["uniform", "distance"])
            model_params = {"n_neighbors": n_neighbors, "weights": weights}
    
    # Training button and progress
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=test_size, random_state=random_state
            )
            
            # Initialize model
            if model_type == "Random Forest":
                model = RandomForestClassifier(**model_params)
            elif model_type == "Logistic Regression":
                model = LogisticRegression(**model_params)
            elif model_type == "SVM":
                model = SVC(**model_params, probability=True)
            elif model_type == "K-Nearest Neighbors":
                model = KNeighborsClassifier(**model_params)
            
            # Train model
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1_score": f1_score(y_test, y_pred, average='weighted')
            }
            
            # Store in session state
            model_key = f"{model_type}_{datetime.now().strftime('%H%M%S')}"
            st.session_state.trained_models[model_key] = {
                "model": model,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba,
                "params": model_params,
                "type": model_type
            }
            st.session_state.model_metrics[model_key] = metrics
            st.session_state.current_model = model_key
            
            st.success(f"✅ Model '{model_key}' trained successfully!")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
    
    # Display trained models
    if st.session_state.trained_models:
        st.subheader("🏆 Trained Models")
        
        models_df = pd.DataFrame([
            {
                "Model": key,
                "Type": value["type"],
                "Accuracy": f"{st.session_state.model_metrics[key]['accuracy']:.3f}",
                "F1-Score": f"{st.session_state.model_metrics[key]['f1_score']:.3f}",
                "Trained": key.split('_')[-1]
            }
            for key, value in st.session_state.trained_models.items()
        ])
        
        st.dataframe(models_df, use_container_width=True)

# MODEL EVALUATION PAGE
elif page == "📈 Model Evaluation":
    st.header("📈 Model Evaluation")
    
    if not st.session_state.trained_models:
        st.warning("⚠️ No trained models found. Please train a model first!")
        st.stop()
    
    # Model selection for evaluation
    selected_model = st.selectbox(
        "Select Model to Evaluate:",
        list(st.session_state.trained_models.keys()),
        index=len(st.session_state.trained_models) - 1
    )
    
    model_data = st.session_state.trained_models[selected_model]
    metrics = st.session_state.model_metrics[selected_model]
    
    # Model information
    st.subheader(f"📊 Model: {selected_model}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Parameters:**")
        for param, value in model_data["params"].items():
            st.write(f"- {param}: {value}")
    
    with col2:
        st.markdown("**Dataset Split:**")
        st.write(f"- Training samples: {len(model_data['X_train'])}")
        st.write(f"- Test samples: {len(model_data['X_test'])}")
        st.write(f"- Features: {len(model_data['X_train'].columns)}")
    
    # Performance metrics
    st.subheader("🎯 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
    
    # Confusion Matrix
    st.subheader("🔍 Confusion Matrix")
    
    cm = confusion_matrix(model_data['y_test'], model_data['y_pred'])
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual")
    )
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(range(3)), ticktext=st.session_state.target_names),
        yaxis=dict(tickmode="array", tickvals=list(range(3)), ticktext=st.session_state.target_names)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.subheader("📋 Detailed Classification Report")
    report = classification_report(
        model_data['y_test'], 
        model_data['y_pred'], 
        target_names=st.session_state.target_names,
        output_dict=True
    )
    
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)
    
    # Feature Importance (if available)
    if hasattr(model_data['model'], 'feature_importances_'):
        st.subheader("🎯 Feature Importance")
        
        importance_df = pd.DataFrame({
            'Feature': model_data['X_train'].columns,
            'Importance': model_data['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance",
            color='Importance',
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction Probabilities (if available)
    if model_data['y_pred_proba'] is not None:
        st.subheader("🎲 Prediction Probabilities")
        
        # Show first 10 predictions
        prob_df = pd.DataFrame(
            model_data['y_pred_proba'][:10],
            columns=[f"P({name})" for name in st.session_state.target_names]
        )
        prob_df['Actual'] = [st.session_state.target_names[i] for i in model_data['y_test'][:10]]
        prob_df['Predicted'] = [st.session_state.target_names[i] for i in model_data['y_pred'][:10]]
        
        st.dataframe(prob_df, use_container_width=True)

# PREDICTIONS PAGE
elif page == "🔮 Predictions":
    st.header("🔮 Make Predictions")
    
    if not st.session_state.trained_models:
        st.warning("⚠️ No trained models found. Please train a model first!")
        st.stop()
    
    # Model selection
    selected_model = st.selectbox(
        "Select Model for Predictions:",
        list(st.session_state.trained_models.keys()),
        index=len(st.session_state.trained_models) - 1
    )
    
    model_data = st.session_state.trained_models[selected_model]
    model = model_data['model']
    
    st.subheader(f"🎯 Using Model: {selected_model}")
    
    # Input form
    st.subheader("📝 Input Features")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
            sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
        
        with col2:
            petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
            petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
        
        submitted = st.form_submit_button("🚀 Make Prediction", type="primary")
        
        if submitted:
            # Prepare input data
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else None
            
            # Display results
            st.subheader("🎊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Species", st.session_state.target_names[prediction])
            
            if prediction_proba is not None:
                with col2:
                    confidence = np.max(prediction_proba)
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with col3:
                    second_best = np.sort(prediction_proba)[-2]
                    margin = confidence - second_best
                    st.metric("Margin", f"{margin:.1%}")
                
                # Probability chart
                st.subheader("📊 Prediction Probabilities")
                
                prob_df = pd.DataFrame({
                    'Species': st.session_state.target_names,
                    'Probability': prediction_proba
                })
                
                fig = px.bar(
                    prob_df,
                    x='Species',
                    y='Probability',
                    title="Prediction Probabilities",
                    color='Probability',
                    color_continuous_scale="viridis"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    # Batch predictions
    st.subheader("📊 Batch Predictions")
    
    if st.checkbox("Enable Batch Predictions"):
        uploaded_file = st.file_uploader(
            "Upload CSV file with iris features",
            type=['csv'],
            help="CSV should have columns: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)"
        )
        
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.write("📄 Uploaded Data:")
                st.dataframe(batch_data.head(), use_container_width=True)
                
                if st.button("🚀 Run Batch Predictions"):
                    # Make predictions
                    batch_predictions = model.predict(batch_data)
                    batch_probabilities = model.predict_proba(batch_data) if hasattr(model, 'predict_proba') else None
                    
                    # Prepare results
                    results_df = batch_data.copy()
                    results_df['Predicted_Species'] = [st.session_state.target_names[pred] for pred in batch_predictions]
                    
                    if batch_probabilities is not None:
                        for i, species in enumerate(st.session_state.target_names):
                            results_df[f'P({species})'] = batch_probabilities[:, i]
                        results_df['Confidence'] = np.max(batch_probabilities, axis=1)
                    
                    st.write("🎯 Prediction Results:")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "batch_predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# CUSTOM DATA PAGE
elif page == "📁 Custom Data":
    st.header("📁 Custom Data Upload")
    
    st.markdown("""
    Upload your own dataset to explore Streamlit's data handling capabilities.
    Your dataset should be a CSV file with numerical features and an optional target column.
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with numerical data"
    )
    
    if uploaded_file is not None:
        try:
            custom_data = pd.read_csv(uploaded_file)
            
            st.subheader("📊 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", len(custom_data))
            with col2:
                st.metric("Columns", len(custom_data.columns))
            with col3:
                missing_pct = (custom_data.isnull().sum().sum() / custom_data.size) * 100
                st.metric("Missing %", f"{missing_pct:.1f}%")
            with col4:
                numeric_cols = len(custom_data.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Columns", numeric_cols)
            
            # Data preview
            st.subheader("👀 Data Preview")
            st.dataframe(custom_data.head(20), use_container_width=True)
            
            # Statistical summary
            st.subheader("📈 Statistical Summary")
            st.dataframe(custom_data.describe(), use_container_width=True)
            
            # Data quality check
            st.subheader("🔍 Data Quality")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Missing Values per Column:**")
                missing_data = custom_data.isnull().sum()
                if missing_data.sum() > 0:
                    missing_df = pd.DataFrame({
                        'Column': missing_data.index,
                        'Missing': missing_data.values,
                        'Percentage': (missing_data.values / len(custom_data)) * 100
                    })
                    st.dataframe(missing_df[missing_df['Missing'] > 0])
                else:
                    st.success("✅ No missing values found!")
            
            with col2:
                st.markdown("**Data Types:**")
                dtype_df = pd.DataFrame({
                    'Column': custom_data.columns,
                    'Type': [str(dtype) for dtype in custom_data.dtypes],
                    'Unique Values': [custom_data[col].nunique() for col in custom_data.columns]
                })
                st.dataframe(dtype_df)
            
            # Visualization options
            numeric_columns = custom_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) >= 2:
                st.subheader("📊 Data Visualization")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_axis = st.selectbox("X-axis:", numeric_columns)
                with col2:
                    y_axis = st.selectbox("Y-axis:", [col for col in numeric_columns if col != x_axis])
                with col3:
                    color_by = st.selectbox("Color by:", ["None"] + custom_data.columns.tolist())
                
                # Create scatter plot
                if color_by == "None":
                    fig = px.scatter(
                        custom_data,
                        x=x_axis,
                        y=y_axis,
                        title=f"{y_axis} vs {x_axis}"
                    )
                else:
                    fig = px.scatter(
                        custom_data,
                        x=x_axis,
                        y=y_axis,
                        color=color_by,
                        title=f"{y_axis} vs {x_axis} (colored by {color_by})"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation heatmap for numeric data
                if len(numeric_columns) > 2:
                    st.subheader("🔗 Correlation Analysis")
                    numeric_data = custom_data[numeric_columns]
                    fig_corr = create_correlation_heatmap(numeric_data)
                    st.plotly_chart(fig_corr, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.info("Please make sure your file is a valid CSV format.")

# Sidebar info and controls
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Session")
st.sidebar.info(f"Trained Models: {len(st.session_state.trained_models)}")

if st.session_state.trained_models:
    st.sidebar.success("✅ Models Ready")
    if st.sidebar.button("🗑️ Clear All Models"):
        st.session_state.trained_models = {}
        st.session_state.model_metrics = {}
        st.session_state.current_model = None
        st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This demo showcases Streamlit fundamentals including:\n\n"
    "• Interactive widgets\n"
    "• Session state management\n"
    "• Data visualization\n"
    "• File upload/download\n"
    "• Model training & evaluation\n"
    "• Custom styling"
)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using Streamlit | Iris Dataset Demo"
    "</div>",
    unsafe_allow_html=True
)