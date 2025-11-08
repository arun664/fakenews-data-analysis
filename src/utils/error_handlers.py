"""
Comprehensive Error Handling for Dashboard Visualizations

This module provides centralized error handling, user-friendly error messages,
and recovery strategies for common dashboard errors.
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Optional, Callable, Any
import traceback
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dashboard_errors.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class DashboardError(Exception):
    """Base exception for dashboard errors"""
    pass


class DataNotFoundError(DashboardError):
    """Raised when required data file is not found"""
    pass


class DataValidationError(DashboardError):
    """Raised when data validation fails"""
    pass


class VisualizationError(DashboardError):
    """Raised when visualization rendering fails"""
    pass


def handle_missing_data(
    data_name: str,
    file_path: str,
    task_command: Optional[str] = None,
    show_placeholder: bool = True
):
    """
    Display user-friendly message for missing data.
    
    Args:
        data_name: Human-readable name of the data
        file_path: Path where data should be located
        task_command: Command to generate the data
        show_placeholder: Whether to show placeholder visualization
    """
    st.warning(f"📂 {data_name} data not available")
    
    st.info(f"""
    **Data file not found:** `{file_path}`
    
    This visualization requires data that hasn't been generated yet.
    """)
    
    if task_command:
        st.code(f"# To generate this data, run:\n{task_command}", language='bash')
    
    if show_placeholder:
        st.markdown("""
        ---
        **What this visualization will show:**
        - Comparison of fake vs real content patterns
        - Statistical analysis and significance testing
        - Interactive charts for data exploration
        
        Please run the required analysis task to see the full visualization.
        """)
    
    logger.warning(f"Missing data: {data_name} at {file_path}")


def handle_validation_error(
    data_name: str,
    expected_columns: list,
    actual_columns: list,
    show_debug: bool = False
):
    """
    Display user-friendly message for data validation errors.
    
    Args:
        data_name: Human-readable name of the data
        expected_columns: List of expected column names
        actual_columns: List of actual column names
        show_debug: Whether to show debug information
    """
    st.error(f"❌ Data validation failed for {data_name}")
    
    missing_columns = set(expected_columns) - set(actual_columns)
    extra_columns = set(actual_columns) - set(expected_columns)
    
    if missing_columns:
        st.warning(f"**Missing columns:** {', '.join(missing_columns)}")
    
    if extra_columns and show_debug:
        st.info(f"**Extra columns:** {', '.join(extra_columns)}")
    
    st.info("""
    **Possible causes:**
    - Data schema has changed
    - Analysis task needs to be re-run
    - Data preprocessing incomplete
    
    **Solution:**
    1. Re-run the analysis task to regenerate data
    2. Check that all preprocessing steps completed successfully
    3. Verify data file integrity
    """)
    
    logger.error(f"Validation error for {data_name}: missing {missing_columns}")


def handle_visualization_error(
    viz_name: str,
    error: Exception,
    show_traceback: bool = False
):
    """
    Display user-friendly message for visualization errors.
    
    Args:
        viz_name: Name of the visualization
        error: The exception that occurred
        show_traceback: Whether to show full traceback
    """
    st.error(f"❌ Error rendering {viz_name}")
    
    st.warning(f"**Error type:** {type(error).__name__}")
    st.warning(f"**Error message:** {str(error)}")
    
    if show_traceback:
        with st.expander("Show detailed error information"):
            st.code(traceback.format_exc())
    
    st.info("""
    **Troubleshooting steps:**
    1. Refresh the page to retry
    2. Check that data is properly formatted
    3. Verify all required columns are present
    4. Review the error log for details
    
    If the problem persists, please check the visualization guide.
    """)
    
    logger.error(f"Visualization error in {viz_name}: {error}", exc_info=True)


def safe_visualization(viz_name: str, show_errors: bool = True):
    """
    Decorator for safe visualization rendering with error handling.
    
    Args:
        viz_name: Name of the visualization
        show_errors: Whether to show error messages to user
    
    Example:
        @safe_visualization("Sentiment Distribution")
        def render_sentiment_chart(data):
            # Visualization code here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                if show_errors:
                    handle_missing_data(
                        viz_name,
                        str(e.filename) if hasattr(e, 'filename') else 'Unknown',
                        task_command=f"# Check analysis_results/ directory"
                    )
                logger.error(f"File not found in {viz_name}: {e}")
                return None
            except KeyError as e:
                if show_errors:
                    st.error(f"❌ Missing required data field: {e}")
                    st.info("Please re-run the analysis task to regenerate data.")
                logger.error(f"Key error in {viz_name}: {e}")
                return None
            except ValueError as e:
                if show_errors:
                    st.error(f"❌ Data format error: {e}")
                    st.info("Please check data types and formats.")
                logger.error(f"Value error in {viz_name}: {e}")
                return None
            except Exception as e:
                if show_errors:
                    handle_visualization_error(viz_name, e, show_traceback=True)
                logger.error(f"Unexpected error in {viz_name}: {e}", exc_info=True)
                return None
        return wrapper
    return decorator


def validate_data_structure(
    data: Any,
    required_keys: list,
    data_name: str,
    raise_error: bool = False
) -> bool:
    """
    Validate data structure and show appropriate error messages.
    
    Args:
        data: Data to validate (dict or DataFrame)
        required_keys: List of required keys/columns
        data_name: Human-readable name of the data
        raise_error: Whether to raise exception on validation failure
    
    Returns:
        True if valid, False otherwise
    
    Raises:
        DataValidationError: If raise_error=True and validation fails
    """
    if data is None:
        if raise_error:
            raise DataValidationError(f"{data_name} is None")
        return False
    
    import pandas as pd
    
    if isinstance(data, dict):
        actual_keys = list(data.keys())
        missing = set(required_keys) - set(actual_keys)
    elif isinstance(data, pd.DataFrame):
        actual_keys = list(data.columns)
        missing = set(required_keys) - set(actual_keys)
    else:
        if raise_error:
            raise DataValidationError(f"{data_name} has invalid type: {type(data)}")
        return False
    
    if missing:
        handle_validation_error(data_name, required_keys, actual_keys)
        if raise_error:
            raise DataValidationError(f"{data_name} missing keys: {missing}")
        return False
    
    return True


def safe_data_load(
    loader_func: Callable,
    data_name: str,
    required_keys: Optional[list] = None,
    fallback_value: Any = None
) -> Any:
    """
    Safely load data with error handling.
    
    Args:
        loader_func: Function to load data
        data_name: Human-readable name of the data
        required_keys: Optional list of required keys to validate
        fallback_value: Value to return on error
    
    Returns:
        Loaded data or fallback_value on error
    """
    try:
        data = loader_func()
        
        if data is None:
            handle_missing_data(
                data_name,
                "Unknown path",
                task_command=f"# Run appropriate analysis task for {data_name}"
            )
            return fallback_value
        
        if required_keys:
            if not validate_data_structure(data, required_keys, data_name):
                return fallback_value
        
        logger.info(f"Successfully loaded {data_name}")
        return data
    
    except FileNotFoundError as e:
        handle_missing_data(
            data_name,
            str(e.filename) if hasattr(e, 'filename') else 'Unknown'
        )
        return fallback_value
    
    except Exception as e:
        st.error(f"❌ Error loading {data_name}: {e}")
        logger.error(f"Error loading {data_name}: {e}", exc_info=True)
        return fallback_value


def create_error_boundary(component_name: str):
    """
    Create an error boundary context manager for components.
    
    Args:
        component_name: Name of the component
    
    Example:
        with create_error_boundary("Sentiment Analysis"):
            render_sentiment_visualizations()
    """
    class ErrorBoundary:
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                handle_visualization_error(component_name, exc_val, show_traceback=True)
                return True  # Suppress exception
            return False
    
    return ErrorBoundary()


def log_performance_warning(
    operation: str,
    duration: float,
    threshold: float = 3.0
):
    """
    Log performance warning if operation exceeds threshold.
    
    Args:
        operation: Name of the operation
        duration: Time taken in seconds
        threshold: Warning threshold in seconds
    """
    if duration > threshold:
        logger.warning(
            f"Performance warning: {operation} took {duration:.2f}s "
            f"(threshold: {threshold}s)"
        )
        
        if duration > threshold * 2:
            st.warning(
                f"⚠️ {operation} is taking longer than expected ({duration:.1f}s). "
                f"Consider optimizing data sampling or caching."
            )


def handle_partial_data(
    data_name: str,
    available_fields: list,
    missing_fields: list
):
    """
    Handle cases where data is partially available.
    
    Args:
        data_name: Human-readable name of the data
        available_fields: List of available fields
        missing_fields: List of missing fields
    """
    st.info(f"ℹ️ {data_name} is partially available")
    
    with st.expander("Show data availability details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Available:**")
            for field in available_fields:
                st.markdown(f"✓ {field}")
        
        with col2:
            st.markdown("**Missing:**")
            for field in missing_fields:
                st.markdown(f"✗ {field}")
    
    st.info("""
    Some visualizations may be limited due to missing data.
    Re-run the analysis task to generate complete data.
    """)
    
    logger.info(f"Partial data for {data_name}: missing {missing_fields}")


def create_fallback_visualization(viz_name: str, message: str = None):
    """
    Create a placeholder visualization when data is unavailable.
    
    Args:
        viz_name: Name of the visualization
        message: Optional custom message
    """
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    fig.add_annotation(
        text=message or f"{viz_name}<br><br>Data not available<br><br>Run analysis task to generate",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
        align="center"
    )
    
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='white',
        height=400
    )
    
    return fig


# Error message templates
ERROR_MESSAGES = {
    'file_not_found': """
    📂 **Data File Not Found**
    
    The required data file could not be located. This usually means the analysis task hasn't been run yet.
    
    **Next steps:**
    1. Run the appropriate analysis task
    2. Check the `analysis_results/` directory
    3. Verify file permissions
    """,
    
    'invalid_format': """
    ⚠️ **Invalid Data Format**
    
    The data file exists but has an unexpected format or structure.
    
    **Next steps:**
    1. Re-run the analysis task
    2. Check for data corruption
    3. Verify data schema matches expectations
    """,
    
    'insufficient_data': """
    📊 **Insufficient Data**
    
    The dataset doesn't have enough records for meaningful visualization.
    
    **Requirements:**
    - Minimum 30 records per category
    - Both fake and real categories present
    - Valid numeric values
    """,
    
    'performance_issue': """
    ⚡ **Performance Issue Detected**
    
    This operation is taking longer than expected.
    
    **Optimization suggestions:**
    - Enable data sampling
    - Increase cache duration
    - Reduce visualization complexity
    """
}


def show_error_message(error_type: str, **kwargs):
    """
    Show standardized error message.
    
    Args:
        error_type: Type of error from ERROR_MESSAGES
        **kwargs: Additional context for the error message
    """
    if error_type in ERROR_MESSAGES:
        st.error(ERROR_MESSAGES[error_type].format(**kwargs))
    else:
        st.error(f"An error occurred: {error_type}")
    
    logger.error(f"Error displayed: {error_type} with context {kwargs}")
