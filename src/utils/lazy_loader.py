"""
Lazy loading framework for heavy content
"""
import streamlit as st
import pandas as pd


class LazyLoader:
    """Lazy loading framework for heavy content"""
    
    @staticmethod
    def show_section_loading(section_name):
        """Show loading screen for a section"""
        st.markdown(f"""
        <script>
            showGlobalLoading('Loading {section_name}...', true);
        </script>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def hide_section_loading():
        """Hide loading screen"""
        st.markdown("""
        <script>
            hideGlobalLoading();
        </script>
        """, unsafe_allow_html=True)
    
    @staticmethod
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_heavy_data(data_path, sample_size=None):
        """Load heavy data with optional sampling"""
        try:
            data = pd.read_parquet(data_path)
            if sample_size and len(data) > sample_size:
                data = data.sample(n=sample_size, random_state=42)
            return data
        except Exception as e:
            st.error(f"Error loading data from {data_path}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def lazy_component(component_func, *args, **kwargs):
        """Wrapper for lazy loading components"""
        try:
            return component_func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error loading component: {e}")
            return None
