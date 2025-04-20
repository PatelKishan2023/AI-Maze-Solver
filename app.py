import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from matplotlib.animation import FuncAnimation

# Import algorithm modules
import maze_generator
import search_algorithms
import visualization
import performance_metrics

# Import algorithm educational content
from algorithm_descriptions import display_algorithm_info, get_algorithm_details
from algorithm_comparisons import display_algorithm_comparison

# Set page configuration
st.set_page_config(
    page_title="Maze Algorithm Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'maze' not in st.session_state:
    st.session_state.maze = None
    st.session_state.start_pos = None
    st.session_state.end_pos = None
    st.session_state.results = {}
    st.session_state.running = False
    st.session_state.step_states = {}
    st.session_state.current_step = {}
    st.session_state.step_index = {}
    st.session_state.step_by_step_mode = False
    st.session_state.algorithm_states = {}

# Main title
st.title("🧩 Maze Algorithm Visualizer")
st.markdown("Visualize and compare different search algorithms solving mazes")

# Sidebar with controls
with st.sidebar:
    st.header("Controls")
    
    # Maze generation parameters
    st.subheader("Maze Generation")
    maze_size = st.slider("Maze Size", min_value=5, max_value=30, value=15, step=1)
    maze_complexity = st.slider("Maze Complexity", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
    
    if st.button("Generate New Maze"):
        with st.spinner("Generating maze..."):
            st.session_state.maze, st.session_state.start_pos, st.session_state.end_pos = maze_generator.generate_maze(maze_size, maze_complexity)
            st.session_state.results = {}
            st.session_state.running = False
            st.session_state.step_states = {}
            st.session_state.current_step = {}
            st.session_state.step_index = {}
        st.success("Maze generated!")
    
    # Algorithm selection
    st.subheader("Algorithm Selection")
    # Only include algorithms that have detailed information available
    available_algorithms = ["Linear Search", "Binary Search", "Jump Search", 
                            "Interpolation Search", "DFS", "BFS", "A*", 
                            "Dijkstra's Algorithm", "Bidirectional Search"]
    
    selected_algorithms = st.multiselect(
        "Select Algorithms to Compare",
        available_algorithms,
        default=["BFS", "DFS", "A*"]
    )
    
    # Visualization controls
    st.subheader("Visualization Controls")
    step_by_step = st.checkbox("Step-by-Step Mode", value=False)
    st.session_state.step_by_step_mode = step_by_step
    
    viz_speed = st.slider("Visualization Speed", min_value=1, max_value=10, value=5, step=1)
    sleep_time = 1.0 / viz_speed if viz_speed > 0 else 0
    
    # Algorithm execution controls
    start_button = st.button("Start", disabled=st.session_state.running or st.session_state.maze is None)
    stop_button = st.button("Stop", disabled=not st.session_state.running)
    reset_button = st.button("Reset Results")
    
    if reset_button:
        st.session_state.results = {}
        st.session_state.running = False
        st.session_state.step_states = {}
        st.session_state.current_step = {}
        st.session_state.step_index = {}
        st.rerun()

# Main content area
if st.session_state.maze is None:
    st.info("Generate a maze using the sidebar controls to get started!")
    
    # Sample visualization to show what the app does
    st.subheader("What This App Does")
    st.markdown("""
    This application allows you to:
    1. Generate random mazes of different sizes and complexity
    2. Visualize how different search algorithms solve the maze
    3. Compare performance metrics like execution time, path length, and nodes explored
    4. Learn about algorithm theory, implementation details, and applications
    
    **Get started by generating a maze using the sidebar controls!**
    """)
    
    # Show algorithm categories
    st.subheader("Available Algorithm Categories")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Classic Search Algorithms")
        st.markdown("- Linear Search\n- Binary Search\n- Jump Search\n- Interpolation Search")
        
    with col2:
        st.markdown("#### Graph Traversal Algorithms")
        st.markdown("- Depth-First Search (DFS)\n- Breadth-First Search (BFS)\n- Uniform Cost Search\n- Bidirectional Search")
        
    with col3:
        st.markdown("#### Informed Search Algorithms")
        st.markdown("- A* Search\n- Best-First Search\n- Hill Climbing\n- Iterative Deepening A* (IDA*)")
        
else:
    # Display the maze
    maze_col, info_col = st.columns([2, 1])
    
    with maze_col:
        st.subheader("Maze Visualization")
        
        # Display the generated maze with results if available
        if st.session_state.results:
            viz_data = {}
            for algo_name, result in st.session_state.results.items():
                viz_data[algo_name] = result
                
            fig = visualization.visualize_maze(
                st.session_state.maze, 
                st.session_state.start_pos, 
                st.session_state.end_pos, 
                viz_data
            )
            st.pyplot(fig)
        # Display the step-by-step visualization if in that mode
        elif st.session_state.step_by_step_mode and st.session_state.current_step:
            viz_data = {}
            for algo_name, state in st.session_state.current_step.items():
                viz_data[algo_name] = state
                
            fig = visualization.visualize_maze(
                st.session_state.maze, 
                st.session_state.start_pos, 
                st.session_state.end_pos, 
                viz_data
            )
            st.pyplot(fig)
        # Display just the maze with no algorithm data
        else:
            fig = visualization.visualize_maze(
                st.session_state.maze, 
                st.session_state.start_pos, 
                st.session_state.end_pos
            )
            st.pyplot(fig)
    
    with info_col:
        st.subheader("Maze Information")
        st.markdown(f"**Maze Size:** {st.session_state.maze.shape[0]}×{st.session_state.maze.shape[0]}")
        st.markdown(f"**Complexity:** {maze_complexity}")
        st.markdown(f"**Start Position:** {st.session_state.start_pos}")
        st.markdown(f"**End Position:** {st.session_state.end_pos}")
        
        if st.session_state.results:
            st.subheader("Quick Results")
            results_df = []
            
            for algo_name, result in st.session_state.results.items():
                if 'metrics' in result:
                    metrics = result['metrics']
                    path_length = metrics['path_length'] if metrics['found_solution'] else "No path"
                    results_df.append({
                        "Algorithm": algo_name,
                        "Path Found": "✅" if metrics['found_solution'] else "❌",
                        "Path Length": path_length,
                        "Time (ms)": round(metrics['time'], 2)
                    })
            
            if results_df:
                st.dataframe(pd.DataFrame(results_df), use_container_width=True)
                
    # Handle algorithm execution
    if start_button:
        if not selected_algorithms:
            st.warning("Please select at least one algorithm to run.")
        else:
            st.session_state.running = True
            st.session_state.results = {}
            st.session_state.step_states = {}
            st.session_state.current_step = {}
            st.session_state.step_index = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run each selected algorithm
            for i, algo_name in enumerate(selected_algorithms):
                status_text.text(f"Running {algo_name}...")
                progress_bar.progress((i / len(selected_algorithms)))
                
                start_time = time.time()
                
                # Get the appropriate algorithm function
                algo_func = getattr(search_algorithms, algo_name.lower().replace(" ", "_").replace("-", "_").replace("'", ""), None)
                
                if algo_func:
                    try:
                        # Run the algorithm with or without step tracking
                        if st.session_state.step_by_step_mode:
                            result, states = algo_func(
                                st.session_state.maze,
                                st.session_state.start_pos,
                                st.session_state.end_pos,
                                return_states=True
                            )
                            st.session_state.step_states[algo_name] = states
                            st.session_state.step_index[algo_name] = 0
                            
                            # Initialize current step
                            if len(states) > 0:
                                st.session_state.current_step[algo_name] = states[0]
                        else:
                            result = algo_func(
                                st.session_state.maze,
                                st.session_state.start_pos,
                                st.session_state.end_pos
                            )
                            
                        # Calculate execution time
                        execution_time = (time.time() - start_time) * 1000  # in ms
                        
                        # Calculate performance metrics
                        metrics = performance_metrics.calculate_metrics(result, execution_time)
                        
                        # Store result
                        st.session_state.results[algo_name] = {
                            'result': result,
                            'metrics': metrics
                        }
                    except Exception as e:
                        st.error(f"Error running {algo_name}: {str(e)}")
                else:
                    st.warning(f"Algorithm function for {algo_name} not found.")
            
            progress_bar.progress(1.0)
            status_text.text("All algorithms completed!")
            
            if st.session_state.step_by_step_mode:
                st.session_state.running = False
                st.rerun()
            else:
                # Show the results directly
                st.session_state.running = False
                st.rerun()
                
    # Handle step-by-step visualization
    if st.session_state.step_by_step_mode and st.session_state.step_states:
        st.subheader("Step-by-Step Visualization Controls")
        
        step_cols = st.columns(len(st.session_state.step_states))
        
        for i, (algo_name, states) in enumerate(st.session_state.step_states.items()):
            with step_cols[i]:
                st.markdown(f"**{algo_name}**")
                step_index = st.session_state.step_index.get(algo_name, 0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"◀️ Prev", key=f"prev_{algo_name}", disabled=step_index <= 0):
                        st.session_state.step_index[algo_name] = max(0, step_index - 1)
                        st.session_state.current_step[algo_name] = states[st.session_state.step_index[algo_name]]
                        st.rerun()
                        
                with col2:
                    if st.button(f"▶️ Next", key=f"next_{algo_name}", disabled=step_index >= len(states) - 1):
                        st.session_state.step_index[algo_name] = min(len(states) - 1, step_index + 1)
                        st.session_state.current_step[algo_name] = states[st.session_state.step_index[algo_name]]
                        st.rerun()
                        
                with col3:
                    if st.button(f"⏭️ End", key=f"end_{algo_name}", disabled=step_index >= len(states) - 1):
                        st.session_state.step_index[algo_name] = len(states) - 1
                        st.session_state.current_step[algo_name] = states[st.session_state.step_index[algo_name]]
                        st.rerun()
                        
                st.markdown(f"Step: {step_index + 1}/{len(states)}")
    
    # Show performance metrics if we have results
    if st.session_state.results:
        st.header("Performance Metrics")
        
        # Create a dataframe of metrics
        metrics_data = []
        
        for algo_name, result in st.session_state.results.items():
            metrics = result['metrics']
            path_length = metrics['path_length'] if metrics['path_length'] is not None else "No path"
            nodes_explored = metrics['nodes_explored']
            execution_time = metrics['time']
            found_solution = metrics['found_solution']
            efficiency = metrics['efficiency'] if found_solution else "N/A"
            
            metrics_data.append({
                'Algorithm': algo_name,
                'Path Length': path_length,
                'Nodes Explored': nodes_explored,
                'Time (ms)': execution_time,
                'Found Solution': found_solution,
                'Efficiency': efficiency
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Display the metrics table
        st.subheader("Metrics Comparison")
        st.dataframe(metrics_df, use_container_width=True)
        
        # Visualization of metrics
        st.subheader("Metrics Visualization")
        
        metric_options = ["Time (ms)", "Nodes Explored", "Path Length", "Efficiency"]
        selected_metric = st.selectbox("Select Metric to Visualize", metric_options)
        
        # Generate visualization of selected metric
        if metrics_data:
            fig = visualization.visualize_performance(metrics_df, selected_metric)
            st.pyplot(fig)

# Educational content
st.header("📚 Algorithm Education Center")
st.markdown("""
    Select an algorithm to learn more about how it works, its time and space complexity, 
    advantages and disadvantages, pseudocode, and real-world applications.
""")

# Tabs for different educational content
educational_tab, comparison_tab = st.tabs(["Algorithm Details", "Algorithm Comparison"])

with educational_tab:
    # Create a dropdown to select algorithm
    selected_learning_algo = st.selectbox(
        "Select an algorithm to learn about:",
        available_algorithms
    )
    
    # Display detailed information about the selected algorithm
    display_algorithm_info(selected_learning_algo)

with comparison_tab:
    # Compare multiple algorithms
    st.subheader("Compare Algorithms")
    st.markdown("""
        Select algorithms to compare their characteristics, complexities, and features.
    """)
    
    comparison_algos = st.multiselect(
        "Select algorithms to compare:",
        available_algorithms,
        default=["BFS", "DFS", "A*"]
    )
    
    # Show algorithm comparison
    display_algorithm_comparison(comparison_algos)

# Footer
st.markdown("---")
st.markdown("""
    **Educational Tool for Algorithm Performance Analysis**
    
    Created for comparing and visualizing search algorithms in various complexity scenarios.
""")
