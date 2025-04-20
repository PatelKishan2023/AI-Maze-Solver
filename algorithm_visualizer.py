import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def display_algorithm_step(algorithm_name, step_index, use_generated_maze=False):
    """
    Display a visualization of an algorithm step
    
    Parameters:
    - algorithm_name: Name of the algorithm
    - step_index: The step number to display
    - use_generated_maze: Whether to use the maze from session state
    
    Returns:
    - Matplotlib figure
    """
    # Predefined exploration patterns for different algorithm types
    patterns = {
        # Breadth-First Search - expands in level order (like a wave)
        "BFS": [
            {"visited": {(1, 1)}, "current": (1, 1)},
            {"visited": {(1, 1), (1, 2), (2, 1)}, "current": (1, 2)},
            {"visited": {(1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1)}, "current": (2, 1)},
            {"visited": {(1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1), (2, 3), (3, 2), (4, 1)}, "current": (1, 3)},
            {"visited": {(1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1), (2, 3), (3, 2), (4, 1), (1, 4), (3, 3)}, "current": (2, 2)},
            {"visited": {(1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1), (2, 3), (3, 2), (4, 1), (1, 4), (3, 3), (4, 2), (5, 1)}, "current": (3, 1)},
        ],
        
        # Depth-First Search - goes as far as possible along one path before backtracking
        "DFS": [
            {"visited": {(1, 1)}, "current": (1, 1)},
            {"visited": {(1, 1), (1, 2)}, "current": (1, 2)},
            {"visited": {(1, 1), (1, 2), (1, 3)}, "current": (1, 3)},
            {"visited": {(1, 1), (1, 2), (1, 3), (1, 4)}, "current": (1, 4)},
            {"visited": {(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)}, "current": (1, 5)},
            {"visited": {(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 5)}, "current": (2, 5)},
        ],
        
        # A* Search - tends toward the goal using a heuristic
        "A*": [
            {"visited": {(1, 1)}, "current": (1, 1)},
            {"visited": {(1, 1), (2, 2)}, "current": (2, 2)},
            {"visited": {(1, 1), (2, 2), (3, 3)}, "current": (3, 3)},
            {"visited": {(1, 1), (2, 2), (3, 3), (4, 4)}, "current": (4, 4)},
            {"visited": {(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)}, "current": (5, 5)},
            {"visited": {(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)}, "current": (6, 6)},
        ],
        
        # Dijkstra's expands outward by cost (similar to BFS in unweighted mazes)
        "Dijkstra's Algorithm": [
            {"visited": {(1, 1)}, "current": (1, 1)},
            {"visited": {(1, 1), (1, 2), (2, 1)}, "current": (1, 2)},
            {"visited": {(1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1)}, "current": (2, 1)},
            {"visited": {(1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1), (1, 4), (2, 3), (3, 2), (4, 1)}, "current": (2, 2)},
            {"visited": {(1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1), (1, 4), (2, 3), (3, 2), (4, 1), (2, 4), (3, 3), (4, 2), (5, 1)}, "current": (3, 1)},
            {"visited": {(1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1), (1, 4), (2, 3), (3, 2), (4, 1), (2, 4), (3, 3), (4, 2), (5, 1), (1, 5), (3, 4), (4, 3)}, "current": (1, 3)},
        ],
        
        # Bidirectional Search - searches from both start and goal
        "Bidirectional Search": [
            {"visited": {(1, 1), (8, 8)}, "current": (1, 1)},
            {"visited": {(1, 1), (8, 8), (1, 2), (2, 1), (7, 8), (8, 7)}, "current": (8, 8)},
            {"visited": {(1, 1), (8, 8), (1, 2), (2, 1), (7, 8), (8, 7), (1, 3), (2, 2), (7, 7), (6, 8)}, "current": (1, 2)},
            {"visited": {(1, 1), (8, 8), (1, 2), (2, 1), (7, 8), (8, 7), (1, 3), (2, 2), (7, 7), (6, 8), (3, 1), (2, 3), (6, 7)}, "current": (7, 8)},
            {"visited": {(1, 1), (8, 8), (1, 2), (2, 1), (7, 8), (8, 7), (1, 3), (2, 2), (7, 7), (6, 8), (3, 1), (2, 3), (6, 7), (5, 5), (5, 6)}, "current": (5, 5)},
            {"visited": {(1, 1), (8, 8), (1, 2), (2, 1), (7, 8), (8, 7), (1, 3), (2, 2), (7, 7), (6, 8), (3, 1), (2, 3), (6, 7), (5, 5), (5, 6), (4, 5), (5, 4)}, "current": (5, 5)},
        ],
        
        # Linear Search - scans row by row
        "Linear Search": [
            {"visited": {(1, 1)}, "current": (1, 1)},
            {"visited": {(1, 1), (1, 2), (1, 3), (1, 4)}, "current": (1, 4)},
            {"visited": {(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8)}, "current": (1, 8)},
            {"visited": {(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 1), (2, 2), (2, 3)}, "current": (2, 3)},
            {"visited": {(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6)}, "current": (2, 6)},
            {"visited": {(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (3, 1)}, "current": (3, 1)},
        ],
        
        # Binary Search - divides the space in half (adapted for 2D maze)
        "Binary Search": [
            {"visited": {(4, 4)}, "current": (4, 4)},
            {"visited": {(4, 4), (2, 2)}, "current": (2, 2)},
            {"visited": {(4, 4), (2, 2), (1, 1)}, "current": (1, 1)},
            {"visited": {(4, 4), (2, 2), (1, 1), (3, 3)}, "current": (3, 3)},
            {"visited": {(4, 4), (2, 2), (1, 1), (3, 3), (6, 6)}, "current": (6, 6)},
            {"visited": {(4, 4), (2, 2), (1, 1), (3, 3), (6, 6), (8, 8)}, "current": (8, 8)},
        ],
        
        # Jump Search - jumps ahead by fixed steps
        "Jump Search": [
            {"visited": {(1, 1)}, "current": (1, 1)},
            {"visited": {(1, 1), (3, 3)}, "current": (3, 3)},
            {"visited": {(1, 1), (3, 3), (5, 5)}, "current": (5, 5)},
            {"visited": {(1, 1), (3, 3), (5, 5), (7, 7)}, "current": (7, 7)},
            {"visited": {(1, 1), (3, 3), (5, 5), (7, 7), (6, 6)}, "current": (6, 6)},
            {"visited": {(1, 1), (3, 3), (5, 5), (7, 7), (6, 6), (6, 7), (7, 6)}, "current": (6, 7)},
        ],
        
        # Best-First Search - uses heuristic to always visit most promising node
        "Best-First Search": [
            {"visited": {(1, 1)}, "current": (1, 1)},
            {"visited": {(1, 1), (2, 2)}, "current": (2, 2)},
            {"visited": {(1, 1), (2, 2), (3, 3)}, "current": (3, 3)},
            {"visited": {(1, 1), (2, 2), (3, 3), (4, 4)}, "current": (4, 4)},
            {"visited": {(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 5)}, "current": (5, 5)},
            {"visited": {(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 5), (7, 6), (8, 7)}, "current": (8, 7)},
        ],
        
        # Beam Search - keeps only the best k paths
        "Beam Search": [
            {"visited": {(1, 1)}, "current": (1, 1)},
            {"visited": {(1, 1), (1, 2), (2, 1), (2, 2)}, "current": (2, 2)},
            {"visited": {(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (3, 2)}, "current": (3, 3)},
            {"visited": {(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (3, 2), (4, 4), (4, 3)}, "current": (4, 4)},
            {"visited": {(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5), (5, 4)}, "current": (5, 5)},
            {"visited": {(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5), (5, 4), (6, 6), (6, 5)}, "current": (6, 6)},
        ],
        
        # Hill Climbing - always moves toward better state
        "Hill Climbing": [
            {"visited": {(1, 1)}, "current": (1, 1)},
            {"visited": {(1, 1), (2, 2)}, "current": (2, 2)},
            {"visited": {(1, 1), (2, 2), (3, 3)}, "current": (3, 3)},
            {"visited": {(1, 1), (2, 2), (3, 3), (4, 4)}, "current": (4, 4)},
            {"visited": {(1, 1), (2, 2), (3, 3), (4, 4), (4, 5), (5, 5)}, "current": (5, 5)},
            {"visited": {(1, 1), (2, 2), (3, 3), (4, 4), (4, 5), (5, 5), (6, 6), (6, 7)}, "current": (6, 7)},
        ],
        
        # IDA* - iterative deepening A*
        "IDA*": [
            {"visited": {(1, 1)}, "current": (1, 1)},
            {"visited": {(1, 1), (1, 2), (2, 1), (2, 2)}, "current": (2, 2)},
            {"visited": {(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (2, 3), (3, 3)}, "current": (3, 3)},
            {"visited": {(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (2, 3), (3, 3), (4, 3), (3, 4), (4, 4)}, "current": (4, 4)},
            {"visited": {(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (2, 3), (3, 3), (4, 3), (3, 4), (4, 4), (5, 4), (4, 5), (5, 5)}, "current": (5, 5)},
            {"visited": {(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (2, 3), (3, 3), (4, 3), (3, 4), (4, 4), (5, 4), (4, 5), (5, 5), (6, 5), (5, 6), (6, 6)}, "current": (6, 6)},
        ],
    }
    
    # Get the appropriate pattern for the algorithm
    if algorithm_name in patterns:
        # Direct match
        steps = patterns[algorithm_name]
    elif "Linear" in algorithm_name:
        steps = patterns["Linear Search"]
    elif "Binary" in algorithm_name:
        steps = patterns["Binary Search"]
    elif "Jump" in algorithm_name:
        steps = patterns["Jump Search"]
    elif "Depth-First" in algorithm_name or "DFS" in algorithm_name:
        steps = patterns["DFS"]
    elif "Breadth-First" in algorithm_name or "BFS" in algorithm_name:
        steps = patterns["BFS"]
    elif "A*" in algorithm_name or "A-Star" in algorithm_name:
        steps = patterns["A*"]
    elif "IDA*" in algorithm_name or "Iterative Deepening" in algorithm_name:
        steps = patterns["IDA*"]
    elif "Best-First" in algorithm_name:
        steps = patterns["Best-First Search"]
    elif "Bidirectional" in algorithm_name:
        steps = patterns["Bidirectional Search"]
    elif "Dijkstra" in algorithm_name or "Uniform Cost" in algorithm_name:
        steps = patterns["Dijkstra's Algorithm"]
    elif "Beam" in algorithm_name:
        steps = patterns["Beam Search"]
    elif "Hill Climbing" in algorithm_name:
        steps = patterns["Hill Climbing"]
    elif "Boggle" in algorithm_name:
        # For Boggle use A* pattern (diagonal movements)
        steps = patterns["A*"]
    elif "Hash" in algorithm_name:
        # For Hash Table Lookup use a more direct pattern
        steps = patterns["Jump Search"]
    elif "Exponential" in algorithm_name:
        # Exponential search jumps with increasing step sizes
        steps = patterns["Jump Search"]
    elif "Fibonacci" in algorithm_name:
        # Fibonacci search has a pattern similar to a smarter binary search
        steps = patterns["Binary Search"]
    elif "Interpolation" in algorithm_name:
        # Interpolation makes educated guesses about positions
        steps = patterns["A*"]
    elif "Sublist" in algorithm_name or "Pattern" in algorithm_name:
        # Sublist/pattern search looks for patterns in the maze
        steps = patterns["Linear Search"]
    elif "Ternary" in algorithm_name:
        # Ternary search divides the space into three parts
        steps = patterns["Binary Search"]
    else:
        # Default to BFS for any other algorithm
        steps = patterns["BFS"]
        
    max_steps = len(steps)
    step_index = min(step_index, max_steps - 1)
    
    # Check if we should use the generated maze
    if use_generated_maze and 'maze' in st.session_state and st.session_state.maze is not None:
        # Use the maze from session state
        maze = st.session_state.maze.copy()
        start = st.session_state.start_pos
        end = st.session_state.end_pos
        maze_size = maze.shape[0]
    else:
        # Create a sample maze for demonstration
        maze_size = 10
        maze = np.ones((maze_size, maze_size))
        
        # Create a simple maze pattern
        for i in range(1, maze_size - 1):
            for j in range(1, maze_size - 1):
                if (i % 2 == 1 and j % 2 == 1) or (i % 2 == 0 and j % 2 == 0 and i > 2 and j > 2):
                    maze[i, j] = 0
        
        # Define start and end points
        start = (1, 1)
        end = (maze_size - 2, maze_size - 2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create a colormap for the maze
    maze_colors = np.zeros(maze.shape + (3,))
    maze_colors[maze == 1] = [0, 0, 0]  # Black for walls
    maze_colors[maze == 0] = [1, 1, 1]  # White for paths
    
    # Plot the maze
    ax.imshow(maze_colors, interpolation='nearest')
    
    # Mark start and end positions
    ax.plot(start[1], start[0], 'go', markersize=12, label="Start")
    ax.plot(end[1], end[0], 'ro', markersize=12, label="End")
    
    # Get the current state
    current_state = steps[step_index]
    visited = list(current_state['visited'])
    current = current_state['current']
    
    # If we're using a real maze, we need to adapt the demo coordinates to our maze
    if use_generated_maze:
        # Scale the demo coordinates to fit our maze
        scale_factor_r = maze.shape[0] / 10
        scale_factor_c = maze.shape[1] / 10
        
        # Scale visited positions
        scaled_visited = []
        for r, c in visited:
            new_r = min(int(r * scale_factor_r), maze.shape[0] - 1)
            new_c = min(int(c * scale_factor_c), maze.shape[1] - 1)
            # Only add if it's a valid path
            if maze[new_r, new_c] == 0:
                scaled_visited.append((new_r, new_c))
        visited = scaled_visited
        
        # Scale current position
        r, c = current
        current = (min(int(r * scale_factor_r), maze.shape[0] - 1), 
                  min(int(c * scale_factor_c), maze.shape[1] - 1))
        # If current position is a wall, find closest path
        if maze[current[0], current[1]] == 1:
            # Find closest path cell
            min_dist = float('inf')
            for i in range(maze.shape[0]):
                for j in range(maze.shape[1]):
                    if maze[i, j] == 0:
                        dist = abs(i - current[0]) + abs(j - current[1])
                        if dist < min_dist:
                            min_dist = dist
                            current = (i, j)
    
    # Plot visited cells
    for r, c in visited:
        if (r, c) != start and (r, c) != end and 0 <= r < maze_size and 0 <= c < maze_size:
            # Make sure we're not plotting walls
            if maze[r, c] == 0:
                ax.plot(c, r, 'o', color='blue', alpha=0.3, markersize=8)
    
    # Plot current position (only if it's a valid path)
    if 0 <= current[0] < maze_size and 0 <= current[1] < maze_size and maze[current[0], current[1]] == 0:
        ax.plot(current[1], current[0], 'o', color='blue', markersize=10, label="Current")
    
    # Set grid
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-0.5, maze_size, 1))
    ax.set_yticks(np.arange(-0.5, maze_size, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Set title with step count
    ax.set_title(f"{algorithm_name} - Step {step_index+1}/{len(steps)}")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    return fig

def display_algorithm_visualization(algorithm_name, use_generated_maze=False):
    """
    Display a visualization of how the algorithm works
    
    Parameters:
    - algorithm_name: Name of the algorithm
    - use_generated_maze: Whether to use the generated maze from session state
    """
    st.subheader("Algorithm Visualization")
    
    # All patterns have 6 steps
    max_steps = 6
    
    # Create placeholder for visualization
    placeholder = st.empty()
    
    # Session state for current step
    if 'visualizer_step' not in st.session_state:
        st.session_state.visualizer_step = 0
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶️ Play"):
            # Increment through all steps
            for i in range(st.session_state.visualizer_step, max_steps):
                st.session_state.visualizer_step = i
                fig = display_algorithm_step(algorithm_name, i, use_generated_maze)
                placeholder.pyplot(fig)
                time.sleep(0.5)  # Pause between frames
    
    with col2:
        if st.button("⏭️ Next Step"):
            if st.session_state.visualizer_step < max_steps - 1:
                st.session_state.visualizer_step += 1
    
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.visualizer_step = 0
    
    # Slider for step selection
    step = st.slider("Step", 1, max_steps, st.session_state.visualizer_step + 1)
    st.session_state.visualizer_step = step - 1
    
    # Display the visualization
    fig = display_algorithm_step(algorithm_name, st.session_state.visualizer_step, use_generated_maze)
    placeholder.pyplot(fig)