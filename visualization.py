import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import random

def visualize_maze(maze, start, end, algorithm_data=None):
    """
    Visualize the maze and algorithm paths
    
    Parameters:
    - maze: The maze to visualize
    - start: Starting position (row, col)
    - end: Target position (row, col)
    - algorithm_data: Dictionary mapping algorithm names to their results,
                      or dictionary mapping algorithm names to current states for step-by-step
    
    Returns:
    - fig: Matplotlib figure for display
    """
    maze_size = maze.shape[0]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a colormap for the maze (black for walls, white for paths)
    maze_colors = np.zeros(maze.shape + (3,))
    maze_colors[maze == 1] = [0, 0, 0]  # Black for walls
    maze_colors[maze == 0] = [1, 1, 1]  # White for paths
    
    # Plot the maze
    ax.imshow(maze_colors, interpolation='nearest')
    
    # Mark start and end positions
    ax.plot(start[1], start[0], 'go', markersize=12, label='Start')  # Green circle for start
    ax.plot(end[1], end[0], 'ro', markersize=12, label='End')  # Red circle for end
    
    # Define colors for different algorithms
    algorithm_colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
        '#FF8000', '#8000FF', '#FF0080', '#80FF00', '#0080FF', '#FF8080',
        '#80FF80', '#8080FF', '#FFFF80', '#FF80FF', '#80FFFF', '#FF4000',
        '#4000FF', '#FF0040', '#40FF00', '#00FF40'
    ]
    
    # Create a legend handles list
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Start'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='End')
    ]
    
    # Plot algorithm paths if provided
    if algorithm_data:
        color_idx = 0
        for algo_name, data in algorithm_data.items():
            color = algorithm_colors[color_idx % len(algorithm_colors)]
            color_idx += 1
            
            # Handle step-by-step visualization
            if isinstance(data, dict) and 'current' in data:
                # This is a step visualization state
                visited = data['visited']
                path_dict = data['path']
                current = data['current']
                
                # Plot visited cells with low alpha
                for r, c in visited:
                    if (r, c) != start and (r, c) != end:
                        ax.plot(c, r, 'o', color=color, alpha=0.3, markersize=5)
                
                # Plot current position
                ax.plot(current[1], current[0], 'o', color=color, markersize=8)
                
                # Add algorithm to legend
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=algo_name)
                )
            
            elif isinstance(data, dict) and 'result' in data:
                # This is a final result
                result = data['result']
                if result['found']:
                    # Plot path
                    path = result['path']
                    
                    for i in range(len(path) - 1):
                        r1, c1 = path[i]
                        r2, c2 = path[i + 1]
                        ax.plot([c1, c2], [r1, r2], color=color, linewidth=2)
                    
                    # Add algorithm to legend
                    legend_handles.append(
                        plt.Line2D([0], [0], color=color, linewidth=2, label=f"{algo_name} (steps: {len(path)})")
                    )
                else:
                    # Plot visited cells for unsolved path
                    visited = result['visited']
                    for r, c in visited:
                        if (r, c) != start and (r, c) != end:
                            ax.plot(c, r, 'o', color=color, alpha=0.3, markersize=5)
                    
                    # Add algorithm to legend
                    legend_handles.append(
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f"{algo_name} (no path)")
                    )
            
            else:
                # Plot path if available
                if 'path' in data and data['found']:
                    path = data['path']
                    
                    for i in range(len(path) - 1):
                        r1, c1 = path[i]
                        r2, c2 = path[i + 1]
                        ax.plot([c1, c2], [r1, r2], color=color, linewidth=2)
                    
                    # Add algorithm to legend
                    legend_handles.append(
                        plt.Line2D([0], [0], color=color, linewidth=2, label=f"{algo_name} (steps: {len(path)})")
                    )
                
                # Plot visited cells
                if 'visited' in data:
                    visited = data['visited']
                    for r, c in visited:
                        if (r, c) != start and (r, c) != end and ((r, c) not in data.get('path', [])):
                            ax.plot(c, r, 'o', color=color, alpha=0.3, markersize=5)
    
    # Set grid
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-0.5, maze_size, 1))
    ax.set_yticks(np.arange(-0.5, maze_size, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add legend
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    # Set title
    if algorithm_data:
        ax.set_title(f"Maze ({maze_size}x{maze_size}) with Algorithm Paths", fontsize=14)
    else:
        ax.set_title(f"Maze ({maze_size}x{maze_size})", fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def visualize_performance(performance_data, metric):
    """
    Visualize performance metrics for algorithms
    
    Parameters:
    - performance_data: DataFrame with algorithm performance metrics
    - metric: Metric to visualize (e.g., 'Time (ms)', 'Nodes Explored', 'Path Length')
    
    Returns:
    - fig: Matplotlib figure for display
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Handle non-numeric values in the dataframe
    numeric_df = performance_data.copy()
    
    # Convert string values to appropriate numeric values for visualization
    if metric in ["Path Length", "Efficiency"]:
        # Find numeric values to set a baseline
        numeric_values = []
        for x in numeric_df[metric]:
            try:
                if not isinstance(x, str):
                    numeric_values.append(float(x))
            except (ValueError, TypeError):
                pass
        
        # Set replacement values
        if metric == "Path Length":
            replace_value = 0  # No path = 0 length
        else:  # Efficiency
            if numeric_values:
                replace_value = max(numeric_values) * 1.2  # Worse than the worst
            else:
                replace_value = 100  # Default if no numeric values
        
        # Apply the replacement
        for i in range(len(numeric_df)):
            try:
                if isinstance(numeric_df.at[i, metric], str):
                    numeric_df.at[i, metric] = replace_value
                else:
                    numeric_df.at[i, metric] = float(numeric_df.at[i, metric])
            except (ValueError, TypeError):
                numeric_df.at[i, metric] = replace_value
    
    # Sort data by the selected metric (special handling for Efficiency - lower is better)
    if metric == "Efficiency":
        sorted_data = numeric_df.sort_values(by=metric, ascending=True)
    else:
        sorted_data = numeric_df.sort_values(by=metric, ascending=False)
    
    # Set up colors
    num_algorithms = len(sorted_data)
    colors = plt.cm.viridis(np.linspace(0, 1, num_algorithms))
    
    # Create bar chart
    bars = ax.bar(sorted_data['Algorithm'], numeric_df.loc[sorted_data.index, metric], color=colors)
    
    # Add values on top of bars using original values from performance_data
    for i, bar in enumerate(bars):
        height = bar.get_height()
        original_value = performance_data.iloc[sorted_data.index[i]][metric]
        
        # Use the original value for display
        if isinstance(original_value, str):
            display_value = original_value  # Use string as is (e.g., "No path" or "N/A")
        elif metric == "Time (ms)":
            display_value = f'{original_value:.2f}'  # More precision for time
        elif metric == "Path Length":
            display_value = f'{int(original_value)}'  # Integer for path length
        else:
            display_value = f'{original_value:.1f}'  # Default formatting
            
        ax.text(bar.get_x() + bar.get_width()/2., height,
                display_value,
                ha='center', va='bottom', rotation=0)
    
    # Set title and labels
    ax.set_title(f'Algorithm Performance - {metric}')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel(metric)
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig
