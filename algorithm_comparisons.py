import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_algorithms(algorithms=None):
    """
    Create a comparison table of selected algorithms
    
    Parameters:
    - algorithms: List of algorithm names to compare (None for all)
    
    Returns:
    - Pandas DataFrame with comparison data
    """
    # Define algorithm characteristics for comparison
    comparison_data = {
        "Algorithm": [],
        "Guarantees Shortest Path": [],
        "Time Complexity": [],
        "Space Complexity": [],
        "Handles Weighted Edges": [],
        "Implementation Difficulty": []
    }
    
    # Get the list of algorithms to compare
    from algorithm_descriptions import algorithm_detailed_info
    
    algo_list = algorithms if algorithms else list(algorithm_detailed_info.keys())
    
    # Algorithm characteristics (predefined based on theory)
    characteristics = {
        "Linear Search": {
            "shortest_path": "No",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "weighted_edges": "No",
            "difficulty": "Easy"
        },
        "Binary Search": {
            "shortest_path": "No",
            "time_complexity": "O(log n)*",
            "space_complexity": "O(n)",
            "weighted_edges": "No",
            "difficulty": "Medium"
        },
        "DFS": {
            "shortest_path": "No",
            "time_complexity": "O(V+E)",
            "space_complexity": "O(V)",
            "weighted_edges": "No",
            "difficulty": "Easy"
        },
        "BFS": {
            "shortest_path": "Yes",
            "time_complexity": "O(V+E)",
            "space_complexity": "O(V)",
            "weighted_edges": "No",
            "difficulty": "Easy"
        },
        "Uniform Cost Search": {
            "shortest_path": "Yes",
            "time_complexity": "O(b^(C/ε))",
            "space_complexity": "O(b^(C/ε))",
            "weighted_edges": "Yes",
            "difficulty": "Medium"
        },
        "A*": {
            "shortest_path": "Yes†",
            "time_complexity": "O(b^d)",
            "space_complexity": "O(b^d)",
            "weighted_edges": "Yes",
            "difficulty": "Hard"
        },
       
       
        "Bidirectional Search": {
            "shortest_path": "Yes‡",
            "time_complexity": "O(b^(d/2))",
            "space_complexity": "O(b^(d/2))",
            "weighted_edges": "Yes‡",
            "difficulty": "Hard"
        },
       
       
       
        "Dijkstra's Algorithm": {
            "shortest_path": "Yes",
            "time_complexity": "O((V+E)log V)",
            "space_complexity": "O(V)",
            "weighted_edges": "Yes",
            "difficulty": "Medium"
        },
    
    }
    
    # Fill in the comparison data
    for algo in algo_list:
        if algo in characteristics:
            comparison_data["Algorithm"].append(algo)
            comparison_data["Guarantees Shortest Path"].append(characteristics[algo]["shortest_path"])
            comparison_data["Time Complexity"].append(characteristics[algo]["time_complexity"])
            comparison_data["Space Complexity"].append(characteristics[algo]["space_complexity"])
            comparison_data["Handles Weighted Edges"].append(characteristics[algo]["weighted_edges"])
            comparison_data["Implementation Difficulty"].append(characteristics[algo]["difficulty"])
    
    # Create a DataFrame
    df = pd.DataFrame(comparison_data)
    
    return df

def generate_complexity_chart(algorithms=None):
    """
    Generate a chart comparing the time complexity of different algorithms
    
    Parameters:
    - algorithms: List of algorithm names to compare (None for all)
    
    Returns:
    - Matplotlib figure
    """
    # Define complexity rankings (lower is better)
    complexity_rank = {
        "O(1)": 1,
        "O(log log n)": 2,
        "O(log n)": 3,
        "O(log₃ n)": 3.5,
        "O(√n)": 4,
        "O(n)": 5,
        "O(n log n)": 6,
        "O(n+m)": 5.5,
        "O(V+E)": 5.5,
        "O((V+E)log V)": 6.5,
        "O(b*w)": 7,
        "O(b^(d/2))": 8,
        "O(b^(C/ε))": 9,
        "O(b^d)": 10,
        "O(8^L)": 10.5,
        "Varies": 7,
    }
    
    # Get comparison data
    df = compare_algorithms(algorithms)
    
    # Extract algorithms and their time complexities
    algos = df["Algorithm"].tolist()
    complexities = df["Time Complexity"].tolist()
    
    # Clean complexity strings
    clean_complexities = []
    for c in complexities:
        # Remove qualifiers like * or †
        c_clean = c.split('*')[0].split('†')[0].split('‡')[0].strip()
        clean_complexities.append(c_clean)
    
    # Rank the algorithms by complexity
    complexity_scores = []
    for c in clean_complexities:
        # Handle special cases
        if "O(1)" in c:
            score = complexity_rank["O(1)"]
        elif c in complexity_rank:
            score = complexity_rank[c]
        else:
            score = 7  # Default rank for unrecognized complexities
        complexity_scores.append(score)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort algorithms by complexity score
    sorted_indices = np.argsort(complexity_scores)
    sorted_algos = [algos[i] for i in sorted_indices]
    sorted_scores = [complexity_scores[i] for i in sorted_indices]
    sorted_complexities = [clean_complexities[i] for i in sorted_indices]
    
    # Create a color map based on complexity (green = good, red = bad)
    colors = plt.cm.RdYlGn_r(np.array(sorted_scores) / max(complexity_scores))
    
    # Plot the bar chart
    bars = ax.barh(sorted_algos, sorted_scores, color=colors)
    
    # Add complexity as text
    for i, (score, complexity) in enumerate(zip(sorted_scores, sorted_complexities)):
        ax.text(score + 0.1, i, complexity, va='center')
    
    ax.set_title('Algorithm Time Complexity Comparison')
    ax.set_xlabel('Relative Efficiency (Lower is Better)')
    
    # Create custom x-tick labels to better explain the complexity scale
    ticks = [0, 2, 4, 6, 8, 10, 12]
    labels = ["", "Constant\nO(1)", "Logarithmic\nO(log n)", "Linear\nO(n)", "Polynomial\nO(n log n)", "Exponential\nO(b^d)", ""]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(0, 12)
    
    # Hide y-axis ticks but keep labels
    ax.tick_params(axis='y', which='both', left=False)
    
    plt.tight_layout()
    
    return fig

def generate_feature_comparison(algorithms=None):
    """
    Generate a feature comparison chart for selected algorithms
    
    Parameters:
    - algorithms: List of algorithm names to compare (None for all)
    
    Returns:
    - Matplotlib figure
    """
    # Define algorithm features with scores (0-5 scale)
    features = {
        "Linear Search": {
            "Speed": 1,
            "Memory Efficiency": 4,
            "Path Quality": 1,
            "Implementation Simplicity": 5,
            "Adaptability": 2
        },
        "Binary Search": {
            "Speed": 3,
            "Memory Efficiency": 4,
            "Path Quality": 2,
            "Implementation Simplicity": 3,
            "Adaptability": 2
        },
        "Jump Search": {
            "Speed": 3,
            "Memory Efficiency": 4,
            "Path Quality": 2,
            "Implementation Simplicity": 3,
            "Adaptability": 2
        },
        "Interpolation Search": {
            "Speed": 4,
            "Memory Efficiency": 3,
            "Path Quality": 2,
            "Implementation Simplicity": 2,
            "Adaptability": 3
        },
        "Exponential Search": {
            "Speed": 3,
            "Memory Efficiency": 4,
            "Path Quality": 2,
            "Implementation Simplicity": 2,
            "Adaptability": 3
        },
        "Fibonacci Search": {
            "Speed": 3,
            "Memory Efficiency": 4,
            "Path Quality": 2,
            "Implementation Simplicity": 2,
            "Adaptability": 2
        },
        "Ternary Search": {
            "Speed": 3,
            "Memory Efficiency": 3,
            "Path Quality": 2,
            "Implementation Simplicity": 3,
            "Adaptability": 2
        },
        "Sublist Search": {
            "Speed": 3,
            "Memory Efficiency": 3,
            "Path Quality": 2,
            "Implementation Simplicity": 2,
            "Adaptability": 3
        },
        "Hash Table Lookup": {
            "Speed": 5,
            "Memory Efficiency": 2,
            "Path Quality": 1,
            "Implementation Simplicity": 4,
            "Adaptability": 4
        },
        "DFS": {
            "Speed": 4,
            "Memory Efficiency": 5,
            "Path Quality": 2,
            "Implementation Simplicity": 4,
            "Adaptability": 3
        },
        "BFS": {
            "Speed": 3,
            "Memory Efficiency": 2,
            "Path Quality": 5,
            "Implementation Simplicity": 4,
            "Adaptability": 3
        },
        "Uniform Cost Search": {
            "Speed": 3,
            "Memory Efficiency": 2,
            "Path Quality": 5,
            "Implementation Simplicity": 3,
            "Adaptability": 4
        },
        "A*": {
            "Speed": 4,
            "Memory Efficiency": 2,
            "Path Quality": 5,
            "Implementation Simplicity": 2,
            "Adaptability": 5
        },
        "Best-First Search": {
            "Speed": 5,
            "Memory Efficiency": 3,
            "Path Quality": 2,
            "Implementation Simplicity": 3,
            "Adaptability": 4
        },
        "Iterative Deepening DFS": {
            "Speed": 2,
            "Memory Efficiency": 5,
            "Path Quality": 5,
            "Implementation Simplicity": 2,
            "Adaptability": 3
        },
        "Bidirectional Search": {
            "Speed": 5,
            "Memory Efficiency": 3,
            "Path Quality": 5,
            "Implementation Simplicity": 2,
            "Adaptability": 3
        },
        "Beam Search": {
            "Speed": 4,
            "Memory Efficiency": 4,
            "Path Quality": 3,
            "Implementation Simplicity": 3,
            "Adaptability": 4
        },
        "Hill Climbing": {
            "Speed": 5,
            "Memory Efficiency": 5,
            "Path Quality": 1,
            "Implementation Simplicity": 5,
            "Adaptability": 2
        },
        "IDA*": {
            "Speed": 3,
            "Memory Efficiency": 5,
            "Path Quality": 5,
            "Implementation Simplicity": 2,
            "Adaptability": 4
        },
        "Dijkstra's Algorithm": {
            "Speed": 3,
            "Memory Efficiency": 2,
            "Path Quality": 5,
            "Implementation Simplicity": 3,
            "Adaptability": 4
        },
        "Boggle Search": {
            "Speed": 2,
            "Memory Efficiency": 3,
            "Path Quality": 4,
            "Implementation Simplicity": 3,
            "Adaptability": 4
        }
    }
    
    # Get the list of algorithms to compare
    if algorithms is None or len(algorithms) == 0:
        # Default to a reasonable subset for comparison
        algorithms = ["BFS", "DFS", "A*", "Dijkstra's Algorithm", "Bidirectional Search"]
    elif len(algorithms) > 6:
        # Limit to avoid crowded chart
        algorithms = algorithms[:6]
    
    # Extract feature scores for selected algorithms
    feature_names = ["Speed", "Memory Efficiency", "Path Quality", "Implementation Simplicity", "Adaptability"]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get feature scores for each algorithm
    width = 0.15  # width of bars
    x = np.arange(len(feature_names))  # feature positions
    
    # Create bars for each algorithm
    for i, algo in enumerate(algorithms):
        if algo in features:
            scores = [features[algo][feature] for feature in feature_names]
            offset = width * (i - len(algorithms)/2 + 0.5)
            ax.bar(x + offset, scores, width, label=algo)
    
    # Set chart properties
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names)
    ax.set_yticks(range(6))
    ax.set_yticklabels(['', 'Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    ax.set_ylim(0, 5.5)
    ax.set_title('Algorithm Feature Comparison')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.tight_layout()
    
    return fig

def display_algorithm_comparison(algorithms=None):
    """
    Display algorithm comparison information
    
    Parameters:
    - algorithms: List of algorithm names to compare (None for all)
    """
    st.header("Algorithm Comparison")
    
    # Create tabs for different comparison views
    tab1, tab2, tab3 = st.tabs(["Comparison Table", "Time Complexity", "Feature Comparison"])
    
    with tab1:
        # Display the comparison table
        df = compare_algorithms(algorithms)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("""
        **Notes:**
        * † Guarantees shortest path only if the heuristic is admissible (never overestimates)
        * ‡ Guarantees shortest path if implemented with BFS in both directions
        * \* Time complexity shown is for the original algorithm in sorted arrays, maze adaptations will differ
        
        **Time Complexity Notation:**
        * **V**: Number of vertices (cells in the maze)
        * **E**: Number of edges (connections between cells)
        * **b**: Branching factor (average number of neighbors per cell)
        * **d**: Depth of the solution (path length from start to goal)
        """)
    
    with tab2:
        # Display time complexity chart
        st.subheader("Time Complexity Comparison")
        
        fig = generate_complexity_chart(algorithms)
        st.pyplot(fig)
        
        st.markdown("""
        **Understanding Time Complexities:**
        * **V**: Number of vertices (cells in the maze)
        * **E**: Number of edges (connections between cells)
        * **b**: Branching factor (average number of neighbors per cell)
        * **d**: Depth of the solution (path length from start to goal)
        * **w**: Beam width (for Beam Search)
        * **n**: Number of elements in the search space (cells in the maze)
        * **ε**: Minimum edge cost
        * **C**: Cost of the optimal solution
        """)
    
    with tab3:
        # Display feature comparison chart
        st.subheader("Feature Comparison")
        
        fig = generate_feature_comparison(algorithms)
        st.pyplot(fig)
        
        st.markdown("""
        **Feature Descriptions:** (Scale: 1-5, where 5 is best)
        * **Speed**: How fast the algorithm runs in average cases, based on time complexity
        * **Memory Efficiency**: How much memory the algorithm needs, based on space complexity
        * **Path Quality**: How optimal the found path is (shortest path = 5)
        * **Implementation Simplicity**: How easy it is to code the algorithm (lower cognitive complexity = higher score)
        * **Adaptability**: How well the algorithm adapts to different maze types and heuristics
        """)
    
    # Additional explanation of metrics
    st.markdown("""
    ### Understanding Performance Metrics
    
    When comparing algorithms, it's important to consider multiple factors:
    
    1. **Time Complexity**: The theoretical growth rate of the algorithm's running time relative to input size
    2. **Space Complexity**: The growth rate of memory used by the algorithm
    3. **Completeness**: Whether the algorithm is guaranteed to find a solution if one exists
    4. **Optimality**: Whether the algorithm always finds the shortest/optimal path
    5. **Practical Performance**: How well the algorithm performs in real-world scenarios
    
    Different algorithms make different trade-offs. For example, DFS uses less memory than BFS but doesn't guarantee the shortest path. A* is often faster than Dijkstra's Algorithm but requires a good heuristic function.
    """)
