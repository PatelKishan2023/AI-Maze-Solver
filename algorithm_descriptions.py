import streamlit as st

# Comprehensive algorithm descriptions
algorithm_detailed_info = {
    "Linear Search": {
        "overview": """
        Linear Search is the simplest search algorithm that works by checking each element in a data structure sequentially until it finds the target or reaches the end.
        
        In the context of maze solving, Linear Search examines cells one by one in a predetermined order until it finds the target. While simple to implement, it's inefficient for large mazes.
        """,
        
        "how_it_works": """
        Linear Search traverses the maze in a systematic, predetermined order, checking each cell one by one:
        
        1. Start at a predefined position (often top-left)
        2. Check if the current cell is the target
        3. If not, move to the next cell in the predetermined sequence
        4. Continue until the target is found or all cells are visited
        
        In our maze implementation, we check cells in row-major order (row by row) and only consider accessible paths.
        """,
        
        "complexity": {
            "time": "O(n) - where n is the total number of cells in the maze",
            "space": "O(n) - to store visited cells and the path"
        },
        
        "pseudocode": """
        function LinearSearch(maze, start, end):
            visited = {start}
            path = {}
            
            for each cell (r, c) in maze (row by row):
                if cell is a wall or already visited:
                    continue
                
                current = (r, c)
                
                # Check if this cell can connect to a visited cell
                for each neighbor of current:
                    if neighbor in visited:
                        visited.add(current)
                        path[current] = neighbor
                        
                        if current == end:
                            return reconstruct_path(path, start, end)
            
            return "No path found"
        """,
        
        "advantages": [
            "Very simple to implement",
            "Works well for small mazes",
            "Low memory overhead",
            "No complex data structures required"
        ],
        
        "disadvantages": [
            "Very inefficient for large mazes",
            "Doesn't use any heuristics or intelligent searching",
            "Will explore almost the entire maze before finding distant targets",
            "Does not guarantee shortest path"
        ],
        
        "applications": [
            "Searching in very small data sets",
            "When simplicity is more important than efficiency",
            "As a baseline comparison for more sophisticated algorithms",
            "Educational purposes to demonstrate basic search concepts"
        ]
    },
    
    "Binary Search": {
        "overview": """
        Binary Search is typically used for sorted arrays, repeatedly dividing the search space in half. In the context of maze solving, it's an adaptation that systematically divides the maze into sections.
        
        Since mazes aren't sorted structures, this is not a traditional binary search but rather an approach inspired by binary search principles.
        """,
        
        "how_it_works": """
        In our maze adaptation:
        
        1. Identify a "middle" reference point in the maze
        2. Prioritize exploration in the direction of the target relative to this point
        3. Recursively explore regions closer to the target first
        
        This approach tries to reduce the search space by making intelligent decisions about which direction to explore based on the target's position.
        """,
        
        "complexity": {
            "time": "O(n) worst case for maze traversal, but can be much better in practice",
            "space": "O(n) to track visited cells and the path"
        },
        
        "pseudocode": """
        function BinarySearchMaze(maze, start, end, mid_point):
            queue = [start]
            visited = {start}
            path = {}
            
            while queue is not empty:
                current = queue.pop(0)
                
                if current == end:
                    return reconstruct_path(path, start, end)
                
                neighbors = get_neighbors(maze, current)
                
                # Sort neighbors based on their position relative to mid_point and end
                # (prioritize exploration toward the target)
                neighbors = prioritize_moves(neighbors, end, mid_point)
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        path[neighbor] = current
                        queue.append(neighbor)
            
            return "No path found"
        """,
        
        "advantages": [
            "More directed than Linear Search",
            "Can find paths more efficiently by prioritizing promising directions",
            "Works well in mazes with clear paths toward the target"
        ],
        
        "disadvantages": [
            "Not a true binary search as mazes aren't sorted",
            "Can still explore many unnecessary cells",
            "Does not guarantee the shortest path",
            "May perform poorly in mazes with many obstacles between start and end"
        ],
        
        "applications": [
            "Maze-solving when the general direction to the target is known",
            "Pathfinding in structured environments",
            "Demonstrating directional search concepts"
        ]
    },
    
    "Jump Search": {
        "overview": """
        Jump Search works by jumping ahead fixed steps and then linear scanning backward. In the context of maze solving, it's adapted to explore by jumping ahead fixed distances along potential paths.
        
        This algorithm bridges the gap between Linear Search and more efficient algorithms like Binary Search.
        """,
        
        "how_it_works": """
        In our maze adaptation:
        
        1. Start at the beginning position
        2. Jump a fixed number of steps (typically sqrt(n) where n is the maze size)
        3. If the jump lands on a valid path, check if it's closer to the target
        4. If promising, explore that area with smaller jumps or linear search
        5. Continue jumping and exploring until the target is found
        
        Jump Search tries to cover large distances quickly while still thoroughly exploring promising areas.
        """,
        
        "complexity": {
            "time": "O(√n) for sorted arrays, O(n) worst case for mazes",
            "space": "O(n) to store visited cells and the path"
        },
        
        "pseudocode": """
        function JumpSearchMaze(maze, start, end):
            visited = {start}
            path = {}
            queue = [start]
            jump_size = int(sqrt(maze_size))
            
            while queue is not empty:
                current = queue.pop(0)
                
                if current == end:
                    return reconstruct_path(path, start, end)
                
                # Try jumping in each direction
                for direction in [(jump_size,0), (0,jump_size), (-jump_size,0), (0,-jump_size)]:
                    jump_r = current[0] + direction[0]
                    jump_c = current[1] + direction[1]
                    
                    # Check if jump is valid
                    if is_valid_position(maze, jump_r, jump_c) and (jump_r, jump_c) not in visited:
                        # Found a valid jump, add to queue
                        visited.add((jump_r, jump_c))
                        path[(jump_r, jump_c)] = current
                        queue.append((jump_r, jump_c))
                
                # Also add immediate neighbors to ensure connectivity
                for neighbor in get_neighbors(maze, current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        path[neighbor] = current
                        queue.append(neighbor)
            
            return "No path found"
        """,
        
        "advantages": [
            "Faster than Linear Search for large mazes",
            "Explores large areas quickly",
            "Simple to implement compared to more complex algorithms",
            "Good balance between exploration speed and thoroughness"
        ],
        
        "disadvantages": [
            "Not guaranteed to find the shortest path",
            "Jump size selection is crucial for performance",
            "Can be inefficient in mazes with many walls or narrow passages",
            "May miss solutions if jump size is too large"
        ],
        
        "applications": [
            "Exploring large open environments",
            "Quick approximate pathfinding",
            "Scenarios where some efficiency is needed but not optimal paths"
        ]
    },
    
    "Interpolation Search": {
        "overview": """
        Interpolation Search estimates where a target might be based on its value and the range of values being searched. In maze solving, it's adapted to estimate the direction of the target based on current position.
        
        This algorithm is particularly effective when the distribution of elements is uniform.
        """,
        
        "how_it_works": """
        In our maze adaptation:
        
        1. Estimate the relative position of the target compared to the current position
        2. Prioritize exploration in directions that appear to lead toward the target
        3. Use this estimation to guide the search more precisely than Binary Search
        
        The algorithm calculates which direction is most likely to lead to the target based on relative positions.
        """,
        
        "complexity": {
            "time": "O(log log n) for uniformly distributed sorted arrays, O(n) worst case for mazes",
            "space": "O(n) to store visited cells and the path"
        },
        
        "pseudocode": """
        function InterpolationSearchMaze(maze, start, end):
            visited = {start}
            path = {}
            queue = [start]
            
            while queue is not empty:
                current = queue.pop(0)
                
                if current == end:
                    return reconstruct_path(path, start, end)
                
                neighbors = get_neighbors(maze, current)
                
                # Calculate position ratio to estimate best direction
                r_ratio = (end[0] - current[0]) / maze_size
                c_ratio = (end[1] - current[1]) / maze_size
                
                # Sort neighbors based on how likely they lead toward the target
                neighbors.sort(key=lambda n: abs((n[0] - current[0])/maze_size - r_ratio) + 
                                          abs((n[1] - current[1])/maze_size - c_ratio))
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        path[neighbor] = current
                        queue.append(neighbor)
            
            return "No path found"
        """,
        
        "advantages": [
            "More efficient than Linear Search and Binary Search in many cases",
            "Adapts well to the specific maze geometry",
            "Makes intelligent decisions about which directions to explore",
            "Can find paths quickly when there are clear routes to the target"
        ],
        
        "disadvantages": [
            "Complex to implement correctly",
            "May perform poorly in mazes with misleading paths",
            "Does not guarantee the shortest path",
            "Estimation can be inaccurate in complex maze structures"
        ],
        
        "applications": [
            "Pathfinding when the distribution of obstacles is known",
            "Navigation in environments with predictable structures",
            "Scenarios where efficient estimation-based search is beneficial"
        ]
    },
    
    "DFS": {
        "overview": """
        Depth-First Search (DFS) is a graph traversal algorithm that explores as far as possible along each branch before backtracking.
        
        DFS starts at the root node and explores each branch completely before moving to the next branch. This approach dives deep into the maze, exploring a single path fully before trying alternatives.
        """,
        
        "how_it_works": """
        DFS explores the maze by:
        
        1. Start at the beginning position
        2. Explore a neighboring unexplored cell
        3. Continue exploring from that cell, going deeper and deeper
        4. Backtrack only when reaching a dead end (no unexplored neighbors)
        5. Continue until finding the target or exhausting all possibilities
        
        This is typically implemented using a stack (or recursion) to keep track of nodes to visit.
        """,
        
        "complexity": {
            "time": "O(V + E) where V is vertices (cells) and E is edges (connections between cells)",
            "space": "O(V) in the worst case for the stack/recursion depth"
        },
        
        "pseudocode": """
        function DFS(maze, start, end):
            stack = [start]
            visited = {start}
            path = {}
            
            while stack is not empty:
                current = stack.pop()  # Take the most recently added cell
                
                if current == end:
                    return reconstruct_path(path, start, end)
                
                for each neighbor of current:
                    if neighbor is not a wall and not visited:
                        visited.add(neighbor)
                        path[neighbor] = current
                        stack.append(neighbor)
            
            return "No path found"
        """,
        
        "advantages": [
            "Memory efficient compared to BFS",
            "Can find a solution without exploring the entire maze",
            "Good for exploring all possible paths",
            "Simple to implement using a stack or recursion",
            "Can work well for mazes with few branches"
        ],
        
        "disadvantages": [
            "Does not guarantee the shortest path",
            "Can get stuck exploring a long path when the solution is nearby",
            "May perform poorly in mazes with many branching paths",
            "Risk of stack overflow with recursion in very large mazes",
            "Less efficient than other algorithms for finding shortest paths"
        ],
        
        "applications": [
            "Topological sorting",
            "Finding connected components",
            "Maze generation algorithms",
            "Path finding in puzzle games",
            "Exploring all possible states in game trees"
        ]
    },
    
    "BFS": {
        "overview": """
        Breadth-First Search (BFS) is a graph traversal algorithm that explores all neighbors at the current depth before moving to the next depth level.
        
        BFS starts at the root node and explores all neighboring nodes before moving to the next level of neighbors. This approach guarantees finding the shortest path in unweighted graphs.
        """,
        
        "how_it_works": """
        BFS explores the maze in concentric waves:
        
        1. Start at the beginning position
        2. Explore all neighboring cells
        3. Then explore all cells two steps away
        4. Continue expanding level by level until finding the target
        
        This is implemented using a queue to keep track of nodes to visit, ensuring that nodes are visited in the order they are discovered.
        """,
        
        "complexity": {
            "time": "O(V + E) where V is vertices (cells) and E is edges (connections between cells)",
            "space": "O(V) for the queue and visited set in the worst case"
        },
        
        "pseudocode": """
        function BFS(maze, start, end):
            queue = [start]
            visited = {start}
            path = {}
            
            while queue is not empty:
                current = queue.pop(0)  # Take the earliest added cell
                
                if current == end:
                    return reconstruct_path(path, start, end)
                
                for each neighbor of current:
                    if neighbor is not a wall and not visited:
                        visited.add(neighbor)
                        path[neighbor] = current
                        queue.append(neighbor)
            
            return "No path found"
        """,
        
        "advantages": [
            "Guarantees the shortest path in unweighted graphs",
            "Explores nodes level by level, which is intuitive",
            "Will find a solution if one exists",
            "Good for finding all nodes at a certain distance",
            "Generally more efficient than DFS for finding shortest paths"
        ],
        
        "disadvantages": [
            "Uses more memory than DFS",
            "May explore many unnecessary nodes before finding the target",
            "Less memory-efficient for very large mazes",
            "Not optimal for weighted graphs (use Dijkstra's or A* instead)",
            "Slower than more informed search algorithms like A*"
        ],
        
        "applications": [
            "Finding shortest paths in unweighted graphs",
            "GPS navigation systems with equal-cost roads",
            "Social network friend/connection analysis",
            "Web crawling",
            "Network broadcasting"
        ]
    },
    
    "A*": {
        "overview": """
        A* (A-star) is an informed search algorithm that combines elements of both Dijkstra's algorithm and greedy best-first search.
        
        A* uses a heuristic function to estimate the cost to reach the goal from each node, allowing it to prioritize paths that appear most promising. This makes it one of the most efficient pathfinding algorithms.
        """,
        
        "how_it_works": """
        A* works by:
        
        1. Start at the beginning position
        2. For each cell, calculate:
           - g(n): The cost to reach this cell from start
           - h(n): The estimated cost to reach the target from this cell (heuristic)
           - f(n): The total estimated cost (g(n) + h(n))
        3. Explore cells in order of lowest f(n) value
        4. Continue until reaching the target
        
        The algorithm uses a priority queue to select the most promising cell to explore next.
        """,
        
        "complexity": {
            "time": "O(b^d) worst case, where b is the branching factor and d is the depth",
            "space": "O(b^d) to store nodes in the priority queue"
        },
        
        "pseudocode": """
        function A_Star(maze, start, end):
            open_set = priority_queue({start: 0})  # Cell: f(n) value
            closed_set = {}
            g_score = {start: 0}  # Cost from start to the node
            f_score = {start: heuristic(start, end)}  # Estimated total cost
            path = {}
            
            while open_set is not empty:
                current = open_set.pop_min()  # Get node with lowest f_score
                
                if current == end:
                    return reconstruct_path(path, start, end)
                
                closed_set.add(current)
                
                for each neighbor of current:
                    if neighbor is a wall or in closed_set:
                        continue
                    
                    tentative_g = g_score[current] + 1  # Cost to neighbor
                    
                    if neighbor not in open_set or tentative_g < g_score[neighbor]:
                        path[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                        open_set.add(neighbor, f_score[neighbor])
            
            return "No path found"
        """,
        
        "advantages": [
            "Combines the benefits of Dijkstra's algorithm and best-first search",
            "Guarantees the shortest path if the heuristic is admissible",
            "Generally explores fewer nodes than BFS or Dijkstra's",
            "Can handle weighted graphs efficiently",
            "Very adaptable through different heuristic functions"
        ],
        
        "disadvantages": [
            "More complex to implement than BFS or DFS",
            "Performance heavily depends on the quality of the heuristic",
            "Can be memory-intensive for large search spaces",
            "May not be optimal if the heuristic overestimates distances",
            "Requires more computation per node than simpler algorithms"
        ],
        
        "applications": [
            "GPS navigation and route planning",
            "Video game pathfinding",
            "Robotics navigation",
            "Network routing algorithms",
            "Puzzle solving (8-puzzle, 15-puzzle, etc.)"
        ]
    },
    
    "Dijkstra's Algorithm": {
        "overview": """
        Dijkstra's algorithm is a graph search algorithm that finds the shortest path between a starting node and all other nodes in a weighted graph.
        
        For maze solving, Dijkstra's algorithm is similar to BFS when all edges have equal weight, but it can handle weighted paths. It always finds the optimal solution by exploring paths in order of their total distance from the start.
        """,
        
        "how_it_works": """
        Dijkstra's algorithm works by:
        
        1. Start at the beginning position
        2. Assign infinite distance to all other nodes and zero to the start
        3. Mark all nodes as unvisited
        4. For the current node, calculate distances to all unvisited neighbors
        5. Update neighbor distances if a shorter path is found
        6. Mark current node as visited and select the unvisited node with smallest distance
        7. Repeat steps 4-6 until the target is reached or all nodes are visited
        
        The algorithm uses a priority queue to efficiently select the next node to visit.
        """,
        
        "complexity": {
            "time": "O((V+E)log V) using a priority queue, where V is vertices and E is edges",
            "space": "O(V) for the priority queue and distance array"
        },
        
        "pseudocode": """
        function Dijkstra(maze, start, end):
            distances = {node: Infinity for all nodes in maze}
            distances[start] = 0
            priority_queue = {start: 0}  # Node: distance
            visited = {}
            path = {}
            
            while priority_queue is not empty:
                current = priority_queue.pop_min()  # Get node with minimum distance
                
                if current == end:
                    return reconstruct_path(path, start, end)
                
                if current in visited:
                    continue
                    
                visited.add(current)
                
                for each neighbor of current:
                    if neighbor is a wall or in visited:
                        continue
                    
                    new_distance = distances[current] + 1  # Weight is 1 for unweighted maze
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        path[neighbor] = current
                        priority_queue.add(neighbor, new_distance)
            
            return "No path found"
        """,
        
        "advantages": [
            "Guarantees the shortest path even in weighted graphs",
            "Works well for finding paths between specific nodes",
            "Can handle different edge weights (for different terrain types)",
            "Optimal for single-source shortest path problems",
            "Relatively simple to implement"
        ],
        
        "disadvantages": [
            "Explores nodes in all directions (no directional guidance)",
            "Less efficient than A* for most pathfinding problems",
            "Requires more processing than BFS for unweighted graphs",
            "Can be slow for very large graphs",
            "Does not use any heuristic to guide the search"
        ],
        
        "applications": [
            "Network routing protocols",
            "Geographic mapping and directions",
            "Transportation planning",
            "Flight scheduling",
            "Circuit design with varying connection costs"
        ]
    },
    
    "Bidirectional Search": {
        "overview": """
        Bidirectional Search is an algorithm that runs two searches simultaneously: one from the start node and one from the goal node. When the two searches meet, a path is found.
        
        This approach can significantly reduce the search space compared to a single-direction search, making it very efficient for finding paths in large mazes.
        """,
        
        "how_it_works": """
        Bidirectional Search works by:
        
        1. Initiate two searches: one from start node, one from goal node
        2. Alternate between the two searches (usually breadth-first)
        3. After each step, check if the two searches have a node in common
        4. If a common node is found, reconstruct the path by combining the two paths
        5. Continue until a path is found or both searches exhaust all possibilities
        
        The algorithm is most efficient when both searches use BFS, guaranteeing the shortest path.
        """,
        
        "complexity": {
            "time": "O(b^(d/2)) where b is branching factor and d is path length",
            "space": "O(b^(d/2)) for storing the two search frontiers"
        },
        
        "pseudocode": """
        function BidirectionalSearch(maze, start, end):
            # Forward search from start
            forward_queue = [start]
            forward_visited = {start: None}  # Node: parent
            
            # Backward search from end
            backward_queue = [end]
            backward_visited = {end: None}  # Node: parent
            
            while forward_queue and backward_queue:
                # Expand forward search
                current = forward_queue.pop(0)
                for each neighbor of current:
                    if neighbor is not a wall and neighbor not in forward_visited:
                        forward_visited[neighbor] = current
                        forward_queue.append(neighbor)
                        
                        # Check if backward search has reached this node
                        if neighbor in backward_visited:
                            return reconstruct_bidirectional_path(
                                forward_visited, backward_visited, neighbor, start, end)
                
                # Expand backward search
                current = backward_queue.pop(0)
                for each neighbor of current:
                    if neighbor is not a wall and neighbor not in backward_visited:
                        backward_visited[neighbor] = current
                        backward_queue.append(neighbor)
                        
                        # Check if forward search has reached this node
                        if neighbor in forward_visited:
                            return reconstruct_bidirectional_path(
                                forward_visited, backward_visited, neighbor, start, end)
            
            return "No path found"
        """,
        
        "advantages": [
            "Much faster than unidirectional search for large mazes",
            "Can reduce time complexity exponentially (b^(d/2) vs b^d)",
            "Guarantees the shortest path when using BFS in both directions",
            "Works well for problems with a known goal state",
            "Particularly effective for long paths"
        ],
        
        "disadvantages": [
            "More complex to implement than single-direction algorithms",
            "Requires careful coordination between the two searches",
            "Uses more memory than depth-first approaches",
            "Intersection detection can add computational overhead",
            "Less effective for problems with many potential goal states"
        ],
        
        "applications": [
            "Finding routes in large transportation networks",
            "Database query optimization",
            "DNA sequence alignment",
            "Robot motion planning in large environments",
            "Finding connections in social networks"
        ]
    }
}

# Function to display detailed algorithm information
def display_algorithm_info(algorithm_name):
    """
    Display detailed information about an algorithm in a structured format
    """
    if algorithm_name not in algorithm_detailed_info:
        st.warning(f"Detailed information for {algorithm_name} is not available.")
        return
    
    info = algorithm_detailed_info[algorithm_name]
    
    # Create expandable sections for different aspects
    with st.expander("Overview", expanded=True):
        st.write(info["overview"])
    
    with st.expander("How It Works"):
        st.write(info["how_it_works"])
    
    # Create columns for complexity information
    with st.expander("Time & Space Complexity"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Time Complexity:**")
            st.write(info["complexity"]["time"])
        with col2:
            st.markdown("**Space Complexity:**")
            st.write(info["complexity"]["space"])
    
    # Show pseudocode in a code block
    with st.expander("Pseudocode"):
        st.code(info["pseudocode"], language="python")
    
    # Advantages and disadvantages in expandable sections
    with st.expander("Advantages & Disadvantages"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Advantages:**")
            for advantage in info["advantages"]:
                st.markdown(f"- {advantage}")
        
        with col2:
            st.markdown("**Disadvantages:**")
            for disadvantage in info["disadvantages"]:
                st.markdown(f"- {disadvantage}")
    
    # Real-world applications
    with st.expander("Real-world Applications"):
        st.markdown("**Applications:**")
        for application in info["applications"]:
            st.markdown(f"- {application}")

# Function to get algorithm details
def get_algorithm_details(algorithm_name):
    """
    Return detailed information for a specific algorithm
    """
    if algorithm_name in algorithm_detailed_info:
        return algorithm_detailed_info[algorithm_name]
    else:
        return None