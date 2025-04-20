import numpy as np
import heapq
from collections import deque
import math
import time

# Descriptions for each algorithm
algorithm_descriptions = {
    "Linear Search": """
        Linear Search scans a maze by checking each cell one by one until it finds the target.
        It's the simplest search algorithm but inefficient for large mazes.
        Time Complexity: O(n) where n is the number of cells.
    """,
    
    "Binary Search": """
        Binary Search is typically used for sorted arrays, not mazes. In the context of this visualization,
        we implement a modified version that divides the maze into sections and searches them systematically.
        Note that this is not traditional binary search, as mazes aren't sorted structures.
        Time Complexity: Not applicable in traditional sense for maze solving.
    """,
    
    "Jump Search": """
        Jump Search works by jumping ahead by fixed steps and then backtracking linearly.
        For maze solving, it explores by jumping ahead fixed distances along paths.
        Time Complexity: O(√n) for sorted arrays, but adaptation needed for mazes.
    """,
    
    "Interpolation Search": """
        Interpolation Search uses a position formula based on the estimated position of the target.
        For maze solving, it tries to estimate the direction of the goal based on current position.
        Time Complexity: O(log log n) for uniformly distributed sorted arrays, but adaptation needed for mazes.
    """,
    
    "Exponential Search": """
        Exponential Search involves two steps: finding a range where the element exists, and doing
        a binary search in that range. For mazes, it explores exponentially increasing areas.
        Time Complexity: O(log n) for sorted arrays, but adaptation needed for mazes.
    """,
    
    "Fibonacci Search": """
        Fibonacci Search divides the search space using Fibonacci numbers.
        In maze solving, it prioritizes exploring certain directions based on Fibonacci patterns.
        Time Complexity: O(log n) for sorted arrays, but adaptation needed for mazes.
    """, 
    
    "Ternary Search": """
        Ternary Search divides the search space into three parts instead of two (as in binary search).
        For maze solving, it explores in three potential directions at each step.
        Time Complexity: O(log₃ n) for sorted arrays, but adaptation needed for mazes.
    """,
    
    "Sublist Search": """
        Sublist Search (Rabin-Karp) is typically used for pattern matching in strings.
        For maze solving, it looks for specific patterns of open paths.
        Time Complexity: O(n+m) for string search, but adaptation needed for mazes.
    """,
    
    "Hash Table Lookup": """
        Hash Table Lookup uses a hash function to quickly access items.
        For maze solving, it hashes visited positions to quickly check if a cell has been visited.
        Time Complexity: O(1) average case for lookups.
    """,
    
    "DFS": """
        Depth-First Search (DFS) explores as far as possible along a branch before backtracking.
        It uses a stack (or recursion) to keep track of nodes to visit.
        DFS may not find the shortest path but is memory-efficient.
        Time Complexity: O(V+E) where V is vertices and E is edges.
    """,
    
    "BFS": """
        Breadth-First Search (BFS) explores all neighbors at the present depth before moving to nodes at the next depth.
        It uses a queue to keep track of nodes to visit and always finds the shortest path in unweighted graphs.
        Time Complexity: O(V+E) where V is vertices and E is edges.
    """,
    
    "Uniform Cost Search": """
        Uniform Cost Search expands the node with the lowest path cost.
        It's similar to Dijkstra's algorithm and always finds the optimal solution.
        Time Complexity: O(b^(C/ε)) where b is the branching factor, C is the cost of the optimal solution,
        and ε is the minimum cost between nodes.
    """,
    
    "A*": """
        A* Search uses both the cost to reach the node and a heuristic that estimates the cost to the goal.
        It's optimal if the heuristic is admissible (never overestimates) and is often more efficient than Dijkstra's.
        Time Complexity: O(b^d) in worst case, where b is the branching factor and d is the depth.
    """,
    
    "Best-First Search": """
        Best-First Search is a greedy algorithm that always expands the node closest to the goal according to a heuristic.
        Unlike A*, it doesn't consider the cost to reach the node, only the estimated cost to the goal.
        Time Complexity: O(b^d) in worst case, where b is the branching factor and d is the depth.
    """,
    
    "Iterative Deepening DFS": """
        Iterative Deepening DFS combines the space efficiency of DFS with the completeness of BFS.
        It performs a series of DFS searches with increasing depth limits.
        Time Complexity: O(b^d) where b is the branching factor and d is the depth.
    """,
    
    "Bidirectional Search": """
        Bidirectional Search runs two simultaneous searches: one from the start and one from the goal.
        When the two searches meet, a path is found. This can be much faster than a single search.
        Time Complexity: O(b^(d/2)) where b is the branching factor and d is the depth.
    """,
    
    "Beam Search": """
        Beam Search is a heuristic search that only keeps a limited number of best states at each level (the beam width).
        It's a space-optimized version of best-first search but is not guaranteed to find the optimal solution.
        Time Complexity: O(b*w) where b is the branching factor and w is the beam width.
    """,
    
    "Hill Climbing": """
        Hill Climbing is a local search that always moves in the direction of increasing value.
        It can get stuck in local maxima and doesn't guarantee an optimal solution.
        Time Complexity: Varies based on landscape and implementation.
    """,
    
    "IDA*": """
        Iterative Deepening A* (IDA*) combines iterative deepening with A* search.
        It performs a series of depth-limited searches with increasing f-value cutoffs.
        Time Complexity: O(b^d) in worst case, where b is the branching factor and d is the depth.
    """,
    
    "Dijkstra's Algorithm": """
        Dijkstra's Algorithm finds the shortest path in a weighted graph.
        For maze solving, it treats all moves as having equal cost (similar to BFS for unweighted graphs).
        Time Complexity: O((V+E)log V) using a priority queue.
    """,
    
    "Boggle Search": """
        Boggle Search is used to find words in a grid by connecting adjacent letters.
        For maze solving, it explores all possible paths from the start using a trie or similar structure.
        Time Complexity: O(8^L) where L is the length of the longest word, but adaptation needed for mazes.
    """
}

# Helper functions
def get_neighbors(maze, pos):
    """Get all valid neighboring positions"""
    r, c = pos
    maze_size = maze.shape[0]
    neighbors = []
    
    # Define directions: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if (0 <= nr < maze_size and 0 <= nc < maze_size and maze[nr, nc] == 0):
            neighbors.append((nr, nc))
    
    return neighbors

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def euclidean_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# Search Algorithms
def linear_search(maze, start, end, return_states=False):
    """
    Linear search implementation for maze solving
    
    Parameters:
    - maze: The maze to solve
    - start: Starting position (row, col)
    - end: Target position (row, col)
    - return_states: Whether to return the states of the algorithm for step-by-step visualization
    
    Returns:
    - result: Dictionary containing path, visited nodes, etc.
    - states (optional): List of states for step-by-step visualization
    """
    maze_size = maze.shape[0]
    visited = set([start])  # Start with the start position as visited
    path = {}
    states = [] if return_states else None
    found_path = False
    
    # Store initial state
    if return_states:
        states.append({
            'visited': visited.copy(),
            'path': dict(path),
            'current': start
        })
    
    # Iterate through the maze row by row, column by column
    for r in range(maze_size):
        for c in range(maze_size):
            # Skip walls and already visited positions
            if maze[r, c] == 1 or (r, c) in visited:
                continue
            
            current = (r, c)
            connect_to = None
            
            # Check if this position can be connected to any visited position
            for neighbor in get_neighbors(maze, current):
                if neighbor in visited:
                    connect_to = neighbor
                    break
            
            # If we can connect, add it to our path
            if connect_to:
                visited.add(current)
                path[current] = connect_to
                
                # Store current state if needed
                if return_states:
                    states.append({
                        'visited': visited.copy(),
                        'path': dict(path),
                        'current': current
                    })
                
                # Check if we found the target
                if current == end:
                    found_path = True
                    break
        
        # Exit if we found a path
        if found_path:
            break
    
    # If we found the end position, reconstruct the path
    if end in visited and end in path:
        # Reconstruct path
        final_path = []
        current = end
        while current != start:
            final_path.append(current)
            if current not in path:
                # This shouldn't happen, but just in case
                found_path = False
                break
            current = path[current]
        
        if found_path:
            final_path.append(start)
            final_path.reverse()
            
            result = {
                'path': final_path,
                'visited': visited,
                'found': True
            }
        else:
            result = {
                'path': [],
                'visited': visited,
                'found': False
            }
    else:
        # No path found
        result = {
            'path': [],
            'visited': visited,
            'found': False
        }
    
    if return_states:
        return result, states
    return result

def binary_search(maze, start, end, return_states=False):
    """
    Binary search-inspired implementation for maze solving
    
    Note: True binary search is for sorted arrays, this is an adaptation for mazes
    """
    maze_size = maze.shape[0]
    visited = set([start])
    path = {}
    queue = [start]
    states = [] if return_states else None
    
    # Get middle point of the maze as a reference
    mid_r, mid_c = maze_size // 2, maze_size // 2
    
    # Function to prioritize moves based on binary search intuition
    def prioritize_moves(moves, target):
        # Sort moves based on whether they're moving toward or away from the middle point,
        # depending on where the target is relative to the middle
        if target[0] < mid_r:  # Target is above the middle
            moves.sort(key=lambda pos: pos[0])  # Prefer moving up
        else:  # Target is below the middle
            moves.sort(key=lambda pos: -pos[0])  # Prefer moving down
            
        if target[1] < mid_c:  # Target is to the left of the middle
            moves.sort(key=lambda pos: pos[1])  # Prefer moving left
        else:  # Target is to the right of the middle
            moves.sort(key=lambda pos: -pos[1])  # Prefer moving right
            
        return moves
    
    while queue:
        current = queue.pop(0)
        
        # Store current state if needed
        if return_states:
            states.append({
                'visited': visited.copy(),
                'path': dict(path),
                'current': current
            })
        
        # Check if we found the target
        if current == end:
            # Reconstruct path
            final_path = []
            while current != start:
                final_path.append(current)
                current = path[current]
            final_path.append(start)
            final_path.reverse()
            
            result = {
                'path': final_path,
                'visited': visited,
                'found': True
            }
            
            if return_states:
                return result, states
            return result
        
        # Get neighbors
        neighbors = get_neighbors(maze, current)
        
        # Prioritize moves based on binary search intuition
        neighbors = prioritize_moves(neighbors, end)
        
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                path[neighbor] = current
                queue.append(neighbor)
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def jump_search(maze, start, end, return_states=False):
    """
    Jump search-inspired implementation for maze solving
    
    Note: True jump search is for sorted arrays, this is an adaptation for mazes
    """
    maze_size = maze.shape[0]
    visited = set([start])
    path = {}
    queue = [start]
    states = [] if return_states else None
    
    # Jump size - approximately sqrt(maze_size)
    jump_size = max(1, int(math.sqrt(maze_size)))
    
    while queue:
        current = queue.pop(0)
        
        # Store current state if needed
        if return_states:
            states.append({
                'visited': visited.copy(),
                'path': dict(path),
                'current': current
            })
        
        # Check if we found the target
        if current == end:
            # Reconstruct path
            final_path = []
            while current != start:
                final_path.append(current)
                current = path[current]
            final_path.append(start)
            final_path.reverse()
            
            result = {
                'path': final_path,
                'visited': visited,
                'found': True
            }
            
            if return_states:
                return result, states
            return result
        
        # Get regular neighbors for local exploration
        neighbors = get_neighbors(maze, current)
        
        # Also consider "jump" neighbors
        r, c = current
        jump_directions = [(-jump_size, 0), (0, jump_size), (jump_size, 0), (0, -jump_size)]
        
        for dr, dc in jump_directions:
            jump_r, jump_c = r + dr, c + dc
            
            # Check if jump destination is valid
            if (0 <= jump_r < maze_size and 0 <= jump_c < maze_size and 
                maze[jump_r, jump_c] == 0 and (jump_r, jump_c) not in visited):
                
                # Perform linear search to connect current position to jump position
                can_reach = False
                intermediate_path = []
                
                # Try to find a path to the jump position
                if dr != 0:  # Vertical jump
                    step = 1 if dr > 0 else -1
                    for i in range(r + step, jump_r + step, step):
                        if i < 0 or i >= maze_size or maze[i, c] == 1:
                            break
                        intermediate_path.append((i, c))
                    else:
                        can_reach = True
                else:  # Horizontal jump
                    step = 1 if dc > 0 else -1
                    for j in range(c + step, jump_c + step, step):
                        if j < 0 or j >= maze_size or maze[r, j] == 1:
                            break
                        intermediate_path.append((r, j))
                    else:
                        can_reach = True
                
                # If we can reach the jump position, add it and intermediate positions
                if can_reach:
                    for pos in intermediate_path:
                        if pos not in visited:
                            visited.add(pos)
                            prev = intermediate_path[intermediate_path.index(pos) - 1] if intermediate_path.index(pos) > 0 else current
                            path[pos] = prev
                    
                    # Add jump position
                    if (jump_r, jump_c) not in visited:
                        visited.add((jump_r, jump_c))
                        path[(jump_r, jump_c)] = intermediate_path[-1] if intermediate_path else current
                        queue.append((jump_r, jump_c))
        
        # Add regular neighbors
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                path[neighbor] = current
                queue.append(neighbor)
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def interpolation_search(maze, start, end, return_states=False):
    """
    Interpolation search-inspired implementation for maze solving
    
    Note: True interpolation search is for sorted arrays, this is an adaptation for mazes
    """
    maze_size = maze.shape[0]
    visited = set([start])
    path = {}
    queue = [start]
    states = [] if return_states else None
    
    while queue:
        current = queue.pop(0)
        
        # Store current state if needed
        if return_states:
            states.append({
                'visited': visited.copy(),
                'path': dict(path),
                'current': current
            })
        
        # Check if we found the target
        if current == end:
            # Reconstruct path
            final_path = []
            while current != start:
                final_path.append(current)
                current = path[current]
            final_path.append(start)
            final_path.reverse()
            
            result = {
                'path': final_path,
                'visited': visited,
                'found': True
            }
            
            if return_states:
                return result, states
            return result
        
        # Get neighbors
        neighbors = get_neighbors(maze, current)
        
        # Sort neighbors based on estimated progress toward target
        # (interpolation-inspired heuristic)
        r, c = current
        target_r, target_c = end
        
        # Interpolation formula adaptation for 2D grid
        def estimate_progress(neighbor):
            nr, nc = neighbor
            # Calculate how far along each axis we are between start and end
            # This is an adaptation of the interpolation search formula for a 2D grid
            dr_progress = abs((nr - start[0]) / max(1, abs(target_r - start[0])))
            dc_progress = abs((nc - start[1]) / max(1, abs(target_c - start[1])))
            # Return average progress
            return (dr_progress + dc_progress) / 2
        
        # Sort neighbors by highest estimated progress
        neighbors.sort(key=estimate_progress, reverse=True)
        
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                path[neighbor] = current
                queue.append(neighbor)
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def exponential_search(maze, start, end, return_states=False):
    """
    Exponential search-inspired implementation for maze solving
    
    Note: True exponential search is for sorted arrays, this is an adaptation for mazes
    """
    maze_size = maze.shape[0]
    visited = set([start])
    path = {}
    states = [] if return_states else None
    
    # Current search radius
    radius = 1
    
    # Continue expanding the search radius exponentially
    while radius < maze_size * 2:
        # BFS within current radius
        queue = deque([(start, 0)])  # (position, distance from start)
        local_visited = set([start])
        
        while queue:
            current, distance = queue.popleft()
            
            # Store current state if needed
            if return_states:
                states.append({
                    'visited': visited.copy(),
                    'path': dict(path),
                    'current': current
                })
            
            # Check if we found the target
            if current == end:
                # Reconstruct path
                final_path = []
                curr = current
                while curr != start:
                    final_path.append(curr)
                    curr = path[curr]
                final_path.append(start)
                final_path.reverse()
                
                result = {
                    'path': final_path,
                    'visited': visited,
                    'found': True
                }
                
                if return_states:
                    return result, states
                return result
            
            # Only explore within current radius
            if distance >= radius:
                continue
            
            # Get neighbors
            for neighbor in get_neighbors(maze, current):
                if neighbor not in local_visited:
                    local_visited.add(neighbor)
                    visited.add(neighbor)
                    path[neighbor] = current
                    queue.append((neighbor, distance + 1))
        
        # Double the radius for next iteration (exponential growth)
        radius *= 2
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def fibonacci_search(maze, start, end, return_states=False):
    """
    Fibonacci search-inspired implementation for maze solving
    
    Note: True Fibonacci search is for sorted arrays, this is an adaptation for mazes
    """
    maze_size = maze.shape[0]
    visited = set([start])
    path = {}
    states = [] if return_states else None
    
    # Generate Fibonacci numbers up to maze_size
    fib = [1, 1]
    while fib[-1] + fib[-2] <= maze_size * 2:
        fib.append(fib[-1] + fib[-2])
    
    # Reverse to start with largest Fibonacci number
    fib.reverse()
    
    # Perform BFS with increasing search areas based on Fibonacci numbers
    for radius in fib:
        # BFS within current radius
        queue = deque([(start, 0)])  # (position, distance from start)
        local_visited = set([start])
        
        while queue:
            current, distance = queue.popleft()
            
            # Store current state if needed
            if return_states:
                states.append({
                    'visited': visited.copy(),
                    'path': dict(path),
                    'current': current
                })
            
            # Check if we found the target
            if current == end:
                # Reconstruct path
                final_path = []
                curr = current
                while curr != start:
                    final_path.append(curr)
                    curr = path[curr]
                final_path.append(start)
                final_path.reverse()
                
                result = {
                    'path': final_path,
                    'visited': visited,
                    'found': True
                }
                
                if return_states:
                    return result, states
                return result
            
            # Only explore within current radius
            if distance >= radius:
                continue
            
            # Get neighbors
            for neighbor in get_neighbors(maze, current):
                if neighbor not in local_visited:
                    local_visited.add(neighbor)
                    visited.add(neighbor)
                    path[neighbor] = current
                    queue.append((neighbor, distance + 1))
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def ternary_search(maze, start, end, return_states=False):
    """
    Ternary search-inspired implementation for maze solving
    
    Note: True ternary search is for sorted arrays, this is an adaptation for mazes
    """
    maze_size = maze.shape[0]
    visited = set([start])
    path = {}
    queue = [start]
    states = [] if return_states else None
    
    # Divide maze into thirds
    third1 = maze_size // 3
    third2 = 2 * (maze_size // 3)
    
    # Function to prioritize moves based on ternary search intuition
    def prioritize_moves(moves, target):
        # Calculate which third the target is in
        target_r, target_c = target
        
        # Divide the target location into one of 9 sections (3x3 grid)
        r_section = 0 if target_r < third1 else (1 if target_r < third2 else 2)
        c_section = 0 if target_c < third1 else (1 if target_c < third2 else 2)
        
        # Sort moves to prefer those moving toward the target's section
        def section_distance(pos):
            pos_r, pos_c = pos
            pos_r_section = 0 if pos_r < third1 else (1 if pos_r < third2 else 2)
            pos_c_section = 0 if pos_c < third1 else (1 if pos_c < third2 else 2)
            return abs(r_section - pos_r_section) + abs(c_section - pos_c_section)
        
        moves.sort(key=section_distance)
        return moves
    
    while queue:
        current = queue.pop(0)
        
        # Store current state if needed
        if return_states:
            states.append({
                'visited': visited.copy(),
                'path': dict(path),
                'current': current
            })
        
        # Check if we found the target
        if current == end:
            # Reconstruct path
            final_path = []
            while current != start:
                final_path.append(current)
                current = path[current]
            final_path.append(start)
            final_path.reverse()
            
            result = {
                'path': final_path,
                'visited': visited,
                'found': True
            }
            
            if return_states:
                return result, states
            return result
        
        # Get neighbors
        neighbors = get_neighbors(maze, current)
        
        # Prioritize moves based on ternary search intuition
        neighbors = prioritize_moves(neighbors, end)
        
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                path[neighbor] = current
                queue.append(neighbor)
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def sublist_search(maze, start, end, return_states=False):
    """
    Sublist search (pattern matching) inspired implementation for maze solving
    
    This algorithm tries to identify patterns of open paths in the maze
    """
    maze_size = maze.shape[0]
    visited = set([start])
    path = {}
    queue = [start]
    states = [] if return_states else None
    
    # Find patterns of connected paths in the maze
    pattern_length = 3  # Length of path pattern to look for
    path_patterns = set()
    
    # Extract patterns of open cells
    for r in range(maze_size - pattern_length + 1):
        for c in range(maze_size - pattern_length + 1):
            # Extract a pattern (small subgrid)
            pattern = []
            for dr in range(pattern_length):
                for dc in range(pattern_length):
                    pattern.append((r + dr, c + dc, maze[r + dr, c + dc]))
            
            # Only consider patterns with at least one open cell
            if any(val == 0 for _, _, val in pattern):
                path_patterns.add(tuple(pattern))
    
    # BFS with pattern matching
    while queue:
        current = queue.pop(0)
        
        # Store current state if needed
        if return_states:
            states.append({
                'visited': visited.copy(),
                'path': dict(path),
                'current': current
            })
        
        # Check if we found the target
        if current == end:
            # Reconstruct path
            final_path = []
            while current != start:
                final_path.append(current)
                current = path[current]
            final_path.append(start)
            final_path.reverse()
            
            result = {
                'path': final_path,
                'visited': visited,
                'found': True
            }
            
            if return_states:
                return result, states
            return result
        
        # Get neighbors
        neighbors = get_neighbors(maze, current)
        
        # Try to match current area with known patterns to prioritize moves
        r, c = current
        
        # Prioritize moves that are part of a pattern directed toward the target
        def pattern_score(neighbor):
            nr, nc = neighbor
            score = 0
            
            # Check if this neighbor is part of a pattern that points toward target
            for pattern in path_patterns:
                # Check if current position matches any position in the pattern
                for pr, pc, val in pattern:
                    if val == 0 and (pr, pc) == (r, c):
                        # Found pattern that includes current position
                        # Count open cells in pattern that are closer to target
                        for pr2, pc2, val2 in pattern:
                            if val2 == 0 and manhattan_distance((pr2, pc2), end) < manhattan_distance(current, end):
                                score += 1
            
            return score
        
        # Sort neighbors by pattern score
        neighbors.sort(key=pattern_score, reverse=True)
        
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                path[neighbor] = current
                queue.append(neighbor)
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def hash_table_lookup(maze, start, end, return_states=False):
    """
    Hash table-based search for maze solving
    
    Uses a hash table (Python dictionary) for O(1) lookups of visited cells
    """
    maze_size = maze.shape[0]
    visited = {}  # Use dictionary instead of set for O(1) lookups
    visited[start] = None  # None represents no parent
    queue = [start]
    states = [] if return_states else None
    
    while queue:
        current = queue.pop(0)
        
        # Store current state if needed
        if return_states:
            states.append({
                'visited': set(visited.keys()),
                'path': {pos: parent for pos, parent in visited.items() if parent is not None},
                'current': current
            })
        
        # Check if we found the target
        if current == end:
            # Reconstruct path
            final_path = []
            while current != start:
                final_path.append(current)
                current = visited[current]
            final_path.append(start)
            final_path.reverse()
            
            result = {
                'path': final_path,
                'visited': set(visited.keys()),
                'found': True
            }
            
            if return_states:
                return result, states
            return result
        
        # Get neighbors
        for neighbor in get_neighbors(maze, current):
            # Use dictionary for O(1) lookup
            if neighbor not in visited:
                visited[neighbor] = current
                queue.append(neighbor)
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': set(visited.keys()),
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def dfs(maze, start, end, return_states=False):
    """
    Depth-First Search implementation for maze solving
    """
    maze_size = maze.shape[0]
    visited = set([start])
    path = {}
    stack = [start]
    states = [] if return_states else None
    
    while stack:
        current = stack.pop()
        
        # Store current state if needed
        if return_states:
            states.append({
                'visited': visited.copy(),
                'path': dict(path),
                'current': current
            })
        
        # Check if we found the target
        if current == end:
            # Reconstruct path
            final_path = []
            while current != start:
                final_path.append(current)
                current = path[current]
            final_path.append(start)
            final_path.reverse()
            
            result = {
                'path': final_path,
                'visited': visited,
                'found': True
            }
            
            if return_states:
                return result, states
            return result
        
        # Get neighbors
        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                path[neighbor] = current
                stack.append(neighbor)
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def bfs(maze, start, end, return_states=False):
    """
    Breadth-First Search implementation for maze solving
    """
    maze_size = maze.shape[0]
    visited = set([start])
    path = {}
    queue = deque([start])
    states = [] if return_states else None
    
    while queue:
        current = queue.popleft()
        
        # Store current state if needed
        if return_states:
            states.append({
                'visited': visited.copy(),
                'path': dict(path),
                'current': current
            })
        
        # Check if we found the target
        if current == end:
            # Reconstruct path
            final_path = []
            while current != start:
                final_path.append(current)
                current = path[current]
            final_path.append(start)
            final_path.reverse()
            
            result = {
                'path': final_path,
                'visited': visited,
                'found': True
            }
            
            if return_states:
                return result, states
            return result
        
        # Get neighbors
        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                path[neighbor] = current
                queue.append(neighbor)
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def uniform_cost_search(maze, start, end, return_states=False):
    """
    Uniform Cost Search implementation for maze solving
    
    Note: In a maze with equal edge weights, this is equivalent to BFS
    """
    maze_size = maze.shape[0]
    visited = set()
    path = {}
    # Priority queue with (cost, position) tuples
    pq = [(0, start)]
    cost_so_far = {start: 0}
    states = [] if return_states else None
    
    while pq:
        current_cost, current = heapq.heappop(pq)
        
        # Skip if already visited
        if current in visited:
            continue
        
        visited.add(current)
        
        # Store current state if needed
        if return_states:
            states.append({
                'visited': visited.copy(),
                'path': dict(path),
                'current': current
            })
        
        # Check if we found the target
        if current == end:
            # Reconstruct path
            final_path = []
            while current != start:
                final_path.append(current)
                current = path[current]
            final_path.append(start)
            final_path.reverse()
            
            result = {
                'path': final_path,
                'visited': visited,
                'found': True
            }
            
            if return_states:
                return result, states
            return result
        
        # Get neighbors
        for neighbor in get_neighbors(maze, current):
            # Cost to reach neighbor is current cost + 1 (all moves cost 1 in this maze)
            new_cost = current_cost + 1
            
            # If we haven't visited this neighbor or found a cheaper path
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, neighbor))
                path[neighbor] = current
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def a_star(maze, start, end, return_states=False):
    """
    A* Search implementation for maze solving
    """
    maze_size = maze.shape[0]
    visited = set()
    path = {}
    
    # Priority queue with (f_score, position) tuples
    # f_score = g_score (cost so far) + h_score (heuristic)
    open_set = [(manhattan_distance(start, end), 0, start)]  # (f_score, g_score, position)
    g_score = {start: 0}  # Cost from start to node
    f_score = {start: manhattan_distance(start, end)}  # Estimated total cost
    
    states = [] if return_states else None
    
    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        
        # Skip if already visited
        if current in visited:
            continue
        
        visited.add(current)
        
        # Store current state if needed
        if return_states:
            states.append({
                'visited': visited.copy(),
                'path': dict(path),
                'current': current
            })
        
        # Check if we found the target
        if current == end:
            # Reconstruct path
            final_path = []
            while current != start:
                final_path.append(current)
                current = path[current]
            final_path.append(start)
            final_path.reverse()
            
            result = {
                'path': final_path,
                'visited': visited,
                'found': True
            }
            
            if return_states:
                return result, states
            return result
        
        # Get neighbors
        for neighbor in get_neighbors(maze, current):
            # Tentative g score
            tentative_g = current_g + 1  # All moves cost 1
            
            # If we haven't visited this neighbor or found a cheaper path
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                # Update path
                path[neighbor] = current
                
                # Update scores
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + manhattan_distance(neighbor, end)
                
                # Add to open set
                heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def best_first_search(maze, start, end, return_states=False):
    """
    Best-First Search implementation for maze solving
    """
    maze_size = maze.shape[0]
    visited = set([start])
    path = {}
    
    # Priority queue with (heuristic, position) tuples
    pq = [(manhattan_distance(start, end), start)]
    
    states = [] if return_states else None
    
    while pq:
        _, current = heapq.heappop(pq)
        
        # Store current state if needed
        if return_states:
            states.append({
                'visited': visited.copy(),
                'path': dict(path),
                'current': current
            })
        
        # Check if we found the target
        if current == end:
            # Reconstruct path
            final_path = []
            while current != start:
                final_path.append(current)
                current = path[current]
            final_path.append(start)
            final_path.reverse()
            
            result = {
                'path': final_path,
                'visited': visited,
                'found': True
            }
            
            if return_states:
                return result, states
            return result
        
        # Get neighbors
        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                path[neighbor] = current
                heapq.heappush(pq, (manhattan_distance(neighbor, end), neighbor))
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def iterative_deepening_dfs(maze, start, end, return_states=False):
    """
    Iterative Deepening DFS implementation for maze solving
    """
    maze_size = maze.shape[0]
    max_depth = maze_size * maze_size  # Maximum possible path length
    
    all_visited = set()
    states = [] if return_states else None
    
    # Helper function for depth-limited DFS
    def dls(current, depth, visited, path):
        # Add current node to visited
        visited.add(current)
        all_visited.add(current)
        
        # Store current state if needed
        if return_states:
            states.append({
                'visited': all_visited.copy(),
                'path': dict(path),
                'current': current
            })
        
        # Check if we found the target
        if current == end:
            return True, path
        
        # If reached depth limit
        if depth == 0:
            return False, path
        
        # Get neighbors
        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                path[neighbor] = current
                found, new_path = dls(neighbor, depth - 1, visited.copy(), path.copy())
                if found:
                    return True, new_path
        
        return False, path
    
    # Iterative deepening
    for depth in range(1, max_depth + 1):
        visited = set([start])
        path = {}
        
        found, final_path = dls(start, depth, visited, path)
        
        if found:
            # Reconstruct path
            result_path = []
            current = end
            while current != start:
                result_path.append(current)
                current = final_path[current]
            result_path.append(start)
            result_path.reverse()
            
            result = {
                'path': result_path,
                'visited': all_visited,
                'found': True
            }
            
            if return_states:
                return result, states
            return result
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': all_visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def bidirectional_search(maze, start, end, return_states=False):
    """
    Bidirectional Search implementation for maze solving
    """
    maze_size = maze.shape[0]
    
    # Forward search from start
    forward_visited = {start: None}  # Maps positions to their parent
    forward_queue = deque([start])
    
    # Backward search from end
    backward_visited = {end: None}  # Maps positions to their parent
    backward_queue = deque([end])
    
    intersection = None  # Will store the meeting point
    states = [] if return_states else None
    all_visited = set([start, end])
    
    while forward_queue and backward_queue and not intersection:
        # Forward BFS step
        if forward_queue:
            current = forward_queue.popleft()
            
            # Store current state if needed
            if return_states:
                forward_path = {}
                for pos, parent in forward_visited.items():
                    if parent is not None:
                        forward_path[pos] = parent
                
                states.append({
                    'visited': all_visited.copy(),
                    'path': forward_path,
                    'current': current
                })
            
            # Check for intersection with backward search
            if current in backward_visited:
                intersection = current
                break
            
            # Expand forward search
            for neighbor in get_neighbors(maze, current):
                if neighbor not in forward_visited:
                    forward_visited[neighbor] = current
                    forward_queue.append(neighbor)
                    all_visited.add(neighbor)
        
        # Backward BFS step
        if backward_queue:
            current = backward_queue.popleft()
            
            # Store current state if needed
            if return_states:
                backward_path = {}
                for pos, parent in backward_visited.items():
                    if parent is not None:
                        backward_path[pos] = parent
                
                states.append({
                    'visited': all_visited.copy(),
                    'path': backward_path,
                    'current': current
                })
            
            # Check for intersection with forward search
            if current in forward_visited:
                intersection = current
                break
            
            # Expand backward search
            for neighbor in get_neighbors(maze, current):
                if neighbor not in backward_visited:
                    backward_visited[neighbor] = current
                    backward_queue.append(neighbor)
                    all_visited.add(neighbor)
    
    # If an intersection was found, reconstruct the path
    if intersection:
        # Build path from start to intersection
        path_from_start = []
        current = intersection
        while current != start:
            path_from_start.append(current)
            current = forward_visited[current]
        path_from_start.append(start)
        path_from_start.reverse()
        
        # Build path from intersection to end
        path_to_end = []
        current = intersection
        while current != end:
            current = backward_visited[current]
            path_to_end.append(current)
        
        # Combine the paths
        full_path = path_from_start + path_to_end
        
        # Create a combined path dictionary for visualization
        combined_path = {}
        for i in range(len(full_path) - 1):
            combined_path[full_path[i+1]] = full_path[i]
        
        result = {
            'path': full_path,
            'visited': all_visited,
            'found': True
        }
        
        if return_states:
            return result, states
        return result
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': all_visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def beam_search(maze, start, end, return_states=False, beam_width=3):
    """
    Beam Search implementation for maze solving
    """
    maze_size = maze.shape[0]
    visited = set([start])
    path = {}
    
    # Current beam - use a list for better control
    beam = [start]
    
    states = [] if return_states else None
    
    while beam:
        # Get all neighbors of current beam
        candidates = []
        
        for node in beam:
            # Store current state if needed
            if return_states:
                states.append({
                    'visited': visited.copy(),
                    'path': dict(path),
                    'current': node
                })
            
            # Check if we found the target
            if node == end:
                # Reconstruct path
                final_path = []
                current = node
                while current != start:
                    final_path.append(current)
                    current = path[current]
                final_path.append(start)
                final_path.reverse()
                
                result = {
                    'path': final_path,
                    'visited': visited,
                    'found': True
                }
                
                if return_states:
                    return result, states
                return result
            
            # Get neighbors
            for neighbor in get_neighbors(maze, node):
                if neighbor not in visited:
                    candidates.append((neighbor, manhattan_distance(neighbor, end), node))
        
        if not candidates:
            break
        
        # Sort candidates by heuristic (distance to target)
        candidates.sort(key=lambda x: x[1])
        
        # Take the best beam_width candidates
        new_beam = []
        for i in range(min(beam_width, len(candidates))):
            neighbor, _, parent = candidates[i]
            new_beam.append(neighbor)
            visited.add(neighbor)
            path[neighbor] = parent
        
        beam = new_beam
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def hill_climbing(maze, start, end, return_states=False):
    """
    Hill Climbing implementation for maze solving
    """
    import random  # For random selection of neighbors
    
    maze_size = maze.shape[0]
    visited = set([start])
    path = {}
    current = start
    
    states = [] if return_states else None
    
    # Keep track of last N positions to detect cycles
    recent_positions = []
    recent_positions.append(current)
    
    # Random restarts if stuck
    max_restarts = 5
    restart_count = 0
    
    while current != end and restart_count < max_restarts:
        # Store current state if needed
        if return_states:
            states.append({
                'visited': visited.copy(),
                'path': dict(path),
                'current': current
            })
        
        # Get neighbors
        neighbors = [(neighbor, manhattan_distance(neighbor, end)) 
                    for neighbor in get_neighbors(maze, current)
                    if neighbor not in visited]
        
        if not neighbors:
            # No unvisited neighbors, try a random restart
            restart_count += 1
            unvisited = [(r, c) for r in range(maze_size) for c in range(maze_size)
                         if maze[r, c] == 0 and (r, c) not in visited]
            
            if not unvisited:
                break  # All cells visited, no solution
            
            # Random restart from a random unvisited cell
            current = unvisited[random.randint(0, len(unvisited)-1)]
            recent_positions = []
            recent_positions.append(current)
            continue
        
        # Sort neighbors by distance to end (ascending)
        neighbors.sort(key=lambda x: x[1])
        
        # Get the best neighbor
        best_neighbor, best_distance = neighbors[0]
        
        # Current distance to end
        current_distance = manhattan_distance(current, end)
        
        if best_distance < current_distance:
            # Move to the better neighbor
            path[best_neighbor] = current
            current = best_neighbor
            visited.add(current)
            recent_positions.append(current)
        else:
            # All neighbors are worse, check if we're in a cycle
            if current in recent_positions and recent_positions.count(current) > 1:
                # We're in a cycle, try a random suboptimal move
                if len(neighbors) > 1:
                    # Pick a suboptimal neighbor
                    subopt_idx = random.randint(1, len(neighbors)-1)
                    subopt_neighbor, _ = neighbors[subopt_idx]
                    path[subopt_neighbor] = current
                    current = subopt_neighbor
                    visited.add(current)
                    recent_positions.append(current)
                else:
                    # No suboptimal moves available, try a random restart
                    restart_count += 1
                    unvisited = [(r, c) for r in range(maze_size) for c in range(maze_size)
                                if maze[r, c] == 0 and (r, c) not in visited]
                    
                    if not unvisited:
                        break  # All cells visited, no solution
                    
                    # Random restart from a random unvisited cell
                    current = unvisited[random.randint(0, len(unvisited)-1)]
                    recent_positions = []
                    recent_positions.append(current)
            else:
                # We're not in a cycle yet, just move to the best neighbor
                path[best_neighbor] = current
                current = best_neighbor
                visited.add(current)
                recent_positions.append(current)
    
    # Check if we found the target
    if current == end:
        # Reconstruct path
        final_path = []
        temp_current = current
        while temp_current != start:
            final_path.append(temp_current)
            if temp_current not in path:
                # Handle case where there's no path to start
                result = {
                    'path': [],
                    'visited': visited,
                    'found': False
                }
                if return_states:
                    return result, states
                return result
            temp_current = path[temp_current]
        final_path.append(start)
        final_path.reverse()
        
        result = {
            'path': final_path,
            'visited': visited,
            'found': True
        }
    else:
        result = {
            'path': [],
            'visited': visited,
            'found': False
        }
    
    if return_states:
        return result, states
    return result

def ida_star(maze, start, end, return_states=False):
    """
    Iterative Deepening A* (IDA*) implementation for maze solving
    """
    maze_size = maze.shape[0]
    all_visited = set()
    states = [] if return_states else None
    
    # Initial cost limit
    cost_limit = manhattan_distance(start, end)
    
    # Helper function for IDA* search
    def search(current, g, path, visited, cost_limit):
        f = g + manhattan_distance(current, end)
        
        # If cost exceeds limit
        if f > cost_limit:
            return False, path, f
        
        # Add current node to visited
        visited.add(current)
        all_visited.add(current)
        
        # Store current state if needed
        if return_states:
            states.append({
                'visited': all_visited.copy(),
                'path': dict(path),
                'current': current
            })
        
        # Check if we found the target
        if current == end:
            return True, path, f
        
        min_cost = float('inf')
        
        # Get neighbors
        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                path[neighbor] = current
                found, new_path, cost = search(neighbor, g + 1, path.copy(), visited.copy(), cost_limit)
                if found:
                    return True, new_path, cost
                min_cost = min(min_cost, cost)
        
        return False, path, min_cost
    
    # Iterative deepening
    while cost_limit < maze_size * maze_size:
        visited = set([start])
        path = {}
        
        found, final_path, new_cost = search(start, 0, path, visited, cost_limit)
        
        if found:
            # Reconstruct path
            result_path = []
            current = end
            while current != start:
                result_path.append(current)
                current = final_path[current]
            result_path.append(start)
            result_path.reverse()
            
            result = {
                'path': result_path,
                'visited': all_visited,
                'found': True
            }
            
            if return_states:
                return result, states
            return result
        
        if new_cost == float('inf'):
            break  # No solution
        
        # Update cost limit
        cost_limit = new_cost
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': all_visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def dijkstra(maze, start, end, return_states=False):
    """
    Dijkstra's Algorithm implementation for maze solving
    
    Note: For unweighted mazes, this is equivalent to BFS
    """
    maze_size = maze.shape[0]
    visited = set()
    path = {}
    
    # Priority queue with (cost, position) tuples
    pq = [(0, start)]
    costs = {start: 0}  # Cost from start to each node
    
    states = [] if return_states else None
    
    while pq:
        current_cost, current = heapq.heappop(pq)
        
        # Skip if already visited
        if current in visited:
            continue
        
        visited.add(current)
        
        # Store current state if needed
        if return_states:
            states.append({
                'visited': visited.copy(),
                'path': dict(path),
                'current': current
            })
        
        # Check if we found the target
        if current == end:
            # Reconstruct path
            final_path = []
            while current != start:
                final_path.append(current)
                current = path[current]
            final_path.append(start)
            final_path.reverse()
            
            result = {
                'path': final_path,
                'visited': visited,
                'found': True
            }
            
            if return_states:
                return result, states
            return result
        
        # Get neighbors
        for neighbor in get_neighbors(maze, current):
            # Calculate new cost (all edges have cost 1)
            new_cost = current_cost + 1
            
            # If we haven't visited this neighbor or found a cheaper path
            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                path[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

def boggle_search(maze, start, end, return_states=False):
    """
    Boggle Search-inspired implementation for maze solving
    
    This searches in all 8 directions including diagonals (Boggle style)
    """
    maze_size = maze.shape[0]
    visited = set([start])
    path = {}
    stack = [start]
    states = [] if return_states else None
    
    # Define directions: all 8 directions including diagonals
    directions = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),  # up, up-right, right, down-right
        (1, 0), (1, -1), (0, -1), (-1, -1)  # down, down-left, left, up-left
    ]
    
    # Get neighbors in all 8 directions
    def get_boggle_neighbors(maze, pos):
        r, c = pos
        neighbors = []
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < maze_size and 0 <= nc < maze_size and maze[nr, nc] == 0):
                neighbors.append((nr, nc))
        
        return neighbors
    
    while stack:
        current = stack.pop()
        
        # Store current state if needed
        if return_states:
            states.append({
                'visited': visited.copy(),
                'path': dict(path),
                'current': current
            })
        
        # Check if we found the target
        if current == end:
            # Reconstruct path
            final_path = []
            while current != start:
                final_path.append(current)
                current = path[current]
            final_path.append(start)
            final_path.reverse()
            
            result = {
                'path': final_path,
                'visited': visited,
                'found': True
            }
            
            if return_states:
                return result, states
            return result
        
        # Get neighbors (including diagonals)
        for neighbor in get_boggle_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                path[neighbor] = current
                stack.append(neighbor)
    
    # If we get here, no path was found
    result = {
        'path': [],
        'visited': visited,
        'found': False
    }
    
    if return_states:
        return result, states
    return result

# Function to get algorithm by name
def get_algorithm_by_name(name):
    """
    Get the appropriate algorithm function by name
    """
    algorithms = {
        "Linear Search": linear_search,
        "Binary Search": binary_search,
        "Jump Search": jump_search,
        "Interpolation Search": interpolation_search,
        "Exponential Search": exponential_search,
        "Fibonacci Search": fibonacci_search,
        "Ternary Search": ternary_search,
        "Sublist Search": sublist_search,
        "Hash Table Lookup": hash_table_lookup,
        "DFS": dfs,
        "BFS": bfs,
        "Uniform Cost Search": uniform_cost_search,
        "A*": a_star,
        "Best-First Search": best_first_search,
        "Iterative Deepening DFS": iterative_deepening_dfs,
        "Bidirectional Search": bidirectional_search,
        "Beam Search": beam_search,
        "Hill Climbing": hill_climbing,
        "IDA*": ida_star,
        "Dijkstra's Algorithm": dijkstra,
        "Boggle Search": boggle_search
    }
    
    return algorithms.get(name)
