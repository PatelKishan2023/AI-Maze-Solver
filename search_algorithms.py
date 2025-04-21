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
    
    "A*": """
        A* Search uses both the cost to reach the node and a heuristic that estimates the cost to the goal.
        It's optimal if the heuristic is admissible (never overestimates) and is often more efficient than Dijkstra's.
        Time Complexity: O(b^d) in worst case, where b is the branching factor and d is the depth.
    """,

    
    "Bidirectional Search": """
        Bidirectional Search runs two simultaneous searches: one from the start and one from the goal.
        When the two searches meet, a path is found. This can be much faster than a single search.
        Time Complexity: O(b^(d/2)) where b is the branching factor and d is the depth.
    """,
    
    "Dijkstra's Algorithm": """
        Dijkstra's Algorithm finds the shortest path in a weighted graph.
        For maze solving, it treats all moves as having equal cost (similar to BFS for unweighted graphs).
        Time Complexity: O((V+E)log V) using a priority queue.
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
        "DFS": dfs,
        "BFS": bfs,
        "A*": a_star,
        "Best-First Search": best_first_search,
        "Iterative Deepening DFS": iterative_deepening_dfs,
        "Bidirectional Search": bidirectional_search,
        "Dijkstra's Algorithm": dijkstra,
    }
    
    return algorithms.get(name)
