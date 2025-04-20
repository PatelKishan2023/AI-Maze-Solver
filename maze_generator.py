import numpy as np
from random import randint, choice, shuffle

def generate_maze(size, complexity):
    """
    Generate a random maze
    
    Parameters:
    - size: Size of the maze (size x size)
    - complexity: Value between 0.1 and 0.9 that determines maze complexity
    
    Returns:
    - maze: 2D numpy array where 1 represents walls and 0 represents paths
    - start_pos: Starting position (row, col)
    - end_pos: Target position (row, col)
    """
    # Initialize maze with walls
    maze = np.ones((size, size), dtype=int)
    
    # Define directions for DFS maze generation
    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]  # right, down, left, up
    
    # Start at a random odd position
    r, c = 1, 1
    maze[r, c] = 0
    stack = [(r, c)]
    
    # DFS to carve passages
    while stack:
        r, c = stack[-1]
        neighbors = []
        
        # Find all neighbors with walls
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size - 1 and 0 <= nc < size - 1 and maze[nr, nc] == 1:
                # Check if the cell two units away is a wall
                neighbors.append((nr, nc, r + dr//2, c + dc//2))
        
        if neighbors:
            # Choose a random neighbor
            nr, nc, wr, wc = choice(neighbors)
            # Remove wall between current cell and chosen neighbor
            maze[wr, wc] = 0
            # Mark the neighbor as visited
            maze[nr, nc] = 0
            # Add neighbor to stack
            stack.append((nr, nc))
        else:
            # Backtrack
            stack.pop()
    
    # Add some additional paths based on complexity
    # Higher complexity means fewer additional paths (more complex maze)
    additional_paths = int((1 - complexity) * (size * size * 0.1))
    
    for _ in range(additional_paths):
        # Find a wall that can be removed without breaking the maze structure
        attempts = 0
        while attempts < 100:  # Limit attempts to avoid infinite loop
            r, c = randint(1, size - 2), randint(1, size - 2)
            if maze[r, c] == 1:
                # Only remove if it doesn't create a 2x2 open area
                if sum([maze[r+1, c], maze[r-1, c], maze[r, c+1], maze[r, c-1]]) >= 3:
                    maze[r, c] = 0
                    break
            attempts += 1
    
    # Set start and end positions at opposite corners
    # Start at top-left region
    start_row, start_col = 0, 0
    while maze[start_row, start_col] == 1:
        if start_row < size // 3 and start_col < size // 3:
            start_row = randint(0, size // 3)
            start_col = randint(0, size // 3)
        else:
            start_row = randint(0, size - 1)
            start_col = randint(0, size - 1)
    
    # End at bottom-right region
    end_row, end_col = size - 1, size - 1
    while maze[end_row, end_col] == 1 or (end_row == start_row and end_col == start_col):
        if end_row > 2 * size // 3 and end_col > 2 * size // 3:
            end_row = randint(2 * size // 3, size - 1)
            end_col = randint(2 * size // 3, size - 1)
        else:
            end_row = randint(0, size - 1)
            end_col = randint(0, size - 1)
    
    # Ensure the maze is solvable by creating a clear path if needed
    if not is_solvable(maze.copy(), (start_row, start_col), (end_row, end_col)):
        # Create a simple path from start to end
        r, c = start_row, start_col
        target_r, target_c = end_row, end_col
        
        # Carve a path in general direction of target
        while r != target_r or c != target_c:
            if r < target_r:
                r += 1
            elif r > target_r:
                r -= 1
            elif c < target_c:
                c += 1
            elif c > target_c:
                c -= 1
            
            if 0 <= r < size and 0 <= c < size:
                maze[r, c] = 0
    
    return maze, (start_row, start_col), (end_row, end_col)

def is_solvable(maze, start, end):
    """
    Check if there is a path from start to end in the maze
    
    Parameters:
    - maze: The maze to check
    - start: Starting position (row, col)
    - end: Target position (row, col)
    
    Returns:
    - True if the maze is solvable, False otherwise
    """
    # BFS to check if there's a path
    queue = [start]
    visited = set([start])
    size = maze.shape[0]
    
    # Define directions: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    while queue:
        r, c = queue.pop(0)
        
        if (r, c) == end:
            return True
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < size and 0 <= nc < size and 
                maze[nr, nc] == 0 and 
                (nr, nc) not in visited):
                queue.append((nr, nc))
                visited.add((nr, nc))
    
    return False
