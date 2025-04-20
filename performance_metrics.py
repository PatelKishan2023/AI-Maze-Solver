def calculate_metrics(result, execution_time):
    """
    Calculate performance metrics for a search algorithm
    
    Parameters:
    - result: Dictionary containing path, visited nodes, etc.
    - execution_time: Time taken by the algorithm in milliseconds
    
    Returns:
    - metrics: Dictionary of performance metrics
    """
    # Ensure execution time is never zero or negative
    if execution_time <= 0:
        execution_time = 0.1  # Minimum 0.1ms to avoid division by zero
    
    metrics = {
        'time': execution_time,  # milliseconds
        'nodes_explored': len(result['visited']) if 'visited' in result else 0,
        'path_length': len(result['path']) if 'path' in result and result['found'] else None,
        'found_solution': result['found'] if 'found' in result else False
    }
    
    # Calculate efficiency score (lower is better)
    if metrics['found_solution'] and metrics['path_length']:
        # Formula: time * nodes_explored / path_length
        # This rewards algorithms that find shorter paths with less exploration time
        metrics['efficiency'] = metrics['time'] * metrics['nodes_explored'] / metrics['path_length']
    else:
        metrics['efficiency'] = float('inf')  # Infinite score for algorithms that fail
    
    return metrics
