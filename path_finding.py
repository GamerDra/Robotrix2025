import heapq

def dijkstra(graph, start):
    """
    Dijkstra's algorithm to find the shortest path from a start node to all other nodes in a graph.

    :param graph: A dictionary where keys are nodes and values are lists of tuples (neighbor, weight).
    :param start: The starting node.
    :return: A dictionary of shortest distances from the start node to each node.
    """
    # Priority queue to store (distance, node)
    priority_queue = []
    # Dictionary to store the shortest distance to each node
    shortest_distances = {node: float('inf') for node in graph}
    shortest_distances[start] = 0

    # Add the start node to the queue
    heapq.heappush(priority_queue, (0, start))

    while priority_queue:
        # Get the node with the smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)

        # If the current distance is greater than the recorded shortest distance, skip
        if current_distance > shortest_distances[current_node]:
            continue

        # Explore neighbors
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight

            # If a shorter path to the neighbor is found
            if distance < shortest_distances[neighbor]:
                shortest_distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return shortest_distances

# Example usage
if __name__ == "__main__":
    # Define the graph as an adjacency list
    graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('A', 1), ('C', 2), ('D', 6)],
        'C': [('A', 4), ('B', 2), ('D', 3)],
        'D': [('B', 6), ('C', 3)]
    }

    start_node = 'A'
    shortest_paths = dijkstra(graph, start_node)
    
    print(f"Shortest paths from {start_node}: {shortest_paths}")
