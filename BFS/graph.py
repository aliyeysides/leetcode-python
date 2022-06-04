from collections import defaultdict


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, vertex, edge):
        self.graph[vertex].append(edge)

    def BFS(self, start) -> list[int]:
        visited, queue, result = [], [], []

        queue.append(start)
        visited.append(start)

        while queue:
            node = queue.pop(0)
            result.append(node)
            for edge in self.graph[node]:
                if edge not in visited:
                    visited.append(edge)
                    queue.append(edge)

        return result


g = Graph()

g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)

print(g.BFS(2))

"""
Time Complexity: O(V+E), where V is the number of nodes and E is the number of edges.
Auxiliary Space: O(V)
"""
