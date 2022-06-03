from collections import defaultdict


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, vertex, edge):
        self.graph[vertex].append(edge)

    def DFSearch(self, node, visited, result):
        result.append(node)
        visited.append(node)
        for edge in self.graph[node]:
            if edge not in visited:
                self.DFSearch(edge, visited, result)

    def DFS(self, node) -> list[int]:
        visited, result = [], []
        self.DFSearch(node, visited, result)
        return result


g = Graph()
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)

print(g.DFS(2))

"""
Time complexity: O(V + E), where V is the number of vertices and E is the number of edges in the graph.
Space Complexity: O(V), since an extra visited array of size V is required.
"""
