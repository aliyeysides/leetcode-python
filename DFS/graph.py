from collections import defaultdict


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, vertex, edge):
        self.graph[vertex].append(edge)

    def DFSearch(self, node, visited):
        visited.append(node)
        for edge in self.graph[node]:
            if edge not in visited:
                self.DFSearch(edge, visited)

    def DFS(self, node):
        visited = []
        self.DFSearch(node, visited)
        return visited

    def DisconnectedDFS(self) -> list[int]:
        visited = []
        for vertex in self.graph:
            if vertex not in visited:
                self.DFSearch(vertex, visited)
        return visited

    def DFSWhile(self, node):
        visited, stack = [], []
        stack.append(node)
        visited.append(node)

        while stack:
            n = stack[-1]
            if n not in visited:
                visited.append(n)
            pop_stack = True
            for edge in self.graph[n]:
                if edge not in visited:
                    visited.append(edge)
                    stack.append(edge)
                    pop_stack = False
                    break
            if pop_stack:
                stack.pop()
        return visited


g = Graph()
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)

print(g.DFS(2))
print(g.DFSWhile(2))
print(g.DisconnectedDFS())

"""
Time complexity: O(V + E), where V is the number of vertices and E is the number of edges in the graph.
Space Complexity: O(V), since an extra visited array of size V is required.
"""
