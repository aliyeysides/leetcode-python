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
g.addEdge(0, 4)
g.addEdge(4, 3)

print(g.DFS(0))
