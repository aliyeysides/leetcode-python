from collections import defaultdict


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, vertex, edge):
        self.graph[vertex].append(edge)

    def BFS(self, s) -> list[int]:
        visited = []
        queue = []
        res = []

        queue.append(s)
        visited.append(s)

        while queue:
            n = queue.pop(0)
            res.append(n)
            for edge in self.graph[n]:
                if edge not in visited:
                    visited.append(edge)
                    queue.append(edge)

        return res


g = Graph()

g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)

print(g.BFS(2))
