class DFSGraph:
    def dfsOfGraph(self, V, adj):
        visit, result = [], []

        def dfs(node, visit, result):
            result.append(node)
            visit.append(node)
            for i in adj[node]:
                if i not in visit:
                    dfs(i, visit, result)

        dfs(0, visit, result)

        return result


dfsG = DFSGraph()
print(dfsG.dfsOfGraph(5, {0: [1, 2, 4], 1: [], 2: [], 3: [], 4: [3]}))
