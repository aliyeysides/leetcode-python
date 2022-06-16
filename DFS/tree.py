# class Node:
#     def __init__(self, val, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# class Tree:
#     def build_tree(self, nodes):
#         def dfs(nodes):
#             val = next(nodes)
#             if val == 'x':
#                 return
#             node = Node(val)
#             node.left = dfs(nodes)
#             node.right = dfs(nodes)

#             return node
#         dfs(iter(nodes.split()))

#     def print_dfs(self, root):
#         if not root:
#             return None

#         l, r = self.print_dfs(root.left), self.print_dfs(root.right)

#         return max(l, r) + 1


# tree = Tree()
# print(tree.build_tree("1 2 3 x x x 4 5 x 6"))
# print(tree.dfs(Node(1)))
