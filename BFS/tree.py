class Node:
    def __init__(self, v):
        self.val = v
        self.left = None
        self.right = None

    def levelOrder(self, root):
        if root is Node:
            return

        res = []
        queue = []
        queue.append(root)

        while queue:
            node = queue.pop(0)
            res.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return res


root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)

print(root.levelOrder(root))

"""
Time Complexity: O(n) where n is the number of nodes in the binary tree
Auxiliary Space: O(n) where n is the number of nodes in the binary tree
"""
