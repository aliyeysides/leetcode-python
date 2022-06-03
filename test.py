from collections import defaultdict
from typing import Optional


class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        seen = {}
        for i, value in enumerate(nums):
            remaining = target - value

            if remaining in seen:
                return [seen[remaining], i]

            seen[value] = i


test = Solution()

# print(test.twoSum([3, 2, 3], 6))


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        res = dummy = ListNode()
        carry = 0
        while l1 or l2:
            v1, v2 = 0, 0
            if l1:
                v1, l1 = l1.val, l1.next
            if l2:
                v2, l2 = l2.val, l2.next

            val = carry + v1 + v2
            res.next = ListNode(val % 10)
            res, carry = res.next, val//10

        if carry:
            res.next = ListNode(carry)

        return dummy.next


test = Solution()

# print(test.addTwoNumbers([5, 4], [6, 3]))


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        used = {}
        max_length = start = 0
        for i, c in enumerate(s):
            if c in used and start <= used[c]:
                start = used[c] + 1
            else:
                max_length = max(max_length, i - start + 1)

            used[c] = i

        return max_length


test = Solution()

# print(test.lengthOfLongestSubstring("tmmzuxt"))


class Solution(object):
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1 or numRows >= len(s):
            return s

        L = [''] * numRows
        index, step = 0, 1

        for x in s:
            L[index] += x
            if index == 0:
                step = 1
            elif index == numRows - 1:
                step = -1
            index += step

        return ''.join(L)


def myAtoi(s: str) -> int:
    if len(s) == 0:
        return 0
    ls = list(s.strip())

    sign = -1 if ls[0] == '-' else 1
    if ls[0] in ['-', '+']:
        del ls[0]
    ret, i = 0, 0
    while i < len(ls) and ls[i].isdigit():
        ret = ret*10 + ord(ls[i]) - ord('0')
        i += 1
    return max(-2**31, min(sign * ret, 2**31-1))


# print(myAtoi('-2147483649'))


class Solution:
    def isValid(self, s: str) -> bool:
        opening = {
            '(': ')',
            '{': '}',
            '[': ']'
        }

        stack = []
        for c in s:
            if c in opening:
                stack.append(c)
            elif len(stack) == 0 or opening[stack.pop()] != c:
                return False

        return len(stack) == 0


class Solution:
    def generateParenthesis(self, n: int) -> list[str]:
        def generate(p, left, right, parens=[]):
            if left:
                generate(p + '(', left-1, right)
            if right > left:
                generate(p + ')', left, right-1)
            if not right:
                parens += p,
            return parens
        return generate('', n, n)


class Solution:
    def search(self, nums, target):
        if not nums:
            return -1

        low, high = 0, len(nums) - 1

        while low <= high:
            mid = (low + high) // 2
            if target == nums[mid]:
                return mid

            if nums[low] <= nums[mid]:
                if nums[low] <= target <= nums[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                if nums[mid] <= target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1

        return -1


class Solution:
    def longestPalindrome(self, s: str) -> str:
        res = ""
        resLen = 0

        for i in range(len(s)):
            l, r = i, i
            while l >= 0 and r < len(s) and s[l] == s[r]:
                if resLen <= len(s[l:r+1]):
                    res = s[l:r+1]
                    resLen = len(res)

                l -= 1
                r += 1

            l, r = i, i + 1
            while l >= 0 and r < len(s) and s[l] == s[r]:
                print('test')
                if resLen <= len(s[l:r+1]):
                    res = s[l:r+1]
                    resLen = len(res)

                l -= 1
                r += 1

        return res


class Solution:
    # Function to return Breadth First Traversal of given graph.
    def bfsOfGraph(self, V, adj):
        visited = []
        queue = []
        res = []

        visited.append(0)
        queue.append(0)

        while queue:
            vertex = queue.pop(0)
            res.append(vertex)
            for edge in adj[vertex]:
                if edge not in visited:
                    queue.append(edge)
                    visited.append(edge)
        return res


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, vertex, edge):
        self.graph[vertex].append(edge)

    def BFS(self, s):
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

# print(g.BFS(2))


class Node:
    def __init__(self, v):
        self.val = v
        self.left = None
        self.right = None

    def levelOrder(self, root):
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

# print(root.levelOrder(root))
