from typing import Optional, List
from collections import deque, defaultdict, Counter
import sys
from math import inf


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

################################################################


class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        seen = {}
        for i, value in enumerate(nums):
            remaining = target - value

            if remaining in seen:
                return [seen[remaining], i]

            seen[value] = i


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

        lower, upper = 0, len(nums) - 1
        while lower <= upper:
            mid = lower + (upper - lower) // 2
            if target == nums[mid]:
                return mid

            if nums[lower] <= nums[mid]:
                if nums[lower] <= target <= nums[mid]:
                    upper = mid
                else:
                    lower = mid + 1
            else:
                if nums[mid] <= target <= nums[upper]:
                    lower = mid + 1
                else:
                    upper = mid
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
                if resLen <= len(s[l:r+1]):
                    res = s[l:r+1]
                    resLen = len(res)

                l -= 1
                r += 1

        return res


class Solution:
    def maxArea(self, height: list[int]) -> int:
        record = 0
        l, r = 0, len(height) - 1
        dist = len(height) - 1
        for w in range(dist, 0, -1):
            if height[l] < height[r]:
                record = max(record, w * height[l])
                l += 1
            else:
                record = max(record, w * height[r])
                r -= 1

        return record


class Solution:
    def threeSum(self, nums: list[int]):
        res = []
        nums.sort()
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l, r = i+1, len(nums)-1
            while l < r:
                sum = nums[i] + nums[l] + nums[r]
                if sum < 0:
                    l += 1
                elif sum > 0:
                    r -= 1
                else:
                    res.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l+1]:
                        l += 1
                    while l < r and nums[r] == nums[r-1]:
                        r -= 1
                    l += 1
                    r -= 1
        return res


class Solution:
    def removeNthFromEnd(self, head, n):
        def remove(head):
            if not head:
                return 0, head
            i, head.next = remove(head.next)
            return i+1, (head, head.next)[i+1 == n]
        return remove(head)[1]


class Solution:
    def getIntersectionNode(self, headA, headB):
        if headA is None or headB is None:
            return None

        pa = headA
        pb = headB

        while pa is not pb:
            pa = headB if pa is None else pa.next
            pb = headA if pb is None else pb.next

        return pa


class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        res = head = ListNode()

        while list1 and list2:
            if list1.val <= list2.val:
                res.next = list1
                list1 = list1.next
            else:
                res.next = list2
                list2 = list2.next

            res = res.next

        res.next = list1 or list2
        return head.next


class Solution:
    def mergeKLists(self, lists: list[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return None
        if len(lists) == 1:
            return lists[0]
        mid = len(lists) // 2
        l, r = self.mergeKLists(lists[:mid]), self.mergeKLists(lists[mid:])
        return self.merge(l, r)

    def merge(self, l, r):
        head = tail = ListNode()
        while l and r:
            if l.val <= r.val:
                tail.next = l
                l = l.next
            else:
                tail.next = r
                r = r.next
            tail = tail.next
        tail.next = l or r

        return head.next


class Solution:
    def merge(self, nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        pa, pb, write_index = m-1, n-1, m + n - 1

        while pb >= 0:
            if pa >= 0 and nums1[pa] > nums2[pb]:
                nums1[write_index] = nums1[pa]
                pa -= 1
            else:
                nums1[write_index] = nums2[pb]
                pb -= 1

            write_index -= 1


class Solution:
    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        def dfs(start_index, path, remaining):
            if remaining == 0:
                ans.append(path)
                return

            for i in range(start_index, len(candidates)):
                if remaining - candidates[i] >= 0:
                    dfs(i, path + [candidates[i]], remaining - candidates[i])

            return ans

        ans = []
        return dfs(0, [], target)


class Solution:
    def removePalindromeSub(self, S: str) -> int:
        if not S:
            return 0
        return 1 if S == S[::-1] else 2


class Solution:
    def rotate(self, matrix: list[list[int]]) -> None:
        n = len(matrix)
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for row in matrix:
            row[:] = row[::-1]


class Solution:
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        seen = {}
        for val in strs:
            ana = ''.join(sorted(val))

            if ana in seen:
                seen[ana].append(val)
            else:
                seen[ana] = [val]

        return seen.values()


class Solution:
    def maxSubArray(self, nums: list[int]) -> int:
        curSum = maxSum = nums[0]
        for num in nums[1:]:
            curSum = max(num, curSum + num)
            maxSum = max(curSum, maxSum)

        return maxSum


class Solution:
    def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
        if len(matrix) == 0:
            return []

        new_matrix = []
        for col in range(len(matrix[0])-1, -1, -1):
            new_row = []
            for row in range(1, len(matrix), 1):
                new_row.append(matrix[row][col])
            new_matrix.append(new_row)

        return matrix[0] + self.spiralOrder(new_matrix)


class Solution:
    def minimumTotal(self, triangle: list[list[int]]) -> int:
        if not triangle:
            return
        for i in range(1, len(triangle)):
            for j in range(len(triangle[i])):
                if j == 0:
                    triangle[i][j] += triangle[i-1][j]
                elif j == len(triangle[i]) - 1:
                    triangle[i][j] += triangle[i-1][j-1]
                else:
                    triangle[i][j] += min(triangle[i-1][j-1], triangle[i-1][j])
        return min(triangle[-1])


class Solution:
    def mergeSort(self, unsorted_list: list[int]) -> list[int]:
        if len(unsorted_list) == 1:
            return unsorted_list

        mid = len(unsorted_list) // 2
        l, r = self.mergeSort(unsorted_list[:mid]), self.mergeSort(
            unsorted_list[mid:])
        pa = pb = 0
        res = []
        while pa < len(l) and pb < len(r):
            if l[pa] < r[pb]:
                res.append(l[pa])
                pa += 1
            else:
                res.append(r[pb])
                pb += 1

        while pa < len(l):
            res.append(l[pa])
            pa += 1

        while pb < len(r):
            res.append(r[pb])
            pb += 1

        return res


class Solution:
    def canJump(self, nums: list[int]) -> bool:
        m = 0
        for i, n in enumerate(nums):
            if i > m:
                return False
            m = max(m, i+n)
        return True


class Solution:
    ans = 0

    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        def dfs(node: TreeNode) -> int:
            if not node:
                return 0
            val = dfs(node.left) + dfs(node.right)
            if val == 0:
                return 3
            if val < 3:
                return 0
            self.ans += 1
            return 1

        return self.ans + 1 if dfs(root) > 2 else self.ans


class Solution:
    def merge(self, intervals: list[list[int]]) -> list[list[int]]:
        out = []
        for i in sorted(intervals, key=lambda i: i[0]):
            if out and i[0] <= out[-1][1]:
                out[-1][1] = max(out[-1][1], i[1])
            else:
                out.append(i)
        return out


class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        def dfs(down, right, memo):
            if down == 1 or right == 1:
                return 1

            if (down, right) in memo:
                return memo[(down, right)]

            memo[(down, right)] = dfs(down-1, right, memo) + \
                dfs(down, right-1, memo)

            return memo[(down, right)]

        return dfs(m, n, {})


class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1] * n for _ in range(m)]

        for r in range(1, m):
            for c in range(1, n):
                dp[r][c] = dp[r-1][c] + dp[r][c-1]

        return dp[-1][-1]


class Solution:
    def map_gate_distances(dungeon_map: list[list[int]]) -> list[list[int]]:
        rows, cols = len(dungeon_map), len(dungeon_map[0])
        queue = deque()
        INF = 2147483647
        directions = [(-1, 0), (0, 1), (1, 0), (-1, 0)]

        for r in range(rows):
            for c in range(cols):
                if dungeon_map[r][c] == 0:
                    queue.append((r, c))

        while queue:
            row, col = queue.popleft()
            for row_offset, col_offset in directions:
                total_row = row + row_offset
                total_col = col + col_offset
                if total_row >= 0 and total_row < rows and total_col >= 0 and total_col < cols:
                    if dungeon_map[total_row][total_col] == INF:
                        dungeon_map[total_row][total_col] = dungeon_map[row][col] + 1
                        queue.append((total_row, total_col))

        return dungeon_map


class Solution:
    def shipWithinDays(self, weights: list[int], days: int) -> int:
        def feasible(cap: int):
            trips = 1
            total = 0
            for w in weights:
                total += w
                if total > cap:
                    trips += 1
                    total = w
                    if trips > days:
                        return False
            return True

        left, right = max(weights), sum(weights)
        while left < right:
            mid = left + (right - left) // 2
            if feasible(mid):
                right = mid
            else:
                left = mid + 1
        return left


class Solution:
    def openLock(self, deadends, target):
        def neighbors(node):
            for i in range(4):
                x = int(node[i])
                for d in (-1, 1):
                    y = (x + d) % 10
                    yield node[:i] + str(y) + node[i+1:]

        dead = set(deadends)
        queue = deque([('0000', 0)])
        seen = {'0000'}
        while queue:
            node, depth = queue.popleft()
            if node == target:
                return depth
            if node in dead:
                continue
            for nei in neighbors(node):
                if nei not in seen:
                    seen.add(nei)
                    queue.append((nei, depth+1))
        return -1


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: list[str]) -> int:
        all_comb_dict = defaultdict(list)
        for word in wordList:
            for i in range(len(word)):
                k = word[:i] + '*' + word[i+1:]
                all_comb_dict[k].append(word)

        queue = deque([(beginWord, 1)])
        seen = {beginWord}
        while queue:
            node, depth = queue.popleft()
            if node == endWord:
                return depth
            for i in range(len(node)):
                intr = node[:i] + '*' + node[i+1:]
                for nei in all_comb_dict[intr]:
                    if nei not in seen:
                        seen.add(nei)
                        queue.append((nei, depth+1))

        return 0


class Solution:
    def num_steps(init_pos: list[list[int]]) -> int:
        rows, cols = len(init_pos), len(init_pos[0])

        def neighbors(pos, start):
            r, c = start
            dir = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            for row, col in dir:
                copy = list(list(i) for i in pos)
                row_offset = row + r
                col_offset = col + c
                if row_offset >= 0 and row_offset < rows and col_offset >= 0 and col_offset < cols:
                    copy[row_offset][col_offset], copy[r][c] = copy[r][c], copy[row_offset][col_offset]
                    yield copy

        queue = deque([(init_pos, 0)])
        seen = {tuple(tuple(i) for i in init_pos)}
        while queue:
            node, depth = queue.popleft()
            if node == [[1, 2, 3], [4, 5, 0]]:
                return depth
            zero = (0, 0)
            for r in range(rows):
                for c in range(cols):
                    if node[r][c] == 0:
                        zero = (r, c)
            for nei in neighbors(node, zero):
                k = tuple(tuple(i) for i in nei)
                if k not in seen:
                    seen.add(k)
                    queue.append((nei, depth+1))

        return -1


class Solution:
    def totalSum(powers):
        n = len(powers)

        def dfs(start_index, total, memo):
            if start_index == n:
                return total % ((10 ** 9) + 7)

            if total in memo:
                return memo[total]

            for i in range(start_index + 1, n + 1):
                total += min(powers[start_index:i]) * \
                    sum(powers[start_index:i])
                ans = dfs(start_index+1, total, {})
                memo[total] = ans

            return ans

        return dfs(0, 0, {})


class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        INF = sys.maxsize

        def dfs(node, lower, upper):
            if not node:
                return True

            if lower < node.val < upper:
                return dfs(node.left, lower, node.val) and dfs(node.right, node.val, upper)
            else:
                return False

        return dfs(root, -INF, INF)


class Solution:
    def remove_duplicates(self, arr: list[int]) -> int:
        slow = 0
        for fast in range(len(arr)):
            if arr[fast] != arr[slow]:
                slow += 1
                arr[slow] = arr[fast]
        return slow + 1


class Solution:
    def findAnagrams(self, s: str, p: str) -> list[int]:
        ns, np = len(s), len(p)
        s_count, p_count = [0] * 26, [0] * 26
        ans = []

        if ns < np:
            return ans

        for c in p:
            p_count[ord(c) - ord('a')] += 1

        for end in range(ns):
            s_count[ord(s[end]) - ord('a')] += 1
            if end >= np:
                s_count[ord(s[end - np]) - ord('a')] -= 1

            if s_count == p_count:
                ans.append(end - np + 1)

        return ans


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not t or not s or s < t:
            return ""

        dict_t = Counter(t)
        window = {}
        l, r = 0, 0
        formed = 0
        required = len(dict_t)
        ans = sys.maxsize, None, None
        n = len(s)

        while r < n:
            ch = s[r]
            window[ch] = window.get(ch, 0) + 1

            if window[ch] == dict_t[ch]:
                formed += 1

            while l <= r and formed == required:
                ch = s[l]
                window[ch] -= 1

                if r - l + 1 < ans[0]:
                    ans = (r - l + 1, l, r)

                if window[ch] < dict_t[ch]:
                    formed -= 1

                l += 1

            r += 1

        return "" if ans[0] == sys.maxsize else ans[ans[1]:ans[2]+1]


class Solution:
    def prefixSum(self, arr: list[int], target: int) -> tuple(int, int):
        prefix_sum = {0: 0}
        cur_sum = 0
        for i, val in enumerate(arr):
            cur_sum += val
            diff = cur_sum - target
            if diff in prefix_sum:
                return (prefix_sum[diff], i + 1)

            prefix_sum[cur_sum] = i + 1

    def prefixSumCount(self, arr: list[int], target: int) -> int:
        prefix_sum = Counter()
        prefix_sum[0] = 1
        cur_sum = 0
        count = 0
        for i in range(len(arr)):
            cur_sum += arr[i]
            diff = cur_sum - target
            if diff in prefix_sum:
                count += prefix_sum[diff]

            prefix_sum[cur_sum] += 1


class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            if fast is slow:
                return True
            slow = slow.next
        return False


class Solution:
    def maximum_score(arr1: List[int], arr2: List[int]) -> int:
        n1, n2 = len(arr1), len(arr2)
        p1, p2 = 0, 0
        sum1, sum2 = 0, 0
        total = 0
        MODULO_AMT = 10 ** 9 + 7

        while p1 < n1 or p2 < n2:
            if p1 < n1 and p2 < n2 and arr1[p1] == arr2[p2]:
                total += max(sum1, sum2) + arr1[p1]
                sum1 = 0
                sum2 = 0
                p1 += 1
                p2 += 1
                continue

            if p1 == n1 or (p2 != n2 and arr1[p1] > arr2[p2]):
                sum2 += arr2[p2]
                p2 += 1
            else:
                sum1 += arr1[p1]
                p1 += 1

        total += max(sum1, sum2)
        return total % MODULO_AMT


class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        rows, cols = len(board), len(board[0])
        path = []

        def neighbors(root):
            r, c = root
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            for row, col in directions:
                row_offset = r + row
                col_offset = c + col
                if row_offset >= 0 and row_offset < rows and col_offset >= 0 and col_offset < cols:
                    yield (row_offset, col_offset)

        def dfs(root, path):
            row, col = root
            if ''.join(path) == word:
                return True

            next_letter = word[len(path)]

            board[row][col] = '#'
            for r, c in neighbors(root):
                if board[r][c] == next_letter:
                    if dfs((r, c), path + [next_letter]):
                        return True

            board[row][col] = path[-1]

            return False

        for row in range(rows):
            for col in range(cols):
                cell = (row, col)
                if board[row][col] == word[0]:
                    if dfs(cell, path + [board[row][col]]):
                        return True

        return False


class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        def dfs(node1, node2):
            if not node1 and not node2:
                return True
            if not node1 or not node2:
                return False

            if node1.val == node2.val:
                return dfs(node1.left, node2.left) and dfs(node1.right, node2.right)
            else:
                return False

        return dfs(p, q)


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if inorder:
            index = inorder.index(preorder.pop(0))
            root = TreeNode(inorder[index])
            root.left = self.buildTree(preorder, inorder[:index])
            root.right = self.buildTree(preorder, inorder[index+1:])
            return root


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price, max_profit = float('inf'), 0
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)
        return max_profit


class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return max([0] + nums)

        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = max(dp[0], nums[1])
        for i in range(2, n):
            dp[i] = max(dp[i-1], nums[i] + dp[i-2])
        return dp[-1]


class Solution:
    def robII(self, nums: List[int]) -> int:
        def helper(nums):
            n = len(nums)
            dp = [0] * n
            dp[0] = nums[0]
            dp[1] = max(nums[0], nums[1])
            for i in range(2, n):
                print(i, n)
                dp[i] = max(dp[i-1], nums[i] + dp[i-2])
            return dp[-1]

        if len(nums) <= 2:
            return max([0] + nums)
        return max(helper(nums[1:]), helper(nums[:-1]))


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [inf] * (amount + 1)
        dp[0] = 0
        for i in range(1, range + 1):
            for coin in coins:
                if i - coin >= 0:
                    dp[i] = min(dp[i-coin], dp[i])
        return dp[-1] if dp[-1] != inf else -1


class Solution:
    def min_path_sum(self, grid: list[list[int]]) -> int:
        rows, cols = len(grid), len(grid[0])

        for r in range(rows):
            for c in range(cols):
                if r == 0 and c == 0:
                    continue
                elif r == 0 and c != 0:
                    grid[r][c] += grid[r][c-1]
                elif r != 0 and c == 0:
                    grid[r][c] += grid[r-1][c]
                else:
                    grid += min(grid[r-1][c], grid[r][c-1])

        return grid[-1][-1]


class Solution:
    def plumber(grid: List[List[int]]) -> int:

        for r in range(len(grid)):
            grid[r].append(-1)

        rows, cols = len(grid), len(grid[0])

        for r in range(rows):
            total = 0
            start = 0
            for c in range(cols):
                if grid[r][c] != -1:
                    total += grid[r][c]
                    if r > 0:
                        prevMax = max(prevMax, grid[r-1][c])
                else:
                    for i in range(start, c):
                        if r == 0:
                            grid[r][i] = total
                        elif prevMax == -1:
                            grid[r][i] = -1
                        else:
                            grid[r][i] = prevMax + total
                    start = c + 1
                    total = 0
                    prevMax = -1
        return max(grid[-1])


class Solution:
    def longest_sub_len(nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        record = 0

        for i in range(n):
            for j in range(i):
                if nums[j] >= nums[i]:
                    continue
                dp[i] = max(dp[i], dp[j] + 1)
            record = max(record, dp[i])

        return record


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        record = 0

        for num in nums:
            if num - 1 not in num_set:
                cur_num = num
                cur_streak = 1

                while cur_num + 1 in num_set:
                    cur_streak += 1
                    cur_num += 1

                record = max(record, cur_streak)

        return record


class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        def dfs(root, visited):
            if not root:
                return root

            if root in visited:
                return visited[root]

            clone = Node(root.val, [])
            visited[root] = clone

            if root.neighbors:
                clone.neighbors = [dfs(nei, visited) for nei in root.neighbors]

            return clone

        return dfs(node, {})


class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        def dfs(start_index, memo):
            if start_index == len(s):
                return True

            if start_index in memo:
                return memo[start_index]

            ok = False
            for word in wordDict:
                if s[start_index:].startswith(word):
                    if dfs(start_index + len(word), memo):
                        ok = True
                        break

            memo[start_index] = ok
            return memo[start_index]
        return dfs(0, {})


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head

        p = self.reverseList(head.next)
        head.next.next = head
        head.next = None

        return p


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head
        while curr:
            tmp = curr.next

            curr.next = prev
            prev = curr
            curr = tmp
        return prev


class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow


class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        prev, curr = None, slow
        while curr:
            curr.next, prev, curr = prev, curr, curr.next

        while prev.next:
            head.next, head = prev, head.next
            prev.next, prev = head, prev.next
