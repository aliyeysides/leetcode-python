from typing import Optional
from collections import deque, defaultdict


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
                tail = tail.next
            else:
                tail.next = r
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
        res = []
        self.dfs(candidates, target, [], res)
        return res

    def dfs(self, nums, target, path, res):
        if target < 0:
            return
        if target == 0:
            res.append(path)
            return
        for i in range(len(nums)):
            remaining = target - nums[i]
            self.dfs(nums[i:], remaining, path+[nums[i]], res)


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
            if node == target: return depth
            if node in dead: continue
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
            if node == endWord: return depth
            for i in range(len(node)):
                intr = node[:i] + '*' + node[i+1:]
                for nei in all_comb_dict[intr]:
                    if nei not in seen:
                        seen.add(nei)
                        queue.append((nei, depth+1))

        return 0
    