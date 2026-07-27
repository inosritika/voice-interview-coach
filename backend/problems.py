"""Original interview-problem bank — coding (solve), debug, and system design.

Why original: LeetCode / InterviewBit problem *text* is copyrighted, so we can't
ship their wording. Instead we cover the same well-known PATTERNS interviewers
actually use today (hashing, two-pointers, sliding window, stacks, BFS/DFS, heaps,
intervals; classic buggy snippets; the standard system-design prompts) in our own
words. Easy to extend — just append to PROBLEMS.

Each problem carries a PRIVATE `interviewer_brief`: the intended approach, the
planted bug, or the key trade-offs. It's given to the interviewer (via the system
prompt) so it can guide well and spot mistakes — the candidate never sees it. The
public fields (everything except the brief) are what the lobby and editor show.

This is a VOICE interview with a shared scratchpad: the candidate writes in an
editor both sides can see and talks through it. No code is executed — the
interviewer reasons about the code the way a human interviewer looking over your
shoulder would.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Problem:
    id: str
    fmt: str            # "solve" | "debug" | "design"
    topic: str          # "dsa" | "system_design"
    title: str
    difficulty: str     # "easy" | "medium" | "hard"
    prompt: str         # shown to the candidate (statement + example, digits not words)
    starter_code: str   # pre-fills the editor (skeleton | buggy code | design template)
    language: str       # "python" | "text"
    interviewer_brief: str  # PRIVATE — intended solution / planted bug / trade-offs
    tags: tuple[str, ...] = ()   # topic tags for filtering: "array", "graph", "heap"…
    minutes: int = 0             # suggested time budget; 0 = derive from difficulty
    companies: tuple[str, ...] = ()  # companies known to favour this pattern (for biased picks)

    # Rough time budgets by difficulty, like a real interview clock.
    _DEFAULT_MIN = {"easy": 15, "medium": 25, "hard": 35, "custom": 25}

    def budget(self) -> int:
        return self.minutes or self._DEFAULT_MIN.get(self.difficulty, 20)

    def public(self) -> dict:
        """Everything the client may see — the brief is stripped out."""
        d = asdict(self)
        d.pop("interviewer_brief", None)
        d["minutes"] = self.budget()
        return d


# ---- SOLVE: write a solution ------------------------------------------------

_SOLVE = [
    Problem(
        id="first-unique",
        fmt="solve", topic="dsa", title="First unique in a stream", difficulty="easy",
        prompt=(
            "Given a list of integer IDs, return the first ID that appears exactly once "
            "in the list. If every ID repeats, return -1.\n\n"
            "Example: for [5, 3, 5, 2, 3, 4] the answer is 2 (5 and 3 repeat; 2 is the "
            "first that doesn't). For [1, 1, 2, 2] the answer is -1."
        ),
        starter_code=(
            "def first_unique(ids: list[int]) -> int:\n"
            "    # Talk through your approach, then sketch it here.\n"
            "    pass\n"
        ),
        language="python",
        interviewer_brief=(
            "Optimal: one pass to count frequencies in a dict, a second pass to return the "
            "first id with count 1. O(n) time, O(n) space. Weak answers scan the rest of the "
            "list for each element (O(n^2)). Push on: what if the stream is huge / online? "
            "(keep counts + an ordered structure). Edge cases: empty list, all-repeat."
        ),
    ),
    Problem(
        id="pair-sum",
        fmt="solve", topic="dsa", title="Pair with a target sum", difficulty="easy",
        prompt=(
            "Given a list of integers and a target, return the indices of two distinct "
            "elements that add up to the target, or an empty list if none exist.\n\n"
            "Example: nums = [2, 7, 11, 15], target = 9 -> [0, 1] because 2 + 7 = 9."
        ),
        starter_code=(
            "def pair_sum(nums: list[int], target: int) -> list[int]:\n"
            "    pass\n"
        ),
        language="python",
        interviewer_brief=(
            "Optimal: hash map of value -> index in one pass; for each x check if "
            "target - x was seen. O(n) time / space. The brute force is the O(n^2) double "
            "loop — get them to improve it. Edge cases: duplicates (e.g. [3,3], target 6), "
            "no solution, negative numbers."
        ),
    ),
    Problem(
        id="longest-unique-substring",
        fmt="solve", topic="dsa", title="Longest substring without repeats", difficulty="medium",
        prompt=(
            "Given a string, return the length of the longest substring that contains no "
            "repeated characters.\n\n"
            "Example: \"abcabcbb\" -> 3 (the substring \"abc\"). \"bbbbb\" -> 1. \"pwwkew\" -> 3 (\"wke\")."
        ),
        starter_code=(
            "def longest_unique(s: str) -> int:\n"
            "    pass\n"
        ),
        language="python",
        interviewer_brief=(
            "Optimal: sliding window with a set (or last-seen-index map). Move right pointer, "
            "shrink from the left when a repeat appears; track max window. O(n). Common mistake: "
            "resetting the window to 0 instead of moving left just past the previous occurrence. "
            "Probe complexity and why the window never moves backward."
        ),
    ),
    Problem(
        id="merge-intervals",
        fmt="solve", topic="dsa", title="Merge overlapping intervals", difficulty="medium",
        prompt=(
            "Given a list of intervals [start, end], merge all that overlap and return the "
            "result sorted by start.\n\n"
            "Example: [[1, 3], [2, 6], [8, 10], [15, 18]] -> [[1, 6], [8, 10], [15, 18]] "
            "(the first two overlap and merge into [1, 6])."
        ),
        starter_code=(
            "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n"
            "    pass\n"
        ),
        language="python",
        interviewer_brief=(
            "Optimal: sort by start (O(n log n)), then sweep — if the current start <= last "
            "merged end, extend the end to max(end, cur_end), else append. Watch for: forgetting "
            "to sort, using < vs <= for touching intervals like [1,2],[2,3], and mutating the "
            "input. Ask for the complexity and why sorting is the bottleneck."
        ),
    ),
    Problem(
        id="kth-largest-stream",
        fmt="solve", topic="dsa", title="Kth largest in a stream", difficulty="medium",
        prompt=(
            "Design a structure that, given a fixed k, accepts a stream of integers and after "
            "each one returns the k-th largest value seen so far.\n\n"
            "Example: k = 3, stream 4, 5, 8, 2 -> after 8 the 3rd largest is 4; after 2 it's 4."
        ),
        starter_code=(
            "class KthLargest:\n"
            "    def __init__(self, k: int):\n"
            "        pass\n\n"
            "    def add(self, value: int) -> int:\n"
            "        pass\n"
        ),
        language="python",
        interviewer_brief=(
            "Optimal: a min-heap of size k. Push each value; if the heap exceeds k, pop the "
            "smallest; the root is the k-th largest. add() is O(log k). Weak answers re-sort "
            "everything each add (O(n log n)). Probe why a MIN heap (not max) and what the root "
            "represents."
        ),
    ),
]

# ---- DEBUG: find and fix the bug --------------------------------------------

_DEBUG = [
    Problem(
        id="debug-binary-search",
        fmt="debug", topic="dsa", title="Binary search that crashes", difficulty="easy",
        prompt=(
            "This binary search should return the index of target in a sorted list, or -1. "
            "It crashes with an index error on some inputs. Find and fix the bug, and explain "
            "what was wrong.\n\n"
            "Example: binary_search([1, 3, 5, 7, 9], 9) returns 4 (fine), but "
            "binary_search([1, 3, 5, 7, 9], 10) — a target larger than everything — crashes "
            "instead of returning -1."
        ),
        starter_code=(
            "def binary_search(nums: list[int], target: int) -> int:\n"
            "    lo, hi = 0, len(nums)\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if nums[mid] == target:\n"
            "            return mid\n"
            "        elif nums[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n"
        ),
        language="python",
        interviewer_brief=(
            "BUG: `hi = len(nums)` together with `while lo <= hi` lets mid reach len(nums), so "
            "nums[mid] is out of range (crashes when target is the last/again element). Fix: "
            "`hi = len(nums) - 1` (keep <=), OR keep hi = len(nums) and change the loop to "
            "`while lo < hi`. Good candidates reason about the invariant (is hi inclusive or "
            "exclusive?). Don't reveal the line — nudge them to trace mid at the boundary."
        ),
    ),
    Problem(
        id="debug-bfs-visited",
        fmt="debug", topic="dsa", title="BFS that never terminates", difficulty="medium",
        prompt=(
            "This breadth-first search should return the shortest number of hops from start to "
            "goal in an unweighted graph (a dict of node -> list of neighbors), or -1. On graphs "
            "with cycles it hangs or is wildly slow. Find and fix the bug.\n\n"
            "Example: graph = {1: [2, 3], 2: [1, 4], 3: [4], 4: []}, start 1, goal 4 -> 2."
        ),
        starter_code=(
            "from collections import deque\n\n"
            "def shortest_hops(graph: dict, start, goal) -> int:\n"
            "    q = deque([(start, 0)])\n"
            "    visited = set()\n"
            "    while q:\n"
            "        node, dist = q.popleft()\n"
            "        if node == goal:\n"
            "            return dist\n"
            "        for nb in graph[node]:\n"
            "            q.append((nb, dist + 1))\n"
            "    return -1\n"
        ),
        language="python",
        interviewer_brief=(
            "BUG: `visited` is created but never used — neighbors are enqueued again and again, "
            "so on a cycle the queue explodes (and revisits inflate work; distances can be wrong). "
            "Fix: mark nodes visited before/when enqueuing: add start to visited up front, and "
            "`if nb not in visited: visited.add(nb); q.append((nb, dist+1))`. Probe why marking at "
            "enqueue time (not dequeue) avoids duplicates in the queue."
        ),
    ),
    Problem(
        id="debug-off-by-one",
        fmt="debug", topic="dsa", title="Factorial that's off by a factor", difficulty="easy",
        prompt=(
            "This should compute n! (n factorial). It returns the wrong answer for every n > 1. "
            "Find and fix the bug.\n\n"
            "Example: factorial(5) should return 120, but this returns 24."
        ),
        starter_code=(
            "def factorial(n: int) -> int:\n"
            "    result = 1\n"
            "    for i in range(1, n):\n"
            "        result *= i\n"
            "    return result\n"
        ),
        language="python",
        interviewer_brief=(
            "BUG: `range(1, n)` stops at n-1, so it computes (n-1)! — off by a factor of n. Fix: "
            "`range(1, n + 1)` (or range(2, n+1)). The example (24 = 4! instead of 5!) is the tell. "
            "Also worth asking: what should factorial(0) return? (1 — the loop handles it correctly)."
        ),
    ),
]

# ---- DESIGN: system / object design (scratchpad is for notes, not runnable code)

_DESIGN = [
    Problem(
        id="design-url-shortener",
        fmt="design", topic="system_design", title="URL shortener (bit.ly)", difficulty="medium",
        prompt=(
            "Design a URL shortener: users submit a long URL and get a short code; visiting the "
            "short code redirects to the original. Assume ~100 million new links/month and reads "
            "roughly 100x writes. Use the scratchpad for your components, data model, and APIs."
        ),
        starter_code=(
            "# Scratchpad — jot components, data model, APIs, trade-offs.\n"
            "# Requirements (functional / non-functional):\n"
            "#   -\n"
            "# API:\n"
            "#   POST /shorten { url } -> { short }\n"
            "#   GET  /{short} -> 302 redirect\n"
            "# Data model:\n"
            "#   -\n"
            "# Key decisions:\n"
            "#   - short-code generation:\n"
            "#   - storage & scale:\n"
            "#   - caching:\n"
        ),
        language="text",
        interviewer_brief=(
            "Look for: clarifying scale first (100M/mo ~ 40 writes/s, ~4k reads/s), a clean API, "
            "and code generation — base62 of an auto-increment id or a hash (with collision "
            "handling). Storage: a KV store (short -> long), read-heavy so a cache (Redis) in "
            "front. Good trade-off discussions: counter vs random/hash codes, custom aliases, "
            "expiry, 301 vs 302. Push on the hot-read path and how they'd shard."
        ),
    ),
    Problem(
        id="design-rate-limiter",
        fmt="design", topic="system_design", title="API rate limiter", difficulty="medium",
        prompt=(
            "Design a rate limiter for an API: each user (by API key) may make at most N requests "
            "per minute; excess requests get a 429. It runs in front of a fleet of app servers. "
            "Use the scratchpad for the algorithm, where state lives, and trade-offs."
        ),
        starter_code=(
            "# Scratchpad\n"
            "# Requirements:\n"
            "#   - limit: N per minute per API key, across all servers\n"
            "# Algorithm options: fixed window / sliding window / token bucket / leaky bucket\n"
            "# Where does the counter live?\n"
            "# Trade-offs:\n"
        ),
        language="text",
        interviewer_brief=(
            "Look for: choosing an algorithm and justifying it — token bucket (smooth, allows "
            "bursts) or sliding-window log/counter (accurate, more memory). The key insight is "
            "SHARED state across servers -> a central store like Redis (INCR + EXPIRE, or a Lua "
            "script for atomicity). Discuss: fixed-window burst problem at the boundary, memory "
            "of a per-request log, what to do if Redis is down (fail open vs closed)."
        ),
    ),
    Problem(
        id="design-notification",
        fmt="design", topic="system_design", title="Notification service", difficulty="medium",
        prompt=(
            "Design a service that sends notifications (email, SMS, push) triggered by events "
            "from other services, at ~10 million sends/day. It must not lose notifications and "
            "must not spam a user with duplicates. Use the scratchpad for the architecture."
        ),
        starter_code=(
            "# Scratchpad\n"
            "# Requirements: channels (email/SMS/push), reliability, no duplicates, scale\n"
            "# High-level components:\n"
            "# Delivery guarantees:\n"
            "# Trade-offs:\n"
        ),
        language="text",
        interviewer_brief=(
            "Look for: an async, queue-based design — producers publish events to a message "
            "queue (Kafka/SQS), workers pull and fan out to channel-specific senders behind "
            "provider APIs. Reliability: durable queue + retries with backoff + a dead-letter "
            "queue. De-dupe: idempotency keys / a seen-set. Discuss: per-channel rate limits, "
            "user preferences/opt-outs, template rendering, and at-least-once vs exactly-once "
            "(hence idempotency)."
        ),
    ),
]

# ---- MORE canonical problems (the "must-know" set, original wording, no DP) --
# These mirror the classics on lists like Top-Interview-150 / Striver's SDE sheet —
# the PROBLEMS are industry-standard; only their exact wording is owned, so these
# are rewritten from scratch.
_MORE = [
    Problem(
        id="valid-parentheses", fmt="solve", topic="dsa", title="Valid parentheses",
        difficulty="easy", tags=("stack", "string"),
        prompt="Given a string containing only the characters ( ) [ ] { }, decide whether "
        "every bracket is properly opened and closed in the right order.\n\n"
        "Example: \"()[]{}\" is valid; \"([)]\" is not (they interleave); \"(]\" is not.",
        starter_code="def is_valid(s: str) -> bool:\n    pass\n", language="python",
        interviewer_brief="Optimal: a stack — push openers, on a closer check it matches the "
        "top and pop; valid iff the stack ends empty. O(n). Common misses: not checking the "
        "stack is empty at the end, or mismatched bracket types. A closer->opener map keeps it clean.",
    ),
    Problem(
        id="reverse-linked-list", fmt="solve", topic="dsa", title="Reverse a linked list",
        difficulty="easy", tags=("linked-list",),
        prompt="Reverse a singly linked list and return the new head. Nodes have .val and .next.\n\n"
        "Example: 1 -> 2 -> 3 -> None becomes 3 -> 2 -> 1 -> None.",
        starter_code="class Node:\n    def __init__(self, val, nxt=None):\n        self.val, self.next = val, nxt\n\n"
        "def reverse(head):\n    pass\n", language="python",
        interviewer_brief="Optimal: iterative three-pointer (prev, cur, nxt), flipping .next as you "
        "go; O(n) time, O(1) space. Watch for losing the rest of the list — save nxt before rewiring. "
        "The recursive version is elegant but O(n) stack.",
    ),
    Problem(
        id="product-except-self", fmt="solve", topic="dsa", title="Product of array except self",
        difficulty="medium", tags=("array",),
        prompt="Given a list of integers, return a list where each position holds the product of "
        "all the OTHER numbers — without using division.\n\nExample: [1, 2, 3, 4] -> [24, 12, 8, 6].",
        starter_code="def product_except_self(nums: list[int]) -> list[int]:\n    pass\n", language="python",
        interviewer_brief="Optimal: two passes — prefix products left-to-right, then suffix products "
        "right-to-left, multiplied together. O(n) time, O(1) extra beyond the output. The no-division "
        "rule is the whole point; ask what breaks with division when there are zeros.",
    ),
    Problem(
        id="group-anagrams", fmt="solve", topic="dsa", title="Group anagrams",
        difficulty="medium", tags=("hashing", "string"),
        prompt="Given a list of words, group together the ones that are anagrams of each other.\n\n"
        "Example: [\"eat\",\"tea\",\"tan\",\"ate\",\"nat\",\"bat\"] -> "
        "[[\"eat\",\"tea\",\"ate\"], [\"tan\",\"nat\"], [\"bat\"]] (order within a group doesn't matter).",
        starter_code="def group_anagrams(words: list[str]) -> list[list[str]]:\n    pass\n", language="python",
        interviewer_brief="Optimal: a dict keyed by the SORTED word (or a 26-length letter-count "
        "tuple) -> list of originals. Sorted key O(n·k log k); count-tuple key O(n·k). Probe why "
        "the sorted string / count is a valid canonical key.",
    ),
    Problem(
        id="top-k-frequent", fmt="solve", topic="dsa", title="Top K frequent elements",
        difficulty="medium", tags=("heap", "hashing"),
        prompt="Given a list of integers and a number k, return the k values that appear most often "
        "(any order).\n\nExample: nums = [1, 1, 1, 2, 2, 3], k = 2 -> [1, 2].",
        starter_code="def top_k_frequent(nums: list[int], k: int) -> list[int]:\n    pass\n", language="python",
        interviewer_brief="Count with a dict, then a size-k min-heap over counts -> O(n log k); or "
        "bucket sort by frequency -> O(n). Weak answer sorts all counts O(n log n). Ask which wins when k << n.",
    ),
    Problem(
        id="number-of-islands", fmt="solve", topic="dsa", title="Number of islands",
        difficulty="medium", tags=("graph", "bfs", "dfs", "matrix"),
        prompt="Given a grid of 1s (land) and 0s (water), count the islands. An island is land "
        "connected horizontally or vertically.\n\nExample:\n1 1 0 0\n1 0 0 1\n0 0 1 1\nhas 3 islands.",
        starter_code="def num_islands(grid: list[list[int]]) -> int:\n    pass\n", language="python",
        interviewer_brief="Optimal: scan the grid; on each unvisited 1, flood-fill (BFS/DFS) the "
        "whole island marking it visited, count++. O(rows·cols). Watch bounds checks and marking "
        "visited (sink to 0 or a visited set) so islands aren't recounted. Probe DFS recursion depth "
        "on a huge grid.",
    ),
    Problem(
        id="level-order", fmt="solve", topic="dsa", title="Binary tree level order",
        difficulty="medium", tags=("tree", "bfs"),
        prompt="Given the root of a binary tree, return its node values level by level, top to "
        "bottom, left to right.\n\nExample: root 3, its children 9 and 20, and 20's children 15 and 7 "
        "-> [[3], [9, 20], [15, 7]].",
        starter_code="class Node:\n    def __init__(self, val, left=None, right=None):\n        "
        "self.val, self.left, self.right = val, left, right\n\ndef level_order(root):\n    pass\n",
        language="python",
        interviewer_brief="Optimal: BFS with a queue, one level per outer iteration — snapshot the "
        "queue size and pop exactly that many. O(n). Common miss: not fixing the level size before "
        "the inner loop, so levels bleed together.",
    ),
    Problem(
        id="validate-bst", fmt="solve", topic="dsa", title="Validate a BST",
        difficulty="medium", tags=("tree",),
        prompt="Decide whether a binary tree is a valid binary search tree: every node's entire left "
        "subtree is strictly smaller and its entire right subtree strictly larger.\n\n"
        "Example: root 5 with left 1 and right 4, where 4 has children 3 and 6 -> NOT valid (3 sits "
        "in 5's right subtree but 3 < 5).",
        starter_code="class Node:\n    def __init__(self, val, left=None, right=None):\n        "
        "self.val, self.left, self.right = val, left, right\n\ndef is_valid_bst(root) -> bool:\n    pass\n",
        language="python",
        interviewer_brief="Optimal: recurse carrying (low, high) bounds, tightening as you descend; "
        "each node must be strictly inside. O(n). The CLASSIC bug is comparing a node only to its "
        "direct children, not the inherited bound — the example is built to catch exactly that. "
        "In-order traversal being strictly increasing also works.",
    ),
    Problem(
        id="min-rooms", fmt="solve", topic="dsa", title="Minimum meeting rooms",
        difficulty="medium", tags=("intervals", "heap", "sorting", "greedy"),
        prompt="Given a list of meeting intervals [start, end], return the minimum number of rooms "
        "needed so no two overlapping meetings share a room.\n\n"
        "Example: [[0, 30], [5, 10], [15, 20]] -> 2 (the [0,30] meeting overlaps both others).",
        starter_code="def min_rooms(intervals: list[list[int]]) -> int:\n    pass\n", language="python",
        interviewer_brief="Optimal: sort starts and ends separately and sweep with two pointers, or "
        "a min-heap of end times; the max number of simultaneously-active meetings is the answer. "
        "O(n log n). Ask why sorting starts and ends independently is valid.",
    ),
    Problem(
        id="min-stack", fmt="solve", topic="dsa", title="Min stack",
        difficulty="medium", tags=("stack", "design"),
        prompt="Design a stack supporting push, pop, top, and get_min — all in O(1).\n\n"
        "Example: push 3, push 1, push 2 -> get_min is 1; pop -> get_min still 1; pop -> get_min is 3.",
        starter_code="class MinStack:\n    def __init__(self):\n        pass\n\n    def push(self, x: int) -> None:\n"
        "        pass\n\n    def pop(self) -> None:\n        pass\n\n    def top(self) -> int:\n        pass\n\n"
        "    def get_min(self) -> int:\n        pass\n", language="python",
        interviewer_brief="Optimal: a second stack of running minimums (push min(x, cur_min) alongside "
        "each element); get_min is its top — all O(1). The trap is scanning for the min on each call "
        "(O(n)). Probe how the min stack stays correct across pops.",
    ),
    Problem(
        id="merge-k-lists", fmt="solve", topic="dsa", title="Merge K sorted lists",
        difficulty="hard", tags=("heap", "linked-list"),
        prompt="Given k sorted lists of integers, merge them into one sorted list.\n\n"
        "Example: [[1, 4, 5], [1, 3, 4], [2, 6]] -> [1, 1, 2, 3, 4, 4, 5, 6].",
        starter_code="def merge_k(lists: list[list[int]]) -> list[int]:\n    pass\n", language="python",
        interviewer_brief="Optimal: a min-heap holding the current front of each list -> O(N log k) "
        "for N total elements; or pairwise tournament merge, also O(N log k). Naive concat-then-sort "
        "is O(N log N). Probe why the heap stays size k.",
    ),
    Problem(
        id="median-stream", fmt="solve", topic="dsa", title="Median of a data stream",
        difficulty="hard", tags=("heap", "stream", "design"),
        prompt="Design a structure that accepts a stream of integers and returns the median of all "
        "values seen so far, at any point.\n\n"
        "Example: add 1, add 2 -> median 1.5; add 3 -> median 2; add 4 -> median 2.5.",
        starter_code="class MedianFinder:\n    def __init__(self):\n        pass\n\n    def add(self, num: int) -> None:\n"
        "        pass\n\n    def median(self) -> float:\n        pass\n", language="python",
        interviewer_brief="Optimal: two heaps — a max-heap for the lower half, a min-heap for the "
        "upper half, kept balanced (sizes differ by <=1). add is O(log n), median is O(1) (a top, or "
        "the average of the two tops). The insight is the balancing; probe how they rebalance each insert.",
    ),
    # two more debug snippets
    Problem(
        id="debug-two-sum", fmt="debug", topic="dsa", title="Two-sum that pairs an element with itself",
        difficulty="easy", tags=("array", "hashing"),
        prompt="This should return the indices of two DISTINCT elements that sum to target, or []. "
        "On some inputs it returns an element paired with itself. Find and fix the bug.\n\n"
        "Example: two_sum([3, 2, 4], 6) should return [1, 2] (2 + 4), but this returns [0, 0].",
        starter_code="def two_sum(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n"
        "        seen[x] = i\n        if target - x in seen:\n            return [seen[target - x], i]\n"
        "    return []\n", language="python",
        interviewer_brief="BUG: it inserts x into `seen` BEFORE checking for the complement, so when "
        "the complement is x itself it matches the same index ([0,0] for 3 at index 0 with target 6). "
        "Fix: check FIRST, then insert — move `seen[x] = i` to after the if.",
    ),
    Problem(
        id="debug-reverse-list", fmt="debug", topic="dsa", title="List reversal that drops nodes",
        difficulty="medium", tags=("linked-list",),
        prompt="This should reverse a singly linked list. Instead it returns a list of just one node. "
        "Find and fix the bug.\n\nExample: reversing 1 -> 2 -> 3 should give 3 -> 2 -> 1, but this returns just 3.",
        starter_code="class Node:\n    def __init__(self, val, nxt=None):\n        self.val, self.next = val, nxt\n\n"
        "def reverse(head):\n    prev = None\n    cur = head\n    while cur:\n        cur.next = prev\n"
        "        prev = cur\n        cur = cur.next\n    return prev\n", language="python",
        interviewer_brief="BUG: `cur.next = prev` destroys the link to the rest of the list BEFORE "
        "it's saved, so `cur = cur.next` walks backward and the loop ends after one node. Fix: save "
        "`nxt = cur.next` first, then rewire, then `cur = nxt`. Classic pointer-ordering bug.",
    ),
    # ---- ML hands-on (so an ML round can be practical, not only theory) ----
    Problem(
        id="ml-precision-recall", fmt="solve", topic="ml", title="Precision, recall and F1",
        difficulty="easy", tags=("metrics", "classification"),
        prompt="Given two equal-length lists of 0/1 labels — the true labels and the predicted "
        "labels — compute precision, recall and F1 for the positive class (1). Return them as a "
        "tuple. Handle the degenerate cases where a denominator is 0.\n\n"
        "Example: y_true = [1, 0, 1, 1, 0], y_pred = [1, 0, 0, 1, 1] -> precision 2/3, recall 2/3, F1 2/3.",
        starter_code="def prf(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:\n    pass\n",
        language="python",
        interviewer_brief="TP = both 1; FP = predicted 1, true 0; FN = predicted 0, true 1. "
        "precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R). The real signal is whether "
        "they guard divide-by-zero (no positive predictions -> precision undefined; convention 0). "
        "Follow up: which matters more for a rare-event detector, and why accuracy is useless there.",
    ),
    Problem(
        id="ml-kmeans-step", fmt="solve", topic="ml", title="One step of k-means",
        difficulty="medium", tags=("clustering", "numpy"),
        prompt="Implement ONE iteration of k-means on 1-D points: given a list of numbers and a "
        "list of k current centroids, assign each point to its nearest centroid, then return the "
        "updated centroids (the mean of each cluster). Decide what to do with an empty cluster.\n\n"
        "Example: points = [1, 2, 9, 10], centroids = [0, 8] -> assignments [0, 0, 1, 1] -> "
        "new centroids [1.5, 9.5].",
        starter_code="def kmeans_step(points: list[float], centroids: list[float]) -> list[float]:\n    pass\n",
        language="python",
        interviewer_brief="Assign by min |p - c|; recompute each centroid as its cluster mean. "
        "The interesting bit is the EMPTY cluster (keep the old centroid, or re-seed to the "
        "farthest point) — most candidates divide by zero there. Follow ups: does k-means always "
        "converge, why is it sensitive to initialization (k-means++), and the O(n·k) per iteration cost.",
    ),
    Problem(
        id="ml-gradient-step", fmt="solve", topic="ml", title="One gradient descent step",
        difficulty="medium", tags=("optimization", "regression"),
        prompt="For simple linear regression y = w*x + b with mean squared error, implement ONE "
        "gradient descent update: given lists x and y, current w and b, and a learning rate, "
        "return the updated (w, b).\n\n"
        "Example: x = [1, 2], y = [2, 4], w = 0, b = 0, lr = 0.1 -> one step moves w and b toward 2 and 0.",
        starter_code="def gd_step(x: list[float], y: list[float], w: float, b: float,\n"
        "            lr: float) -> tuple[float, float]:\n    pass\n",
        language="python",
        interviewer_brief="MSE = mean((w*x+b - y)^2). dW = mean(2*(pred-y)*x), dB = mean(2*(pred-y)); "
        "then w -= lr*dW, b -= lr*dB. Watch for: forgetting to average (sum vs mean changes the "
        "effective lr), and updating w using the ALREADY-updated b. Follow ups: what happens if lr "
        "is too large, and why we'd batch/normalize features.",
    ),
]

# ---- BANK 2: depth across ML / DSA / design at every difficulty ---------------
# The first bank was DSA-heavy (ML and design had 3 each), so asking for "another
# ML coding question" cycled between the same two. All original wording.
_BANK2 = [
    # ---------------- ML coding: easy ----------------
    Problem(
        id="ml-sigmoid", fmt="solve", topic="ml", title="Sigmoid activation",
        difficulty="easy", tags=("activation", "numpy"),
        prompt="Implement the sigmoid function for a list of numbers: sigmoid(z) = 1 / (1 + e^-z). "
        "Return a list of the same length.\n\nExample: [0, 2, -2] -> [0.5, 0.881, 0.119] (rounded).",
        starter_code="import math\n\ndef sigmoid(zs: list[float]) -> list[float]:\n    pass\n",
        language="python",
        interviewer_brief="One-liner: [1/(1+math.exp(-z)) for z in zs]. The real question is NUMERICAL "
        "STABILITY: math.exp(-z) overflows for very negative z. The stable form uses exp(z)/(1+exp(z)) "
        "when z < 0. Follow up: why sigmoid saturates and how that causes vanishing gradients.",
    ),
    Problem(
        id="ml-accuracy", fmt="solve", topic="ml", title="Accuracy from predictions",
        difficulty="easy", tags=("metrics", "classification"),
        prompt="Given true labels and predicted labels (equal-length lists), return the accuracy — "
        "the fraction that match. Return 0.0 for empty input.\n\n"
        "Example: y_true = [1, 0, 1, 1], y_pred = [1, 1, 1, 0] -> 0.5.",
        starter_code="def accuracy(y_true: list[int], y_pred: list[int]) -> float:\n    pass\n",
        language="python",
        interviewer_brief="sum(t == p) / len. Trivial to code — use it to probe WHEN accuracy lies: "
        "class imbalance (99% negatives -> 99% accuracy by predicting all-negative). That's the real "
        "signal here; push them toward precision/recall or balanced accuracy.",
    ),
    Problem(
        id="ml-normalize", fmt="solve", topic="ml", title="Min-max feature scaling",
        difficulty="easy", tags=("preprocessing",),
        prompt="Scale a list of numbers to the range 0 to 1 using min-max normalization: "
        "(x - min) / (max - min). Handle the case where all values are identical.\n\n"
        "Example: [10, 20, 30] -> [0.0, 0.5, 1.0].",
        starter_code="def min_max(xs: list[float]) -> list[float]:\n    pass\n",
        language="python",
        interviewer_brief="Guard max == min (divide by zero) — return zeros or 0.5s. Follow ups: why "
        "scale at all (distance-based models, gradient descent conditioning), min-max vs "
        "standardization, and the classic LEAKAGE trap: fit the scaler on TRAIN only, then apply to test.",
    ),
    Problem(
        id="ml-cosine", fmt="solve", topic="ml", title="Cosine similarity",
        difficulty="easy", tags=("similarity", "embeddings"),
        prompt="Compute the cosine similarity between two equal-length vectors: the dot product "
        "divided by the product of their magnitudes. Handle a zero vector.\n\n"
        "Example: [1, 0] and [1, 1] -> 0.707.",
        starter_code="import math\n\ndef cosine(a: list[float], b: list[float]) -> float:\n    pass\n",
        language="python",
        interviewer_brief="dot/(||a||*||b||); return 0.0 if either norm is 0. Follow ups: why cosine "
        "over Euclidean for embeddings (magnitude-invariant — document length shouldn't matter), and "
        "the relationship to dot product when vectors are already normalized.",
    ),
    # ---------------- ML coding: medium ----------------
    Problem(
        id="ml-softmax", fmt="solve", topic="ml", title="Numerically stable softmax",
        difficulty="medium", tags=("activation",),
        prompt="Implement softmax over a list of scores so the outputs are positive and sum to 1. "
        "It must not overflow on large inputs.\n\n"
        "Example: [1, 2, 3] -> [0.09, 0.245, 0.665] (rounded). [1000, 1001] must not overflow.",
        starter_code="import math\n\ndef softmax(scores: list[float]) -> list[float]:\n    pass\n",
        language="python",
        interviewer_brief="The whole point: subtract the max before exponentiating — "
        "exp(x - max) / sum(exp(x - max)). Mathematically identical, but without it exp(1000) "
        "overflows. If they write the naive version, hand them [1000, 1001] and let them discover it. "
        "Follow up: why softmax over just normalizing raw scores.",
    ),
    Problem(
        id="ml-knn", fmt="solve", topic="ml", title="K-nearest-neighbours classify",
        difficulty="medium", tags=("knn", "classification"),
        prompt="Given training points (each a list of numbers) with integer labels, a query point, "
        "and k, return the majority label among the k nearest training points by Euclidean distance.\n\n"
        "Example: points [[0,0],[0,1],[5,5]], labels [0,0,1], query [0,2], k=2 -> 0.",
        starter_code="def knn(points: list[list[float]], labels: list[int],\n"
        "        query: list[float], k: int) -> int:\n    pass\n",
        language="python",
        interviewer_brief="Distance to every point, sort (or heapq.nsmallest), majority vote of the "
        "top k. O(n·d + n log n) per query — no training cost, all cost at inference. Follow ups: "
        "tie-breaking, why odd k, why features must be scaled, and how it degrades in high dimensions.",
    ),
    Problem(
        id="ml-confusion", fmt="solve", topic="ml", title="Binary confusion matrix",
        difficulty="medium", tags=("metrics", "classification"),
        prompt="Given true and predicted 0/1 labels, return the four counts as a tuple "
        "(true_positives, false_positives, true_negatives, false_negatives).\n\n"
        "Example: y_true = [1, 0, 1, 0], y_pred = [1, 1, 0, 0] -> (1, 1, 1, 1).",
        starter_code="def confusion(y_true: list[int], y_pred: list[int]) -> tuple[int, int, int, int]:\n    pass\n",
        language="python",
        interviewer_brief="One pass counting the four combinations. The value is in the follow-up: "
        "have them derive precision, recall and specificity from these counts, and ask which error "
        "(FP vs FN) is worse for fraud detection vs a medical screening test — and how the threshold moves it.",
    ),
    Problem(
        id="ml-logistic-predict", fmt="solve", topic="ml", title="Logistic regression predict",
        difficulty="medium", tags=("regression", "classification"),
        prompt="Given a weight vector, a bias, and a list of feature rows, return the predicted "
        "probability for each row: sigmoid(w · x + b). Then return hard 0/1 labels at a threshold.\n\n"
        "Example: w = [1, 1], b = 0, rows = [[0,0],[2,2]], threshold 0.5 -> probs [0.5, 0.982] -> labels [0, 1].",
        starter_code="import math\n\ndef predict(w: list[float], b: float, rows: list[list[float]],\n"
        "            threshold: float = 0.5) -> list[int]:\n    pass\n",
        language="python",
        interviewer_brief="Dot product + bias -> sigmoid -> compare to threshold. Watch the boundary "
        "convention at exactly 0.5. Follow ups: why the threshold is a business decision not a model "
        "one, and how moving it trades precision against recall.",
    ),
    # ---------------- ML coding: hard ----------------
    Problem(
        id="ml-kmeans-full", fmt="solve", topic="ml", title="K-means to convergence",
        difficulty="hard", tags=("clustering",),
        prompt="Implement full k-means on 1-D points: given points, k, and a max iteration count, "
        "return the final centroids. Initialize however you like, iterate assign-then-update, and "
        "stop early when assignments stop changing.\n\n"
        "Example: points [1, 2, 9, 10], k = 2 -> centroids near [1.5, 9.5].",
        starter_code="def kmeans(points: list[float], k: int, max_iter: int = 100) -> list[float]:\n    pass\n",
        language="python",
        interviewer_brief="Loop: assign each point to nearest centroid, recompute centroids as cluster "
        "means, stop when assignments are unchanged (or max_iter). O(n·k) per iteration. The traps: "
        "EMPTY clusters (keep old centroid or re-seed), and initialization sensitivity (k-means++). "
        "Follow ups: does it always converge (yes, to a LOCAL optimum), and how you'd choose k (elbow/silhouette).",
    ),
    Problem(
        id="ml-roc-auc", fmt="solve", topic="ml", title="ROC AUC from scores",
        difficulty="hard", tags=("metrics", "ranking"),
        prompt="Given true 0/1 labels and predicted scores, compute the ROC AUC — the probability "
        "that a randomly chosen positive scores higher than a randomly chosen negative. Count ties "
        "as half.\n\nExample: y = [0, 0, 1, 1], scores = [0.1, 0.4, 0.35, 0.8] -> 0.75.",
        starter_code="def roc_auc(y: list[int], scores: list[float]) -> float:\n    pass\n",
        language="python",
        interviewer_brief="Cleanest: for every positive/negative pair count 1 for a win, 0.5 for a tie, "
        "divide by (n_pos * n_neg) — O(n^2) but obviously correct. The efficient version sorts by score "
        "and uses rank sums (Mann-Whitney U): (sum_of_positive_ranks - n_pos(n_pos+1)/2)/(n_pos*n_neg), "
        "O(n log n). Follow ups: why AUC is threshold-free, and why PR-AUC is preferred for rare positives.",
    ),
    # ---------------- DSA: medium ----------------
    Problem(
        id="implement-trie", fmt="solve", topic="dsa", title="Implement a trie",
        difficulty="medium", tags=("trie", "string", "design"),
        prompt="Implement a prefix tree supporting insert(word), search(word) — exact match — and "
        "starts_with(prefix).\n\nExample: insert \"apple\"; search \"apple\" -> True; search \"app\" "
        "-> False; starts_with \"app\" -> True.",
        starter_code="class Trie:\n    def __init__(self):\n        pass\n\n    def insert(self, word: str) -> None:\n"
        "        pass\n\n    def search(self, word: str) -> bool:\n        pass\n\n"
        "    def starts_with(self, prefix: str) -> bool:\n        pass\n",
        language="python",
        interviewer_brief="Nested dicts of char -> child, plus an end-of-word marker (a sentinel key or "
        "a flag). All three ops are O(len(word)). The distinction between search and starts_with IS the "
        "end marker — candidates who skip it get search(\"app\") wrong. Follow up: memory vs a hash set, "
        "and why a trie wins for autocomplete.",
    ),
    Problem(
        id="course-schedule", fmt="solve", topic="dsa", title="Can you finish the courses?",
        difficulty="medium", tags=("graph", "topological-sort", "bfs"),
        prompt="Given a number of courses and a list of [course, prerequisite] pairs, decide whether "
        "it's possible to finish all courses — i.e. whether the dependency graph has no cycle.\n\n"
        "Example: 2 courses, [[1, 0]] -> True. 2 courses, [[1, 0], [0, 1]] -> False (circular).",
        starter_code="def can_finish(num_courses: int, prereqs: list[list[int]]) -> bool:\n    pass\n",
        language="python",
        interviewer_brief="This is cycle detection on a directed graph. Cleanest is Kahn's algorithm: "
        "build in-degrees, queue the zero-in-degree nodes, pop and decrement; if you process all nodes "
        "there's no cycle. DFS with a three-colour (unvisited/in-stack/done) marking also works — the "
        "common bug is using a plain visited set, which can't tell a cross-edge from a back-edge.",
    ),
    Problem(
        id="spiral-matrix", fmt="solve", topic="dsa", title="Spiral matrix traversal",
        difficulty="medium", tags=("matrix",),
        prompt="Given a 2-D matrix, return all its elements in spiral order — right across the top, "
        "down the right side, left across the bottom, up the left side, and inward.\n\n"
        "Example: [[1,2,3],[4,5,6],[7,8,9]] -> [1, 2, 3, 6, 9, 8, 7, 4, 5].",
        starter_code="def spiral(matrix: list[list[int]]) -> list[int]:\n    pass\n",
        language="python",
        interviewer_brief="Four boundaries (top, bottom, left, right) shrinking inward. The bug everyone "
        "hits is on non-square matrices: after the top row and right column you must re-check "
        "top <= bottom / left <= right before the bottom row and left column, or you double-visit. "
        "Give them a 1-row or 1-column input to expose it.",
    ),
    Problem(
        id="rotate-image", fmt="solve", topic="dsa", title="Rotate a matrix in place",
        difficulty="medium", tags=("matrix",),
        prompt="Rotate an n x n matrix 90 degrees clockwise, IN PLACE (no new matrix).\n\n"
        "Example: [[1,2],[3,4]] becomes [[3,1],[4,2]].",
        starter_code="def rotate(matrix: list[list[int]]) -> None:\n    pass\n",
        language="python",
        interviewer_brief="Transpose (swap across the diagonal), then reverse each row — two clean "
        "passes, O(1) extra space. Watch the transpose loop bounds: the inner loop must start at i, "
        "not 0, or you swap everything back. The layer-by-layer four-way rotation also works but is "
        "far easier to get wrong.",
    ),
    Problem(
        id="subsets", fmt="solve", topic="dsa", title="All subsets",
        difficulty="medium", tags=("backtracking", "recursion"),
        prompt="Given a list of distinct integers, return all possible subsets (the power set), in "
        "any order.\n\nExample: [1, 2, 3] -> [[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]].",
        starter_code="def subsets(nums: list[int]) -> list[list[int]]:\n    pass\n",
        language="python",
        interviewer_brief="Backtracking: at each index either include or skip, recursing. Or iterate "
        "bitmasks 0..2^n-1. Either way it's O(n·2^n) — there are 2^n subsets, so that's optimal. The "
        "classic bug is appending the working list by REFERENCE instead of a copy, so every subset "
        "ends up identical (or empty). Ask what happens with duplicates in the input.",
    ),
    # ---------------- DSA: hard ----------------
    Problem(
        id="quickselect", fmt="solve", topic="dsa", title="Kth smallest without sorting",
        difficulty="hard", tags=("quickselect", "sorting", "partition"),
        prompt="Find the k-th smallest element in an unsorted list (k is 1-indexed) in better than "
        "O(n log n) average time — so no full sort.\n\n"
        "Example: [7, 10, 4, 3, 20, 15], k = 3 -> 7.",
        starter_code="def kth_smallest(nums: list[int], k: int) -> int:\n    pass\n",
        language="python",
        interviewer_brief="Quickselect: partition around a pivot; recurse only into the side containing "
        "k. Average O(n) (n + n/2 + n/4 ...), worst case O(n^2) on an adversarial pivot — random pivot "
        "or median-of-medians fixes it. Ask them to justify the average-case sum, and compare with a "
        "size-k heap (O(n log k)) — which wins depends on k.",
    ),
    Problem(
        id="min-window", fmt="solve", topic="dsa", title="Minimum window substring",
        difficulty="hard", tags=("string", "sliding-window", "hashing"),
        prompt="Given strings s and t, return the shortest substring of s that contains every "
        "character of t including duplicates. Return \"\" if none exists.\n\n"
        "Example: s = \"ADOBECODEBANC\", t = \"ABC\" -> \"BANC\".",
        starter_code="def min_window(s: str, t: str) -> str:\n    pass\n",
        language="python",
        interviewer_brief="Sliding window with a need-count map and a `missing` counter: expand right "
        "until nothing is missing, then contract left while still valid, recording the best. O(len(s)). "
        "The two traps: handling DUPLICATE characters in t (counts, not a set), and knowing when the "
        "window is still valid as you shrink. Ask for the complexity and why each pointer only moves forward.",
    ),
    Problem(
        id="word-search", fmt="solve", topic="dsa", title="Word search in a grid",
        difficulty="hard", tags=("backtracking", "matrix", "dfs"),
        prompt="Given a grid of characters and a word, decide whether the word can be formed by "
        "moving through horizontally or vertically adjacent cells. A cell can't be reused within one "
        "word.\n\nExample: grid [['A','B'],['C','D']], word \"ABD\" -> True; word \"ABC\" -> False.",
        starter_code="def exists(board: list[list[str]], word: str) -> bool:\n    pass\n",
        language="python",
        interviewer_brief="DFS from every cell, matching one character at a time, marking the cell "
        "visited before recursing and UNMARKING on the way out — that restore is the whole backtracking "
        "idea and the usual bug. Worst case O(rows·cols·4^len(word)). Ask about pruning (bail on first "
        "character mismatch) and why a visited set must be per-path, not global.",
    ),
    # ---------------- System design ----------------
    Problem(
        id="design-typeahead", fmt="design", topic="system_design", title="Search autocomplete",
        difficulty="medium", tags=("system-design",),
        prompt="Design search autocomplete: as a user types, suggest the top few completions, ranked "
        "by popularity, in under about 100 ms. Assume tens of millions of queries a day. Use the "
        "scratchpad for components, data structures, and how suggestions stay fresh.",
        starter_code="# Scratchpad\n# Requirements: top-k suggestions per prefix, p99 < 100ms, freshness?\n"
        "# Data structure for prefix lookup:\n# Where is it stored / how is it served?\n"
        "# How do popularity counts get updated?\n# Trade-offs:\n",
        language="text",
        interviewer_brief="Look for: a TRIE with the top-k completions precomputed at each node (so a "
        "lookup is O(prefix) with no ranking at request time), served from memory and sharded by prefix. "
        "Counts come from an offline/streaming aggregation of the query log, rebuilt periodically — "
        "explicitly a freshness-vs-cost trade-off. Strong candidates raise caching of hot prefixes, "
        "typo tolerance, and personalization as extensions rather than core.",
    ),
    Problem(
        id="design-chat", fmt="design", topic="system_design", title="Real-time chat",
        difficulty="medium", tags=("system-design",),
        prompt="Design a one-to-one chat system: messages delivered in near real time, nothing lost, "
        "in-order per conversation, and readable later on any device. Assume millions of daily users "
        "and users who go offline. Use the scratchpad for the architecture.",
        starter_code="# Scratchpad\n# Requirements: realtime delivery, durability, ordering, offline users\n"
        "# Connection model (poll / long-poll / websocket)?\n# Storage & partitioning:\n"
        "# Ordering guarantee:\n# Trade-offs:\n",
        language="text",
        interviewer_brief="Look for: persistent WebSocket connections to a gateway layer, a session "
        "registry mapping user -> gateway, and messages PERSISTED before acknowledgement. Storage "
        "partitioned by conversation id with a per-conversation sequence number for ordering (wall-clock "
        "time is not enough). Offline users get a queue/inbox drained on reconnect. Good discussions: "
        "at-least-once + idempotent message ids, read receipts, fan-out for group chat.",
    ),
    Problem(
        id="design-news-feed", fmt="design", topic="system_design", title="Social news feed",
        difficulty="medium", tags=("system-design",),
        prompt="Design a social news feed: each user sees recent posts from people they follow, "
        "loading in well under a second. Assume 100 million daily users, and some accounts with tens "
        "of millions of followers. Use the scratchpad for the architecture.",
        starter_code="# Scratchpad\n# Requirements: feed latency, scale, ranking?\n"
        "# Fan-out on WRITE vs fan-out on READ:\n# Celebrity problem:\n# Storage & cache:\n# Trade-offs:\n",
        language="text",
        interviewer_brief="The core trade-off is fan-out on WRITE (precompute each follower's feed — "
        "fast reads, brutal for celebrities) vs fan-out on READ (merge at query time — cheap writes, "
        "slow reads). The expected answer is HYBRID: push for normal users, pull for celebrity accounts, "
        "merged at read. Feeds cached in memory, capped at N entries. Push on: the celebrity case "
        "explicitly, and ranking vs pure reverse-chronological.",
    ),
    Problem(
        id="design-distributed-cache", fmt="design", topic="system_design", title="Distributed cache",
        difficulty="hard", tags=("system-design",),
        prompt="Design a distributed in-memory cache (a Redis-like service) used by many app servers: "
        "get/set by key, millions of operations per second, and it must survive a node dying. Use the "
        "scratchpad for partitioning, eviction, and failure handling.",
        starter_code="# Scratchpad\n# Requirements: ops/sec, latency, what happens on node loss\n"
        "# Partitioning scheme:\n# Eviction policy:\n# Replication / failover:\n"
        "# Hot key handling:\n# Trade-offs:\n",
        language="text",
        interviewer_brief="Key ideas: CONSISTENT HASHING (with virtual nodes) so losing a node "
        "reshuffles only its share, not everything — that's the main thing to test. Then: LRU eviction "
        "with a TTL, primary/replica per shard with failover, and cache-aside vs write-through. Strong "
        "candidates raise hot keys (replicate or client-side cache), thundering herd on expiry "
        "(request coalescing / jittered TTLs), and that a cache miss must be correct, just slower.",
    ),
    Problem(
        id="design-ride-dispatch", fmt="design", topic="system_design", title="Ride matching backend",
        difficulty="hard", tags=("system-design",),
        prompt="Design the matching backend for a ride-hailing app: drivers stream their location "
        "continuously, riders request a ride, and you must match each rider to a nearby driver within "
        "a few seconds at city scale. Use the scratchpad for the architecture.",
        starter_code="# Scratchpad\n# Requirements: location update rate, match latency, scale per city\n"
        "# How are driver locations indexed for 'nearby' queries?\n# Matching algorithm:\n"
        "# Avoiding double-assignment:\n# Trade-offs:\n",
        language="text",
        interviewer_brief="The crux is the geospatial index: geohash / S2 / quadtree cells so 'drivers "
        "near me' is a lookup of a few cells rather than a scan. Location writes are huge but "
        "low-value — keep them in memory, partitioned by cell, not in a durable DB on the hot path. "
        "Matching must avoid assigning one driver twice: a lock/atomic state transition per driver. "
        "Good extensions: ETA-based rather than straight-line ranking, and surge as a separate concern.",
    ),
]

# ---- BANK 3: the high-frequency canonical set --------------------------------
# These are the problems that show up on every "top interview questions" list
# (Top-Interview-150, Striver's SDE sheet, Blind-75 and friends). The TASKS are
# long-standing and industry-standard; the statements and solution notes here are
# written from scratch, since those sites' wording and editorials are theirs.
# Deliberately no dynamic programming — excluded on request.
_BANK3 = [
    # ---------------- two pointers ----------------
    Problem(
        id="valid-palindrome", fmt="solve", topic="dsa", title="Valid palindrome",
        difficulty="easy", tags=("two-pointers", "string"),
        prompt="Decide whether a string reads the same forwards and backwards, ignoring case and "
        "any character that isn't a letter or digit.\n\n"
        "Example: \"A man, a plan, a canal: Panama\" -> True. \"race a car\" -> False.",
        starter_code="def is_palindrome(s: str) -> bool:\n    pass\n", language="python",
        interviewer_brief="Two pointers from both ends, skipping non-alphanumerics, comparing "
        "lowercased. O(n) time, O(1) space. The lazy answer builds a cleaned string and compares to "
        "its reverse — correct but O(n) space; ask them to do it in place. Edge cases: empty string, "
        "all punctuation.",
    ),
    Problem(
        id="two-sum-sorted", fmt="solve", topic="dsa", title="Two sum in a sorted array",
        difficulty="medium", tags=("two-pointers", "array"),
        prompt="Given a list sorted in ascending order and a target, return the 1-indexed positions "
        "of the two numbers that add to the target. Use O(1) extra space.\n\n"
        "Example: [2, 7, 11, 15], target 9 -> [1, 2].",
        starter_code="def two_sum_sorted(nums: list[int], target: int) -> list[int]:\n    pass\n",
        language="python",
        interviewer_brief="Two pointers at both ends: sum too small -> move left in, too big -> move "
        "right in. O(n) time, O(1) space. The point of the exercise is that SORTEDNESS removes the "
        "need for the hash map. Ask why moving the pointer can never skip the answer — that's the "
        "invariant they should be able to argue.",
    ),
    Problem(
        id="three-sum", fmt="solve", topic="dsa", title="Three numbers that sum to zero",
        difficulty="medium", tags=("two-pointers", "array", "sorting"),
        prompt="Given a list of integers, return all unique triplets that sum to 0. No duplicate "
        "triplets in the output.\n\nExample: [-1, 0, 1, 2, -1, -4] -> [[-1, -1, 2], [-1, 0, 1]].",
        starter_code="def three_sum(nums: list[int]) -> list[list[int]]:\n    pass\n", language="python",
        interviewer_brief="Sort, then for each index run the sorted two-pointer scan on the rest: "
        "O(n^2). The hard part is DE-DUPLICATION — skip repeated values at the fixed index and after "
        "each match, which is exactly where most candidates fumble. Using a set of tuples works but "
        "is a code smell; push for the skip-duplicates version.",
    ),
    Problem(
        id="container-water", fmt="solve", topic="dsa", title="Container with most water",
        difficulty="medium", tags=("two-pointers", "array", "greedy"),
        prompt="Each number in a list is the height of a vertical line. Pick two lines that with the "
        "x-axis hold the most water — area is the shorter height times the distance between them.\n\n"
        "Example: [1, 8, 6, 2, 5, 4, 8, 3, 7] -> 49.",
        starter_code="def max_area(heights: list[int]) -> int:\n    pass\n", language="python",
        interviewer_brief="Two pointers at the ends; always move the SHORTER line inward. O(n). The "
        "insight to draw out: moving the taller line can never help, because width shrinks and the "
        "height is still capped by the shorter one — so nothing is lost by discarding it. If they "
        "propose the O(n^2) double loop, ask what they can prove about the shorter side.",
    ),
    Problem(
        id="trapping-rain", fmt="solve", topic="dsa", title="Trapping rain water",
        difficulty="hard", tags=("two-pointers", "array"),
        prompt="Given heights of bars of width 1, compute how much rain water is trapped between "
        "them.\n\nExample: [0,1,0,2,1,0,1,3,2,1,2,1] -> 6.",
        starter_code="def trap(heights: list[int]) -> int:\n    pass\n", language="python",
        interviewer_brief="Water above bar i = min(maxLeft, maxRight) - height[i]. Naive is O(n^2); "
        "prefix/suffix max arrays give O(n) time and O(n) space; the two-pointer version is O(n) time "
        "and O(1) space — move whichever side has the smaller running max, because that side's max is "
        "the binding constraint. Getting them from the per-bar formula to the two-pointer proof is the "
        "whole interview.",
    ),
    # ---------------- arrays / hashing ----------------
    Problem(
        id="valid-anagram", fmt="solve", topic="dsa", title="Valid anagram",
        difficulty="easy", tags=("hashing", "string"),
        prompt="Decide whether two strings are anagrams — same characters, same counts, any order.\n\n"
        "Example: \"anagram\" and \"nagaram\" -> True. \"rat\" and \"car\" -> False.",
        starter_code="def is_anagram(s: str, t: str) -> bool:\n    pass\n", language="python",
        interviewer_brief="Length check, then compare character counts (a dict or a 26-slot array). "
        "O(n). Sorting both is O(n log n) and acceptable but weaker. Follow ups: unicode (a fixed "
        "26-array breaks), and doing it with one dict by incrementing for s and decrementing for t.",
    ),
    Problem(
        id="best-time-stock", fmt="solve", topic="dsa", title="Best time to buy and sell",
        difficulty="easy", tags=("array", "greedy"),
        prompt="Given daily prices, return the maximum profit from buying on one day and selling on a "
        "LATER day. If no profit is possible, return 0.\n\n"
        "Example: [7, 1, 5, 3, 6, 4] -> 5 (buy at 1, sell at 6).",
        starter_code="def max_profit(prices: list[int]) -> int:\n    pass\n", language="python",
        interviewer_brief="One pass tracking the minimum price so far and the best profit against it. "
        "O(n) time, O(1) space. The ordering constraint (must sell after buying) is what rules out "
        "max - min. Watch the all-decreasing case -> 0, not a negative number.",
    ),
    Problem(
        id="majority-element", fmt="solve", topic="dsa", title="Majority element",
        difficulty="easy", tags=("array", "hashing"),
        prompt="Given a list where one value appears more than half the time, return that value.\n\n"
        "Example: [2, 2, 1, 1, 1, 2, 2] -> 2.",
        starter_code="def majority(nums: list[int]) -> int:\n    pass\n", language="python",
        interviewer_brief="A counting dict is O(n) time / O(n) space and perfectly fine. The elegant "
        "answer is Boyer-Moore voting: keep a candidate and a count, increment on a match, decrement "
        "otherwise, reset the candidate at zero — O(1) space. Ask WHY it works: every non-majority "
        "element can cancel at most one majority element, and there are more than n/2 of those.",
    ),
    Problem(
        id="longest-consecutive", fmt="solve", topic="dsa", title="Longest consecutive sequence",
        difficulty="medium", tags=("hashing", "array"),
        prompt="Given an unsorted list of integers, return the length of the longest run of "
        "consecutive numbers (they need not be adjacent in the list). Aim for O(n).\n\n"
        "Example: [100, 4, 200, 1, 3, 2] -> 4 (the run 1, 2, 3, 4).",
        starter_code="def longest_consecutive(nums: list[int]) -> int:\n    pass\n", language="python",
        interviewer_brief="Put everything in a set; for each value, only start counting if value - 1 "
        "is NOT in the set (i.e. it's the start of a run), then walk upward. That start-check is what "
        "keeps it O(n) overall despite the inner loop — without it you rescan runs and it degrades. "
        "Sorting is O(n log n) and misses the point of the question.",
    ),
    Problem(
        id="merge-sorted-array", fmt="solve", topic="dsa", title="Merge two sorted arrays in place",
        difficulty="easy", tags=("two-pointers", "array"),
        prompt="Given a sorted list `a` with extra space at the end and a sorted list `b`, merge b "
        "into a so a stays sorted. Do it in place.\n\n"
        "Example: a = [1, 2, 3, 0, 0, 0] (3 real values), b = [2, 5, 6] -> [1, 2, 2, 3, 5, 6].",
        starter_code="def merge(a: list[int], m: int, b: list[int], n: int) -> None:\n    pass\n",
        language="python",
        interviewer_brief="Fill from the BACK: pointers at the last real elements of a and b, writing "
        "the larger into the end of a. Going forward would overwrite unread values in a — that's the "
        "whole trick. O(m+n), O(1) space. Edge case: b exhausted early (a's prefix is already correct), "
        "and a exhausted early (must keep draining b).",
    ),
    # ---------------- stack ----------------
    Problem(
        id="eval-rpn", fmt="solve", topic="dsa", title="Evaluate reverse Polish notation",
        difficulty="medium", tags=("stack",),
        prompt="Evaluate an expression in reverse Polish (postfix) notation, given as a list of "
        "tokens. Operators are + - * / and division truncates toward zero.\n\n"
        "Example: [\"2\", \"1\", \"+\", \"3\", \"*\"] -> 9.",
        starter_code="def eval_rpn(tokens: list[str]) -> int:\n    pass\n", language="python",
        interviewer_brief="Stack: push numbers, on an operator pop TWO and push the result. O(n). Two "
        "traps: operand ORDER matters for - and / (second pop is the left operand), and Python's // "
        "floors toward negative infinity, so -7//2 is -4 not -3 — need int(a/b) for truncation. "
        "Negative-number tokens also break a naive isdigit() check.",
    ),
    Problem(
        id="daily-temperatures", fmt="solve", topic="dsa", title="Days until a warmer day",
        difficulty="medium", tags=("stack", "monotonic-stack", "array"),
        prompt="For each day's temperature, return how many days you'd wait for a warmer one, or 0 if "
        "it never gets warmer.\n\nExample: [73, 74, 75, 71, 69, 72, 76, 73] -> [1, 1, 4, 2, 1, 1, 0, 0].",
        starter_code="def daily_temperatures(temps: list[int]) -> list[int]:\n    pass\n", language="python",
        interviewer_brief="Monotonic decreasing stack of INDICES: for each day, pop every index whose "
        "temperature is lower and fill in the gap. O(n) — each index is pushed and popped once. The "
        "brute force is O(n^2). If they've not seen monotonic stacks, walk them to it by asking what "
        "information about earlier days is still 'waiting' to be resolved.",
    ),
    Problem(
        id="largest-rectangle", fmt="solve", topic="dsa", title="Largest rectangle in a histogram",
        difficulty="hard", tags=("stack", "monotonic-stack"),
        prompt="Given bar heights of width 1, find the area of the largest rectangle that fits inside "
        "the histogram.\n\nExample: [2, 1, 5, 6, 2, 3] -> 10 (heights 5 and 6, width 2).",
        starter_code="def largest_rectangle(heights: list[int]) -> int:\n    pass\n", language="python",
        interviewer_brief="For each bar, the biggest rectangle using it as the height extends to the "
        "first shorter bar on each side. A monotonic increasing stack finds both boundaries in one "
        "pass: when popping, the new index is the right boundary and the element below is the left. "
        "O(n). Getting the width arithmetic right at pops (and flushing the stack at the end) is the "
        "hard part — a sentinel 0 appended to the input simplifies it a lot.",
    ),
    # ---------------- linked list ----------------
    Problem(
        id="merge-two-lists", fmt="solve", topic="dsa", title="Merge two sorted linked lists",
        difficulty="easy", tags=("linked-list",),
        prompt="Merge two sorted linked lists into one sorted list and return its head. Nodes have "
        ".val and .next.\n\nExample: 1->2->4 and 1->3->4 becomes 1->1->2->3->4->4.",
        starter_code="class Node:\n    def __init__(self, val, nxt=None):\n        self.val, self.next = val, nxt\n\n"
        "def merge_two(a, b):\n    pass\n", language="python",
        interviewer_brief="A DUMMY head node removes every special case around choosing the first "
        "element — that's the technique to teach here. Walk both lists appending the smaller, then "
        "attach whichever list remains (don't loop it node by node). O(n+m), O(1) extra.",
    ),
    Problem(
        id="linked-list-cycle", fmt="solve", topic="dsa", title="Detect a cycle in a linked list",
        difficulty="easy", tags=("linked-list", "two-pointers"),
        prompt="Decide whether a singly linked list contains a cycle, using O(1) extra space.\n\n"
        "Example: 3->2->0->-4 where -4 points back to 2 -> True.",
        starter_code="class Node:\n    def __init__(self, val, nxt=None):\n        self.val, self.next = val, nxt\n\n"
        "def has_cycle(head) -> bool:\n    pass\n", language="python",
        interviewer_brief="Floyd's tortoise and hare: slow moves 1, fast moves 2; they meet iff there's "
        "a cycle. O(n) time, O(1) space. A visited set works but is O(n) space — the constraint is the "
        "point. Ask why they must meet (the gap closes by one each step inside the loop), and the "
        "follow-up: find the cycle's START (reset one pointer to head, advance both by one).",
    ),
    Problem(
        id="remove-nth-node", fmt="solve", topic="dsa", title="Remove the Nth node from the end",
        difficulty="medium", tags=("linked-list", "two-pointers"),
        prompt="Remove the n-th node counting from the END of a singly linked list and return the "
        "head. Do it in one pass.\n\nExample: 1->2->3->4->5, n = 2 -> 1->2->3->5.",
        starter_code="class Node:\n    def __init__(self, val, nxt=None):\n        self.val, self.next = val, nxt\n\n"
        "def remove_nth(head, n: int):\n    pass\n", language="python",
        interviewer_brief="Two pointers n apart: advance the lead pointer n steps, then move both "
        "until it hits the end — the trailing pointer is now just before the target. A dummy head "
        "handles the case where the node to remove IS the head, which is the bug most candidates hit. "
        "One pass, O(1) space.",
    ),
    Problem(
        id="lru-cache", fmt="solve", topic="dsa", title="LRU cache",
        difficulty="medium", tags=("design", "hashing", "linked-list"),
        prompt="Implement a cache with a fixed capacity supporting get(key) and put(key, value), both "
        "in O(1). When it's full, evict the least recently used entry. A get counts as a use.\n\n"
        "Example: capacity 2; put(1,1); put(2,2); get(1) -> 1; put(3,3) evicts key 2.",
        starter_code="class LRUCache:\n    def __init__(self, capacity: int):\n        pass\n\n"
        "    def get(self, key: int) -> int:\n        pass\n\n"
        "    def put(self, key: int, value: int) -> None:\n        pass\n", language="python",
        interviewer_brief="Hash map (key -> node) plus a doubly linked list ordered by recency: map "
        "gives O(1) lookup, the list gives O(1) move-to-front and O(1) eviction at the tail. Both "
        "halves are needed — ask why a list alone or a dict alone fails. Sentinel head/tail nodes kill "
        "the edge cases. In Python, OrderedDict.move_to_end is the legitimate shortcut; make them "
        "explain what it's doing underneath.",
    ),
    # ---------------- trees ----------------
    Problem(
        id="invert-tree", fmt="solve", topic="dsa", title="Invert a binary tree",
        difficulty="easy", tags=("tree", "recursion"),
        prompt="Mirror a binary tree — swap every node's left and right children — and return the "
        "root.\n\nExample: root 4 with children 2 and 7 becomes root 4 with children 7 and 2 "
        "(recursively all the way down).",
        starter_code="class Node:\n    def __init__(self, val, left=None, right=None):\n"
        "        self.val, self.left, self.right = val, left, right\n\ndef invert(root):\n    pass\n",
        language="python",
        interviewer_brief="Swap the children, recurse both sides, return root. O(n). Iterative BFS/DFS "
        "with an explicit stack is equally valid — worth asking for if they want to avoid recursion "
        "depth on a skewed tree. Base case: None root returns None.",
    ),
    Problem(
        id="max-depth-tree", fmt="solve", topic="dsa", title="Maximum depth of a binary tree",
        difficulty="easy", tags=("tree", "recursion", "dfs"),
        prompt="Return the number of nodes along the longest path from the root down to a leaf.\n\n"
        "Example: root 3 with children 9 and 20, and 20 has children 15 and 7 -> 3.",
        starter_code="class Node:\n    def __init__(self, val, left=None, right=None):\n"
        "        self.val, self.left, self.right = val, left, right\n\ndef max_depth(root) -> int:\n    pass\n",
        language="python",
        interviewer_brief="1 + max(depth(left), depth(right)), with None -> 0. O(n). Use it as a warm-up "
        "then escalate: iterative level-order counting levels, and the space difference (recursion is "
        "O(height), which is O(n) on a degenerate tree).",
    ),
    Problem(
        id="lca-bst", fmt="solve", topic="dsa", title="Lowest common ancestor in a BST",
        difficulty="medium", tags=("tree", "binary-search"),
        prompt="Given a binary SEARCH tree and two nodes' values, return the value of their lowest "
        "common ancestor (a node may be its own ancestor).\n\n"
        "Example: BST rooted at 6 with 2 and 8 as children; LCA of 2 and 8 -> 6; LCA of 2 and 4 -> 2.",
        starter_code="class Node:\n    def __init__(self, val, left=None, right=None):\n"
        "        self.val, self.left, self.right = val, left, right\n\ndef lca(root, p: int, q: int) -> int:\n    pass\n",
        language="python",
        interviewer_brief="Exploit the BST ordering: if both values are smaller than the node go left, "
        "if both larger go right, otherwise this node is the split point and therefore the LCA. O(height), "
        "O(1) iteratively. If they give the general-binary-tree recursion, that's correct but ask what "
        "the BST property buys them.",
    ),
    Problem(
        id="right-side-view", fmt="solve", topic="dsa", title="Binary tree right side view",
        difficulty="medium", tags=("tree", "bfs"),
        prompt="Return the values visible when looking at a binary tree from the right — the last node "
        "of each level, top to bottom.\n\nExample: root 1, children 2 and 3, and 2 has right child 5 "
        "-> [1, 3, 5].",
        starter_code="class Node:\n    def __init__(self, val, left=None, right=None):\n"
        "        self.val, self.left, self.right = val, left, right\n\ndef right_view(root) -> list[int]:\n    pass\n",
        language="python",
        interviewer_brief="Level-order BFS, taking the LAST node of each level (snapshot the queue size "
        "per level). O(n). The subtlety the example is built to catch: the rightmost visible node isn't "
        "always in the right subtree — level 3 here comes from node 2's child. A right-first DFS "
        "recording the first node seen at each new depth also works.",
    ),
    Problem(
        id="serialize-tree", fmt="solve", topic="dsa", title="Serialize and deserialize a binary tree",
        difficulty="hard", tags=("tree", "design", "dfs"),
        prompt="Write two functions: one turning a binary tree into a string, and one rebuilding the "
        "identical tree from that string. Any format is fine as long as it round-trips.\n\n"
        "Example: a tree with root 1 and children 2 and 3 must serialize and come back structurally identical.",
        starter_code="class Node:\n    def __init__(self, val, left=None, right=None):\n"
        "        self.val, self.left, self.right = val, left, right\n\n"
        "def serialize(root) -> str:\n    pass\n\ndef deserialize(data: str):\n    pass\n",
        language="python",
        interviewer_brief="Preorder DFS writing a sentinel (like '#') for None, comma-joined; rebuild by "
        "consuming that stream with an iterator in the same order. The key realisation: the NULL markers "
        "are what make the structure unambiguous — a plain preorder of values alone cannot be inverted. "
        "O(n) both ways. Ask why inorder+preorder pairs are the usual alternative and why that needs "
        "distinct values.",
    ),
    # ---------------- graphs ----------------
    Problem(
        id="clone-graph", fmt="solve", topic="dsa", title="Clone a graph",
        difficulty="medium", tags=("graph", "dfs", "hashing"),
        prompt="Deep-copy a connected undirected graph. Each node has a value and a list of "
        "neighbours; the copy must share no objects with the original.\n\n"
        "Example: node 1 <-> node 2 must produce two brand-new nodes wired the same way.",
        starter_code="class Node:\n    def __init__(self, val, neighbors=None):\n"
        "        self.val, self.neighbors = val, neighbors or []\n\ndef clone(node):\n    pass\n",
        language="python",
        interviewer_brief="DFS/BFS with a map from ORIGINAL node -> COPY. Create the copy on first "
        "visit and register it in the map BEFORE recursing into neighbours — otherwise a cycle recurses "
        "forever. That map is doing double duty as the visited set. O(V+E).",
    ),
    Problem(
        id="rotting-oranges", fmt="solve", topic="dsa", title="Rotting oranges",
        difficulty="medium", tags=("graph", "bfs", "matrix"),
        prompt="In a grid, 0 is empty, 1 is a fresh orange, 2 is rotten. Each minute, a rotten orange "
        "rots any fresh orange directly adjacent to it. Return the minutes until none are fresh, or -1 "
        "if some can never rot.\n\nExample: [[2,1,1],[1,1,0],[0,1,1]] -> 4.",
        starter_code="def oranges_rotting(grid: list[list[int]]) -> int:\n    pass\n", language="python",
        interviewer_brief="MULTI-SOURCE BFS: seed the queue with every rotten orange at once and expand "
        "level by level, each level being a minute. Count fresh oranges up front and decrement; if any "
        "remain at the end return -1. The mistake is BFS from one source at a time — the simultaneity is "
        "the whole point. Edge case: zero fresh oranges -> 0 minutes, not -1.",
    ),
    # ---------------- binary search ----------------
    Problem(
        id="search-rotated", fmt="solve", topic="dsa", title="Search in a rotated sorted array",
        difficulty="medium", tags=("binary-search", "array"),
        prompt="A sorted array was rotated at some unknown pivot. Find the index of a target in "
        "O(log n), or -1.\n\nExample: [4, 5, 6, 7, 0, 1, 2], target 0 -> 4.",
        starter_code="def search_rotated(nums: list[int], target: int) -> int:\n    pass\n", language="python",
        interviewer_brief="Binary search with an extra step: at each mid, ONE half is guaranteed sorted "
        "(compare nums[lo] to nums[mid]). Check whether the target lies inside that sorted half; if so "
        "search it, else search the other. O(log n). Candidates who try to find the pivot first and then "
        "binary search are also correct — that's two passes but fine. Edge case: duplicates break the "
        "guarantee, worth raising.",
    ),
    Problem(
        id="find-min-rotated", fmt="solve", topic="dsa", title="Minimum in a rotated sorted array",
        difficulty="medium", tags=("binary-search", "array"),
        prompt="Find the smallest element of a sorted array that was rotated at an unknown pivot, in "
        "O(log n).\n\nExample: [3, 4, 5, 1, 2] -> 1.",
        starter_code="def find_min(nums: list[int]) -> int:\n    pass\n", language="python",
        interviewer_brief="Binary search comparing nums[mid] to nums[hi]: if nums[mid] > nums[hi] the "
        "minimum is strictly right of mid (lo = mid+1), else it's at mid or left (hi = mid). Loop while "
        "lo < hi and return nums[lo]. Comparing against nums[lo] instead is the classic wrong turn — it "
        "can't distinguish a non-rotated array. O(log n).",
    ),
    Problem(
        id="koko-bananas", fmt="solve", topic="dsa", title="Minimum eating speed",
        difficulty="medium", tags=("binary-search", "greedy"),
        prompt="Given piles of bananas and h hours, find the smallest integer eating speed k such that "
        "eating ceil(pile/k) hours per pile finishes every pile within h hours.\n\n"
        "Example: piles = [3, 6, 7, 11], h = 8 -> 4.",
        starter_code="import math\n\ndef min_speed(piles: list[int], h: int) -> int:\n    pass\n",
        language="python",
        interviewer_brief="Binary search on the ANSWER, not on the array — the search space is 1..max(pile), "
        "and 'can we finish at speed k' is monotonic (true for all larger k). O(n log(max)). Recognising "
        "that the answer space is what's sorted here is the skill being tested. Watch the ceiling division "
        "and that the upper bound must be max(piles), not the sum.",
    ),
    Problem(
        id="median-two-sorted", fmt="solve", topic="dsa", title="Median of two sorted arrays",
        difficulty="hard", tags=("binary-search", "array"),
        prompt="Given two sorted arrays, return the median of their combined elements in O(log(min(m,n))).\n\n"
        "Example: [1, 3] and [2] -> 2.0. [1, 2] and [3, 4] -> 2.5.",
        starter_code="def median_two(a: list[int], b: list[int]) -> float:\n    pass\n", language="python",
        interviewer_brief="Binary search the PARTITION point on the shorter array so that everything "
        "left of the combined cut is <= everything right of it, then the median comes from the boundary "
        "values. Use +/- infinity for the empty side. Genuinely hard — merging is O(m+n) and is a fine "
        "first answer; only push for the log solution once they've stated the simpler one and the "
        "even/odd-length handling.",
    ),
    # ---------------- intervals / greedy / heap ----------------
    Problem(
        id="insert-interval", fmt="solve", topic="dsa", title="Insert into sorted intervals",
        difficulty="medium", tags=("intervals", "array"),
        prompt="Given non-overlapping intervals sorted by start, insert a new interval and merge where "
        "needed. Return the result.\n\n"
        "Example: [[1,3],[6,9]], new [2,5] -> [[1,5],[6,9]].",
        starter_code="def insert_interval(intervals: list[list[int]], new: list[int]) -> list[list[int]]:\n    pass\n",
        language="python",
        interviewer_brief="Three phases in one pass: append everything ending before the new interval "
        "starts, absorb everything that overlaps (min of starts, max of ends), append the rest. O(n) — "
        "no sorting needed since the input is already sorted, which is the difference from plain merge-"
        "intervals. Boundary case: intervals that merely touch ([1,2] and [2,3]) — decide and state it.",
    ),
    Problem(
        id="non-overlapping", fmt="solve", topic="dsa", title="Fewest intervals to remove",
        difficulty="medium", tags=("intervals", "greedy", "sorting"),
        prompt="Given a list of intervals, return the minimum number you must remove so that the rest "
        "don't overlap.\n\nExample: [[1,2],[2,3],[3,4],[1,3]] -> 1 (remove [1,3]).",
        starter_code="def erase_overlap(intervals: list[list[int]]) -> int:\n    pass\n", language="python",
        interviewer_brief="Greedy: sort by END time, keep an interval whenever it starts at/after the "
        "last kept end, count the rest as removals. O(n log n). Sorting by START is the natural-but-wrong "
        "instinct — ask them to find a counterexample (one long interval swallowing several short ones). "
        "This is the classic activity-selection argument.",
    ),
    Problem(
        id="task-scheduler", fmt="solve", topic="dsa", title="Task scheduler with cooldown",
        difficulty="medium", tags=("heap", "greedy", "hashing"),
        prompt="Given task labels and a cooldown n, identical tasks must be at least n intervals apart. "
        "Return the minimum number of intervals (including idles) to finish everything.\n\n"
        "Example: tasks [A, A, A, B, B, B], n = 2 -> 8.",
        starter_code="def least_interval(tasks: list[str], n: int) -> int:\n    pass\n", language="python",
        interviewer_brief="The answer is driven by the MOST frequent task: (maxCount - 1) * (n + 1) + "
        "(number of tasks tied at maxCount), floored at len(tasks) — that floor matters when there are so "
        "many distinct tasks that no idling is needed. A max-heap simulation also works and is easier to "
        "reason about out loud. Push them to explain the formula, not just recite it.",
    ),
    Problem(
        id="gas-station", fmt="solve", topic="dsa", title="Gas station circuit",
        difficulty="medium", tags=("greedy", "array"),
        prompt="Around a circular route, station i gives gas[i] fuel and it costs cost[i] to reach the "
        "next. Return the starting index that lets you complete the circuit, or -1.\n\n"
        "Example: gas = [1,2,3,4,5], cost = [3,4,5,1,2] -> 3.",
        starter_code="def can_complete(gas: list[int], cost: list[int]) -> int:\n    pass\n", language="python",
        interviewer_brief="Two facts: if total gas < total cost the answer is -1; otherwise a solution is "
        "unique-ish and findable in one pass — track a running tank, and whenever it goes negative, no "
        "start in that stretch works, so reset the candidate start to the next index and zero the tank. "
        "O(n). Ask them to justify why discarding the whole stretch is safe — that's the real question.",
    ),
    # ---------------- backtracking ----------------
    Problem(
        id="permutations", fmt="solve", topic="dsa", title="All permutations",
        difficulty="medium", tags=("backtracking", "recursion"),
        prompt="Given distinct integers, return all possible orderings.\n\n"
        "Example: [1, 2, 3] -> the 6 permutations [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1].",
        starter_code="def permutations(nums: list[int]) -> list[list[int]]:\n    pass\n", language="python",
        interviewer_brief="Backtracking: pick an unused element, recurse, un-pick. n! results so O(n·n!) "
        "is optimal. Same trap as subsets — append a COPY of the working list, not the list itself. "
        "Follow up: handling duplicate inputs (sort, then skip a value equal to its predecessor when the "
        "predecessor is unused).",
    ),
    Problem(
        id="combination-sum", fmt="solve", topic="dsa", title="Combination sum",
        difficulty="medium", tags=("backtracking", "recursion"),
        prompt="Given distinct positive candidates and a target, return all unique combinations summing "
        "to the target. A number may be reused any number of times.\n\n"
        "Example: candidates [2, 3, 6, 7], target 7 -> [[2, 2, 3], [7]].",
        starter_code="def combination_sum(candidates: list[int], target: int) -> list[list[int]]:\n    pass\n",
        language="python",
        interviewer_brief="Backtracking where recursing on the SAME index allows reuse, and never going "
        "backwards prevents permutations of the same multiset being counted twice. Prune when the "
        "remainder goes negative. The uniqueness requirement is handled structurally by that "
        "never-go-back rule — candidates who dedupe with a set afterwards have missed it.",
    ),
    Problem(
        id="n-queens", fmt="solve", topic="dsa", title="N-Queens",
        difficulty="hard", tags=("backtracking", "recursion"),
        prompt="Place n queens on an n x n board so none attack each other (no shared row, column or "
        "diagonal). Return the number of distinct solutions.\n\n"
        "Example: n = 4 -> 2. n = 8 -> 92.",
        starter_code="def n_queens(n: int) -> int:\n    pass\n", language="python",
        interviewer_brief="Place one queen per row, recursing row by row. The trick that makes it fast "
        "is O(1) conflict checks with three sets: columns, and the two diagonal keys (row - col) and "
        "(row + col). Un-mark on backtrack. Rescanning the board each placement works but is much slower "
        "— ask them for a constant-time check and let them derive the diagonal identities.",
    ),
    # ---------------- hard graph ----------------
    Problem(
        id="word-ladder", fmt="solve", topic="dsa", title="Word ladder",
        difficulty="hard", tags=("graph", "bfs", "string"),
        prompt="Given a start word, an end word, and a word list, return the number of words in the "
        "shortest transformation sequence where each step changes exactly one letter and every "
        "intermediate word is in the list. Return 0 if impossible.\n\n"
        "Example: \"hit\" -> \"cog\" with [\"hot\",\"dot\",\"dog\",\"lot\",\"log\",\"cog\"] -> 5 "
        "(hit, hot, dot, dog, cog).",
        starter_code="def ladder_length(begin: str, end: str, words: list[str]) -> int:\n    pass\n",
        language="python",
        interviewer_brief="It's a shortest path on an implicit graph, so BFS (not DFS). The key move is "
        "generating neighbours by wildcarding each position ('h*t') rather than comparing against every "
        "word — that's O(L·26) per word instead of O(N·L). Use a set for the word list and mark visited "
        "on enqueue. Strong candidates raise bidirectional BFS from both ends as the real optimisation.",
    ),
    Problem(
        id="alien-dictionary", fmt="solve", topic="dsa", title="Alien dictionary order",
        difficulty="hard", tags=("graph", "topological-sort", "string"),
        prompt="Given a list of words sorted according to an unknown alphabet's ordering, derive a "
        "possible ordering of that alphabet's letters. Return \"\" if the input is contradictory.\n\n"
        "Example: [\"wrt\", \"wrf\", \"er\", \"ett\", \"rftt\"] -> \"wertf\".",
        starter_code="def alien_order(words: list[str]) -> str:\n    pass\n", language="python",
        interviewer_brief="Compare each ADJACENT pair of words, find the first differing character, and "
        "add an edge — that single pair gives you exactly one ordering fact. Then topologically sort "
        "(Kahn's). Two traps that separate strong candidates: the invalid prefix case ([\"abc\",\"ab\"] "
        "is contradictory), and remembering that letters appearing in no edge still belong in the output. "
        "A cycle means return \"\".",
    ),
]

# ---- BANK 4: canonical patterns that were still missing --------------------
# Standard, widely-asked problems (Kadane, prefix-sum, etc.) — original write-ups.
_BANK4 = [
    Problem(
        id="max-subarray", fmt="solve", topic="dsa",
        title="Maximum subarray sum", difficulty="medium",
        prompt=(
            "Given an integer array, return the largest sum of any contiguous subarray "
            "(the subarray must contain at least one number).\n\n"
            "Example: [-2, 1, -3, 4, -1, 2, 1, -5, 4] -> 6, from the subarray [4, -1, 2, 1].\n\n"
            "Can you do it in a single pass?"
        ),
        starter_code="def max_subarray(nums):\n    # return the largest contiguous subarray sum\n    pass\n",
        language="python",
        interviewer_brief=(
            "Kadane's algorithm: best_here = max(x, best_here + x) as you scan; the answer is the "
            "max best_here seen. O(n) time, O(1) space. The trap is the all-negative case — seed "
            "with the first element (or -inf), never 0, or you'll wrongly return 0."
        ),
    ),
    Problem(
        id="subarray-sum-k", fmt="solve", topic="dsa",
        title="Subarrays summing to K", difficulty="medium",
        prompt=(
            "Given an integer array and an integer k, return how many contiguous subarrays "
            "sum to exactly k.\n\n"
            "Example: nums = [1, 1, 1], k = 2 -> 2.   nums = [1, 2, 3], k = 3 -> 2.\n\n"
            "Brute force is O(n^2). Can you do better?"
        ),
        starter_code="def subarray_sum(nums, k):\n    # count contiguous subarrays whose sum is k\n    pass\n",
        language="python",
        interviewer_brief=(
            "Running prefix sum + a hashmap of prefix-sum -> count. For each running sum s, add "
            "count.get(s - k, 0) to the answer, then bump count[s]. Seed count[0] = 1 so subarrays "
            "starting at index 0 are counted. O(n) time and space."
        ),
    ),
    Problem(
        id="add-two-numbers", fmt="solve", topic="dsa",
        title="Add two numbers as linked lists", difficulty="medium",
        prompt=(
            "Two non-empty linked lists represent two non-negative integers, digits stored in "
            "reverse order (ones digit first). Add them and return the sum as a linked list in "
            "the same format.\n\n"
            "Example: (2 -> 4 -> 3) which is 342, plus (5 -> 6 -> 4) which is 465, gives "
            "7 -> 0 -> 8 which is 807."
        ),
        starter_code=(
            "class ListNode:\n"
            "    def __init__(self, val=0, next=None):\n"
            "        self.val = val\n"
            "        self.next = next\n\n"
            "def add_two_numbers(l1, l2):\n"
            "    # return the sum as a reverse-order linked list\n"
            "    pass\n"
        ),
        language="python",
        interviewer_brief=(
            "Walk both lists together with a carry: digit = (a + b + carry) % 10, "
            "carry = (a + b + carry) // 10. Use a dummy head to simplify. After both lists end, "
            "if carry is still 1 append a final node. O(max(m, n)) time."
        ),
    ),
    Problem(
        id="move-zeroes", fmt="solve", topic="dsa",
        title="Move zeroes to the end", difficulty="easy",
        prompt=(
            "Given an integer array, move every 0 to the end while keeping the relative order of "
            "the non-zero elements. Do it in place.\n\n"
            "Example: [0, 1, 0, 3, 12] -> [1, 3, 12, 0, 0]."
        ),
        starter_code="def move_zeroes(nums):\n    # modify nums in place; return nothing\n    pass\n",
        language="python",
        interviewer_brief=(
            "Two pointers: a write index for the next non-zero slot. Scan once, and whenever "
            "nums[i] is non-zero write it to nums[write] and advance write; then fill the tail "
            "from write onward with zeros. O(n) time, O(1) space."
        ),
    ),
]

# ---- company associations ---------------------------------------------------
# Which companies are widely known to favour a given canonical pattern. This is
# public, unownable knowledge (the patterns themselves, not anyone's problem
# text), used only to BIAS which of our own problems we surface when a company is
# picked — a problem can be favoured by several, and untagged problems still show
# up under "Generic" and as fallbacks.
_COMPANY_FAVORITES: dict[str, tuple[str, ...]] = {
    "google": (
        "number-of-islands", "clone-graph", "course-schedule", "word-ladder", "alien-dictionary",
        "rotting-oranges", "word-search", "trapping-rain", "merge-intervals", "insert-interval",
        "non-overlapping", "min-window", "longest-consecutive", "three-sum", "largest-rectangle",
        "lru-cache", "top-k-frequent", "median-stream", "serialize-tree", "spiral-matrix",
        "max-subarray", "subarray-sum-k", "design-typeahead", "design-distributed-cache",
        "design-url-shortener",
    ),
    "meta": (
        "valid-palindrome", "valid-parentheses", "merge-intervals", "min-rooms", "three-sum",
        "group-anagrams", "top-k-frequent", "kth-largest-stream", "lca-bst", "right-side-view",
        "level-order", "validate-bst", "subsets", "product-except-self", "min-stack",
        "remove-nth-node", "add-two-numbers", "subarray-sum-k", "move-zeroes", "number-of-islands",
        "clone-graph", "design-news-feed", "design-chat", "design-typeahead",
    ),
    "openai": (
        "ml-gradient-step", "ml-logistic-predict", "ml-softmax", "ml-kmeans-step", "ml-kmeans-full",
        "ml-roc-auc", "ml-knn", "ml-confusion", "lru-cache", "min-window", "median-stream",
        "design-rate-limiter", "design-distributed-cache", "design-ride-dispatch", "max-subarray",
        "koko-bananas",
    ),
    "anthropic": (
        "ml-precision-recall", "ml-roc-auc", "ml-confusion", "ml-logistic-predict", "ml-knn",
        "ml-softmax", "ml-gradient-step", "validate-bst", "design-rate-limiter", "design-notification",
        "design-distributed-cache", "min-window", "median-stream", "alien-dictionary", "word-ladder",
    ),
}
# invert to per-problem: id -> (company, ...)
_COMPANIES: dict[str, tuple[str, ...]] = {}
for _co, _ids in _COMPANY_FAVORITES.items():
    for _pid in _ids:
        _COMPANIES[_pid] = _COMPANIES.get(_pid, ()) + (_co,)

# Backfill topic tags on the original entries without editing each one.
_TAGS = {
    "first-unique": ("array", "hashing"), "pair-sum": ("array", "hashing"),
    "longest-unique-substring": ("string", "sliding-window"),
    "merge-intervals": ("intervals", "sorting"), "kth-largest-stream": ("heap", "stream"),
    "debug-binary-search": ("binary-search",), "debug-bfs-visited": ("graph", "bfs"),
    "debug-off-by-one": ("math",), "design-url-shortener": ("system-design",),
    "design-rate-limiter": ("system-design",), "design-notification": ("system-design",),
}
import dataclasses as _dc  # noqa: E402

PROBLEMS: list[Problem] = [
    _dc.replace(p, tags=p.tags or _TAGS.get(p.id, ()), companies=_COMPANIES.get(p.id, ()))
    for p in (_SOLVE + _DEBUG + _DESIGN + _MORE + _BANK2 + _BANK3 + _BANK4)
]
_BY_ID: dict[str, Problem] = {p.id: p for p in PROBLEMS}

# ---- mid-interview problem switching ----------------------------------------
# The candidate can ask for a different problem out loud ("next one", "give me a
# medium graph problem", "something harder"). We detect that deterministically and
# swap the problem in the editor — the interviewer never has to obey the request
# by improvising a problem that then doesn't match what's on screen.
# Deliberately CONSERVATIVE: a false positive interrupts the candidate mid-answer
# (observed: "…and move on to the next element" wrongly swapped the problem). Every
# alternative requires an explicit CHANGE intent — a change-word next to
# problem/question/one, or an explicit switch/skip verb — never a bare "move on"
# or a mere mention of "the problem".
# Only QUALIFIER words may sit between "a" and "problem/question". Allowing ANY
# word made "give me a hint about the problem" a false switch; capping the count
# instead broke real requests like "give me a hard binary search problem" (three
# qualifier words). Restricting the vocabulary satisfies both.
_QUAL = (
    r"(?:easy|medium|hard|harder|easier|simple|simpler|tough|tougher|difficult|challenging"
    r"|new|different|another|other|fresh|next|short|quick|real|proper"
    r"|coding|code|programming|theory|theoretical|conceptual|practical"
    r"|ml|machine|learning|system|design|dsa|algorithm|algorithms|data|structure|structures"
    r"|graph|graphs|tree|trees|array|arrays|string|strings|stack|heap|queue|linked|list|lists"
    r"|binary|search|two|pointer|pointers|sliding|window|backtracking|greedy|dynamic"
    r"|interval|intervals|matrix|grid|trie|recursion|recursive|sorting|hashing|hash"
    r"|monotonic|topological|clustering|metric|metrics|regression|classification)"
)
_SWITCH_RE = re.compile(
    r"""(?ix) \b (
        (?:the\ )? next \ (?:QUALIFIER\ ){0,3}? (?:problem|question|one|challenge)
      | (?:a\ |an\ )? (?:another|different|new|other|fresh) \ (?:QUALIFIER\ ){0,3}? (?:problem|question|one|challenge)
      # a request verb -> a/an/another -> (opt. qualifier words) -> problem/question/one:
      | (?: give\ me | can\ (?:we|i) | could\ (?:we|i) | let'?s | we\ can | i\ want | i'?d\ like | i\ would\ like )
          \ (?: (?:to\ )? (?:do|try|have|get|see|solve|attempt|work\ on) \ )?
          (?:a|an|another) \ (?:QUALIFIER\ ){0,4}? (?:problem|question|one|challenge)
      | (?:a|an) \ (?:\w+\ )? (?:harder|easier|tougher|simpler) \ (?:problem|question|one)
      | (?:problem|question|one) \ (?:that'?s\ |that\ is\ )? (?:harder|easier|tougher|simpler)
      # "give me the wrong code", "can you show me some buggy code" — a format
      # request that never says "problem"/"question":
      | (?: give\ me | show\ me | can\ (?:we|i|you) | could\ (?:we|i|you) | let'?s | i\ want | i'?d\ like )
          \ (?:\w+\ ){0,3}? (?:wrong|buggy|broken|incorrect|faulty|bad) \ code
      | (?:debug|debugging) \ (?:question|problem|one|round|exercise)
      | something \ (?:harder|easier|else|different)
      | (?:make\ it|go) \ (?:harder|easier|tougher|simpler)
      # "can you increase the difficulty (of the questions)" — this fell through to
      # the LLM before, which then improvised a theory question instead of switching:
      | (?:increase|raise|bump|turn\ up|step\ up|crank\ up|lower|reduce|decrease)
          \ (?:the\ )? (?:difficulty|level|complexity)
      | (?:harder|tougher|easier|simpler) \ (?:level|questions?)
      | more \ (?:difficult|challenging)
      # "move (on) to ... problem/question/one" — needs the noun, so "move on to the
      # next ELEMENT" (an algorithm step) does NOT match:
      | move \ (?:on\ )? to \ (?:the\ next\ |a\ |an\ |another\ |the\ )? (?:\w+\ ){0,2}? (?:problem|question|one|challenge)
      | (?:let'?s|let\ us|lets|can\ we|could\ we|shall\ we|can\ i) \ move \ on   # explicit control phrase
      | switch \ (?:the\ )? (?:problem|question) | change \ (?:the\ )? (?:problem|question)
      | skip \ (?:this|that|it) (?:\ (?:problem|question|one))?
    ) \b """.replace("QUALIFIER", _QUAL),
    re.VERBOSE,
)
_LEVELS = ["easy", "medium", "hard"]
_TAG_WORDS = {
    # longest/most specific phrases first — the first hit wins
    "two pointer": "two-pointers", "two-pointer": "two-pointers",
    "sliding window": "sliding-window", "binary search": "binary-search",
    "linked list": "linked-list", "linkedlist": "linked-list",
    "monotonic": "monotonic-stack", "backtrack": "backtracking",
    "topological": "topological-sort", "system design": "system-design",
    "array": "array", "string": "string", "hash": "hashing", "graph": "graph", "tree": "tree",
    "heap": "heap", "stack": "stack", "queue": "heap", "trie": "trie", "greedy": "greedy",
    "interval": "intervals", "recursion": "recursion", "matrix": "matrix", "grid": "matrix",
    "sort": "sorting", "clustering": "clustering", "metric": "metrics",
}


def _target_difficulty(text: str, current: "Problem") -> str | None:
    low = text.lower()
    for d in _LEVELS:
        if re.search(rf"\b{d}\b", low):
            return d
    cur = current.difficulty if current.difficulty in _LEVELS else "medium"
    i = _LEVELS.index(cur)
    # Relative requests: step one level up or down from where we are.
    if re.search(
        r"\b(harder|tougher|more difficult|more challenging|difficult|"
        r"(?:increase|raise|bump|turn up|step up|crank up)\s+(?:the\s+)?(?:difficulty|level|complexity))\b",
        low,
    ):
        return _LEVELS[min(i + 1, 2)]
    if re.search(
        r"\b(easier|simpler|"
        r"(?:lower|reduce|decrease)\s+(?:the\s+)?(?:difficulty|level|complexity))\b",
        low,
    ):
        return _LEVELS[max(i - 1, 0)]
    return None


# Which KIND of problem they asked for: "give me one where wrong code is given"
# should hand back a DEBUG problem, not another write-it-from-scratch one.
_FMT_HINTS = (
    ("debug", r"\b(debug(?:ging)?|buggy|(?:find|spot)\s+the\s+(?:bug|issue|error|mistake)|"
              r"wrong\s+code|broken\s+code|incorrect\s+code|faulty|fix\s+(?:the\s+)?code|"
              r"correct\s+the\s+code|what'?s\s+wrong\s+with)\b"),
    ("design", r"\b(system\s+design|design\s+(?:a|an|the)\s+\w+|architecture|scalability)\b"),
    ("solve", r"\b(write\s+(?:the\s+)?code|implement\s+(?:a|an|the)|from\s+scratch|coding\s+problem)\b"),
)
# Asking ABOUT the flow ("are we moving to the next question?") is a question for
# the interviewer to answer — NOT an instruction to switch problems.
_INQUIRY_RE = re.compile(r"\b(?:are|is|was|were|do|does|did)\s+(?:we|you|this|that|it)\b", re.IGNORECASE)
_REQUESTY_RE = re.compile(
    r"\b(give\s+me|can\s+(?:we|i)|could\s+(?:we|i)|let'?s|let\s+us|i\s+want|i'?d\s+like|"
    r"please|switch|skip|move\s+on\s+to|next\s+problem)\b",
    re.IGNORECASE,
)


def _target_fmt(text: str) -> str | None:
    for fmt, rx in _FMT_HINTS:
        if re.search(rx, text, re.IGNORECASE):
            return fmt
    return None


# Switching normally stays inside the round's subject (an ML round shouldn't jump
# to DSA). But if they NAME another subject, honor it. "coding" is deliberately
# absent — in an ML round "a coding question" means an ML coding question.
_TOPIC_HINTS = (
    ("ml", r"\b(ml|machine\s+learning|deep\s+learning|model|classifier|regression)\b"),
    ("system_design", r"\b(system\s+design|architecture|scalability|design\s+(?:a|an|the))\b"),
    ("dsa", r"\b(dsa|algorithms?|data\s+structures?|leetcode)\b"),
)


def _target_topic(text: str) -> str | None:
    for topic, rx in _TOPIC_HINTS:
        if re.search(rx, text, re.IGNORECASE):
            return topic
    return None


def match_switch(
    text: str, current: "Problem | None", exclude: "set[str] | None" = None,
    company: str | None = None,
) -> "Problem | None":
    """If the candidate asked for a different problem, return a NEW one that fits
    their format/difficulty/topic hints; else None. Only for coding rounds.

    `exclude` is the set of problem ids already shown this session — we prefer
    something they haven't seen, so "next question" genuinely advances instead of
    cycling back through the same two problems."""
    if not text or current is None:
        return None
    # A bare noun-phrase answer to "what would you like?" — "a dynamic programming
    # problem, please" — has no request verb, so _SWITCH_RE misses it. Accept it
    # only when the WHOLE utterance is just that phrase (anchored), so a sentence
    # that merely contains "...a graph problem..." mid-answer can never match.
    bare = re.fullmatch(
        r"(?i)\s*(?:i(?:'d| would)? like\s+)?(?:maybe\s+)?(?:a|an|another)\s+"
        rf"(?:{_QUAL}\s+){{0,4}}(?:problem|question|one|challenge)"
        r"[\s.,!]*(?:please|now|next)?[\s.,!]*",
        text,
    )
    if not bare and not _SWITCH_RE.search(text):
        return None
    # Guard: an inquiry about the flow, with no actual request in it, is a question
    # to answer — not a command. ("...or are we moving back to the next question?")
    if _INQUIRY_RE.search(text) and not _REQUESTY_RE.search(text):
        return None
    diff = _target_difficulty(text, current)
    low = text.lower()
    tag = next((t for w, t in _TAG_WORDS.items() if w in low), None)
    want_fmt = _target_fmt(text)
    want_topic = _target_topic(text) or current.topic
    fmts = ["design"] if want_topic == "system_design" else ["solve", "debug"]
    pool = [p for p in PROBLEMS if p.fmt in fmts and p.id != current.id]
    # Stay in the round's subject unless they named a different one explicitly.
    pool = [p for p in pool if p.topic == want_topic] or pool
    # Format is the strongest signal — honor it even if it means ignoring difficulty.
    if want_fmt:
        pool = [p for p in pool if p.fmt == want_fmt] or pool
    # Topic BEFORE difficulty: if they ask for "a hard graph problem" and no hard
    # graph problem exists, a medium GRAPH one serves them better than a hard
    # problem about something else entirely.
    if tag:
        pool = [p for p in pool if tag in p.tags] or pool
    if diff:
        pool = [p for p in pool if p.difficulty == diff] or pool
    # Soft company bias: if a target company was set, prefer the problems it favours,
    # but only when that doesn't empty the pool — an explicit tag/difficulty request
    # still wins.
    company = (company or "").strip()
    if company and company != "generic":
        pool = [p for p in pool if company in p.companies] or pool
    # Last: among everything that matches what they asked for, prefer a problem
    # they haven't already been given. Only recycle once the pool is exhausted.
    seen = set(exclude or ())
    pool = [p for p in pool if p.id not in seen] or pool
    return random.choice(pool) if pool else None


def coding_switch_line(problem: "Problem") -> str:
    """Deterministic spoken line when we swap the problem mid-interview."""
    return (
        f"Sure — here's a new one on your screen: {problem.title}. You've got about "
        f"{problem.budget()} minutes. Take a read and walk me through your approach."
    )


def get_problem(pid: str | None) -> Problem | None:
    return _BY_ID.get((pid or "").strip())


def list_problems(
    fmt: str | None = None, topic: str | None = None, company: str | None = None
) -> list[Problem]:
    out = [p for p in PROBLEMS]
    if fmt:
        out = [p for p in out if p.fmt == fmt]
    if topic:
        out = [p for p in out if p.topic == topic]
    # A company just re-orders (favourites first) — it never hides problems, so the
    # picker still offers everything and untagged rounds are unaffected.
    company = (company or "").strip()
    if company and company != "generic":
        out.sort(key=lambda p: company not in p.companies)
    return list(out)


def random_problem(fmt: str | None = None, topic: str | None = None) -> Problem | None:
    pool = list_problems(fmt, topic)
    return random.choice(pool) if pool else None


def coding_opening(problem: Problem) -> str:
    """A deterministic spoken opening that points the candidate at the on-screen
    problem — the editor already shows the full text, so this just orients them."""
    mins = problem.budget()
    line = {
        "solve": f"On your screen is a problem — {problem.title}. You've got about {mins} minutes. "
        "Take a moment to read it, then walk me through how you're thinking about it.",
        "debug": f"On your screen is some code with a bug in it — {problem.title}. You've got about "
        f"{mins} minutes. Have a read, trace it in your head, and tell me what you find.",
        "design": f"On your screen is a system design prompt — {problem.title}. We've got about "
        f"{mins} minutes. Start by asking any clarifying questions, then talk me through your approach.",
    }.get(problem.fmt)
    return "Hi, thanks for joining. " + (line or f"Let's work through {problem.title}.")


def custom_problem(title: str, prompt: str, starter_code: str = "", fmt: str = "solve") -> Problem:
    """Wrap a user-pasted problem as a Problem (no interviewer brief — the model
    works from the prompt/code alone). fmt decides the language/scratchpad flavor."""
    fmt = fmt if fmt in ("solve", "debug", "design") else "solve"
    return Problem(
        id="custom",
        fmt=fmt,
        topic="system_design" if fmt == "design" else "dsa",
        title=(title or "Your problem").strip(),
        difficulty="custom",
        prompt=(prompt or "").strip(),
        starter_code=starter_code or "",
        language="text" if fmt == "design" else "python",
        interviewer_brief="",
    )
