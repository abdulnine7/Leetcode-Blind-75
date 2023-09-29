# Leetcode-Blind-75
Clear, concise explanation and optimal solutions to every single challenge on the LeetCode Blind 75 sheet.


## Arrays & Hashing
### 1. Contains Duplicate 
Given an integer array `nums`, return `true` if any value appears **at least twice** in the array, and return `false` if every element is distinct.

##### Solution: 
- Using `hashset` to keep track of unique elements.
```python
def containsDuplicate(self, nums: List[int]) -> bool:
    hashset = set()

    for n in nums:
        if n in hashset:
            return True
        hashset.add(n)
    return False
```
- Inside the loop, it checks whether the current element `n` is already in the `hashset`.
- If `n` is found in the `hashset`, that means it's a duplicate element, so the function returns `True`, indicating that the input array contains duplicates.
___

### 2. Valid Anagram

Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`, and `false` otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

##### Solution: 
- Using two dictionaries with count of each characters.
```python
def isAnagram(self, s: str, t: str) -> bool:
    if len(s) != len(t):
        return False

    countS, countT = {}, {}

    for i in range(len(s)):
        countS[s[i]] = 1 + countS.get(s[i], 0)
        countT[t[i]] = 1 + countT.get(t[i], 0)
    return countS == countT
```
- It initializes two dictionaries, `countS` and `countT`, to keep track of character counts in the strings.

- It iterates through both strings simultaneously using a `for` loop.
___

### 3. Two Sum
Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

##### Solution: 
- Using dictionary called `prevMap` to store previously encountered values. 
```python
def twoSum(self, nums: List[int], target: int) -> List[int]:
    prevMap = {}  # val -> index

    for i, val in enumerate(nums):
        diff = target - val
        if diff in prevMap:
            return [prevMap[diff], i]
        prevMap[val] = i
```
- Iterate and checks if the `diff` value exists in the `prevMap`. If it does, that means a complementary value has been seen before.
-  If the `diff` is not found in `prevMap`, it adds the current element `val` to `prevMap` with its index `i`
___

### 4. Group Anagrams
Given an array of strings strs, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

##### Solution: 
- Using a dictionary `ans` to store all the strings with the `key` as tuple of frequency count of characters of the string.
```python
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    ans = collections.defaultdict(list)

    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord("a")] += 1
        ans[tuple(count)].append(s)
    return ans.values()
```
- `ans` dictionary will contain different group of anagram for each key.
- Time complexity `O(n*k)`; `n` is length of `strs` and `k` is length of largest string.
___

### 5. Top K Frequent Elements
Given an integer array `nums` and an integer `k`, return the `k` most frequent elements. You may return the answer in any order.

##### Solution: 

- This can easily done by counting the frequency in a dictionary `d` and then adding the elements to a max heap of size `k`. Time complexity will be, `O(n) + O(d*log(k))`.
- Here we will use different approach for the solution to solve it in `O(n)` time.
- We create a frequency dictionary `count` and we also create a array of list `freq` that contains all the numbers of certain frequency at the `index` that is equal to their frequency.

```python
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    count = {}
    freq = [[] for i in range(len(nums) + 1)]

    for n in nums:
        count[n] = 1 + count.get(n, 0)

    for n, c in count.items():
        freq[c].append(n)

    res = []
    #Traversing from the last to find top k frequent elements
    for i in range(len(freq) - 1, 0, -1): 
        for n in freq[i]:
            res.append(n)
            if len(res) == k:
                return res
```
- When we have the list of numbers that have frequency equal to the `index` in the `freq` array of lists.
- So we traverse freq array from the last to get top `k` frequent elements.
___

### 6. Product of Array Except Self
Given an integer array `nums`, return an array answer such that `answer[i]` is equal to the product of all the elements of `nums` except `nums[i]`.

The product of any prefix or suffix of `nums` is **guaranteed** to fit in a **32-bit** integer.

You must write an algorithm that runs in `O(n)` time and without using the division operation.

##### Solution: 
- First calculate the product of each element from `0` to `i-1` and store it at index `i` in the `res` array.
- We traverse from the end and keep track of product from `n-1` to `i+1` while we update the `i`<sup>th</sup> position in the `res` array.
- Finally we have product of all the elements except self in the `res` array.
```python
def productExceptSelf(self, nums: List[int]) -> List[int]:
    res = [1] * (len(nums))

    #Calculate and store product from 0 to i-1 at index i
    for i in range(1, len(nums)):
        res[i] = res[i-1] * nums[i-1]

    postfix = 1
    for i in range(len(nums) - 1, -1, -1):
        # postfix will have product from n-1 to i+1
        # res[i] will have product from 0 to i-1
        res[i] *= postfix 
        postfix *= nums[i]

    return res
```
___

### 7. Encode and Decode Strings

Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.

Please implement `encode` and `decode`

```
Input: ["lint","code","love","you"]
Output: ["lint","code","love","you"]

Explanation:
One possible encode method is: "lint#code#love#you"
```

##### Solution: 
- Very easy need no explaination. Bye.

```python
class Solution:
    def encode(self, strs):
        return "#".join(strs)

    def decode(self, str):
        return str.split("#")
```
___

### 8. Longest Consecutive Sequence

Given an unsorted array of integers `nums`, return the length of the *longest consecutive elements sequence*.

You must write an algorithm that runs in `O(n)` time.

##### Solution: 
- In this we have to count the length of longest consecutive sequence. in `O(n)` time.
- This can be easily done by sorting the array and counting but it will take `O(nlogn)` time.
- In this solution, we convert the `nums` in `hashset` and try to find the start of any sequence.
- If `nums[i]-1` does not exist in `hashset` that means `nums[i]` is an start of sqeuence.
- Once we find a start we check how many elements are there in that particular sequence, and count the length.
```python
def longestConsecutive(self, nums: List[int]) -> int:
    numSet = set(nums)
    longest = 0

    for x in numSet:
        # check if its the start of a sequence
        if (x - 1) not in numSet:
            length = 1
            # Count how many consecutive exist in sequence 
            while (x + length) in numSet:
                length += 1
            # Then take max
            longest = max(length, longest)
    return longest
```
- Worst case we visit each element twice, so `O(n)`

___

## Two Pointers

### 9.

A phrase is a **palindrome** if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string `s`, return `true` if it is a **palindrome**, or `false` otherwise.

#### Solution:
- In this we visit the string form both side and check if characters are same.
- Here we ignore character `c` that are not alphabet or numeric. Using function `alphanum(c)`.

```python
def isPalindrome(self, s: str) -> bool:
    l, r = 0, len(s) - 1
    while l < r:
        while l < r and not self.alphanum(s[l]):
            l += 1
        while l < r and not self.alphanum(s[r]):
            r -= 1
        if s[l].lower() != s[r].lower():
            return False
        l += 1
        r -= 1
    return True

# Could write own alpha-numeric function
def alphanum(self, c):
    return (
        ord("A") <= ord(c) <= ord("Z")
        or ord("a") <= ord(c) <= ord("z")
        or ord("0") <= ord(c) <= ord("9")
    )
```
- Time complexity, `O(n)`

---

### 10. 3Sum

Given an integer array nums, return all the triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

Notice that the solution set must not contain duplicate triplets.

#### Solution:
- As array has both `-ve` and `+ve` numbers, firstly we sort the array. 
- If array size <3 means no triplets. If sorted array 1<sup>st</sup> element is +ve means no sum of 3 numbers = 0.
- The basic thinking logic for this is: **Fix any one number in sorted array and find the other two numbers after it**. The other two numbers can be easily found using two pointers (as array is sorted) and two numbers should have sum = -1*(fixed number).
- Search between two pointers, just similiar to binary search. `threeSum = num[i] + num[left] + num[right]`.
    - If `threeSum` is -ve, means, we need more +ve numbers to make it 0, increament low (left++).
    - If `threeSum` is +ve, means, we need more -ve numbers to make it 0, decreament high (right--).
    - If `threeSum` is 0, that means we have found the required triplet, push it in result array.
- Now again, to avoid duplicate triplets, we have to navigate to last occurences of `num[left]` and `num[right]` respectively.
```python
def threeSum(self, nums: List[int]) -> List[List[int]]:
    res = []
    nums.sort()

    if len(nums) < 3:  # Base case 1
        return []
    if nums[0] > 0:  # Base case 2
        return []

    for i, x in enumerate(nums): # Traversing the array to fix the number.
        # If the fix number is +ve, stop, we can't make it zero by searching after it.
        if x > 0:
            break

        if i > 0 and x == nums[i - 1]:
            continue

        l, r = i + 1, len(nums) - 1 # Make two pointers, left and right
        while l < r:    # Search between two pointers, similar to binary search.
            threeSum = x + nums[l] + nums[r]
            if threeSum > 0:
                r -= 1
            elif threeSum < 0:
                l += 1
            else:
                res.append([x, nums[l], nums[r]])
                l += 1
                r -= 1

                # Navigate to the last occurrences of nums[l] and nums[r] to avoid duplicates
                while nums[l] == nums[l - 1] and l < r:
                    l += 1
                while nums[r] == nums[r + 1] and l < r:
                    r -= 1
                    
    return res
```
- Time complexity O(n<sup>2</sup>)

---

### 11. Container With Most Water


You are given an integer array `height` of length `n`. There are `n` vertical lines drawn such that the two endpoints of the `i`<sup>th</sup> line are `(i, 0)` and `(i, height[i])`.

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the *maximum amount* of water a container can store.

**Notice** that you may not slant the container.

<img src="https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/17/question_11.jpg"  width="500">

#### Solution:
- First we set left and right pointer at the extream ends. 
- And we measure water level with minimum of two heights, `height[left]` and `height[right]` with width, `right-left`
- In order to have more height of the container we move left or right pointer. Whichever is shorter in height we change that pointer to next.

```python
def maxArea(self, height: List[int]) -> int:
    l, r = 0, len(height) - 1
    water = 0

    while l < r:
        # Calculate the water between the current left and right pointers and take max
        water = max(water, (r - l) * min(height[l], height[r]))

        # Move the pointer that points to the shorter height towards the center,
        # as moving the taller height wouldn't increase the water.
        if height[l] < height[r]:
            l += 1
        elif height[r] <= height[l]:
            r -= 1
        
    return water
```
- Time complexity, `O(n)`

---

## Sliding Window

## Stack

## Binary Search

## Linked List

## Trees

## Tries

## Heap / Priority Queue

## Backtracking

## Graphs

## Advanced Graphs

## 1-D Dynamic Programming

## 2-D Dynamic Programming

## Greedy

## Intervals

## Math & Geometry

## Bit Manipulation