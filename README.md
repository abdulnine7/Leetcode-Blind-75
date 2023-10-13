# Leetcode-Blind-75
Clear, concise explanation and optimal solutions to every single challenge on the LeetCode Blind 75 sheet.


## Arrays & Hashing
### [217. Contains Duplicate](https://leetcode.com/problems/contains-duplicate/) <sup style="color:#2DB55D">Easy</sup>
Given an integer array `nums`, return `true` if any value appears **at least twice** in the array, and return `false` if every element is distinct.

####  Solution: 
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

### [242. Valid Anagram](https://leetcode.com/problems/valid-anagram/) <sup style="color:#2DB55D">Easy</sup>

Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`, and `false` otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

####  Solution: 
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

### [1. Two Sum](https://leetcode.com/problems/two-sum/) <sup style="color:#2DB55D">Easy</sup>
Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

####  Solution: 
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

### [49. Group Anagrams](https://leetcode.com/problems/group-anagrams/) <sup style="color:#FFB801">Medium</sup>
Given an array of strings strs, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

####  Solution: 
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

### [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/) <sup style="color:#FFB801">Medium</sup>
Given an integer array `nums` and an integer `k`, return the `k` most frequent elements. You may return the answer in any order.

####  Solution: 

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

### [238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/) <sup style="color:#FFB801">Medium</sup>
Given an integer array `nums`, return an array answer such that `answer[i]` is equal to the product of all the elements of `nums` except `nums[i]`.

The product of any prefix or suffix of `nums` is **guaranteed** to fit in a **32-bit** integer.

You must write an algorithm that runs in `O(n)` time and without using the division operation.

####  Solution: 
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

### [271. Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/) <sup style="color:#FFB801">Medium</sup>

Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.

Please implement `encode` and `decode`

```
Input: ["lint","code","love","you"]
Output: ["lint","code","love","you"]

Explanation:
One possible encode method is: "lint#code#love#you"
```

####  Solution: 
- Very easy need no explaination. Bye.

```python
class Solution:
    def encode(self, strs):
        return "#".join(strs)

    def decode(self, str):
        return str.split("#")
```
___

### [128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/) <sup style="color:#FFB801">Medium</sup>

Given an unsorted array of integers `nums`, return the length of the *longest consecutive elements sequence*.

You must write an algorithm that runs in `O(n)` time.

####  Solution: 
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

### [125. Valid Palindrome](https://leetcode.com/problems/valid-palindrome/) <sup style="color:#2DB55D">Easy</sup>

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
```
```python
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

### [15. 3Sum](https://leetcode.com/problems/3sum/) <sup style="color:#FFB801">Medium</sup>

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
```
```python
    for i, x in enumerate(nums): # Traversing the array to fix the number.
        # If the fix number is +ve, stop, we can't make it zero by searching after it.
        if x > 0:
            break

        # If number is getting repeated, ignore the lower loop and continue.
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

### [11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/) <sup style="color:#FFB801">Medium</sup>


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

### [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) <sup style="color:#2DB55D">Easy</sup>

You are given an array `prices` where `prices[i]` is the price of a given stock on the `i`<sup>th</sup> day.

You want to maximize your profit by choosing a **single** day to buy one stock and choosing a **different day in the future** to sell that stock.

Return the *maximum profit you can achieve from this transaction*. If you cannot achieve any profit, return `0`.

Example: \
Input: prices = [7,1,5,3,6,4] \
Output: 5 \
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6)

#### Solution:
- We use sliding window approach with two pointer, `left` for buying and `right` for selling.
- We fixate the buying day (`left`) and iterate over prices to find a selling day (`right`).
- If we come across a selling day (`right`) that has price lower than buying day (`left`), we update our buying day (`left`) to this index in the array `prices`. As we need to buy at lowest possible price, for maximizing the profit.
- In each iteration we also check if the profit is bigger than earlier profit by taking `max`.
```python
def maxProfit(self, prices):
    left = 0
    max_profit = 0

    for right in range(1, len(prices)):
        # This means loss, we found a price lower than earlier buying price
        # so we set this as our buying price, ie we update left to this location
        if prices[right] < prices[left]: 
            left = right
    
        currentProfit = prices[right] - prices[left] #our current Profit
        max_profit = max(currentProfit,max_profit)

    return max_profit
```
- Time complexity, `O(n)`
---

### [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) <sup style="color:#FFB801">Medium</sup>

Given a string `s`, find the length of the **longest substring** without repeating characters.

Example:\
Input: s = "abcabcbb"\
Output: 3\
Explanation: The answer is "abc", with the length of 3.

#### Solution:

- In this again we use 2 pointers and a character set to have/record unique elements. 
- While traversing the `right` pointer when we come across a element that is already in set, we pop elements of set `charSet` and move `left` pointer untill the element is popped out. And our window left to right contain all unique elements.
- At each iteration we check and record if length `left` to `right` is **maximum**.
```python
def lengthOfLongestSubstring(self, s: str) -> int:
    charSet = set()
    left = 0
    result = 0

    for right in range(len(s)):

        while s[right] in charSet:
            charSet.remove(s[left])
            left += 1

        charSet.add(s[right])
        result = max(result, right - left + 1)
    return result
```
- Time, `O(n)`
---
### [424. Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/) <sup style="color:#FFB801">Medium</sup>

You are given a string `s` and an integer `k`. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most `k` times.

Return the length of the *longest substring containing the same letter* you can get after performing the above operations.

Example: \
Input: s = "ABAB", k = 2 \
Output: 4 \
Explanation: Replace the two 'A's with two 'B's or vice versa.


#### Solution:

- In this again we use 2 pointers and a *frequency dictionary* (`count`) to keep count of characters in current window.
- We track the window with `left` and `right` pointers, and we also keep record `maxf` ie *maximum frequency of any character in the current window*.
- In each iteration the size of window is increased as `right` is increased, But if voilation happend in that iteration `left` is increased so size is reduced by 1 and window size remain same.
- In this we wont update the `maxf` when we reduce the window size, as we are not tracking `maxf` belong to which character's frequency.
- Hence by the end we will have window of maximum length that can be made by ignoring `k` characters.
```python
def characterReplacement(self, s: str, k: int) -> int:
    count = {}
    left = 0
    maxf = 0

    for right in range(len(s)):
        
        # Update count of character in frequency counter in current window
        count[s[right]] = 1 + count.get(s[right], 0)
        maxf = max(maxf, count[s[right]])

        # If voilation detected, ie. the length of the current window minus the maximum 
        # frequency exceeds 'k', it means we need to shrink the window from the left
        if (right - left + 1) - maxf > k:
            count[s[left]] -= 1 
            left += 1

    return (right - left + 1)
```
- Time complexity, `O(n)`
---
### [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/) <sup style="color:#FF2D55">Hard</sup>

Given two strings `s` and `t` of lengths `m` and `n` respectively, return the **minimum window
substring** of `s` such that every character in `t` (**including duplicates**) is included in the window. If there is no such substring, `return` the empty string `""`.

The testcases will be generated such that the answer is **unique**.

#### Solution:

- In this we maintain 2 dictionaries, 2 variables `have` and `need`
    - `countT` to keep frequency count of chatacters of `t`
    - `window` to keep frequenc count of characters of current window
    - `have` is count of characters that is there in window that are in `t`
    - `need` need is required number of characters we need in the window that are in `t`
- We also maintain `res` for result string (start, end) and `resLen` for length of result string
- Rest of the explaination in comments:

```python
def minWindow(self, s: str, t: str) -> str:
    if t == "":
        return ""

    countT, window = {}, {}

    # We get frequency count of t in countT
    for c in t:
        countT[c] = 1 + countT.get(c, 0)

    # Iniitialize variables needed
    have, need = 0, len(countT)
    res, resLen = [-1, -1], float("infinity")
    left = 0

    # Iterate  the right pointer 
    for right in range(len(s)):
        # Add character count to window as they reveal
        c = s[right]
        window[c] = 1 + window.get(c, 0)

        # if a character that is in t matches count in both dictionary
        # that means we have that character, hence we update have 
        if c in countT and window[c] == countT[c]:
            have += 1

        # 1. When we have all the characters we need we update result
        # 2. we have while loop here instead of IF because to collect all the characters
        #    and make have==need we collected more more characters in window than needed
        #    in minimum so we pop characters on left and update results while, have==need
        while have == need:
            if (right - left + 1) < resLen:
                res = [left, right]
                resLen = right - left + 1
        
            # pop from the left of our window and update have if we pop a character from
            # the window that is also in t. Also update left in each pop.
            window[s[left]] -= 1
            if s[left] in countT and window[s[left]] < countT[s[left]]:
                have -= 1
            left += 1
    
    # Get he start and end from res and return the substring
    left, right = res
    return s[left : right + 1] if resLen != float("infinity") else ""
```
- Time complexity, `O(m+n)`, here `m` and `n` are size of `s` and `t`.
---

## Stack

### [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) <sup style="color:#2DB55D">Easy</sup>

Given a string `s` containing just the characters `'('`,` ')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.

#### Solution:

- Here we create a dictionary (`Map`) of parenthesis, we use *closing parenthesis as key* and its corresponding *opening parenthesis as value*. And a list `stack` to track opening parenthesis.
- We then iterate string `s` and for each character `c`.
- If we come come across a opening parenthesis we add it to the `stack`.
- If we come across closing parenthesis we check stack top if we have its corresponding opening parenthesis, if not we return `False`.

```python
def isValid(self, s: str) -> bool:
    Map = {")": "(", "]": "[", "}": "{"}
    stack = []

    for c in s:
        # character c is opening parenthesis
        if c not in Map:
            stack.append(c)
            continue

        # character c is closing parenthesis, hence we check if at the
        # top of stack we have its corresponding opening parenthesis
        if not stack or stack[-1] != Map[c]:
            return False

        stack.pop()

    return not stack
```
- Time complexity, `O(n)`
---

## Binary Search

Note: 
- In Binary Search when we find the middle value between the left and right bounds (their average) we can equivalently do: `mid = left + (right - left) // 2`, if we are concerned left + right would cause overflow (which would occur **if we are searching a massive array using a language like Java or C** that has fixed size integer types).
- In python we can simply do  `mid = (left + right ) // 2`

### [153. Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/) <sup style="color:#FFB801">Medium</sup>

Suppose an array of length `n` sorted in ascending order is rotated between `1` and `n` times. For example, the array `nums = [0,1,2,4,5,6,7]` might become:

- `[4,5,6,7,0,1,2]` if it was rotated `4` times.
- `[0,1,2,4,5,6,7]` if it was rotated `7` times.

Notice that rotating an array `[a[0], a[1], a[2], ..., a[n-1]]` 1 time results in the array `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]`.

Given the sorted rotated array `nums` of **unique elements**, return the *minimum element* of this array.

You must write an algorithm that runs in `O(log n)` time.

#### Solution:

- There is a point `pivot` in array which is the minimum of array.
- In this solution the main idea for our checks is to converge the `start` and `end` bounds on the start of the `pivot/minimum`, and never disqualify the index for a possible minimum value.
- `if nums[mid] > nums[end]` is True, 
    - We KNOW that the `pivot/minimum` value must have occurred somewhere to the right of `mid`, which is why the values wrapped around and became smaller, hence we update, `start = mid + 1` 
    - Eg. `array=[3,4,5,6,7,8,9,1,2]` where `array[mid]=7` and `array[end]=2`
- else we have `if nums[mid] <= nums[end]`,
    - We KNOW the `pivot/minimum` must be at `mid` or to the left of `mid`. we know the numbers continued increasing to the right of `mid`, so they never reached the pivot and wrapped around. Therefore, we know the pivot must be at `index <= mid`. Hence we update, `end = mid` .
    - Eg. `array=[8,9,1,2,3,4,5,6,7]` where `array[mid]=3` and `array[end]=7`
- At the end we will have `start = end` which will be our pivot index. 

```python
def findMin(self, nums: List[int]) -> int:
    start, end = 0, len(nums) - 1
    
    while start  <  end :
        mid = (start + end ) // 2

        # right has the pivot 
        if nums[mid] > nums[end]:
            start = mid + 1
            
        # left has the pivot 
        else:
            end = mid 
            
    return nums[start]
```
- Time complexity, `O(log n)`
---

### [33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/) <sup style="color:#FFB801">Medium</sup>

There is an integer array `nums` sorted in ascending order (with **distinct** values).

Prior to being passed to your function, `nums` is **possibly rotated** at an unknown pivot index `k` `(1 <= k < nums.length)` such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (**0-indexed**). For example, `[0,1,2,4,5,6,7]` might be rotated at pivot index `3` and become `[4,5,6,7,0,1,2]`.

Given the array `nums` **after** the possible rotation and an integer `target`, return the *index of* `target` if it is in `nums`, or `-1` if it is not in `nums`.

You must write an algorithm with **O(log n)** runtime complexity.

#### Solution:

- Just like every binary search we take the `mid` and check if it is the target. If not then we move further.
- If we have `nums[start] <= nums[mid]` means left portion of array is sorted.
    - Eg. array=[<u>**3,4,5,6,7**</u>,8,9,1,2] where `array[start]=3` and `array[mid]=7`
    - So we now know ***if target is greater than mid element or if it is smaller than start element*** that means it should be on right side of mid element. Hence `start = mid + 1` .
    - Otherwise, it should be on left side of mid element. Hence we do `end = mid - 1`
- If we have `nums[start] > nums[mid]` means right portion of array is sorted.
    - Eg. array=[8,9,1,2,<u>**3,4,5,6,7**</u>] where `array[start]=8` and `array[mid]=3`
    - So we now know ***if target is less than mid element or if it is greater than end element*** that means it should be on left side of mid element. Hence `end = mid - 1` .
    - Otherwise, it should be on right side of mid element. Hence we do `start = mid + 1`
    

```python
def search(self, nums: List[int], target: int) -> int:
    start, end = 0, len(nums) - 1

    while start <= end:
        mid = (start + end) // 2

        # target is found
        if target == nums[mid]:
            return mid

        # left sorted portion
        if nums[start] <= nums[mid]:
            if target > nums[mid] or target < nums[start]:
                start = mid + 1
            else:
                end = mid - 1

        # right sorted portion
        else:
            if target < nums[mid] or target > nums[end]:
                end = mid - 1
            else:
                start = mid + 1
    return -1
```
- Time complexity, `O(log n)`
---

## Linked List

### [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) <sup style="color:#2DB55D">Easy</sup>

Given the `head` of a singly linked list, reverse the list, and return the *reversed* list.

#### Solution:

- In this we use the `curr` node as pointer to iterate over the linked list.
- In each iteration we, put the current node behind previous node untill there are no current nodes. In these following steps, 
    - We record the node next to current node and keep it in `temp`.
    - Then we set the next of current to the previous recorded node.
    - And mark the current node previous for next iteration.
    - And then we move the current to the next node that is pointed by `temp`.
- In the end we will have variable `prev/curr` pointing to the last node of linked list, which is now reversed so the last node is new head of linked list, hence we return `prev`

```python
def reverseList(self, head: ListNode) -> ListNode:
    prev, curr = None, head

    while curr:
        temp = curr.next
        curr.next = prev
        prev = curr
        curr = temp
    return prev
```
- Time complexity `O(n)` as we visit each node only once.
---

### [21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/) <sup style="color:#2DB55D">Easy</sup>

You are given the *heads* of two sorted linked lists `list1` and `list2`.

Merge the two lists into one **sorted** list. The list should be made by splicing together the nodes of the first two lists.

Return the *head of the merged linked list*.

#### Solution:

- We create a new node a set two different pointers to it. One to keep pointing to it and one to iterate over and add elements to new list it.
- Then we iterate over two lists `list1` and `list2` and add elements to our list in *sorted order*.
- Once any of the list is exhausted we add the rest of the other list to our list.
- Then we return next of the node that we created at the start, so that we only have elements from `list1` and `list2` in our new sorted list.

```python
def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
    head = curr = ListNode()

    while list1 and list2:
        if list1.val < list2.val:
            curr.next = list1
            list1 = list1.next
        else:
            curr.next = list2
            list2 = list2.next
        node = node.next

    # Add the remaining of the list that still has elements
    curr.next = list1 or list2

    return head.next
```
- Time complexity, `O(n)` where `n` is length of shorter linked list among the two.
---

### [143. Reorder List](https://leetcode.com/problems/reorder-list/) <sup style="color:#FFB801">Medium</sup>

You are given the head of a singly linked-list. The list can be represented as:

L<sub>0</sub> → L<sub>1</sub> → L<sub>2</sub> → ..... → L<sub>n-1</sub> → L<sub>n</sub>

Reorder the list to be on the following form:

L<sub>0</sub> → L<sub>n</sub> → L<sub>1</sub> → L<sub>n-1</sub> → L<sub>2</sub> → L<sub>n-2</sub> → ....

You may not modify the values in the list's nodes. Only nodes themselves may be changed.

Example:\
<img src="https://assets.leetcode.com/uploads/2021/03/09/reorder2-linked-list.jpg"  width="500">

#### Solution:

- We craft a very simple 4 step solution for this,
    - Find the middle of the linked list *(Using Fast and slow method)*
    - Set the `first` & `second` pointer to first and second half.
    - We reverse the second half. *(Like in above problem 206)*
    - Finally, we merge the two halves. 

```python
def reorderList(self, head: ListNode) -> None:
    # find middle
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # set the first and second pointer to first and second half
    second = slow.next
    slow.next = None # We do this to break the 1st half connection with 2nd
    first = head

    # reverse second half
    prev = None
    while second:
        tmp = second.next
        second.next = prev
        prev = second
        second = tmp
    seecond = prev


    # merge two halfs first and second
    # we will have more eleeent in first in case of odd number of elements
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first, second = tmp1, tmp2
```
- Time complexity, `O(n)`
---

### [19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/) <sup style="color:#FFB801">Medium</sup>

Given the `head` of a *linked list*, remove the **k<sup>th</sup>** node from the end of the list and return its head.

#### Solution:

- Let's say the *n = length of linked list*.
- In this we use two different pointers to head (`fast` and `slow`) node then,and we traverse `k` steps with `fast`.
- Now once `fast` is `k` steps ahead, we traverse both `slow` and `fast` untill fast reaches end node of linked list. That means fast is on last node, `n-1` .
- Hence fast has traversed, `n-1` and slow has traversed, `n-1-k`.
- This means that, `slow` is 1 step away from k<sup>th</sup> node from the end.
- So we do `slow.next = slow.next.next` to delete the k<sup>th</sup> node from the end.


```python
def removeNthFromEnd(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
    fast, slow = head, head

    # traverse k steps with fast
    for _ in range(k): 
        fast = fast.next

    # if fast is null, means k = length of list, so we delete head,
    # as the kth node from end will be 1st node in the list.
    if not fast: 
        return head.next

    while fast.next: 
        fast, slow = fast.next, slow.next

    # delete the node
    slow.next = slow.next.next

    return head
```
- Time complexity, `O(n)`
---

### [141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)  <sup style="color:#2DB55D">Easy</sup>

Given `head`, the head of a* linked list*, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to. **Note that `pos` is not passed as a parameter.**

Return `true` if there is a cycle in the linked list. Otherwise, return `false`.

#### Solution:

- Here we use the **Floyd’s Cycle-Finding Algorithm (tortoise and the hare algorithm)**
    - Traverse linked list using two pointers.
    - Move one pointer(`slow`) by one and another pointer(`fast`) by two.
    - If these pointers meet at the same node then there is a loop. If pointers do not meet then linked list doesn’t have a loop.


```python
def hasCycle(self, head: ListNode) -> bool:
    slow, fast = head, head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```
- Time complexity, `O(n)`
---

### [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/) <sup style="color:#FF2D55">Hard</sup>

You are given an array of `k` linked-lists `lists`, each linked-list is sorted in ascending order.

*Merge all the linked-lists into one sorted linked-list and return it.*
#### Solution:

- In this we merge consecutive the lists in `lists` into and create a new `mergedLists` array containing half of the `lists`.
- Then we set `lists = mergedLists`, and then do the same operation. We repeat untill the size of `lists` is reduced to `1` and all the `lists` are merged.
- Then we return the final list, `lists[0]` that is remaining in `lists`.

```python

def mergeKLists(self, lists: List[ListNode]) -> ListNode:
    if not lists or len(lists) == 0:
        return None

    while len(lists) > 1:
        mergedLists = []
        for i in range(0, len(lists), 2):
            l1 = lists[i]
            l2 = lists[i + 1] if (i + 1) < len(lists) else None
            mergedLists.append(self.mergeList(l1, l2))
        lists = mergedLists
    return lists[0]
```
```python
def mergeList(self, l1, l2):
    dummy = ListNode()
    tail = dummy

    while l1 and l2:
        if l1.val < l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next
    if l1:
        tail.next = l1
    if l2:
        tail.next = l2
    return dummy.next
```
- Time complexity, `O(n)`
---

## Trees

### [226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)  <sup style="color:#2DB55D">Easy</sup>

Given the `root` of a binary tree, invert the tree, and return its `root`.

#### Solution:

- In this we have to change tree to its mirror image. That means all the *left nodes will be right* **and** *right nodes will be left*.
- To do this we call `invertTree` function to interchange `left` and `right` branches and inside the function we make recursive call on `left` and `right` subtree, so that their childrens are interchanged too.

```python
def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return None
    
    # swap the children
    root.left, root.right = root.right, root.left
    
    # make 2 recursive calls
    self.invertTree(root.left)
    self.invertTree(root.right)
    return root
```
- Time complexity, `O(n)` where n is number of nodes in the tree.
---

### [104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/) <sup style="color:#2DB55D">Easy</sup>

Given the `root` of a binary tree, `return` its **maximum depth**.

A binary tree's **maximum depth** is the number of nodes along the longest path from the root node down to the farthest leaf node.


#### Solution:

1. Depth First Search (DFS) - Recursive
    - This one is fairly simple, we recursively check of `left` or `right` subtree is longer and we return `1 + longer of the two branches` to the level above.

2. Depth First Search (DFS) - Iterative
    - Here we uses a `stack` to keep track of the nodes in the tree. We starts by adding the `root` node and its depth (which is 1) to the `stack`.
    - Then we go in while loop, In each iteration of the loop, we pop the top node and its depth from the stack. if depth is greater the `result` we update `result`.
    - We then adds the left and right child nodes of the current node to the stack, along with their respective depths (which are the depth of the current node plus 1).
    
3. Breadth First Search (BFS)
    - In BFS we track the *levels* deep we go to iterate over all the elements.
    - In each level we increase the `level` counter.
    - At the end we return the `level` variable.

```python
# RECURSIVE DFS
def maxDepth(self, root: TreeNode) -> int:
    if not root:
        return 0

    # Get the depth of left and right subtree
    left, right = self.maxDepth(root.left), self.maxDepth(root.right)

    return 1 + max(left, right)
```

```python
# ITERATIVE DFS
def maxDepth(self, root: TreeNode) -> int:
    stack = [[root, 1]]
    result = 0

    while stack:
        node, depth = stack.pop()

        if node:
            result = max(result, depth)
            stack.append([node.left, depth + 1])
            stack.append([node.right, depth + 1])
    return result
```

```python
# BFS
def maxDepth(self, root: TreeNode) -> int:
    q = deque()
    if root:
        q.append(root)

    level = 0

    while q:

        # Visiting all the nodes at a certain level and adding their
        # children to queue to visit in next level
        for i in range(len(q)):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        level += 1
    return level

```
- Time complexity, `O(n)` in all three cases, as we are visiting each node only once. `n` is number of nodes in the tree.
---

### [100. Same Tree](https://leetcode.com/problems/same-tree/) <sup style="color:#2DB55D">Easy</sup>

Given the `roots` of two binary trees `p` and `q`, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

#### Solution:

- In this we check if roots are same of `p` and `q`.
- Then we recursively check of left and right branches are same for `p` and `q`.

```python
def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
    
    if not p and not q:
        return True

    if p and q and p.val == q.val:
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
    else:
        return False
```
- Time complexity, `O(n)`
---

### [572. Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/) <sup style="color:#2DB55D">Easy</sup>

Given the roots of two binary trees `root` and `subRoot`, return `true` if there is a subtree of root with the same structure and node values of `subRoot` and `false` otherwise.

A subtree of a binary tree `tree` is a tree that consists of a node in `tree` and all of this node's descendants. The tree `tree` could also be considered as a subtree of itself.

<img src="https://assets.leetcode.com/uploads/2021/04/28/subtree1-tree.jpg"  width="350">

#### Solution:

- In this we chech if `s` and `t` are same tree using the function `isSameTree(p,q)` that we made in Problem No. 100.
- If `s` and `t` are not same we check if `s.left` is same as `t` **or** `s.right` is same as `t`.
- Like this we recursively check if any branch in the tree `s` is same as tree `t`.

```python
def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:

    if not t:
        return True
    if not s:
        return False

    if self.isSameTree(s, t): # We use this function from above problem #100
        return True

    return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)
```
- Time complexity, `O(m⋅n)` , where `m` and `n` are the number of nodes in `s` and `t`, respectively. 

#### Note: 
-  We can improve the time complexity of this code by using a more efficient algorithm. One way to do this is to use a pre-order traversal of both trees and compare the resulting strings.
- The time complexity to check if the string for `t` is a substring of the string for `s` is `O(m⋅n)`, where `m` and `n` are the lengths of the strings for `s` and `t`, respectively. However, in practice, the time complexity is often much lower than `O(m⋅n)` due to the use of efficient string matching algorithms such as the **Knuth-Morris-Pratt algorithm** or the **Boyer-Moore algorithm**. These algorithms can reduce the time complexity to `O(m+n)` in the worst case.
```python
def preorder(node):
    if not node:
        return "null"
    return "#" + str(node.val) + " " + preorder(node.left) + " " + preorder(node.right)
```
---

### [235. Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) <sup style="color:#FFB801">Medium</sup>

Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

<img src="https://afteracademy.com/images/lca-of-binary-tree-example2-a42f5b48898c146d.png" width="500" alt="">

#### Solution:

- In this we go in a loop untill both `p` and `q` nodes are on opposite side.
- Once we come across that `p` and `q` are on different side we return that node.

```python
def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:

    while True:
        # Both p and q are on right side, we dig more deep
        if root.val < p.val and root.val < q.val:
            root = root.right

        # Both p and q are on left side, we dig more deep
        elif root.val > p.val and root.val > q.val:
            root = root.left

        # We found the node 'root' where p and q are on opposite side,
        # hence 'root' is the common ancestor of the two.
        else:
            return root
```
- Time complexity, `O(log n)`
---

### [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/) <sup style="color:#FFB801">Medium</sup>

Given the `root` of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

| Example Tree | Output |
| :----: | :----: |
| <img src="https://assets.leetcode.com/uploads/2021/02/19/tree1.jpg" width="180" alt=""> | `[[3],[9,20],[15,7]]` |


#### Solution:

- In this we do simple **BFS search**, and at each level we create an array of elements `val` at that level.

```python
def levelOrder(self, root: TreeNode) -> List[List[int]]:
    result = []
    q = collections.deque()

    # BFS Search
    if root:
        q.append(root)
    while q:
        val = [] 

        for i in range(len(q)): # At each level fill the new array 'val'
            node = q.popleft()
            val.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)

        result.append(val) # Apppend array 'val' for each level to result array
    return result
```
- Time complexity, `O(n)`
---

### [98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/) <sup style="color:#FFB801">Medium</sup>

Given the `root` of a binary tree, *determine if it is a valid binary search tree (BST)*.

- A **valid BST** is defined as follows:
    - The **left** subtree of a node contains only nodes with keys **less than** the node's key.
    - The **right** subtree of a node contains only nodes with keys **greater than** the node's key.
    - Both the left and right subtrees must also be binary search trees.


#### Solution:

- In this problem, a simple way to check is do In-order traverse and add the elements to an array and check if they are in ascending order or not.

```python
def isValidBST(self, root: Optional[TreeNode]) -> bool:

    def inorder(root, arr):
        if root == None: return

        if root.left:
            inorder(root.left, arr)

        arr.append(root.val) # Visit the node in-order

        if root.right:
            inorder(root.right, arr)

    a = []
    inorder(root, a)
    return all([a[i] < a[i+1] for i in range(len(a)-1)]) # Check if all elements are ascending
```

- Another way to do this is create a function `valid()` , where we will pass it `left` and `right` ancestor for the given node.
- When we start with `root` we say its `left` and `right` ancestor are `-inf` and `inf` respectively.
- When we go visit `left` node we pass the value of left ancestor of parent node and the right ancestor will become the parent node.
- When we go visit `right` node we pass the value of parent node as left ancestor and the right ancestor will be remain the right ancestor of parent node.

```python
def isValidBST(self, root: TreeNode) -> bool:

    def valid(node, left, right):
        if not node:
            return True

        if left >= node.val or right <= node.val:
            return False

        return valid(node.left, left, node.val) and valid(node.right, node.val, right)

    return valid(root, float("-inf"), float("inf"))
```
- Time complexity, `O(n)` in all cases as we need to visit each node to verify.
---

### [230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/) <sup style="color:#FFB801">Medium</sup>

Given the `root` of a binary search tree, and an integer `k`, return the *kth smallest value (**1-indexed**) of all the values of the nodes in the tree*.

#### Solution:

- In this we do **in-order traversal using `stack`**, (just like queue in BFS), but instead of visiting the element at each level we visit the smallest element and reduce the count `k` each time.
- First we keep going `left` and add elements to end of stack, such that our stack is in ascending order.
- When there are no more left to go we pop element and reduce `k` by `1`.
- Then we move right and keep going left and maintain the stack in the ascending order `(in-order)`
- we keep on doing this as long as we can go left or we have elements in the `stack`. Until we reduce the `k` to zero and find the K<sup>th</sup> smallest number.

```python
def kthSmallest(self, root: TreeNode, k: int) -> int:
    stack = []
    curr = root

    while stack or curr:

        while curr:
            stack.append(curr)
            curr = curr.left

        curr = stack.pop()
        k -= 1

        if k == 0:
            return curr.val
        curr = curr.right
```
- Time complexity, `O(log n)`
---

### [105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/) <sup style="color:#FFB801">Medium</sup>

Given two integer arrays `preorder` and `inorder` where `preorder` is the preorder traversal of a binary tree and `inorder` is the inorder traversal of the same tree, construct and return the binary tree.

#### Solution:

- In this we use the tactic **divide and conquer**.
- Here we first locate the `mid` of `inorder` by searching the 0<sup>th</sup> element in `preorder`.
- Once we found the mid we know **preorder for left subtree** will be from 1<sup>st</sup> element of preorder and it will have `mid` number of elements in total and **inorder for left subtree** will be left of the `mid` in inorder array.
- Similarly for the right subtree.

```python
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None

        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0]) # Find mid of inorder

        root.left = self.buildTree(preorder[1 : mid + 1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1 :], inorder[mid + 1 :])
        
        return root
```
- Time complexity, `O(n)` as we have to visit all the elements.
---

### [124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/) <sup style="color:#FF2D55">Hard</sup>

A **path** in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence **at most once**. Note that the path does not need to pass through the root.

The **path sum** of a path is the sum of the node's values in the path.

Given the `root` of a binary tree, *return the maximum **path sum** of any **non-empty** path.*

#### Solution:

- In this we visit each node of tree in DFS and *return* maximum sum of path either in left or right direction.
- But before we return we also check if we take this as the split node, will we have max path going in left and right direction from here. We check if max path in both direction from here is gonna be our max path sum.
- To track the max path we use the variable `res`.

```python
def maxPathSum(self, root: TreeNode) -> int:
    self.res = root.val

    def dfs(root):
        if not root:
            return 0

        leftMax = dfs(root.left)
        rightMax = dfs(root.right)
        leftMax = max(leftMax, 0)
        rightMax = max(rightMax, 0)

        # compute max path sum WITH split
        self.res = max(self.res, root.val + leftMax + rightMax)

        # return max path sum without split
        return root.val + max(leftMax, rightMax)

    dfs(root)

    return self.res
```
- Time complexity, `O(n)` as we visit all the nodes.
---

### [297. Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/) <sup style="color:#FF2D55">Hard</sup>

Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

Clarification: The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.

#### Solution:

- In this we do pre-order DFS traverse to the binary tree and create an array and then convert that array to an string to serialize.
- To Deserialize, we do reverse dfs search where we, conver the string to array and then maintain variable `i` to track the array traversal.
- In Deserialize, we recursively calls dfs to construct the left and right subtrees of the current node, and assigns them to the left and right attributes of the current node.

```python
def serialize(self, root):
    res = []

    def dfs(node):
        if not node:
            res.append("N")
            return
        res.append(str(node.val))
        dfs(node.left)
        dfs(node.right)

    dfs(root)
    return ",".join(res)
```
```python
def deserialize(self, data):
    vals = data.split(",")
    self.i = 0

    def dfs():
        # If we come across N that means the current node is null
        # Hence it wont have left and right subtree, so we return None here.
        if vals[self.i] == "N":
            self.i += 1
            return None

        node = TreeNode(int(vals[self.i])) # Create and new node
        self.i += 1 # Increase the counter

        node.left = dfs()
        node.right = dfs()

        # Once the tree is created we return root.
        return node

    return dfs()
```
- Time complexity, `O(n)` for both the functions.
---

## Tries

### [208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/description/) <sup style="color:#FFB801">Medium</sup>

A **trie** (pronounced as "try") or **prefix tree** is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

- `Trie()` Initializes the trie object.
- `void insert(String word)` Inserts the string word into the trie.
- `boolean search(String word)` Returns `true` if the string `word` is in the trie (i.e., was inserted before), and false otherwise.
- `boolean startsWith(String prefix)` Returns `true` if there is a previously inserted string `word` that has the prefix `prefix`, and `false` otherwise.


#### Solution:

- Explaination is in code comments.

```python
# Create a class for each character as Node and its followup character its children
# We use the variable end to denote that its is the end of one perfect word stored in Trie
class Node:
    def __init__(self):
        self.children = [None] * 26
        self.end = False
```

```python
class Trie:
    # Initialize the root of Trie, it has 26 children to start a word using any character.
    def __init__(self):
        self.root = Node()
```
```python
    # Insert the word in the Trie
    def insert(self, word: str) -> None:
        curr = self.root 

        # Iterate over each character of word and insert it in Trie if not already there
        for c in word:
            i = ord(c) - ord("a") # Find i for each lower case character (0 to 25)

            if curr.children[i] == None:
                # Creating a node at index i is equal to inserting the character at i
                curr.children[i] = Node() 
            curr = curr.children[i]
        
        # Update the end character of word as `end = TRUE` denoting an end of perfect word
        curr.end = True
```
```python
    # search for a perfect word in Trie
    def search(self, word: str) -> bool:
        curr = self.root

        # Iterate over each character of word to search
        for c in word:
            i = ord(c) - ord("a")

            # if any character not in Trie in correct order return False.
            if curr.children[i] == None:
                return False
            curr = curr.children[i]
        
        # if all character present, check if last character is end i.e `end = TRUE`
        return curr.end
```
```python
    # `startWith` function is just like search, without need to check for the `end = TRUE`
    def startsWith(self, prefix: str) -> bool:
        curr = self.root

        for c in prefix:
            i = ord(c) - ord("a")

            if curr.children[i] == None:
                return False
            curr = curr.children[i]

        return True

```
- Time complexity, `O(1)` to initialize Trie. But it will be `O(n)` for insert(), search() and startWith() where `n` is length of `word` passed to the respective function.
---
#### Another way to Implement Trie (Prefix Tree):
Earlier we were allocating array of size 26 for each new character, resulting in waste of memory. To avoid this we can implement Trie like this:
```python
class Trie:

    def __init__(self):
        self.trie = {}
```
```python
    def insert(self, word: str) -> None:
        t = self.trie
		# The idea is to insert words in a linked fashion. For Ex "cars" 
		# will be inserted as, Trie = {'c':{'a':{'r':{'s':{}}}}
        for w in word:
          if w not in t:
            t[w] = {}
          t = t[w]
		# Another key named "end" will distinguish that the word  "cars" actually exists
		# Without the end key it simply means that the traversed part is just prefix
        t["end"] = True 
```
```python
    def search(self, word: str) -> bool:
        t = self.trie
		# Traverse through the word
        for w in word:
          if w in t:
            t = t[w]
          else:
            return False
		# As the end key denotes the existence of the word
        return "end" in t
```
```python
    def startsWith(self, prefix: str) -> bool:
        t = self.trie
		# Traverse through the word
        for w in prefix:
          if w in t:
            t = t[w]
          else:
            return False
		# Here it is okay to find whether we had traversed the entire prefix or not
        return True
```
---

### [211. Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/description/) <sup style="color:#FFB801">Medium</sup>

- Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the `WordDictionary` class:

- `WordDictionary()` Initializes the object.
- `void addWord(word)` Adds word to the data structure, it can be matched later.
- `bool search(word)` Returns `true` if there is any string in the data structure that matches `word` or `false` otherwise. `word` may contain dots `'.'` where dots can be matched with any letter.

#### Solution:
- Just like the above Trie, here we create a trie using nodes of `TrieNode` class.
- For `addWord(word)` function we implement it same as any other Trie insert function.
- But for `search(word)` function we need to implement a search, in which we consider `'.'` as a wild card character that matches any character.
- So we use `dfs` search in this, whenever we come across the character `'.'` we do `dfs` search over all the children of that character.

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # a : TrieNode
        self.word = False
```
```python
class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    # Insert `word` in the Trie
    def addWord(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.word = True
```
```python
    # Search `word` in the Trie
    def search(self, word: str) -> bool:

        # Using dfs to search the `word`, here `j` is the index where we start search from.
        def dfs(j, root):
            cur = root

            for i in range(j, len(word)):
                c = word[i]

                # if c is wild card character do dfs search over all current childrens
                # to find: if remaining part of word is present.
                if c == ".":

                    for child in cur.children.values():
                         # if one of the dfs return TRUE that means word exist
                        if dfs(i + 1, child):
                            return True 

                    return False # We Return FALSE if all dfs returns FALSE

                else:

                    # if c is not wild character and also not present in current children
                    if c not in cur.children:
                        return False

                    cur = cur.children[c] # if c is in current node child move current ahead
            return cur.word

        return dfs(0, self.root)

```
- Time complexity, `O(n)` for addWord() function where `n` is length of word.
- For search function, since we have limitation on number of `'.'` wild card character as 2, so 2 times we can go maximum of 26 children branches. Hence time complexity will be, `O(2 * 26 * n)` i.e `O(n)`.
- If we have no limitation on the `'.'` character for search, then the wild card characters can be as much as `n` then time complexity can go as high as , `O(n`<sup>`2`</sup>`)`
---

### [212. Word Search II](https://leetcode.com/problems/word-search-ii/description/) <sup style="color:#FF2D55">Hard</sup>

Given an `m x n` board of characters and a list of strings `words`, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.


#### Solution:

- In this of we look close, we can do **Brutforce** way to solve this, where we do DFS to check each `word` in `words` starting at every *index* `i,j` in the `board`. In this Brutforce solution, hence run time will be `O(m×n)×Time_to_DFS` => means, `(m×n×k)` where `k` i average length of `word` in `words`, `k` can range from `1 to m×n`.
- To save Time we use **Trie** to do this in optimal manner.
- We add all the `word` in `words` array to a **Trie** data structure.
- Then we Iterate over all the places `board[i][j]` of the board. And we do DFS search at each index.
- But instead of searching just one single word, we search over the entire Trie and try to find all the words that can be reached from that particular places `board[i][j]` of the board.
- Then we eliminate those already found words in further search at other board places. 

```python
# We create a TrieNode class that will be use to hold Trie Data Structure.
class TrieNode:
    def __init__(self):
        self.children = {} # Dictionary for next character nodes. 
        self.isWord = False # To keep trach of end of word node.
        self.refs = 0 # Reference to track count of times a character node is referred.

    # Just like regular add words to Trie
    def addWord(self, word):
        cur = self
        cur.refs += 1 # When we add a character we increase its reference count
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
            cur.refs += 1 # When we add a character we increase its reference count
        cur.isWord = True

    # Remove `word` is just reducing to reference count for each character of the word
    def removeWord(self, word):
        cur = self
        cur.refs -= 1
        for c in word:
            if c in cur.children:
                cur = cur.children[c]
                cur.refs -= 1
        cur.isWord = False
```
```python
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:

        # Create a Trie and add all the words in it
        root = TrieNode() 
        for w in words:
            root.addWord(w)
        
        ROWS, COLS = len(board), len(board[0])
        # Created result and `visit` set to keep track of visited places of board while DFS
        res, visit = set(), set() 
```
```python
        # Create the DFS Function, it takes input r, c as row and col of the board
        # to start the DFS. Other input `node`, it is node in Trie from where we
        # do DFS search and input `word` is prefix of word that has been traversed.
        def dfs(r, c, node, word):
            if ( # here we check if we can go further in DFS search
                r not in range(ROWS) 
                or c not in range(COLS)
                or board[r][c] not in node.children
                or node.children[board[r][c]].refs < 1
                or (r, c) in visit
            ):
                return

            # visit the position board[r][c] add it to prefix, `word`
            visit.add((r, c))
            node = node.children[board[r][c]]
            word += board[r][c]

            # if we find end of any word, we add it to result and remove it from Trie
            if node.isWord:
                node.isWord = False
                res.add(word)
                root.removeWord(word)

            # do further DFS in all 4 directions,
            dfs(r + 1, c, node, word)
            dfs(r - 1, c, node, word)
            dfs(r, c + 1, node, word)
            dfs(r, c - 1, node, word)

            # After all the DFS is done we remove the nodes from visited set, so that 
            # we have all nodes unvisited when we do DFS starting at other `board` position.
            # We put this as last statement of DFS so that it runs in the end, after all the
            # recursive calls have been processed.
            visit.remove((r, c))
```
```python
        # Call the dfs at each position of the board
        for r in range(ROWS):
            for c in range(COLS):
                dfs(r, c, root, "")

        return list(res)

```
- The time complexity of this code can be analyzed in parts:

- **Trie Construction (root.addWord(w))**: For every word in `words`, adding a word to the Trie takes `O(K)` where `K` is the length of the word. Since there are `words` words, the time complexity for this step is `O(N×K)`, where `N` is the number of words in `words`.

- **DFS Traversal**: The DFS (dfs function) is called on every cell of the board (of size `m×n`). In the worst case, DFS would visit every cell starting from each cell (so it would be `m×n×m×n`). But considering the TrieNode structure, the DFS will not explore paths not present in the Trie. The trie has a total of `N×K` nodes (or less). So, for each cell, we'll visit at most `O(N×K)` nodes in the trie. So, for all cells combined, it's `O(m×n×N×K)`.

- Thus, the overall time complexity of the findWords function is: `O(m×n×N×K)` , where, 
    - `m×n` is the size of the board.
    - `N` is the number of words.
    - `K` is the average length of a word.

Note: This is a rough upper bound. **The real-world performance will be better due to early pruning when DFS paths don't exist in the Trie.**

---

## Heap / Priority Queue

### [295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/description/) <sup style="color:#FF2D55">Hard</sup>

The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.
 r = [2,3,4]`, the median is `3`.
    For example, for `arr = [2,3]`, the median is `(2 + 3) / 2 = 2.5`.

Implement the MedianFinder class:

- `MedianFinder()` initializes the MedianFinder object.
- `void addNum(int num) `adds the integer num from the data stream to the data structure.
- `double findMedian()` returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.


#### Solution:

- In this we maintain two heaps small and large which are max heap and min heap respectively.

```python
import heapq

class MedianFinder:
    def __init__(self):
        # We will maintain two heaps:
        # 1. small: A max heap for the smaller half of numbers.
        #    To simulate a max heap, we negate numbers before inserting/retrieving.
        # 2. large: A min heap for the larger half of numbers. (Python default)
        self.small, self.large = [], []
```
```python
    def addNum(self, num: int) -> None:
        # Decide where to put the new number.
        if self.large and num > self.large[0]:
            heapq.heappush(self.large, num)
        else:
            heapq.heappush(self.small, -1 * num)

        # Rebalance the heaps to ensure the difference in their sizes is not more than 1.
        if len(self.small) > len(self.large) + 1:
            val = -1 * heapq.heappop(self.small)
            heapq.heappush(self.large, val)

        if len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -1 * val)
```
```python
    def findMedian(self) -> float:
        # If 'small' has more elements, the median is the largest element in 'small'.
        if len(self.small) > len(self.large):
            return -1 * self.small[0] 
        # If 'large' has more elements, the median is the smallest element in 'large'.
        elif len(self.large) > len(self.small):
            return self.large[0]
        # If the number of elements is equal in both heaps, the median is the average.
        return (-1 * self.small[0] + self.large[0]) / 2.0

```
- Time complexity, `O(log n)` for both `addNum()` and `findMedian()` functions.
  
## Backtracking

### [39. Combination Sum](https://leetcode.com/problems/combination-sum/description/) <sup style="color:#FFB801">Medium</sup>

Given an array of distinct integers `candidates` and a target integer `target`, return a list of all unique combinations of `candidates` where the chosen numbers sum to `target`. You may return the combinations in any order.

The same number may be chosen from `candidates` an unlimited number of times. Two combinations are unique if the **frequency** of at least one of the chosen numbers is different.

The test cases are generated such that the number of *unique combinations* that sum up to target is less than `150` combinations for the given input.

#### Solution:

- In this we do backtracking usng DFS search, here we maintain a combination `curr` until get get `total` equal to `target`. If we found such combination we add it to result `res`.

```python
def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    res = []

    # Define a recursive DFS function.
    # i: the current index in candidates to consider.
    # cur: the current combination being built.
    # total: the sum of the numbers in the current combination.
    def dfs(i, cur, total):

        if total == target:  # found a valid combination.
            res.append(cur.copy())
            return

        if i >= len(candidates) or total > target:
            return

        cur.append(candidates[i])
        # Recursively call the dfs function keeping 'i' same because we can reuse same number.
        dfs(i, cur, total + candidates[i])

        # After exploring with the current number, we remove it (backtrack).
        cur.pop()

        # Now, we skip the current candidate and move on to the next one.
        dfs(i + 1, cur, total)

    dfs(0, [], 0) # start DFS from 0th index, empty combination and `total = 0`

    return res

```
- Given `n` as the number of candidates, in the worst case, each candidate can be chosen or not chosen (2 choices for each candidate). This results in 2<sup>n</sup> possible combinations. Thus, the time complexity is `O(2`<sup>`n`</sup>`)` in the worst-case scenario.
---

### [79. Word Search](https://leetcode.com/problems/word-search/description/) <sup style="color:#FFB801">Medium</sup>

Given an `m x n` grid of characters `board` and a string `word`, return `true` if `word` exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

#### Solution:

- In this we run 4 directional DFS at each index location of the `board` to search for the word `word`.
- In our DFS function we take input `r,c,i` for row, col and index of `word` to search from.
- DFS to check if `word[i:]` i.e substring of `word` from `i` to `len(word)`, can be made starting for `r,c` index of the `board`.

```python
def exist(self, board: List[List[str]], word: str) -> bool:
    ROWS, COLS = len(board), len(board[0]) # Get the imensions
    path = set() # To track the visited locations
```
```python
    def dfs(r, c, i):
        # Base case: if the entire word is matched
        if i == len(word):
            return True

        # Base case: if out of bounds, character mismatch or cell already visited
        if (
            min(r, c) < 0
            or r >= ROWS
            or c >= COLS
            or word[i] != board[r][c]
            or (r, c) in path
        ):
            return False

        path.add((r, c)) # Mark the location as visited

        # Explore in all 4 directions: down, up, right, and left
        # If any direction return true that means we found a match
        res = (
            dfs(r + 1, c, i + 1)
            or dfs(r - 1, c, i + 1)
            or dfs(r, c + 1, i + 1)
            or dfs(r, c - 1, i + 1)
        )

        path.remove((r, c)) # Backtrack: mark the cell as unvisited
        return res
```
```python
    # To prevent TLE,reverse the word if frequency of the first letter is more than the last letter's
    count = defaultdict(int, sum(map(Counter, board), Counter())) # Count character frequencies

    if count[word[0]] > count[word[-1]]:
        word = word[::-1] # Reversed
    
    # Search at each location of board
    for r in range(ROWS):
        for c in range(COLS):
            if dfs(r, c, 0):
                return True
    return False

    # O(n * m * 4^n)

```
- Time complexity of the code is approximately `O(m×n×L)`, where `m×n` is the size of the board and `L` is the length of the word. 
---

## Graphs

### [200. Number of Islands](https://leetcode.com/problems/number-of-islands/description/) <sup style="color:#FFB801">Medium</sup>

Given an` m x n` 2D binary grid grid which represents a map of `'1'`s (land) and `'0'`s (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

#### Solution:

- In this at each location of grid `i,j` if it is a land i.e `'1'` we change it and all its adjacent land to `'0'` and increase the count of islands by 1. 
- Then we iterate forward and we if find any land again we repeat the process. Untill we changed all `'1'` to `'0'`.
- Thus we will get number of inslands count in the variable `islands`.

```python
def numIslands(self, grid: List[List[str]]) -> int:
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    
    def dfs(i, j):
        # If out of bounds or not an island, return
        if i < 0 or i >= rows or j < 0 or j >= cols or grid[i][j] == '0':
            return
        # Mark the land cell as visited
        grid[i][j] = '0'
        
        # Visit all adjacent land cells
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)

    islands = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                islands += 1
                dfs(i, j)

    return islands

```
- Time complexity, `O(m×n)`. WE will visit each cell at max 2 times, one from the for loops and 1 time in the DFS. Hence worst case time will be 2×m×n i.e `O(m×n)`.
---

### [133. Clone Graph](https://leetcode.com/problems/clone-graph/description/) <sup style="color:#FFB801">Medium</sup>

Given a reference of a node in a **connected** undirected graph.

Return a **deep copy** (clone) of the graph.

Each node in the graph contains a value (`int`) and a list (`List[Node]`) of its neighbors.

```java
class Node {
    public int val;
    public List<Node> neighbors;
}
```

Test case format:

For simplicity, each node's value is the same as the node's index (1-indexed). For example, the first node with `val == 1`, the second node with `val == 2`, and so on. The graph is represented in the test case using an adjacency list.

An adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.

The given node will always be the first node with `val = 1`. You must return the copy of the given node as a reference to the cloned graph.

#### Solution:

- In this we use, Depth First Search function to traverse and clone each node.
- We also maintain the dictionary of cloned nodes `oldToNew`, so that if we come across already cloned node we dont clone them again.

```python
def cloneGraph(self, node: "Node") -> "Node":
    # Dictionary to map original nodes to their respective clones
    oldToNew = {}

    def dfs(node):
        # If the node is already cloned, return the cloned version
        if node in oldToNew:
            return oldToNew[node]

        copy = Node(node.val) # Create new clone node.
        oldToNew[node] = copy # Add to dictionary
        
        # For each neighboring node, recursively clone it and append
        # to the neighbors of the current clone.
        for x in node.neighbors:
            copy.neighbors.append(dfs(x))

        return copy

    # Start the DFS from the given node, or return None if the node is null
    return dfs(node) if node else None

```
- The time complexity is `O(N+E)`, where `N` is the number of nodes, and `E` is the number of edges/connections between them.
---

### [417. Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/description/) <sup style="color:#FFB801">Medium</sup>

There is an `m x n` rectangular island that borders both the Pacific Ocean and Atlantic Ocean. The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches the island's right and bottom edges.

The island is partitioned into a grid of square cells. You are given an `m x n` integer matrix `heights` where `heights[r][c]` represents the height above sea level of the cell at coordinate `(r, c)`.

The island receives a lot of rain, and the rain water can flow to neighboring cells directly *north, south, east, and west* if the neighboring cell's height is less than or equal to the current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.

Return a **2D list** of grid coordinates result where `result[i] = [r`<sub>`i`</sub>`, c`<sub>`i`</sub>`]` denotes that rain water can flow from cell `(r`<sub>`i`</sub>`, c`<sub>`i`</sub>`)` to both the Pacific and Atlantic oceans.

#### Solution:

- In this we create 2 mxn arrays `pacific` and `atlantic` to keep track of cells that can drain to the Pacific and Atlantic.
- Start a DFS from every cell on the Pacific's border (top and left borders) and mark all the cells that can be reached in the Pacific matrix.
- Similarly, start a DFS from every cell on the Atlantic's border (right and bottom borders) and mark all the cells that can be reached in the Atlantic matrix.
- Finally, return the union of pacific and atlantic arrays.

```python
def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
    if not heights:
        return []
    
    m, n = len(heights), len(heights[0])
    pacific = [[0]*n for _ in range(m)]
    atlantic = [[0]*n for _ in range(m)]

    def dfs(r, c, visited, prev_height):

        # If out of bounds, already visited, or current height is smaller than previous
        if (
            r < 0 or r >= m 
            or c < 0 or c >= n 
            or visited[r][c] == 1 
            or heights[r][c] < prev_height
        ):
            return
        
        # Mark the current cell as visited
        visited[r][c] = 1 

        # Explore the neighboring cells
        for x, y in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
            dfs(x, y, visited, heights[r][c])
    
    for i in range(m):
        dfs(i, 0, pacific, heights[i][0]) # Pacific Left Border
        dfs(i, n-1, atlantic, heights[i][n-1]) # Atlantic Right Border
    
    for i in range(n):
        dfs(0, i, pacific, heights[0][i]) # Pacific Top Border
        dfs(m-1, i, atlantic, heights[m-1][i]) # Atlantic Bottom Border
    
    res = []
    for i in range(m):
        for j in range(n):
            if pacific[i][j] == 1 and atlantic[i][j] == 1:
                res.append([i,j])
        
    return res

```
- Time complexity, `O(m×n)`
---

### [207. Course Schedule](https://leetcode.com/problems/course-schedule/description/) <sup style="color:#FFB801">Medium</sup>

There are a total of numCourses courses you have to take, labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [a`<sub>`i`</sub>`, b`<sub>`i`</sub>`]` indicates that you must take course `b`<sub>`i`</sub> first if you want to take course `a`<sub>`i`</sub> .

For example, the pair `[0, 1]`, indicates that to take course `0` you have to first take course `1`.

Return `true` if you can finish all courses. Otherwise, return `false`.

#### Solution:

- This problem can be solved using a **Topological Sort**. The idea is to represent the courses and their prerequisites as a directed graph. A cycle in this graph indicates that it's impossible to finish all courses, because there's a circular dependency between courses.
- To check if a cycle exists in the graph, we can use depth-first search (DFS). If we encounter a node that is already being visited it means there's a cycle, and we can't finish all courses.
- In DFS we will use `visited[i] = -1` when the node is being visited i.e it is kept on hold for another course to be finished as it has prerequisite courses that is not visited(finished) yet.
- And we will use `visited[i] = 1` when certain course is visited i.e all its prerequisite can be done so it can be done too.

```python
def canFinish(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    visited = [0 for _ in range(numCourses)]

    # create graph
    for pair in prerequisites:
        x, y = pair
        graph[x].append(y)

    # return True if there's a cycle
    def dfs(i):

        # if ith node is marked as being visited, then a cycle is found
        if visited[i] == -1:
            return True

        # if it is done visited, then do not visit again
        if visited[i] == 1:
            return False

        # mark as being visited
        visited[i] = -1

        # visit all the neighbors
        for j in graph[i]:
            if dfs(j):
                return True
                
        # after visiting all the neighbors of i, mark it to be done visited
        visited[i] = 1
        return False

    for i in range(numCourses):
        if dfs(i):
            return False
    return True

```
- Time complexity, `O(n)` where `n` is Number of courses. we visit each course once. 
---

### Topological Sort:

    Topological Sort is an algorithm used to linearly order the vertices of a directed acyclic graph (DAG) in such a way that for every directed edge `(u, v)`, vertex `u` comes before vertex `v` in the ordering.

    It's essential to note that Topological Sort is only applicable for DAGs.

Examples:
- **Dependency Resolution**: Scenarios where certain tasks have to be executed in a particular order, and some tasks depend on the completion of other tasks. (Build/Execution Order)
- **Course Scheduling**: Given a set of courses and their prerequisites, determine an order to take the courses.
- **Advanced algorithms**: Some algorithms, especially on graphs, use Topological Sort as a subroutine.

**Topological Sort Algorithm:**
1. Choose a node that has no incoming edges (i.e., its in-degree is 0).
2. Remove the node and all outgoing edges from it.
3. Add the node to the topological ordering.
4. Repeat steps 1-3 until no nodes remain.

**Topological Sort Algorithm using DFS:**
1. Create an empty stack and a visited set.
2. For every vertex v in the graph:
    - If v is not visited:\
        a. Call the DFS helper function starting from vertex v, marking nodes as visited along the way, and pushing nodes to the stack when they have no unvisited neighbors.
3. Pop nodes from the stack; this gives the topological ordering.

**Pseudo-Code:**
```javascript
function topologicalSort(graph):
    stack = empty stack
    visited = empty set

    for each vertex v in graph:
        if v not in visited:
            dfs(v, visited, stack)

    while stack is not empty:
        print(stack.pop())

function dfs(v, visited, stack):
    visited.add(v)

    for each neighbor u of v:
        if u not in visited:
            dfs(u, visited, stack)

    # push `v` to stack, when all its neighbors are visited in above for loop
    stack.push(v) 

```
Eg:\
<img src="https://media.geeksforgeeks.org/wp-content/cdn-uploads/graph.png" width="200px">\
Will Have Topological Sort: `stack = [4,5,0,2,3,1]` (according to above algorithm)\
Note, There can be more than one topological sorting for a graph.

---

### [323 · Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/) <sup style="color:#FFB801">Medium</sup>

In this problem, there is an undirected graph with `n` nodes. There is also an edges array. Where `edges[i] = [a, b]` means that there is an edge between node `a` and node `b` in the graph.

You need to *return* the number of connected components in that graph.

Example: A graph with 10 nodes and 6 edges.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20200421201526/Graph-Connected-Example-1-300x98.png">\
Connected Components: **`4`**


#### Solution:

- We first construct the graph using an adjacency list. 
- We maintain a visited list to keep track of nodes we've already explored.
- For each unvisited node, we initiate a DFS to explore its entire component and mark all nodes in that component as visited. We also increment the components count for each unvisited node (i.e., for each new component).

```python
def countComponents(n, edges):
    # Create an adjacency list to represent the graph
    graph = [[] for _ in range(n)]
    for edge in edges:
        a, b = edge
        graph[a].append(b)
        graph[b].append(a)

    visited = [False] * n

    # DFS function
    def dfs(i):
        if visited[i]:
            return
        visited[i] = True
        for neighbor in graph[i]:
            dfs(neighbor)

    # Check if any node is unvisited, visit it and all its connected components
    # and mark them visited, increment the counter.
    components = 0
    for i in range(n):
        if not visited[i]:
            components += 1
            dfs(i)

    return components

```
- Time complexity, `O(n)` to visit all the nodes.
---

### [261 · Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/) <sup style="color:#FFB801">Medium</sup>

Given `n` nodes labeled from `0` to` n - 1` and a list of `undirected` edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.

#### Solution:

- By Definition of Valid Tree:
    - The graph must be connected: there must be a path between every pair of vertices.
    - The graph must not contain any cycles.

**Using approach: Union-Find (or Disjoint Set Union - DSU)**
- If the graph is a tree, it must have exactly `n - 1` edges. If not, return False immediately.

- The `find` function finds the root of the set that element `x` belongs to. This is done using a very common union-find algorithm.

- The `union` function merges the disjoint sets that elements `x` and `y` belong to. It returns `True` if `x` and `y` were in different sets (and hence no cycle was introduced by adding the edge `(x, y)`), and `False` if `x` and `y` were already in the same set.

- `all(map(union, edges))` returns `True` if all edges connect nodes in different sets (and hence no cycles are introduced), and `False` otherwise.

```python
def valid_tree(n, edges):
    if len(edges) != n - 1: # Check edge count
        return False
    
    # Initialize the parent list for union-find.
    # Initially, every node is its own parent.
    parent = list(range(n)) 
    
    # Find the root of the set that node x belongs to.
    # If a node's parent is itself, it's the root. Otherwise, recursively find the root.
    def find(x):
        return x if parent[x] == x else find(parent[x])
    
    # Join two sets together and return whether they were previously separate.
    # The union function combines the sets that contain nodes x and y.
    # It returns True if they were separate (no cycle introduced), 
    # False if they were already in the same set.
    def union(edge):
        x, y = map(find, edge)  # Find the roots for nodes in the edge
        parent[x] = y           # Make y the parent of x
        return x != y           # True if x and y had different roots, means no cycle
    
    # Check if all edges connect nodes that were previously not connected.
    # If any edge connects nodes in the same set, a cycle is introduced and it's not a tree.
    return all(map(union, edges))

```
- Time Complexity: `O(N + E)`, where `N` is the number of nodes and `E` is the number of edges, as each union/find operation takes `O(1)` time.
---

## Advanced Graphs

### [269 · Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)

There is a new alien language which uses the latin alphabet. However, the order among letters are unknown to you. You receive a list of **non-empty** words from the dictionary, where words are **sorted lexicographically by the rules of this new language**. Derive the order of letters in this language.

#### Solution:

-

```python

```
- Time complexity, `O()`
---

## 1-D Dynamic Programming

## 2-D Dynamic Programming

## Greedy

## Intervals

## Math & Geometry

## Bit Manipulation

### []()

#### Solution:

-

```python

```
- Time complexity, `O()`
---