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


##### Solution:

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

##### Solution:

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

##### Solution:

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

##### Solution:

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

##### Solution:

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

##### Solution:

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

##### Solution:

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

### []()

##### Solution:

-

```python

```
-
---

### []()

##### Solution:

-

```python

```
-
---

### []()

##### Solution:

-

```python

```
-
---

### []()

##### Solution:

-

```python

```
-
---

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

### []()

##### Solution:

-

```python

```
-
---