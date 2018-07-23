class Solution:
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        length = len(s)
        if length == 1:
            if s == '0':
                return 0
            else:
                return 1

        num_decode_methods = []  # the number of different decode methods at each position
        # state = []  # state for each position
        #             # 1 denotes the number at that position only has 1 decode method
        #             # 0 denotes the number at that position has more than 1 decode methods
        for i, char in enumerate(s):
            if i == 0:
                num_decode_methods.append(1)

            if i == 1:
                if char == '0':
                    num_decode_methods.append(num_decode_methods[i - 1])
                else:
                    if int(s[i - 1] + char) <= 26:
                        num_decode_methods.append(2)
                    else:
                        num_decode_methods.append(1)
                print(num_decode_methods)
            else:
                if int(s[i - 1] + char) > 26:
                    num_decode_methods.append(num_decode_methods[i - 1] + 0)

                elif int(s[i - 1] + char) <= 26:

                    if char == '0':
                        num_decode_methods.append(num_decode_methods[i - 1] + 0)
                    else:
                        num_decode_methods.append(num_decode_methods[i - 1] + 1)

            # print (num_decode_methods)
        return num_decode_methods[-1]


if __name__ == '__main__':
    solution = Solution()
    # print(solution.twoSum([1,2,2,3,4,1, 1.5, 1.5], 3))

    # print(solution.delete_identical(solution.twoSum([1,2,2,3,4,1, 1.5, 1.5], 3)))

    print(solution.numDecodings('226'))
    pass
    # print([list(ele) for ele in res if len(ele) == 3])
    # print(solution.delete_identical(solution.threeSum([1,-1,1,2,0,-3])))
