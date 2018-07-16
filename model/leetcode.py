
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        xxx = []


        targets = [-num for num in nums]
        index = 0
        for target in targets:
            double_list = self.twoSum(nums, target)
            # print(double_list)

            if double_list is not None:
                # deleted_double_lst = self.delete_identical(double_list)
                xx = [ele+[index] for ele in double_list]
                new_tri = [set(ele+[index]) for ele in double_list]
                # if set(new_tri)
                res += new_tri
                xxx += xx
            index += 1

        print(res)
        res_index = [list(ele) for ele in res if len(ele) == 3]
        res_val = []
        for ele in res_index:
            ele_val = [nums[ele_index] for ele_index in ele]
            res_val.append(ele_val)
        return res_index, xxx

    @staticmethod
    def delete_identical(lst):
        """
        Each element in lst is a list of a pair of 2 ints, delete identical pairs in lst
        """
        new_lst = []
        for pair in lst:
            if pair not in new_lst and [pair[1],pair[0]] not in new_lst:
                new_lst.append(pair)
        return new_lst
        # new_lst = []
        # for tri in lst:
        #     if tri not in new_lst and [tri[0]]:

    @staticmethod
    def twoSum(nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[[int]]
        """
        res = []
        length = len(nums)
        for i, first_num in enumerate(nums):
            second_num = target - first_num
            if second_num in nums[i+1:]:
                res.append([i, nums.index(second_num)])

        return res


if __name__ == '__main__':
    solution = Solution()
    # print(solution.twoSum([1,2,2,3,4,1, 1.5, 1.5], 3))

    # print(solution.delete_identical(solution.twoSum([1,2,2,3,4,1, 1.5, 1.5], 3)))

    print(solution.threeSum([1,-1,2,0,-3]))
    res, _ = solution.threeSum([1,-1,2,0,-3])
    # print([list(ele) for ele in res if len(ele) == 3])
    # print(solution.delete_identical(solution.threeSum([1,-1,1,2,0,-3])))
