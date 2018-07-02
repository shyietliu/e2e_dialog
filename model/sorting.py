
def bubble_sorting(list_to_be_sorted):
    length = len(list_to_be_sorted)
    finished = 0
    while not finished:
        finished = 1
        for i in range(length-1):
            if list_to_be_sorted[i] > list_to_be_sorted[i+1]:
                temp = list_to_be_sorted[i]
                list_to_be_sorted[i] = list_to_be_sorted[i+1]
                list_to_be_sorted[i+1] = temp
                finished = 0
            print(list_to_be_sorted)


def select_sorting(list_to_be_sorted):
    length = len(list_to_be_sorted)
    for i in range(length-1):
        min_index = i
        # find minimum
        for j in range(i+1, length):
            if list_to_be_sorted[j] < list_to_be_sorted[min_index]:
                min_index = j

        list_to_be_sorted[i], list_to_be_sorted[min_index] = list_to_be_sorted[min_index], list_to_be_sorted[i]
        print(list_to_be_sorted)


def insert_sorting(list_to_be_sorted):
    length = len(list_to_be_sorted)
    if length == 1:
        return list_to_be_sorted

    for i in range(1, length):
        for j in range(i, 0, -1):
            if list_to_be_sorted[j] > list_to_be_sorted[j-1]:
                list_to_be_sorted[j], list_to_be_sorted[j-1] = list_to_be_sorted[j-1], list_to_be_sorted[j]

        print(list_to_be_sorted)
    return list_to_be_sorted


def shell_sort(list):
    n = len(list)
    # 初始步长
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            # 每个步长进行插入排序
            temp = list[i]
            j = i
            # 插入排序
            while j >= gap and list[j - gap] > temp:
                list[j] = list[j - gap]
                j -= gap
            list[j] = temp

            # print('After {0} sorting'.format(i),list)
        # 得到新的步长
        gap = gap // 2
    return list


def quick_sort(list_to_be_sorted):
    lower = []
    equal = []
    higher = []

    if len(list_to_be_sorted) > 1:
        pivot = list_to_be_sorted[0]

        for ele in list_to_be_sorted:

            if ele > pivot:
                higher.append(ele)
            elif ele < pivot:
                lower.append(ele)
            elif ele == pivot:
                equal.append(ele)
        return quick_sort(lower) + equal + quick_sort(higher)
    else:
        return list_to_be_sorted


if __name__ == '__main__':
    list_to_be_sorted = [2, 5, 5, 9, 6, 7, 1, 8, 54, 132, 55]
    print(list_to_be_sorted)
    # lst = insert_sorting(list_to_be_sorted)
    lst = quick_sort(list_to_be_sorted)
    print(lst)
