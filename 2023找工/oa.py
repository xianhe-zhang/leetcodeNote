

prefix_sum, min_val = arr[0], arr[0]
for i in range(1, len(arr)):
    prefix_sum += arr[i]
    min_val = min(min_val, prefix_sum)


