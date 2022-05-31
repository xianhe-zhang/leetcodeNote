import math
v = [1,0,0,0,1,0,1]
n = 7
for i in range(n):
    if v[i] == 1:
        v[i] = -(i+1)
    else:
        v[i] = i
for i in range(1, n):
    v[i] = v[i-1] + v[i]
res = math.inf
for end in range(1, n - 1):
    res = min(res, abs((v[-1] - v[end]) - v[end]) )
print(res)