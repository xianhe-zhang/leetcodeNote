

# Binar Search
# 难点在于if-clause的变化/二分标的的选择
def binary_serch(l, r): 
    while l < r:
        m = (l+r) // 2
        if f(m):
            return m
        if g(m):
            r = m
        else:
            l = m + 1
    return l

# 什么时候lower bound? 什么时候upper bound？
# 注意if-clause中的<=/<符号，这决定了left/right最后相遇的位置。
# 如果是<，那么left/right相遇在target/target作为lower bound的第一顺位
# 如果是<=, 那么left/right相遇在target作为作为lower bound的第一顺位


# BFS

# 这里的｜= 其实是union()
def bfs():
    bfs = [target.val]
    visited = set([target.val])
    for k in range(K):
        for x in bfs:
            for y in connections[x]:
                if y not in visited:
                    visited |= set(bfs) 
                    bfs.append(y)
        # bfs = [y for x in bfs for y in connections[x] if y not in visited]
    return bfs

