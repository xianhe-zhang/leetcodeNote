def f1(x: int):
    res = [3]
    def f2(y: int):
        
        res[0] += y
        print("neibu"+ str(res))
    f2(x)
    print(res)
f1(4)