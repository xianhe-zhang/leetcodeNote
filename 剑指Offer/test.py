# 作者：小狐狸FM
def Move1(lis_now,dirction):
    '''移动左上角的机关，将会让pos1、pos2、pos3顺时针旋转'''
    lis_now[0] = dirction[(dirction.index(lis_now[0]) + 1)% len(dirction)] #移动pos1
    lis_now[1] = dirction[(dirction.index(lis_now[1]) + 1)% len(dirction)] #移动pos2
    lis_now[2] = dirction[(dirction.index(lis_now[2]) + 1)% len(dirction)] #移动pos3
    return lis_now
def Move2(lis_now,dirction):
    '''移动右上角的机关，将会让pos1、pos2、pos4顺时针旋转'''
    lis_now[0] = dirction[(dirction.index(lis_now[0]) + 1)% len(dirction)] #移动pos1
    lis_now[1] = dirction[(dirction.index(lis_now[1]) + 1)% len(dirction)] #移动pos2
    lis_now[3] = dirction[(dirction.index(lis_now[3]) + 1)% len(dirction)] #移动pos4
    return lis_now
def Move3(lis_now,dirction):
    '''移动左下角的机关，将会让pos1、pos3、pos4顺时针旋转'''
    lis_now[0] = dirction[(dirction.index(lis_now[0]) + 1)% len(dirction)] #移动pos1
    lis_now[2] = dirction[(dirction.index(lis_now[2]) + 1)% len(dirction)] #移动pos3
    lis_now[3] = dirction[(dirction.index(lis_now[3]) + 1) % len(dirction)]  # 移动pos4
    return lis_now
def Move4(lis_now,dirction):
    '''移动右下角的机关，将会让pos2、pos3、pos4顺时针旋转'''
    lis_now[1] = dirction[(dirction.index(lis_now[1]) + 1)% len(dirction)] #移动pos2
    lis_now[2] = dirction[(dirction.index(lis_now[2]) + 1)% len(dirction)] #移动pos3
    lis_now[3] = dirction[(dirction.index(lis_now[3]) + 1) % len(dirction)]  # 移动pos4
    return lis_now
if __name__=="__main__": #主函数
    lis_now = ["d", "d", "a", "d"]  # 机关当前的方向
    dirction = ["a", "b", "c", "d"]  # 机关的方向标记，a为正确的方向
    answer = [0,0,0,0] #正确的答案
    # pos1 = "d" # 左上角的机关方向，击打pos1将会使pos1、pos2、pos3顺时针旋转
    # pos2 = "a" # 右上角的机关方向，击打pos2将会使pos1、pos2、pos4顺时针旋转
    # pos3 = "c" # 左下角的机关方向，击打pos3将会使pos1、pos3、pos4顺时针旋转
    # pos4 = "c" # 右下角的机关方向，击打pos4将会使pos2、pos3、pos4顺时针旋转
    # print(dirction[(dirction.index("d") + 1) % len(dirction)]) #测试
    for i in range(4): #循环1
        answer = [0,0,0,0] #重置
        Move1(lis_now,dirction) #触发pos1
        answer[0] = (answer[0] + 1)%4 #记录pos1击打的次数
        if lis_now == ["a", "a", "a", "a"]:  # 找到答案时
            print(answer)
            exit
        for j in range(4): #循环2
            Move2(lis_now,dirction) #触发pos2
            answer[1] = (answer[1] + 1)%4 #记录pos2击打的次数
            if lis_now == ["a", "a", "a", "a"]:  # 找到答案时
                print(answer)
                exit
            for k in range(4): #循环3
                Move3(lis_now, dirction)  # 触发pos3
                answer[2] = (answer[2] + 1)%4 #记录pos3击打的次数
                if lis_now == ["a", "a", "a", "a"]:  # 找到答案时
                    print(answer)
                    exit
                for g in range(4): #循环4
                    Move4(lis_now, dirction)  # 触发pos4
                    answer[3] = (answer[3] + 1)%4 #记录pos4击打的次数
                    if lis_now == ["a","a","a","a"]: #找到答案时
                        print(answer)
                        exit
