# 所有大写
# str.upper()
# 所有小写
# str.lower()
# 把第一个字母转化为大写字母，其余小写
# str.capitalize()
# 把每个单词的第一个字母转化为大写，其余小写 
# str.title()

# raw_input函数
# raw_input([prompt]) 函数从标准输入读取一个行，并返回一个字符串（去掉结尾的换行符）：
str = input("请输入：")
print("你输入的内容是: ", str)

# file object = open(file_name [, access_mode][, buffering]) 
# t 文本 / x 写模式 / r 只读 / w 只用于写入 / a用于追加
fo = open("foo.txt", "w")
print(fo.closed)
print(fo.name)
print(fo.mode)
print(fo.softspace)

fo.close()
fo.write("this is mario")
str = fo.read(10)

# 查找当前位置
position = fo.tell()
print("当前文件位置 : ", position)


# 把指针再次重新定位到文件开头
position = fo.seek(0, 0)
str = fo.read(10)
print("重新读取字符串 : ", str)

import os
os.rename( "test1.txt", "test2.txt" )
os.remove("test2.txt")
os.mkdir("newdir")

"----------------------------------   Pandas   --------------------------------------------------"
