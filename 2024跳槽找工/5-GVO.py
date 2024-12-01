# decode string 重写
# 4 
# 68
# 6
# 1244 
# 295
# 218
# 212
# 359
# 经典餐厅waitlist system
# dijaska 算法
# 第一轮：最烂的一轮，脑子特别不清醒，题目是生成一个车牌号。00000 -> 00001 -> 00002 -> ... -> 99999 -> A0000 -> A0001 -> ... -> A9999 -> ... -> Z9999 -> AA000 -> ... -> ZZ999 -> ... -> AAA00 -> ... -> ZZZ99 -> ... -> AAAA0 -> ... -> ZZZZ9 -> ... -> AAAAA -> ... -> ZZZZZ。输入是一个数字，表示要生成第几个车牌号，输出是一个字符串，对应该数字对应的车牌号。我的解法是暴力法，每个范围依次去找对应的车牌号范围。但写到70%的时候卡住了，面试官给了提示，说可以把问题分成两部分——数字部分按10进制，字母部分按26进制来处理。但当时时间不够，没来得及转换成面试官的方法，最后没能做完。
# 第二轮：有一个data stream找到最新的K个元素的average，但是元素会一直更新，用queue就好。follow-up：同样的data stream，但是要在最新的K个元素中去掉X个最大的元素然后求average。楼主一开始说了个priority queue，面试官说不太efficiency，然后思考了一会儿用sortedList，压哨写完。
# 第三轮：给一个不重复的character集合，然后找出这些characters总共可以组成多少个字符串使得生成的string - 长度不超过L，以及同样的字符连续不超过K。 一开始用暴力法解 复杂度L^N， 然后问能不能优化，最后用dp优化到 L*K 复杂度，还剩了点时间。
# 第四轮BQ ：都是经典题什么和同事conflict或者a project I am proud of。我都是硬套的准备好的故事。


# 第一轮:implement secure linked list，就是insert node at head然后每个node要compute hash value然后这个hashvalue是这个node的value的hashvalue + 下个node的hashvalue。
# 第二轮：给一个array和prefix然后 detect number of word has prefix，然后这个array是sorted。array里面是string word，用binary search解决。
# # 第三轮：bq，聊的还行
# # 第四轮： 利口62，follow up设置障碍，求path的数量穿过所有的障碍

# 第二轮: 亚洲女生，很chill，地里有的 餐厅排队。设计一个waitlist的structure，满足三种功能，join waitlist, leave waitlist, serve customers with certain size. 这题可能每个面试官都会有一些不一样，serve customer这个部分我只需要检查有没有party size和table size完全一样的party就行了。 join 和 leave可能要考虑一下有没有重复加waitlist或者重复leave的可能。
# 第三轮：三哥大叔，话不多，有点口音有时候不太听得懂。给一个list of strings, 要求group strings if they are buddies. Buddies 的定义是： 1. they have same length. 2. the distance between each character is same.
# 举例：“aaa” 和 “bbb”, 都是长度3， 并且a-a-a 的间隔是0-0-0，b-b-b也是，所以是buddies。“zab” 也和 “abc” 是buddies。input只有可能是 “”，或者a-z组合，string里不会有空格。
# 第四轮：看不出哪里人，很supportive，开头就讲了比较看重思考过程，所以一直在交流。题目给一个list of subsequence，subsequence only contains integers. 这些subsequence 都是从某个 master sequence 删除一些element得来的。master sequence是一个没有重复数字的sequence，比如1, 2, 3, 4, 5, 那subsequence就可能是 1，2，5或者 2，3，4。题目要求判断if all subsequence in the list could possibly come from one same master sequence。比如如果出现1， 2， 5 和 2，1 就说明一个master sequence不可能得出这两个subsequence。



#  dp, calculator
# 224
# 694



这个题小弟之前面试微软的时候也遇到过相似的，但是从来没有刷到过。小弟怀疑是Leetcode的原题，如果各位大佬知道，可以说一下。题目不难，小弟虽然没有刷到过，但是也是看完就有思路了。
大致的意思是：大家会不停的使用Google Search。你调用一个function，叫search。search会给你一个timestamp，和用户搜索的内容，比如说“今天天气怎么样？”。timestamp是单调递增的。比如说你第一次call search的时候timestamp是5，第二次可能是6，可能是8，但是不可能是4。那么如果说，两次call search的timestamp是小于60的，并且内容是一样的。那么就不用把他送给server，反之则是需要的。比如说，用户在1的时候，提问了“今天天气怎么样？”，然后又在36的时候，提问了“今天天气怎么样？”，那么36秒的这个就不需要送给server。但是如果在68秒的时候提问了“今天天气怎么样？”，那么就需要送给server。
小弟的思路是用一个queue来保存这些timestamp，然后用一个hashmap来保存这些提问的string。followup小弟不记得了。
然后这轮的面试官是个台湾小哥，面试体验其实也很不错。很有耐心，全程也没有打断我的思路。我第一遍写完是有bug的。然后小哥说，你把我给你的例子从头到尾跑一边看看。我跑的时候发现了bug，然后自己把他改掉了。然后就是follow up。小弟不记得了，应该是没有写出来的。
第三题：这个题小弟在刷地里Meta面试题的时候看到了相似的题。小弟也没有刷到过。小弟也怀疑是Leetcode原题。如果有大佬知道，麻烦提一下。
这轮面试体验极其糟糕，是个大胡子老铁，也没有题。上来问我，你有没有玩过bingo？我说没有。他说是这样的。bingo是5 * 5的棋盘。然后第一排是1-15的数，但是这五个数不能是重复的。第二排是16-30的数，第二排的五个数也不能是重复的。以此类推。我现在给你一个random(x, y)，他可以随机生成一个range从x到y的数。然后你给我构造一个valid的bingo棋盘出来。
小弟其实一开始没明白是啥意思。一开始以为是类似于N皇后这种，横着竖着斜着都有限制的棋盘。然后再三和大胡子老铁确认了啥意思，然后让他能不能给个题目出来。大胡子老铁说没有题目，给你一个valid棋盘的例子看看吧！然后这个时候小弟才反应过来。哦！原来第一排就是从1-15里面sample5个数，这5个数不能是重复的。然后第二排就是从16-30里面sample五个数，这五个数也不能重复的。
然后小弟就开始写了。对于第一排，我构造一个hashset，重复的使用random(1，15)。如果说这个数在hashset里面，我就重复call，如果不在，那我就把他填到棋盘里面去。
大胡子老铁说，你这个不对。你这个可能会重复的使用random(1,15)，每个cell只能用一次。我想了一下，我说，那第一个cell我用random(1,3)，第二个cell我用random(4,6)，第三个cell我用random(7,9)... 大胡子老铁：你这个也不对。因为你没有办法构造出every possible bingo case。然后我就问他要了hint。大胡子老铁说，你想想，如果说你现在取出一个，还有几个数字呢？小弟这个时候想到了，哦！可以用index来sample。我把15个数字排成一个list，然后从1到15里面sample一个数，这个数是这个list的index。选出这个数之后，把这个数从这个list里面pop出去，这个list里面就只有14个数字了。然后再从1到14里面sample一个数。然后大胡子老铁又说，你这个还是不对！因为你用了pop操作，这个操作非常之费时。小弟想了一下，把选出来的数，和list的最后一位swap了一下。这回大胡子老铁总算是满意了。
大胡子老铁的followup是，如果现在有五个bingo 棋盘，你应该怎么构造这五个bingo棋盘，让他们每个都不一样呢？这个followup有个很关键的点，就是其实这五个bingo棋盘，只需要有一个元素和他其他的不一样就可以了。小弟给了很多思路，但没有一个是大胡子老铁满意的。时间到了之后，大胡子老铁说，没关系，小伙子！我这个问题除了这个followup还有4个followup，到目前为止的记录是有人能答出两个followup。


Serialize object to json string. Input could be number, string, array or dictionary. If there is a cycle in an array or dictionary, serialize it to “recursion”. Example:
a = {“k”, b} b = [a, 2]
then serialize(a) = {“k”: [“recursion”, 2]}