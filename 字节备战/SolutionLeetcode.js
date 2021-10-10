// # 冒泡排序
// # 归并
// # 树的右视图 199
// # 统计单词出现次数 剑指offer56-I -II 43 39
// # 二叉树反转180 226
// # 二叉树最小深度 111
// # 岛屿数量 200
// # 打家劫舍 337
// # 上台阶 746
// # 两数之和 1
// # 反转链表 206
// # 数组的子序列最大和 53 剑指offer
// # 数组的topk大数字  剑指offerII 076
 
// 冒泡排序
function bubbleSort(arr){
  var n = arr.length()
  for(let i = 0; i < n; i++){
    for(let j=0; j<n-i-1; j++){
      if(arr[j] > arr[j+1]){
        arr[j+1],arr[j] = arr[j],arr[j+1];
      }
    }
  }
}

// 归并排序
function mergeSort(arr, l, r){
  function mergeSort(arr, l, m, r){

  }
}

//leetcode-199 树的右视图
//BFS - 将每一层的最右边节点放入res
var rightSideView = function(root) {
    let res = [];
    if(root === undefined){
      return 
    }

};
//DFS - 先遍历根节点->右子节点->左子节点，每一个depth第一次遍历到的就是最右节点
var rightSideView = function(root) {
  let res = [];

};

var fnArr = [];
for (var i = 0; i < 10; i ++) {
  fnArr[i] =  function(){
    return i
  };
}
console.log(fnArr[3]())