// 防抖 ✅
// 节流 ✅
// 数组去重  ✅
// promise.all ✅
// promise.any ✅
// promise ✅
// 原型链继承✅ 
// bind/call/apply ✅
// 手写ajax✅
// 深拷贝 ✅
// ！！json数组转换✅
// 数组转tree结构✅
// 函数add 接受任意数量的数字✅
// 驼峰改写✅
// 快排✅
// ES5和ES6求数组最大值✅
// 实现四则运算器✅
// 实现矩形相交面积✅
// 构建一个类✅
// 改代码输出
var obj = {
  value : 'obj',
  fn : function(){ console.log(this.value)}
}
var fn = obj.fn;
fn();

// 手写promise
function loadImg(img) {
  return new Promise(                                 //返回一个promise对象，拥有两个参数
    (resolve, rejected) => {                          //这里用箭头是看什么情况下如何进行各自的处理
      const img = document.createElement('img');      //img成dom结构，就可以利用api了
      img.onload = () => {
        resolve(img);                                 //如果成功返回resolve状态下的img
      }
      img.onerror = () => {
        const err = new Error(`图片加载失败 ${src}`);   //这个按数字键盘前面的·就可以了
        rejected(err);                                //传惨进去
      }
      img.src = src                                   //img对象的src等于我们最开始传入的src？
    }
  );
}


//深拷贝
function deepClone(obj) {
  let result = Array.isArray(obj)?[]:{};                  //判断obj是数组还是对象
  if (obj && typeof obj === "object"){                    //不为空且为对象时，进入
    for(let key in obj){
      if(obj.getOwnProperty(key)){                        //判断key是否是当前obj的属性
        if(obj[key] && typeof obj[key] === "object"){     //并且针对该属性判断是否为对象...如果有的话，意味着还有结构可以去遍历，因此进入递归
          result[key] = deepClone(obj[key]);
        }else{
          result[key] = obj[key];
        }
      }
    }
  }
  return result;
}


//数组去重
//ES5  indexof
function unique(arr) {
  var res = arr.filter(function(item, index, array){
    return array.indexOf(item) === index
  })
  return res
}

//ES6 set
var unique = arr => [... new Set(arr)]

//sort 去重
Array.prototype.unique = function () {
  const newArray = [];

  for (let i = 0; i<this.length; i++) { 
    for (let j=i+1; j<this.length; j++) {
      if(this[i] === this[j]){
        j = ++i;
      }
    }
    newArray.push(this[i]);
  }
  return newArray
}
//双for循环，用push/也可以用splice
Array.prototype.unique = function () {
  const newArray = [];
  this.sort();
  for (let i=0; i<this.length; i++){
    if (this[i] !== this[i+1]) {
      newArray.push(this[i]);
    }
  }
  return newArray;
}

//手写Ajax
function ajax(url, successFn) {
  const xhr = new XMLHttpRequest();
  xhr.open("GET", url, true);
  xhr.onreadystatechange = function () {
    if (xhr.readyState === 4 && xhr.status === 200){
      successFn(xhr.responseText);
    }
  }
  xhr.send(null);
}

//手写bind
Function.prototype.bind = function () {
  if (typeof this !== 'function') {
    return ;
  }
  
  var _self = this;
  var args = Array.prototype.slice.call(arguments, 1)
  return function () {
    return _self.apply(thisArg, args.concat(Array.prototype.slice.call(arguments)));
  }
}


//手写promise.all
let promise1 = new Promise(function (resolve, reject) {});
let promise2 = new Promise(function (resolve, reject) {});
let promise3 = new Promise(function (resolve, reject) {});

let p = Promise.all(promise1, promise2, promise3);

p.then(
  function () {

}, 
  function(){

})

//防抖 -- n秒内没有再次触发的话，执行回调函数
function debounce(fn, delay) {
  let timer = null;
  return function () {
    let context = this;
    let args = arguments;
    if (timer) {
      clearTimeout(timer);
    }
    timer = setTimeout (function(){
      fn.apply(context, args);
    }, delay)
  }
}

//节流 -- 只有满足一定时间的delay才拥有触发fn的能力。
function throttle(fn, delay) {
  return function() {
    let context = this;
    let args = arguments;
    let now = new Date();
    if(now - last >= delay) {
      fn.apply(context, args);
      last = now;
    }
  }
}

// 二分
// js可以不用显示定义形参，通过arguments对象获取对应的参数，这个就是js的特性，弱类型，也有助于多态。
function BinarySearch(nums, target){
  if (nums === target){
    return -1;
  }
  let left = 0;
  let right = nums.length;

  while(left<right){
    let mid = left + (right - left) / 2;
    if(nums[mid] === target){
      return mid;
    }

    if(nums[mid] < target){
      left = mid + 1;
    }else{
      right = mid;
    }
  }
  return -1;
}

// 快排
function quickSort(arr, low, high){
  function partition(arr, low, high){
    let i = low - 1;
    let pivot = arr[high];

    for(let j=low; j<high; j++){
      if(arr[j] <= pivot){
        i++;
        arr[i], arr[j] = arr[j], arr[i];
      }
    }
    return i+1
  }
  if(low < high){
    let pi = partition(arr,low, high);
    quickSort(arr,low,pi-1);
    quickSort(arr,pi+1,high);
  }
}

//原型链继承
function Duck () {}
Duck.prototype = new Animal('duck');
const duck = new Duck();

//驼峰改写
function toUp(str){
  var arr = str.split('-');
  for (var i = 1; i < arr.length; i++){
    arr[i] = arr[i].charAt(0).toUpperCase() + arr[i].substring(1);
  }
  return arr.join('')
}

//json数组转换
// JSON.stringify(), JSON.parseArray()
let jsonStr = '[{"id":"01","open":false,"pId":"0","name":"A部门"},{"id":"01","open":false,"pId":"0","name":"A部门"}]';
let jsonObj = $.parseJSON(jsonStr);//json字符串转化成json对象(jq方法)
//var jsonObj =  JSON.parse(jsonStr)//json字符串转化成json对象（原生方法）
let jsonStr1 = JSON.stringify(jsonObj)//json对象转化成json字符串

//console.log(jsonStr1+"jsonStr1")

//json对象转化成json数组对象
let arr1=[];
for(let i in jsonObj){ //i是key，jsonObj[i]是value
    //var o={};
    //o[i]=jsonObj[i];
    arr1.push(jsonObj[i]);            
}
//console.log(arr1);
//console.log(typeof(arr));
var jsonStr11 = JSON.stringify(arr1)//json数组转化成json字符串
//console.log(jsonStr11);
var jsonArr = [];
for(var i in jsonObj){
        jsonArr[i] = jsonObj[i];
}
//console.log(jsonArr);
//console.log(typeof(jsonArr))


// 数组转tree结构
function listToTree(oldArr){
  oldArr.forEach(element => {
    let parentId = element.parentId;
    if(parentId !== 0){
      oldArr.forEach(ele => {
        if(ele.id == parentId){ //当内层循环的ID== 外层循环的parendId时，（说明有children），需要往该内层id里建个children并push对应的数组；
          if(!ele.children){
            ele.children = [];
          }
          ele.children.push(element);
        }
      });
    }
  });
  console.log(oldArr) //此时的数组是在原基础上补充了children;
  oldArr = oldArr.filter(ele => ele.parentId === 0); //这一步是过滤，按树展开，将多余的数组剔除；
  console.log(oldArr)
  return oldArr;
}

//函数add 接受任意数量的数字
function getData(){
 
  //js的arguments,可以访问所有传入值。
  alert(arguments.length);

var j = arguments;
var sum = 0;

      //遍历arguments中每个元素，并累加
for(i=0;i<j.length;i++){

  //判断参数是否满足三点：number类型，不是NaN，排除除数为0
if( (typeof(j[i])=="number") && (!isNaN(j[i])) && (j[i]!=Infinity) ){
    sum+=j[i];
      }
    }
      alert(sum);
}

/*先输出 传入的参数长度 6
  再输出 参数的求和 21*/
getData(1,2,3,4,5,6);
getData(1,2,'a');//3 


//实现用promise的红绿灯
function red() {
  console.log("红色");
}

function green() {
  console.log("绿色");
}

function yellow() {
  console.log("黄色");
}

const light = function(timer, choice){
  return new Promise(resolve => {
    setTimeout(() => {
      choice();
      resolve();
    }, timer);
  });
}

const stepOn = function(){
  Promise.resolve.then(() => {
    return light(5000, green);
  }).then(() => {
    return light(1000, yellow);
  }).then(() => {
    return light(5000, red);
  }).then(() => {
    return stepOn();
  })
}

stepOn();


//实现矩形相交面积
var rectA = {
  left:-3, 
  bottom:0, 
  right:3, 
  top:4
}
var rectB = {
  left:0, 
  bottom:-1, 
  right:9, 
  top:2
}
function getIntersectArea(rectA, rectB){
  // 返回相交面积 或异常
  var width = Math.min(rectA.right, rectB.right) - Math.max(rectA.left, rectB.left);
  var height = Math.min(rectA.top, rectB.top) - Math.max(rectA.bottom, rectB.bottom);
  if(width <= 0 || height <= 0){
    throw new ArgumentException();
  }else{
    return width * height;
    
  }
}
console.log(getIntersectArea(rectA,rectB) )//true

//看输出与改输出
var fnArr = [];
for (var i = 0; i < 10; i ++) {
  fnArr[i] =  function(){
    return i
  };
}
console.log(fnArr[3]()) //10

var fnArr = [];
for (let i = 0; i < 10; i ++) { //var -> let
  fnArr[i] =  function(){
    return i
  };
}
console.log(fnArr[3]()) //3



//ES5和ES6求数组最大值
//ES5 
var max = Math.max.apply(this,[1,2,3])
//ES6
var max = Math.max(...[1,2,3])


//实现四则运算器
function account()
{
 	var op1=prompt("请输入第一个数：","");
	var op2=prompt("请输入第二个数：","");
	var sign=prompt("请输入运算符号","")
	var result;
	opp1=parseFloat(op1);
	opp2=parseFloat(op2);
	switch(sign)
	{
		case "+":
			result=opp1+opp2;
			break;
		case "-":
			result=opp1-opp2;
			break;
		case "*":
			result=opp1*opp2;
			break;
		default:
			result=opp1/opp2;
	}
	alert("两数运算结果为："+op1+sign+op2+"="+result);
}


//构造一个类 - 利用到ES6里面的class
// 命名类
let Example = class Example {
  constructor(a) {
      this.a = a;
  }
}
let exam1 = Example(); 