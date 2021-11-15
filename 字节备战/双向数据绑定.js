//定义一个空对象
var obj = {}

// init
var val = 'zhang'; 

Object.defineProperty(obj, 'val', {
  get: function() {
    return val;
  },

  set: function (newVal) {
    val = newVal; //定义val修改后的内容
    document.getElementById('a').value = val;
    document.getElementById('b').innerHTML = val
  }
});
document.addEventListener('keyup', function(e) {
  obj.val = e.target.value;
}) 

//通过事件监听keyup触发set方法，而set修改访问器属性的同时，也修改了dom
//修改dom后，显示也随着修改
//这是利用js实现的，如何理解发布订阅模式，user是publisher，设备与js是订阅者。


