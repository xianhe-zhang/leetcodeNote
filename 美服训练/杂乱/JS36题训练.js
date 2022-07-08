// 1.数据类型判断
function typeOf(obj) {
    // call就是把作用域传递给/绑定给另一个方法。在指定的作用域下使用指定的方法
    // 这里就是针对obj使用前面的那一串方法
    let res = Object.prototype.toString.call(obj).split(' ')[1]
    res = res.substring(0, res.length - 1).toLowerCase
    // return Object.prototype.toString.call(obj).slice(8, -1).toLowerCase()
}
typeOf([])          // 'array'
typeOf({})          // 'object'
typeOf(new Date)    // 'date'

// 2. 原型链继承
// 我们创建的每一个函数都有一个prototype（原型）属性，这个属性是一个指针，指向一个对象，即原型对象。
// prototype就是通过调用构造函数而创建的那个对象实例的原型对象
// 使用原型对象的好处是可以让所有对象实例共享它所包含的属性和方法。
function Animal() { 
    this.colors = ['b', 'w']
}
Animal.prototype.getColor = function() {
    return this.colors
}
function Dog() {}
Dog.prototype =  new Animal()

let dog1 = new Dog()
dog1.colors.push('brown')
let dog2 = new Dog()
console.log(dog2.colors)  // ['black', 'white', 'brown']


// 3. 构造函数实现继承
function Animal(name) {
    this.name = name
    this.getName = function() {
        return this.name
    }
}
function Dog(name) {
    // 好像是吧dog的name绑定到了animal的方法中
    Animal.call(this, name)
}
// 使得dog的prototype 指向 animal的构造函数
Dog.prototype =  new Animal()


// 4.class 实现继承 
// ES6新特性？和java好像。
class Animal {
    constructor(name) { 
        this.name = name
    }
    getName() {
        return this.name
    }
}
class Dog extends Animal {
    constructor(name, age) {
        super(name)
        this.age = age
    }
}


// 5. 数组去重
// ES5
    // indexOf() 方法可返回某个指定的字符串值在字符串中首次出现的位置。
    // 如果没有找到匹配的字符串则返回 -1。
    // 注意： indexOf() 方法区分大小写。
function unique(arr) {
    // item = 当前元素的值；index；array就是我们的arr对象
    var res = arr.filter(function(item, index, array) {
        return array.indexOf(item) === index
    })
    return res
}

// ES_6
// 新建arr的set对象，自动去重，然后...展开在[]中，并且直接利用箭头函数进行传参
var unique = arr => [...new set(arr)]



// 6. 数组扁平化
// ES5 - Recursion 挺像算法题的
function flattern(arr) {
    var result = []
    for (var i = 0, len = arr.length; i < len; i++) {
        if(Array.isArray(arr[i])) {
            result = result.concat(flattern(arr[i]))
        } else { 
            result.push(arr[i])
        }
    }
    return result
}

// ES 6
// 如果发现arr里的元素有array的话，那么我们就继续拆开
// ...会把arr中的每一个元素展开，concat会把他们连接起来
// [1,2,[3,4]] -> 1 2 [3,4] ->[1,2,3,4] concat会把这些array连接起来用的！
function flattern(arr) {
    while (arr.some(item => Array.isArray(item))) {
        arr = [].concat(...arr)
    }
    return arr
}


// typeof会返回一个变量的基本类型，instanceof返回的是一个布尔值
// instanceof 可以准确地判断复杂引用数据类型，但是不能正确判断基础数据类型
// 而 typeof 也存在弊端，它虽然可以判断基础数据类型（null 除外），但是引用数据类型中，除了 function 类型以外，其他的也无法判断


// 7. 深浅拷贝
// 浅拷贝 只考虑对象类型
function shallowCopy(obj) {
    if (typeof obj != 'object') return;

    // 如果obj是Array的实例的话
    let newObj = obj instanceof Array ? []:{}
    for (let key in obj) {
        if (obj.hasOwnProperty(key)) {
            newObj[key] = obj[key]
        }
    }
    return newObj
}

// 简单版深拷贝：只考虑普通对象属性，不考虑内置对象和函数。 与shallowcopy类似，但是每一层他都拷贝
function deepClone(obj) {
    if (typeof obj !== 'object') return;
    var newObj = obj instanceof Array ? [] : {};
    for (var key in obj) {
        if (obj.hasOwnProperty(key)) {
            newObj[key] = typeof obj[key] === 'object' ? deepClone(obj[key]) : obj[key];
        }
    }
    return newObj;
}

// 8. 发布订阅模式
class EventEmitter {
    constructor() {
        this.cache = {}
    }
    // 如果存在name，那么新增fn
    // 如果不存在name，直接新增kv
    on(name, fn) {
        if (this.cache[name]) {
            this.cache[name].push(fn)
        } else {
            this.cache[name] = [fn]
        }
    }
    // 取消订阅
    off(name, fn) {
        let tasks = this.cache[name]
        if (tasks) {
            const index = tasks.findIndex(f => f === fn || f.callback === fn)
            if (index >= 0) {
                tasks.splice(index, 1)
            }
        }
    }
    // 发布事件，如果cache中存在该事件，意味着有人订阅了，那么调用里面的args和函数，这个时候就自动调用我们的event handler function
    emit(name, once = false, ...args) {
        if (this.cache[name]) {
            // 创建副本，如果回调函数内继续注册相同事件，会造成死循环
            let tasks = this.cache[name].slice()
            for (let fn of tasks) {
                fn(...args)
            }
            if (once) {
                delete this.cache[name]
            }
        }
    }
}

// 测试
let eventBus = new EventEmitter()
let fn1 = function(name, age) {
	console.log(`${name} ${age}`)
}
let fn2 = function(name, age) {
	console.log(`hello, ${name} ${age}`)
}
// 注册aaa事件，并且将fn传递进去，是为了如果发布了aaa事件，可以执行fn。
eventBus.on('aaa', fn1)
eventBus.on('aaa', fn2)
eventBus.emit('aaa', false, '布兰', 12)
// '布兰 12'
// 'hello, 布兰 12'

// 剩下的再有两天就差不多了。