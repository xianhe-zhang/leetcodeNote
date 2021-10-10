// 项目框架重写

// 如何部署环境？如何执行？
首先我们需要NPM
也需要知道了解Webpack，因为vue是基于webpack进行打包的，开的本地服务器也是
根目录下执行命令：npm run start

而我们的webpack的位置下，默认

// main.js  可以理解为项目的入口，执行main.js从而自动化执行整个流程。
// 主要用于import：css处理器，组件库，vuex，路由，app，和vue
<script>
import Vue from 'vue'
import App from './router'
import VueAwesomeSwiper from 'vue-awesome-swiper'

vue.use(VueAwesomeSwiper)

new Vue({
  el: '#app',
  router,
  store,
  components: {App},
  templeate: '<App/>'
})
</script>

// router/store/assets都是文件夹
  // 其中router是vue-router，store是vuex



// vue-router  ->  index.js
<script>
import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router)

export default new Router({
  routes:[{
    path: '/',
    name: 'Home',
    component: Home
  },
  {}], //这里再跟的都是都是路径，看看我们的router这个组件可以跳转到哪里
  scrollerBehavior()     //自定义方法，可以进行传参和return；
})

</script>

// App.vue
这里很简单，没什么东西。
就是用一个keep-alive标签包裹着一个router-view的标签。
前者是为了避免重复渲染不活跃的内容；后者是路由到默认的路径，及我们的home页面，这个是在vue-router中的路径下设置的。



// 下面我们来看一下具体的页面及其组件
// Home
首先在页面下我们插入home.vue这个核心文件，在里面我们规定了引入什么插件与组件

每一个vue实例中会有几个关键部分
name/components/data/computed/methods/mounted/activated
分别是：
实例名字
利用到的组件
本组件的数据，一般是return出去，起到封装的作用，以免数据被污染，胡乱修改
计算属性
方法
挂载在mounted下面的
activated就是唤醒的时候进行判断的

// vue组件的三大件分别是<template/> <style/>
<script>
  export default{
  computed: {
    pages () {
      const pages = []
      this.list.forEach((item, index) => {  //对list调用了forEach的箭头函数
        const page = Math.floor(index / 8)  //用来规定这个item属于第几页
        if (!pages[page]) {
          pages[page] = [] 
        }
        pages[page].push(item)  
      })
      return pages  //最终pages出来的page[1]有八个icon
    }
  }}
</script>
这项目里涉及到的大部分都是css，但cs这种东西布局太玄学了。

在这种子子组件里面的有一些很有趣的东西，可以学到
 <router-link
        tag="li"
        class="item border-bottom"
        v-for="item of list"
        :key="item.id"
        :to="'/detail/' + item.id"
      >
比如我们给这个router-link一个tag，class的话就是用于css选择的
v-for的话
key值给到的是绑定作用
to就是路由到哪里

在template中的适用v-指令的时候，我们会有缩略的写法。
比如:x是直接绑定
@时间，我忘记是事件监听，还是事件绑定了。

getCityInfor () {
  axios.get('/api/city.json')
    .then(this.handleGetCityInfoSucc) 
}
// 这个就是用来处理axios请求数据的方法，在请求成功后，直接跟一个.then去处理数据，好计谋呀。
