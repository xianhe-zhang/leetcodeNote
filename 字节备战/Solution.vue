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

// App.vue




