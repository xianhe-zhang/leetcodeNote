class Event {
  constructor () {}

  //定义一个容器,用来装事件,因为订阅者可以是多个
  handlers = {}



  //事件添加方法
  addEventListeners (type, handler) {
    //首先判断handler有没有type事件,如果没有则新建一个数组容器。
    if (!(type in this.handlers)) {
      this.handlers = [];
    }
    // 将事件存入
    this.handlers[type].push(handler);
  }

  //触发事件两个参数（事件名，参数/callback）
  dispatchEvent (type, ...params) {
    //若没有注册该事件，则抛出错误
    if (!(type in this.handlers)) {
      return new Error('Not register yet!');
    }
    //遍历触发
    this.handlers[type].forEach(handler => {
      handler(...params);
    });
  }

  //事件移除函数(事件名，删除的事件，若无第二个参数，则删除该事件的订阅和发布)
  removeEventListener (type, handler) {
    if (!(type in this.handlers)) {
      return new Error("Invalid event!");
    }
    if (!handler) {
      //直接移除事件
      delete this.handlers[type];
    } else {
      const idx = this.handlers[type].findIndex(element => {
        element == handler
      })
      //排除异常
      if (idx == undefined) {
        return new Error('No bind event!');
      }
      //移除事件
      this.handlers[type].splice(idx, 1);
      if (this.handlers[type].length == 0) {
        delete this.handlers[type];
      }
    }
  }
}

"这也就是事件监听机制，可以创建出很多事件直接依赖该类，并且可以通过该类中的API为订阅/绑定该类topic的事件dispatch出去相同的操作。"