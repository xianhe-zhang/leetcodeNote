let jsonStr = '[{"id":"01","open":false,"pId":"0","name":"A部门"},{"id":"01","open":false,"pId":"0","name":"A部门"}]';
// let jsonObj = $.parseJSON(jsonStr);//json字符串转化成json对象(jq方法)
var jsonObj =  JSON.parse(jsonStr)//json字符串转化成json对象（原生方法）
let jsonStr1 = JSON.stringify(jsonObj)//json对象转化成json字符串
console.log(jsonStr1);