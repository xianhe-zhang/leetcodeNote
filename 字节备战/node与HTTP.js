"处理get请求"
const http = require('http')
const querystring = require('querystring')

const server = http.createServer((req, res) => {
  console.log("method: ", req.method)  //GET
  const url = req.url
  console.log("url:", url)
  req.query = querystring.parse(url.split('?')[1])
  res.end(
    JSON.stringify(req.query)
  )
})
server.listen(8000)


"处理post请求"
const http = require('http')

const server = http.createServer((req, res) => {
  if (req.method === 'POST') {
    console.log('req content-types: ', req.headers['content-type'])
    //接受
    let postData = ''
    req.on('data', chunk =>  {
      postData += chunk.toString()
    })
    req.on('end', ()=> {
      console.log('postData: ', postData)
      res.end('hello you')
    })
  }
})
console.log('ok')


"nodejs处理路由"
const http = require('http');
const server = http.createServer((req, res) => {
  const url = req.url
  const path = url.split('?')[0]
  res.end(path);
})
server.listen(8000);


//设置返回格式 JSON
res.setHeader('Content-type', 'applicaiton/json')

//返回的数据
const resData = {
  method,
  url,
  path,
  query
}

//返回
if(method === 'GET') {
  res.end(
    JSON.stringify(resData)
  )
}
if (method === 'POST') {
  let postData = ''
  req.on('data', chunk =>  {
    postData += chunk.toString()
  })
  req.on('end', ()=> {
    console.log('postData: ', postData)
    res.end(
      JSON.stringify(resData)
    )
  })
}
