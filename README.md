# ChatBot
微信小程序聊天机器人

- 小程序前端+python flask后端+人工智能算法+服务器端部署uWSGI+Nginx+Ubuntu（腾讯云）
- 算法部分：

![img](https://imglf6.lf127.net/img/cDR1NkVtV3pKMEk1WE9kaHRXOXJEaFZ5ZnlKN3IzUHRpWG96R3lPL2hTNmJwZCtzZC9FVzFBPT0.png)

![img](https://imglf3.lf127.net/img/aEpOalRwZHRiUi8wcmw3enpxYXBLWTJiTDlaekIwcUx3eFVsc2VjaW5VL0NjRDBaM1Z3TTdBPT0.png)

-  训练挑选的机器模型因超过100M，无法上传，文件夹里只有对应的语料库和分词词典，训练模型获取见百度云链接： https://pan.baidu.com/s/1gyD7JIKMUtV4OWHs58P37w  提取码：6zlg
-  flask后端主要是初始化机器模型，并设置路由对小程序发来的请求进行响应

- flask后端和机器模型均已放至服务器端，具体部署见相关markdown

- web服务器目前还在运行，小程序端开启不校验合法域名，可运行程序：

![img](https://imglf4.lf127.net/img/RnA0T1dLRXBhZHpSTnlPeUlSUW44a3hRd1hmVDVaUlFRaGFvNVM3R1U1NENWTzZmYkV5ZEZBPT0.png)



