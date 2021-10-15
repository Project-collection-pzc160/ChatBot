## 服务器部署：Flask+uWSGI+Nginx+Ubuntu(腾讯云)

### 前言

web框架(如django、flask) + web服务器(如uWSGI) + nginx(反向代理、负载均衡)算是python web开发中比较经典的后端架构。因为自己也是第一次部署该架构，编写文档记录学习过程。

### 1.为什么使用这套架构？以及各部分发挥了什么作用？

#### 1.1 web服务器和web框架

二者是web开发的两大版块，**web服务器**用来接受客户端请求，建立连接，转发程序的响应内容。至于转发的内容是什么，交由**web框架**来处理，即处理相关的业务逻辑，如查询数据库、生成实时信息等。

显然，在ChatBot项目中flask是**web框架**，用来生成机器人的聊天内容，而uWSGI是**web服务器**，用来启动程序进行响应并返回内容，当然nginx其实也是**web服务器**，但主要是用来反向代理服务器，在个人的小型项目中，不需要nginx也可以部署应用。

#### 1.2 flask和uWSGI

**flask**是一个python开发的web微框架，微”意味着 Flask 旨在保持核心简单而易于扩展。flask依赖两个外部库： Jinja2模板引擎和 Werkzeug WSGI 工具集，分别对应模板渲染和消息响应功能，其他的功能flask支持用扩展来进行添加。ChatBot项目中主要是用到消息响应的功能，模板渲染因为有小程序作为前端可以不需要。

**uWSGI**是实现了**WSGI**、**uwsgi、http等协议**的一个web服务器。**WSGI**是一个通信协议，只要web服务器和web框架满足**WSGI协议**，它们就能相互搭配；而**uwsgi**是一种线路协议而不是通信协议，常用于在uWSGI服务器与其他web服务器的数据通信。

**注意：**之前提到使用nginx代理是可选项，两种情况下**uWSGI**所使用的协议是不同的，进行部署设置时也要注意这点。当没有使用nginx，仅仅是uWSGI+flask时，客户端是通过HTTP/HTTPS来通信的，所以uWSGI必须使用相应的协议，否则无法通信；当使用nginx进行代理时，nginx与uWSGI是使用uwsgi协议，部署设置时要使用socket，具体参考如下图。**ChatBot项目是基于第二种实现。**

![img](https://img-blog.csdnimg.cn/20190814110155935.png)

#### 1.3 Nginx

Nginx主要有三个作用：**反向代理、负载均衡和动静分离**。

**反向代理：**指以代理服务器来接受internet上的连接请求，然后将请求转发给内部网络上的服务器，并将从服务器上得到的结果返回给internet上请求连接的客户端，此时对外就表现为一个整体的代理服务器，隐藏了内部真实服务器。真实的服务器不能直接被外部网络访问，所以需要一台代理服务器，而代理服务器能被外部网络访问的同时又跟真实服务器在同一个网络环境，当然也可能是同一台服务器，端口不同而已（**ChatBot项目是此类情况，同在一台腾讯云服务器上，端口不同**)。

**负载均衡：**根据请求情况和服务器负载情况，将请求分配给不同的web服务器，保证服务器性能。负载均衡的机制有3种， (1) 循环 - 对应用程序服务器的请求以循环方式分发，
(2) 最少连接 - 下一个请求被分配给活动连接数最少的服务器，
(3) ip-hash - 哈希函数用于确定应为下一个请求选择哪个服务器（基于客户端的IP地址）。
**（跟反向代理一起搭配，ChatBot项目因只有一个web服务器，这部分不深入展开）**

**动静分离：**Nginx本身也是一个静态资源的服务器，Nginx可以放静态文件，先处理静态请求，再将动态请求转发给上游的web服务器，从而减少对web服务器的访问压力。**(ChatBot项目暂未使用此功能，后期可以考虑将前端使用的图片资源放到Nginx上)**

### 2.准备工作

#### 2.1 注册好腾讯云web服务器

#### 2.2 下载WinSCP工具

个人觉得挺好用的，图形化的SFTP客户端，同时支持SCP协议，能够在本地与远程计算机间安全的复制文件，而且在远程计算机上进行各种文件操作也很便捷。**ChatBot项目需要使用训练好的机器模型进行聊天内容的生成，用该工具将模型放到服务器端，同时可对服务器端的文件进行便捷操作，用户文件主要包括机器模型以及后端代码**

![img](https://imglf6.lf127.net/img/ek1BdFFYV3c1VklWcER4TUROU1hUUzU2K3hZTGVUaFZaZ29ZUFNEKzhJc2RBWGdNamxEZS93PT0.png)

![img](https://imglf5.lf127.net/img/OVlLL0llQWxIaUczUmo2d0pJdWlNdWo3NGJqdnVRZ2t0NmYwQXhQR0IwTk85QkNlSUkxRDlBPT0.png)

#### 2.3 将flask项目部署到服务器上

用pycharm专业版与服务器建立会话，将flask代码发布到上面，这样即使之后flask代码进行了更改，也可以很方便地同步到服务器。

### 3.配置

#### 3.1 为flask项目构建虚拟环境及解决依赖问题

安装virtualenv：

```
sudo pip install virtualenv
```

我的项目目录叫 /home/ubuntu/flask，进入项目目录后安装虚拟环境 (虚拟环境名叫 venv )：

```
virtualenv venv
```

在项目目录下就会建立一个新的 venv 目录，里面就是运行python 的基本环境的工具与指令，和包。 然后启用该环境，使用当前命令行状态进入虚拟环境，进入虚拟环境后，一切安装python的操作都会将包和引用装在虚拟环境内，而不会影响到全局的python 环境。

```
source venv/bin/activate （进入虚拟环境）
```

调用 activate 指令后命令符前就会出现 (venv) 字样，可通过 deactivate 退出虚拟环境。

在虚拟环境下安装项目需要的包:

(requirements.txt文件在pycharm的terminal终端输入pip freeze>requirements.txt命令后生成)

```
pip install -r requirement.txt  # 解决依赖问题
```

#### 3.2 配置Nginx

安装Nginx：

```
sudo apt-get install nginx
```

启动 nginx：

```
sudo /etc/init.d/nginx start
```

这时候在浏览器地址栏输入服务器的 ip 地址，看到Welcome to  Nginx就表明 Nginx 已经启动了。

将flask对应代码段修改为如下：

```
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
```

在浏览器输入服务器本机地址，加端口号5001就可以看到网页**(如果是网页项目可以看到，ChatBot项目的flask端只用于响应消息，所以没有特殊的界面）**

修改 Nginx 的默认配置文件：

(用文本编辑器打开，因为该文件是只读文件，加上sudo确保有权限修改)

[linux系统使用vi命令进入文件后怎么修改保存_Axiaoph的博客-CSDN博客](https://blog.csdn.net/qq_44116786/article/details/107289851)

[Linux Vim基本操作（文件的打开和编辑）完全攻略（有图有真相） (biancheng.net)](http://c.biancheng.net/view/805.html)

```
sudo vim /etc/nginx/sites-enabled/default 
```

打开后可以从 Nginx 默认配置中了解一些配置参数，具体可以看 Nginx 的文档。下面给出一个简单的配置：

**(注意：配置文件里不要加上这些注释，在配置uWSGI时试过会报错)**

```
server {                                                                       
listen 80;                   # 服务器监听端口                                                 
server_name x.x.x.x; # 这里写你的域名或者公网IP                                                    
charset      utf-8;          # 编码                                                  
client_max_body_size 75M;                                                      

location / {                                                                   
    include uwsgi_params;         # 导入uwsgi配置                                           
    uwsgi_pass 127.0.0.1:8000;    # 转发端口，需要和uwsgi配置当中的监听端口一致                            
    uwsgi_param UWSGI_PYTHON /home/自己创建的目录/venv;# Python解释器所在的路径（这里为虚拟环境）      
    uwsgi_param UWSGI_CHDIR /home/自己创建的目录;       # 项目根目录     
    uwsgi_param UWSGI_SCRIPT app_flask:app; # 项目的主程序，app_flask为运行文件的名称         
      }                                                                              
}
```

重启 Nginx：

```
sudo /etc/init.d/nginx restart
```

常用Nginx命令：

```
sudo nginx   #启动
sudo nginx -s stop   #停止
sudo nginx -s reload  #平滑启动，在不停止 nginx 的情况下，重启 nginx，重新加载配置文件，用新的工作进程代替旧的工作进程。
killall -9 nginx #杀死进程，端口被占用的时候使用 
```

#### 3.3 配置uWSGI

在安装 uWSGI 前，需要解决 uWSGI 的依赖问题，因为 uWSGI 是一个 C 语言写的应用，所以我们需要 C 编译器，以及 python 开发相关组件：

```
sudo apt-get install build-essential python-dev（这是两个包）
```

进入到项目的虚拟环境中安装uWSGI：

```
pip install uwsgi
```

在项目文件根目录新建配置文件uwsgi.ini，进行编写：

```
sudo vim uwsgi.ini
```

配置内容如下：

```
[uwsgi]
socket = 127.0.0.1:8000      # uwsgi的监听端口 
chdir = /home/ubuntu/flask   # 项目根目录
wsgi-file = app_flask.py     # flask程序的启动文件  
master = True                #开启master
callable = app               # flask在app_flask.py文件中的app名 
processes = 4                #配置进程数
threads = 2                  #配置线程数     
```

启动uWSGI：

```
uwsgi uwsgi.ini
```

可能会报如下错：

```
no internal routing support, rebuild with pcre support
```

执行操作：

```
pip uninstall uwsgi
sudo apt-get remove uwsgi
sudo apt-get install libpcre3 libpcre3-deb
pip install uwsgi -I --no-cache-dir
```

也可能找不到uwsgi命令，那么需要创建软链：

```
find / -name uwsgi  #找到uwsgi，一般应该就是在虚拟环境的bin文件夹里
ln -s /home/ubuntu/flask/venv/bin/uwsgi /usr/bin/uwsgi 
#注意这里/usr/bin/uwsgi必须要root用户才有权限访问，sudo -i切换成root再运行
```

常用uwsgi命令：

```
uwsgi uwsgi.ini  #启动，但要进入响应的目录下
uwsgi --ini /home/ubuntu/flask/uwsgi.ini   #启动
uwsgi --stop uwsgi.pid   #关闭
pkill -f -9 uwsgi    #删除
```

