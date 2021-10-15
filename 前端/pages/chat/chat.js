const app = getApp();
//引入插件：微信同声传译
const plugin = requirePlugin('WechatSI');
//获取全局唯一的语音识别管理器recordRecoManager
const manager = plugin.getRecordRecognitionManager();

Page({
  data: {
    scrollTop: 1000,
    indicatorDots: false,
    autoplay: false,
    interval: 5000,
    duration: 1000,
    msgs: [{
      type: 'text',
      msg: '你好，我是大白，你的私人聊天机器人。我长成这样是为了让人看起来更想拥抱 (づ ●─● )づ，快来和我聊聊叭~',
    },
    ],  //存放消息的数组，初始化了第一个消息
    recordState: false, //录音状态
    inputdata: '',  //输入框中的数据
    userInfo: {}   //用户数据
  },

  //监听页面加载
  onLoad: function () {
    //识别语音,实例化语音识别管理器对象manager
    this.initRecord();

    //控制台调试显示消息数组
    console.log('onLoad test');
    console.log(this.data.msgs);
    //如果全局的用户数据有，直接用；如果没有判断是否授权，授权了就调用函数读取
    if (app.globalData.userInfo) {
      this.setData({
        userInfo: app.globalData.userInfo,
        hasUserInfo: true
      })
    } else if (this.data.canIUse) {
      // 由于 getUserInfo 是网络请求，可能会在 Page.onLoad 之后才返回
      // 所以此处加入 callback 以防止这种情况
      app.userInfoReadyCallback = res => {
        this.setData({
          userInfo: res.userInfo,
          hasUserInfo: true
        })
      }
    } else {
      // 在没有 open-type=getUserInfo 版本的兼容处理
      wx.getUserInfo({
        success: res => {
          app.globalData.userInfo = res.userInfo
          this.setData({
            userInfo: res.userInfo,
            hasUserInfo: true
          })
        }
      })
    }
  },

  //识别语音 -- 初始化
  initRecord: function () {
    const that = this;
    // 有新的识别内容返回，则会调用此事件
    manager.onRecognize = function (res) {
      console.log(res)
    }
    // 正常开始录音识别时会调用此事件
    manager.onStart = function (res) {
      console.log("成功开始录音识别", res)
    }
    // 识别错误事件
    manager.onError = function (res) {
      console.error("error msg", res)
    }
    //识别结束事件
    manager.onStop = function (res) {
      console.log('..............结束录音')
      console.log('录音临时文件地址 -->' + res.tempFilePath); 
      console.log('录音总时长 -->' + res.duration + 'ms'); 
      console.log('文件大小 --> ' + res.fileSize + 'B');
      console.log('语音内容 --> ' + res.result);
      if (res.result == '') {
        wx.showModal({
          title: '提示',
          content: '您未说话，请重新语音',
          showCancel: false,
          success: function (res) {}
        })
        return;
      }
      var text = that.data.inputdata + res.result;
      that.setData({
        inputdata: text
      })
    }
  },

   //语音  --按住说话
   touchStart: function (e) {
    this.setData({
      recordState: true  //录音状态
    })
    // 语音开始识别
    manager.start({
      lang: 'zh_CN',// 识别的语言，目前支持zh_CN en_US zh_HK sichuanhua
    })
  },
  //语音  --松开结束
  touchEnd: function (e) {
    this.setData({
      recordState: false
    })
    // 语音结束识别
    manager.stop();
  },



  //读取用户数据的同时，也对消息进行发送和响应
  getUserInfo: function (e){
    app.globalData.userInfo = e.detail.userInfo
    this.setData({
      userInfo: e.detail.userInfo,
      hasUserInfo: true
    })
    //替换变量
    var input = this.data.inputdata;
    var that = this;
    var msgs = that.data.msgs;
    msgs.push({ msg: input, 'type': 'text' }); //把用户输入消息存入数组
    //更新数据，消息数组更新，输入数据重置为空
    that.setData({
      msgs: msgs,
      inputdata: '',
    })
    //微信数据请求  
    //如果无法返回结果，调用加载界面
    wx.showLoading({  
      title: '对方正在输入...',
      icon:'none'
    })
    //请求发送和响应
    wx.request({
      url: 'http://101.34.160.96/predict',  //服务器更改
      method: "POST",
      data: {
        msg: input
      },
      header: {
        'Content-Type': "application/x-www-form-urlencoded",
        'chartset': 'utf-8'
      },
      success: function (res) {  
        console.log(res.data)
        wx.hideLoading()
        msgs.push({ msg: res.data, 'type': 'text' });
        //需要先更新数据，再更新scrollTop才行，不能同时更新
        that.setData({
          msgs: msgs,
          inputdata: '',
        })
        that.setData({scrollTop: that.data.scrollTop + 500});
      }
    })

  },


  //获取输入框里的数据
  setInputValue: function (e) {
    console.log(e.detail);
    this.setData({
      inputdata:e.detail.value
    })
  },


  onShareAppMessage: function () {
    // 用户点击右上角分享
    return {
      title: '快来一起和大白聊天~', // 分享标题
      desc: '闲着没事，聊两句', // 分享描述
    }
  },
})  