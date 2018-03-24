### ***话不多说直接上代码***
```python
import matplotlib.pyplot as plt
import seaborn as sns

data = test_feature.corr()  #test_feature => pandas.DataFrame#
sns.heatmap(data)
plt.show()
```
### **效果图**

![这里写图片描述](http://img.blog.csdn.net/20180324161411443?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcm9nZXJfcm95ZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### **[顺带分享一篇机器学习实践相案例](https://mp.weixin.qq.com/s?__biz=MzIwMTgwNjgyOQ==&mid=2247486659&idx=1&sn=8a41a9734457a39a7b26d6839aead8e3&chksm=96e90a41a19e83576d46484edfbb144b40a0e2fa6215aedaebd5dca75ce0b1be2e75fd5f6067&mpshare=1&scene=23&srcid=0320lmGvrPmluO72YBiu0bGl#rd)**
