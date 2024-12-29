import matplotlib.pyplot as plt
import numpy as np

def mypolar(data):
    hist, bin_edges = np.histogram(data, bins='auto')

    # 角度
    theta = (bin_edges[1:] + bin_edges[:-1]) / 2 * np.pi / max(bin_edges)

    # 绘制极坐标柱状图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.bar(theta, hist, width=np.pi*2/len(bin_edges), bottom=0)

    # 设置标签和标题
    # ax.set_thetagrids(np.degrees(theta), labels=bin_edges)
    ax.set_title('gragh')

    # 显示图形
    plt.show()    

# 数据
data = np.random.normal(0,1,size=100)


