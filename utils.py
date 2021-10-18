# -*- coding: UTF-8 -*-
import glob
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans, avg_iou

save_path= 'save20'  #
def plotPoints(data, centroids):
    """
        把所有点的画出来
    """
    resfiles = '/Clustering_result_ ' + str(len(centroids)) + '.png'

    plt.scatter(data[:, 0], data[:, 1], marker='.', label='data points', s=5, c=data[:, 2])
    plt.scatter(centroids[:, 0], centroids[:, 1], color='r', label='centroids', s=100, marker="*")
    plt.title('Select % d th centroid' % centroids.shape[0])
    plt.savefig(save_path + resfiles)
    plt.show()


def accShow(accuracy, k):
    plt.plot(k, accuracy, label='k-accuracy', linewidth=1, color='blue', marker='o', markerfacecolor='r', markersize=3)

    for x, y in zip(k, accuracy):
        plt.text(x, y, round(y, 2), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Number of Clusters')
    plt.ylabel('avg IOU(%)')
    plt.savefig(save_path + "/bestCluster1.png")
    plt.legend()
    plt.show()


def show_width_height(data, bins=50):
    """
        用直方图显示边界框 长, 宽, 长/宽 的直方分布
    """
    if data.dtype != np.float32:
        data = data.astype(np.float32)  # 把数据转换为浮点型

    width = data[:, 0]
    height = data[:, 1]
    ratio = width / height

    plt.figure(1, figsize=(20, 6))
    plt.subplot(131)
    plt.hist(width, bins=bins, color='green')
    plt.xlabel('width')
    plt.ylabel('number')
    plt.title('Distribution of Width')

    plt.subplot(132)
    plt.hist(height, bins=bins, color='blue')
    plt.xlabel('Height')
    plt.ylabel('Number')
    plt.title('Distribution of Height')

    plt.subplot(133)
    plt.hist(ratio, bins=bins, color='magenta')
    plt.xlabel('Width / Height')
    plt.ylabel('number')
    plt.title('Distribution of aspect ratio(Width / Height)')
    plt.savefig(save_path +"/shape-distribution.png")
    plt.show()


def sort_cluster(cluster):
    """
        Sort the cluster to with area small to big.
        根据聚类中心点的面积进行从小到大排序
    """
    if cluster.dtype != np.float32:
        cluster = cluster.astype(np.float32)
    area = cluster[:, 0] * cluster[:, 1]
    cluster = cluster[area.argsort()]
    ratio = cluster[:, 0] / cluster[:, 1]

    # TODO 改进
    return np.concatenate([cluster, ratio[:, None]], axis=1)


# TODO 对上面的函数的一个改进, 上面的函数暂时不采用
def sortClusters(cluster):
    if cluster.dtype != np.float32:
        cluster = cluster.astype(np.float32)

    ratio = cluster[:, 0] / cluster[:, 1]  # Width / Height
    cluster, ratio = cluster[ratio.argsort()], ratio[ratio.argsort()]

    # TODO 改进
    return np.concatenate([cluster, ratio[:, None]], axis=1)


def load_dataset(path, normalized=True):
    """
        加载 xml 文件
    """
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))

        for obj in tree.iter("object"):
            if normalized:
                xmin = int(obj.findtext("bndbox/xmin")) / float(width)
                ymin = int(obj.findtext("bndbox/ymin")) / float(height)
                xmax = int(obj.findtext("bndbox/xmax")) / float(width)
                ymax = int(obj.findtext("bndbox/ymax")) / float(height)
            else:
                xmin = int(obj.findtext("bndbox/xmin"))
                ymin = int(obj.findtext("bndbox/ymin"))
                xmax = int(obj.findtext("bndbox/xmax"))
                ymax = int(obj.findtext("bndbox/ymax"))
            if (xmax - xmin) == 0 or (ymax - ymin) == 0:
                continue  # 避免除数为0
            dataset.append([xmax - xmin, ymax - ymin])

    return np.array(dataset)
