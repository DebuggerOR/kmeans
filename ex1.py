import numpy as np
import math
from matplotlib.pyplot import imread
from init_centroids import init_centroids
import matplotlib.pyplot as plt


def show_image(path, centroids):
    k = len(centroids)
    A = imread(path)
    A_norm = A.astype(float) / 255.
    img_size = A_norm.shape
    B_norm = A_norm
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            min_c = 0
            min_dst = float('inf')
            for l in range(k):
                dst = euclid_dst(A_norm[i][j], centroids[l])
                if dst < min_dst:
                    min_c = l
                    min_dst = dst
            B_norm[i][j] = centroids[min_c]
    plt.imshow(B_norm)
    plt.grid(False)
    plt.show()


def load_image(path):
    A = imread(path)
    A_norm = A.astype(float) / 255.
    img_size = A_norm.shape
    X = A_norm.reshape(img_size[0] * img_size[1], img_size[2])
    return X


def euclid_dst(p1, p2):
    dim = len(p1)
    sqr_dst = 0.0
    # calc (a1-b1)^2 + .. + (am-bm)^2
    for i in range(dim):
        sqr_dst += (p1[i] - p2[i]) ** 2
    return math.sqrt(sqr_dst)


def print_avg_loss(clusters, centroids, X):
    # init k and sum
    k = len(clusters)
    sum = 0
    # sum square dst from point to it's centroid
    for i in range(k):
        for point in clusters[i]:
            sum += (euclid_dst(point, centroids[i]))
    # print avg of loss
    print(sum / len(X))


def print_iter(t, centroids):
    # determine dim of points
    dim = len(centroids[0])
    # format is 'iter 1: [0.01, 0.01, 0.01], [0.41, 0.36, 0.32]'
    output = "iter " + str(t) + ":"
    first = 1
    for c in centroids:
        if first:
            output += " ["
            first = 0
        else:
            output += ", ["
        output += str(np.floor(c[0] * 100) / 100)
        for i in range(1, dim):
            output += ", " + str(np.floor(c[i] * 100) / 100)
        output += "]"
    # replace to fix prints
    output = output.replace("0.0,", "0.,")
    output = output.replace("0.0]", "0.]")
    # print iter
    print(output)


def print_iter2(t, cent):
    # official printing func
    output = "iter " + str(t) + ": "
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        output += ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').\
            replace('\n', ' ').replace(' ]',']').replace(' ', ', ')
    else:
        output += ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').\
            replace('\n', ' ').replace(' ]',']').replace(' ', ', ')[1:-1]
    print(output)


def divide_clusters(centroids):
    # k clusters, each is list of points
    clusters = [[] for i in range(k)]
    # for all points in X
    for point in X:
        min_c = 0
        min_dst = float('inf')
        # find closest centroid
        for i in range(k):
            dst = euclid_dst(point, centroids[i])
            if dst < min_dst:
                min_c = i
                min_dst = dst
        # append to cluster of closest centroid
        clusters[min_c].append(point)
    return clusters


def update_centroids(centroids, clusters):
    # calc dim and k
    dim = len(centroids[0])
    k = len(centroids)
    # for k centroids
    for i in range(k):
        # centroid = avg of cluster
        sum_point = [0] * dim
        for point in clusters[i]:
            for j in range(dim):
                sum_point[j] += point[j]
        cluster_size = len(clusters[i])
        if cluster_size > 0:
            for j in range(dim):
                sum_point[j] /= len(clusters[i])
        centroids[i] = sum_point
    return centroids


def kmeans(X, k):
    # init k centroids and num iters
    centroids = init_centroids(X, k)
    num_iters = 10
    # print init centroids
    print_iter(0, centroids)
    # iterate num iterations
    for i in range(num_iters):
        # update clusters and centroids
        clusters = divide_clusters(centroids)
        centroids = update_centroids(centroids, clusters)
        # print_avg_loss(clusters, centroids, X)
        # print_iter(i + 1, centroids)
        print_iter2(i + 1, centroids)
    # show_image('dog.jpeg', centroids)


if __name__ == "__main__":
    image_path = 'dog.jpeg'
    X = load_image(image_path)
    # run kmeans on X with 2, 4, 8, 16 clusters
    k_arr = [2, 4, 8, 16]
    for k in k_arr:
        print("k=" + str(k) + ":")
        kmeans(X, k)
