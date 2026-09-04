__authors__ = ['1708264','1704523','1589611']
__group__ = 'Team_37'

from utils_data import read_dataset, read_extended_dataset, crop_images
import numpy as np
from Kmeans import KMeans, get_colors
from KNN import KNN
import time

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    def retrieval_by_color(images, color_labels, query_colors):
        result = []
        for i in range(len(images)):
            for color in query_colors:
                if color in color_labels[i]:
                    result.append(i)
                    break
        return result
    

    def retrieval_by_shape(images, shape_labels, query_shape):
        result = []
        for i in range(len(images)):
            for shape in query_shape:
                if shape in shape_labels[i]:
                    result.append(i)
                    break
        return result
    

    def retrieval_combined(images, shape_labels, color_labels, query_shape, query_color):
        Icolors = retrieval_by_color(images, color_labels, query_color)
        Ishapes = retrieval_by_shape(images, shape_labels, query_shape)
        combined = []
        for color in Icolors:
            if  color in Ishapes:
               combined.append(color)
        return combined
    

    def get_shape_accuracy(etiquetes_knn, ground_truth):
        correctes = 0
        for knn, ground in zip(etiquetes_knn, ground_truth):
            if knn == ground:
                correctes += 1
        return (correctes / len(etiquetes_knn)) * 100


    def get_color_accuracy(etiquetes_kmeans, ground_truth):
        total = 0
        for kmeans, ground in zip(etiquetes_kmeans, ground_truth):
            correctes = 0
            for g in ground:
                if g in kmeans:
                    correctes += 1
            if len(ground) != 0:
                total += correctes/len(ground)
        return (total / len(ground_truth)) * 100


    # def get_color_accuracy(etiquetes_kmeans, ground_truth):
    #     correctes = 0
    #     malament = False
    #     for kmeans, ground in zip(etiquetes_kmeans, ground_truth):
    #         malament = False
    #         for g in ground:
    #             if g not in kmeans:
    #                 malament = True
    #                 break
    #         if not malament:
    #             correctes += 1

    #     return (correctes / len(ground_truth)) * 100


    # def get_color_accuracy(etiquetes_kmeans, ground_truth):
    #     correctes = 0
    #     for kmeans, ground in zip(etiquetes_kmeans, ground_truth):
    #         correct = 0
    #         if len(kmeans) < len(ground):
    #             for kmean in kmeans:
    #                 if kmean not in ground:
    #                     correct += 1
    #             if len(kmeans) != 0:
    #                 correctes += correct/len(kmeans)
    #         else:
    #             for groun in ground:
    #                 if groun not in ground:
    #                     correct += 1
    #             if len(ground) != 0:
    #                 correctes += correct/len(ground)
    #     return (correctes / len(ground_truth)) * 100




kmeans = []
etiquetes_kmeans = []
options = {}
options['km_init'] = 'first'
# first
# random
# diagonal
options['tolerance'] = 20
options['diffrence'] = 0
# 0	Identical
# < 10	Indistinguishable
# 10–30	Slight difference
# > 50	Probably noticeable
# > 100	Very different colors
options['fitting'] = 'WCD'
# WCD
# ICD
# FD
print("KMeans")
start = time.time()

for i, img in enumerate(cropped_images):
    kmeans.append(KMeans(img,1,options))
    kmeans[i].find_bestK(100)
    etiquetes_kmeans.append(np.unique(get_colors(kmeans[i].centroids)))

end = time.time()
print(end - start,"s")
print(get_color_accuracy(etiquetes_kmeans, color_labels),"%")


print(" ")
print("KNN")

for k in range(1, 10):
    start = time.time()
    knn = KNN(train_imgs, train_class_labels) 
    etknn = knn.predict(imgs, k)
    end = time.time()
    print(end - start,"s")
    print(get_shape_accuracy(etknn, class_labels),"%")
