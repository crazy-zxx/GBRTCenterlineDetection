import glob
import os

import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import minmax_scale
from skimage.morphology import thin
from scipy.ndimage.morphology import distance_transform_edt


def load_dataset(train_file_path, gt_file_path):
    print('loading datasets ...')
    imgs = []
    gts = []
    if os.path.exists(train_file_path) and os.path.exists(gt_file_path):
        imgs_list = glob.glob(train_file_path + '*.tif')
        gts_list = glob.glob(gt_file_path + '*.tif')
        if len(imgs_list) == len(gts_list):
            for imn, gn in zip(imgs_list, gts_list):
                print(imn, ',', gn)
                imgs.append(cv2.imread(imn, cv2.IMREAD_GRAYSCALE))
                gts.append(cv2.imread(gn, cv2.IMREAD_GRAYSCALE))
        else:
            print('file number of imgs and gts not match!')
    return np.array(imgs), np.array(gts)


def load_sep_filters_and_weights(sep_filter_path, wights_path):
    print('loading sep_filters and weights file ...')
    sep_filters = []
    weights = []
    if os.path.exists(sep_filter_path) and os.path.exists(wights_path):
        sep_filters_list = glob.glob(sep_filter_path + '*.txt')
        weights_list = glob.glob(wights_path + '*.txt')
        if len(sep_filters_list) == len(weights_list):
            for fn, wn in zip(sep_filters_list, weights_list):
                print(fn, ',', wn)
                sep_filters.append(np.loadtxt(fn).tolist())
                weights.append(np.loadtxt(wn).tolist())
        else:
            print('file number of sep_filter and weights not match!')
    return np.array(sep_filters), np.array(weights)


def get_sep_features(imgs, sep_filters, weights):
    print('get all sep_features ...')
    sep_filters_shape = sep_filters.shape
    sep_features = []
    num_imgs = len(imgs)
    for i in range(num_imgs):
        sep_features_i = []
        for j in range(sep_filters_shape[0]):
            for k in range(sep_filters_shape[2]):
                A = sep_filters[j, :sep_filters_shape[1] // 2, k]
                B = sep_filters[j, sep_filters_shape[1] // 2:, k]
                sep_features_i.append(cv2.sepFilter2D(imgs[i], ddepth=-1, kernelX=A, kernelY=B).flatten().tolist())
        sep_features.append(np.matmul(np.array(sep_features_i).T, weights[0]))
    return np.vstack(sep_features)


def select_samples(gts, sep_features, labels, pos_sample_num=0, neg_sample_num=0):
    print('select train samples ...')
    dist_gts = labels.reshape((-1, gts.shape[1], gts.shape[2]))
    pos_indices = np.ravel_multi_index(np.where(dist_gts != 0), gts.shape)
    neg_indices = np.ravel_multi_index(np.where(dist_gts == 0), gts.shape)
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)
    sample_indices = range(len(labels))
    if len(pos_indices) >= pos_sample_num and len(neg_indices) >= neg_sample_num:
        pos_sample_indices = np.random.choice(pos_indices, pos_sample_num)
        neg_sample_indices = np.random.choice(neg_indices, neg_sample_num)
        sample_indices = numpy.hstack([pos_sample_indices, neg_sample_indices])

    return sep_features[sample_indices], labels[sample_indices]


def distance_transform(gts, threshold):
    print('get all labels')
    labels = []
    num_gts = len(gts)
    for i in range(num_gts):
        thin_gt = thin(gts[i] // 255)
        thin_gt_reverse = (thin_gt == False).astype(int)
        # edt
        labels.append(distance_transform_edt(thin_gt_reverse).tolist())
    labels = np.array(labels)
    labels[labels > threshold] = threshold
    labels = 1 - np.array(labels)
    labels = (labels - np.min(labels)) / (np.max(labels) - np.min(labels))
    labels = np.exp(6 * labels) - 1
    return labels.flatten()


def train(features, labels, n_estimators):
    print('start train gbrt ...')
    return GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=n_estimators,
                                     subsample=0.5, max_depth=100, verbose=1).fit(features, labels)


def test(reg, X_test):
    print('start predict ...')
    return reg.predict(X_test)


def save_pred(test_imgs, y_pred):
    print('start save predicted images ...')
    num_test_imgs = len(test_imgs)
    test_img_size = test_imgs.shape[1] * test_imgs.shape[2]
    for i in range(num_test_imgs):
        pred_img = minmax_scale(
            y_pred[test_img_size * i:test_img_size * (i + 1)].reshape(test_imgs.shape[1:]), feature_range=(0, 255))
        cv2.imshow(str(i), pred_img)
        cv2.waitKey(0)
        cv2.imwrite('img-%s.jpg' % i, pred_img)
        print('save img-%s.jpg successful !' % i)


def draw_deviance(reg, n_estimators, X_test, y_test):
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    # train curve
    plt.plot(np.arange(n_estimators) + 1, reg.train_score_, 'b-', label='Training Set Deviance')
    # test curve
    test_score = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(X_test)):
        test_score[i] = reg.loss_(y_test, y_pred)
    plt.plot(np.arange(n_estimators) + 1, test_score, 'r-', label='Test Set Deviance')
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show()


def direct_method(train_imgs, train_gts, test_imgs, test_gts, sep_filters, weights,
                  train_pos_sample_num, train_neg_sample_num, n_estimators=10):
    all_sep_features = get_sep_features(train_imgs, sep_filters, weights)
    all_labels = distance_transform(train_gts, 11)
    features, labels = select_samples(train_gts, sep_features=all_sep_features, labels=all_labels,
                                      pos_sample_num=train_pos_sample_num, neg_sample_num=train_neg_sample_num)
    # use all features or not
    # reg = train(all_sep_features, all_labels, n_estimators)
    reg = train(features, labels, n_estimators)

    sep_features = get_sep_features(test_imgs, sep_filters, weights)
    labels = distance_transform(test_gts, 11)
    y_pred = test(reg, sep_features)
    save_pred(test_imgs, y_pred)
    draw_deviance(reg, n_estimators, sep_features, labels)


# TODO:how to implement auto-context ????????
def auto_context(train_imgs, train_gts, test_imgs, test_gts, sep_filters, weights,
                 train_pos_sample_num, train_neg_sample_num, ac_pos_sample_num, ac_neg_sample_num,
                 n_estimators=10, iter_no=0):
    train_sep_features = get_sep_features(train_imgs, sep_filters, weights)
    train_labels = distance_transform(train_gts, 11)
    ac_sep_features = []
    reg = None
    for i in range(iter_no + 1):
        features, labels = select_samples(train_gts, sep_features=train_sep_features, labels=train_labels,
                                          pos_sample_num=train_pos_sample_num, neg_sample_num=train_neg_sample_num)
        if len(ac_sep_features):
            ac_features, ac_labels = select_samples(train_gts, sep_features=ac_sep_features, labels=train_labels,
                                                    pos_sample_num=ac_pos_sample_num, neg_sample_num=ac_neg_sample_num)
            features = np.vstack([features, ac_features])
            labels = np.hstack([labels, ac_labels])
        reg = train(features, labels, n_estimators)
        y_pred = test(reg, train_sep_features)
        # if len(ac_sep_features):
        #     y_pred = test(reg, ac_sep_features)
        # else:
        #     y_pred = test(reg, train_sep_features)
        if i < iter_no:
            ac_sep_features = get_sep_features(y_pred.reshape(train_imgs.shape), sep_filters, weights)

    test_sep_features = get_sep_features(test_imgs, sep_filters, weights)
    test_labels = distance_transform(test_gts, 11)
    ac_sep_features = []
    y_pred = None
    for i in range(iter_no + 1):
        if len(ac_sep_features):
            y_pred = test(reg, ac_sep_features)
        else:
            y_pred = test(reg, test_sep_features)
        if i < iter_no:
            ac_sep_features = get_sep_features(y_pred.reshape(test_imgs.shape), sep_filters, weights)

    save_pred(test_imgs, y_pred)
    draw_deviance(reg, n_estimators, test_sep_features, test_labels)


if __name__ == '__main__':
    train_imgs, train_gts = load_dataset('DS4/TR/', 'DS4/GT/')
    sep_filters, weights = load_sep_filters_and_weights('filter_banks/', 'weights/')
    test_imgs, test_gts = load_dataset('DS4/TI/', 'DS4/ETSTI/')

    # direct_method(train_imgs, train_gts, test_imgs, test_gts, sep_filters, weights, 20000, 10000, 30)
    # auto_context(train_imgs, train_gts, test_imgs, test_gts, sep_filters, weights, 20000, 10000, 2000, 1000, 3, 2)
