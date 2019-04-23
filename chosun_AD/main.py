from data import DataLoader
from excel_class import Printer
from sklearn.linear_model import LogisticRegression
from pygam import LogisticGAM
from sklearn import metrics
from model import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def column(matrix, i):
    return [row[i] for row in matrix]

def column(matrix, i_list):
    return [[row[i] for i in i_list] for row in matrix]

def logistic(x_tr, y_tr, x_tst, y_tst):
    classifier = LogisticRegression(solver='lbfgs',multi_class='auto')
    classifier.fit(x_tr, y_tr)
    y_pred = classifier.predict(x_tst)
    print(y_pred)
    confusion_matrix = metrics.confusion_matrix(y_tst, y_pred)
    print(confusion_matrix)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'\
          .format(classifier.score(x_tst, y_tst)))
    print('Accuracy of logistic regression classifier on train set: {:.2f}' \
          .format(classifier.score(x_tr, y_tr)))

def logistic_GAM(x_tr, y_tr, x_tst, y_tst):
    classifier = LogisticGAM()
    classifier.fit(x_tr, y_tr)
    tr_pred = classifier.predict(x_tr)
    y_pred = classifier.predict(x_tst)
    confusion_matrix = metrics.confusion_matrix(y_tst, y_pred)
    print(confusion_matrix)
    print('Accuracy of logistic regression classifier on test set: {:.2f}' \
          .format(metrics.accuracy_score(y_pred, y_tst)))
    print('Accuracy of logistic regression classifier on train set: {:.2f}' \
          .format(metrics.accuracy_score(tr_pred, y_tr)))
    '''
    XX = generate_X_grid(classifier)
    plt.rcParams['figure.figsize'] = (28, 8)
    fig, axs = plt.subplots(1, len(x_tr.feature_names[0:]))
    titles = boston.feature_names
    for i, ax in enumerate(axs):
        pdep, confi = classifier.partial_dependence(XX, feature=i+1, width=.95)
        ax.plot(XX[:, i], pdep)
        ax.plot(XX[:, i], confi[0][:, 0], c='grey', ls='--')
        ax.plot(XX[:, i], confi[0][:, 1], c='grey', ls='--')
        ax.set_title(titles[i],fontsize=26)

    plt.show()
    '''


def main():
    prt = Printer()
    frd = 4
    loader = DataLoader(frd)
    data = loader.data_read()
    loader.extr_data(data)
    x_tr, tmp_y_tr = loader.get_tr_set()
    x_tst, tmp_y_tst = loader.get_tst_set()
    print('label shape is : ')
    print(np.shape(tmp_y_tr), np.shape(tmp_y_tst))
    print(tmp_y_tr)
    y_tr = pd.get_dummies(tmp_y_tr)
    y_tst = pd.get_dummies(tmp_y_tst)

    prt.p_list(x_tr)
    x_tr_hippo = column(x_tr, [5, 12])
    x_tst_hippo = column(x_tst, [5, 12])

    # prt.p_list(x_tr_hippo)
    # assert False

    # model = Model()
    # model.logistic_regression()
    # model.nn()
    # model.session(x_tr, y_tr, x_tst, y_tst)

    print('training and testing data shape is : ')
    print(np.shape(x_tr), np.shape(x_tst))
    assert False
    # prt.p_list(x_tr)
    # print(tmp_y_tr)
    # print(y_tr)
    logistic_GAM(x_tr, tmp_y_tr, x_tst, tmp_y_tst)
    # logistic(x_tr, tmp_y_tr, x_tst, tmp_y_tst)
    # logistic_GAM(x_tr_hippo, tmp_y_tr, x_tst_hippo, tmp_y_tst)
    # logistic(x_tr_hippo, tmp_y_tr, x_tst_hippo, tmp_y_tst)
    del prt
    # data_gr = []
    # label_gr = []
    # for i in range(3):
    #     tmp_d, tmp_l = loader.get_gr_by_i(i)
    #     data_gr.append(tmp_d)
    #     label_gr.append(tmp_l)
    #
    # dict = {
    #     '1':'red',
    #     '2':'green',
    #     '3':'blue',
    #     '4':'yellow',
    #     '5':'black'
    # }
    # col = ['b','g','r','c','m','y','k','w']
    # col_num = len(col)
    # # plt.grid(True)
    # for label in range(15):
    #     for index in range(3):
    #         for e in data_gr[index]:
    #             #plt.scatter(index,e[label], col[label%col_num])
    #             plt.scatter(index,e[label], marker='.', color=col[label%col_num])
    # plt.xticks([0,1,2],['p_left', 'p-right', 'control'])
    # ticks = [100*i for i in range(15)]
    # ticks_label = [str(100*i) for i in range(15)]
    # print(ticks, ticks_label)
    # plt.yticks(ticks, ticks_label)
    # # plt.xlim(-1,3)
    # # plt.ylim(300,1300)
    # plt.show()
    #
    # for _ in data_gr[0]:
    #    plt.hist(_,10)
    # plt.plot(data_gr[0], label_gr[0])

    return None

if __name__ == '__main__':
    main()