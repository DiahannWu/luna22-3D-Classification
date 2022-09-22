from sklearn import metrics
import math

y_test = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
y_predict = [1, 3, 2, 2, 2, 3, 1, 1, 3, 1, 2, 3]


# print("Confusion matrix")
# print(metrics.confusion_matrix(y_test,y_predict))

'''import seaborn as sns
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

cf_matrix = metrics.confusion_matrix(y_test,y_predict)
sum_true = np.expand_dims(np.sum(cf_matrix, axis=1), axis=1)
precision_matrix = cf_matrix / sum_true
df = pd.DataFrame(precision_matrix)
ax = sns.heatmap(df,cmap="Blues",annot=True)
ax.set_title('confusion matrix')
ax.set_xlabel('predict')
ax.set_ylabel('true')
plt.show()'''

# from sklearn.metrics import classification_report
# report = classification_report(y_test,y_predict)
# print(report)


from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

#用metrics.roc_curve()求出 fpr, tpr, threshold
fpr, tpr, threshold = metrics.roc_curve(y_test, y_predict, pos_label=3)

#用metrics.auc求出roc_auc的值
roc_auc = metrics.auc(fpr,tpr)

#将plt.plot里的内容填写完整
plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)

#将图例显示在右下方
plt.legend(loc = 'lower right')

#画出一条红色对角虚线
plt.plot([0, 1], [0, 1],'r--')

#设置横纵坐标轴范围
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])

#设置横纵名称以及图形名称
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.show()



'''####################################################
# 自己写的混淆矩阵代码
# 混淆矩阵定义
def confusion_matrix(self, preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def plot_maxtrix(self, maxtrix, per_kinds):
    # 分类标签
    lables = ['0', '1', '2']

    Maxt = np.empty(shape=[0, 3])

    m = 0
    for i in range(3):
        print('row sum:', per_kinds[m])
        f = (maxtrix[m, :] * 100) / per_kinds[m]
        Maxt = np.vstack((Maxt, f))
        m = m + 1

    thresh = Maxt.max() / 1

    plt.imshow(Maxt, cmap=plt.cm.Blues)

    for x in range(3):
        for y in range(3):
            info = float(format('%.2f' % (maxtrix[y, x] / per_kinds[y])))
            print('info:', info)
            plt.text(x, y, info, verticalalignment='center', horizontalalignment='center')
    plt.tight_layout()
    plt.yticks(range(3), lables)  # y轴标签
    plt.xticks(range(3), lables, rotation=45)  # x轴标签
    plt.savefig('./test.png', bbox_inches='tight')  # bbox_inches='tight'可确保标签信息显示全
    plt.show()
#############################################################################'''