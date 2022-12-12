import pandas as pd
# 通过二分类概率返回预测结果标签
def getClass(limit, result):
    predict=[]
    for i in range(len(result)):
        if(result[i]>limit):
            predict.append(1)
        else:
            predict.append(0)
    return predict

# 获取精确率和召回率的函数
def getConfusion(test_labels,y_predict):
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(len(test_labels)):
        if test_labels[i]==y_predict[i] and y_predict[i]==1:
            tp=tp+1
        elif test_labels[i]==y_predict[i] and y_predict[i]==0:
            tn=tn+1
        elif test_labels[i]!=y_predict[i] and y_predict[i]==1:
            fp=fp+1
        elif test_labels[i]!=y_predict[i] and y_predict[i]==0:
            fn=fn+1
    return [tp,tn,fp,fn]

    # 获取tpr和fpr的函数
def getTprAndFpr(test_labels,y_predict):
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(len(test_labels)):
        if test_labels[i]==y_predict[i] and y_predict[i]==1:
            tp=tp+1
        elif test_labels[i]==y_predict[i] and y_predict[i]==0:
            tn=tn+1
        elif test_labels[i]!=y_predict[i] and y_predict[i]==1:
            fp=fp+1
        elif test_labels[i]!=y_predict[i] and y_predict[i]==0:
            fn=fn+1
    print(tp,tn,fp,fn)
    # 返回真阳率和假阳率
    return [tp/(tp+fn), fp/(fp+tn)]

# 画出ROC曲线
def rocPlot(test_labels,predictions):
    roc=[]
    tpr=[]
    fpr=[]
    for i in range(9):
        roc.append(getTprAndFpr(test_labels,getClass((i+1)*0.1,predictions)))
    for i in range(len(roc)):
        tpr.append(roc[i][0])
        fpr.append(roc[i][1])
    plt.xlabel("FPR")  # x轴为角度数
    plt.ylabel("PTR")  # sin值大小
    plt.title("RandomForest ROC")
    plt.plot(fpr,tpr)
    
# 获取分类任务评价指标
def getClfEvaluation(confusion):
    return [(confusion[0]+confusion[1])/sum(confusion), confusion[0]/(confusion[0]+confusion[2]), 
confusion[0]/(confusion[0]+confusion[3])]

'''
计算缺失值比例
'''
def missing(df):
    missing_number = df.isnull().sum().sort_values(ascending = False)
    missing_percent = df.isnull().mean().sort_values(ascending = False)
    missing_values = pd.concat([missing_number,missing_percent],axis=1,keys=['missing_number','missing_percent'])
    
    return missing_values