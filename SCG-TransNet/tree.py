from Divide_Select import *
import numpy as np

# 训练集data，属性集A
# 0色泽，1根蒂，2敲声，3纹理，4脐部，5触感，
# 对于密度，每个划分点算作一个特征，共16个划分点，即6~21
A = list(range(22))

def find_most(x):

    return sorted([(np.sum(x == i), i) for i in np.unique(x)])[-1][-1]

def tree_generate(melons, features):
    # 如果所有样本属于同一类别，返回该类别作为叶子节点
    melons_y = [i[7] for i in melons]
    if len(np.unique(melons_y)) == 1:
        return melons_y[0]
    # 如果features是空集或者所有样本在features上取值相同，返回多数类别作为叶子节点
    same_flag = 1
    for i in range(6):          # 括号里填什么？
        if len(np.unique([j[i] for j in melons])) > 1:
            same_flag = 0
    if not features or same_flag == 1:
        return find_most(melons_y)

    # 选出最优特征
    [max_entropy, best_feature] = select_best_feature(melons, features)
    node = {best_feature: {}}
    division = list()
    to_divide = list()
    # 对于离散特征
    if best_feature < 6:
        division = [i[best_feature] for i in data]          # 特征best_feature有division的可能性
        to_divide = [i[best_feature] for i in melons]       # 特征best_feature在melons中有to_divide的分支
    # 对于连续特征
    else:
        for j in [i[6] for i in melons]:
            if j > divide_point[best_feature - 6]:
                to_divide.append(1)
            else:
                to_divide.append(0)
        #to_divide = np.unique(to_divide)
        division = [0, 1]

    data_y = [i[7] for i in data]
    for i in np.unique(division):
        loc = list(np.where(to_divide == i))
        if len(loc[0]) == 0:     # 若该属性取此值的样本集为空,生成叶节点，其类别记为样本最多的类
            test = find_most(melons_y)
            node[best_feature][i] = find_most(melons_y)
        else:
            new_melons = []
            for k in range(len(loc[0])):
                new_melons.append(melons[loc[0][k]])
            if best_feature in features:            # 避免重复删除报错
                features.remove(best_feature)
            node[best_feature][i] = tree_generate(new_melons, features)
    return node

print(tree_generate(data, A))

