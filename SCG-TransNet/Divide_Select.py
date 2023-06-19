import math
data = [[0, 0, 0, 0, 0, 0, 0.697, 1],
        [1, 0, 1, 0, 0, 0, 0.774, 1],
        [1, 0, 0, 0, 0, 0, 0.634, 1],
        [0, 0, 1, 0, 0, 0, 0.608, 1],
        [2, 0, 0, 0, 0, 0, 0.556, 1],
        [0, 1, 0, 0, 1, 1, 0.403, 1],
        [1, 1, 0, 1, 1, 1, 0.481, 1],
        [1, 1, 0, 0, 1, 0, 0.437, 1],
        [1, 1, 1, 1, 1, 0, 0.666, 0],
        [0, 2, 2, 0, 2, 1, 0.243, 0],
        [2, 2, 2, 2, 2, 0, 0.245, 0],
        [2, 0, 0, 2, 2, 1, 0.343, 0],
        [0, 1, 0, 1, 0, 0, 0.639, 0],
        [2, 1, 1, 1, 0, 0, 0.657, 0],
        [1, 1, 0, 0, 1, 1, 0.360, 0],
        [2, 0, 0, 2, 2, 0, 0.593, 0],
        [0, 0, 1, 1, 1, 0, 0.719, 0]]

divide_point = [0.244, 0.294, 0.351, 0.381, 0.420, 0.459, 0.518, 0.574, 0.600, 0.621, 0.636, 0.648, 0.661, 0.681, 0.708,
                0.746]
# 计算信息熵
def Entropy(melons):
    melons_num = len(melons)
    pos_num = 0
    nag_num = 0
    for i in range(melons_num):
        if melons[i][7] == 1:
            pos_num = pos_num + 1
    nag_num = melons_num - pos_num
    p_pos = pos_num / melons_num
    p_nag = nag_num / melons_num
    entropy = -(p_pos * math.log(p_pos, 2) + p_nag * math.log(p_nag, 2))
    return entropy

# 计算第charac项特征的的信息熵
# charac = 0~5
# 输出：[信息增益,第几个特征]
def Entropy_Gain(melons, charac):
    charac_entropy = 0
    entropy_gain = 0
    melons_num = len(melons)

    # 密度特征是连续特征
    if charac >= 6:
        # 对于某一个划分点，划分后的信息增益
        density_entropy = list()
        density0 = list()
        density1 = list()
        class0_small_num = 0  # 是否大于第i个划分点用big和small表示，是否是好瓜用0和1表示
        class0_big_num = 0
        class1_small_num = 0
        class1_big_num = 0

        for i in range(melons_num):
            if melons[i][7] == 1:
                if melons[i][6] > divide_point[charac - 6]:
                    class1_big_num = class1_big_num + 1
                else:
                    class1_small_num = class1_small_num + 1
            else:
                if melons[i][6] > divide_point[charac - 6]:
                    class0_big_num = class0_big_num + 1
                else:
                    class0_small_num = class0_small_num + 1

        # 防止除零报错
        if class0_small_num == 0 and class1_small_num == 0:
            p0_small = 0
            p1_small = 0
        else:
            p0_small = class0_small_num / (class0_small_num + class1_small_num)
            p1_small = class1_small_num / (class0_small_num + class1_small_num)
        if class0_big_num == 0 and class1_big_num == 0:
            p0_big = 0
            p1_big = 0
        else:
            p0_big = class0_big_num / (class0_big_num + class1_big_num)
            p1_big = class1_big_num / (class0_big_num + class1_big_num)

        # 防止log0的报错
        if p0_small != 0 and p1_small != 0:
            entropy_small = -(class0_small_num + class1_small_num) / melons_num * (
                -(p0_small * math.log(p0_small, 2)
                    + p1_small * math.log(p1_small, 2)))
        elif p0_small == 0 and p1_small != 0:
            entropy_small = -(class0_small_num + class1_small_num) / melons_num * (
                -p1_small * math.log(p1_small, 2))
        elif p0_small != 0 and p1_small == 0:
            entropy_small = -(class0_small_num + class1_small_num) / melons_num * (
                -p0_small * math.log(p0_small, 2))
        else:
            entropy_small = 0
        #print(entropy_small)

        if p0_big != 0 and p1_big != 0:
            entropy_big = -(class0_big_num + class1_big_num) / melons_num * (
                -(p0_big * math.log(p0_big, 2) + p1_big *
                    math.log(p1_big, 2)))
        elif p0_big == 0 and p1_big != 0:
            entropy_big = -(class0_big_num + class1_big_num) / melons_num * (
                -p1_big * math.log(p1_big, 2))
        elif p0_big != 0 and p1_big == 0:
            entropy_big = -(class0_big_num + class1_big_num) / melons_num * (
                -p0_big * math.log(p0_big, 2))
        else:
            entropy_big = 0
        entropy_gain = Entropy(melons) + entropy_small + entropy_big

    # 触感特征只有两种情况
    elif charac == 5:
        class0_melons = []
        class1_melons = []
        class_melons = [[], []]
        for i in range(melons_num):
            if melons[i][5] == 0:
                class0_melons.append(melons[i][7])
            else:
                class1_melons.append(melons[i][7])
        class_melons[0] = class0_melons
        class_melons[1] = class1_melons
        #print(class_melons)

        for i in range(2):
            class0_num = 0
            class1_num = 0
            total_num = len(class_melons[i])
            for j in range(total_num):
                if class_melons[i][j] == 0:
                    class0_num = class0_num + 1
                else:
                    class1_num = class1_num + 1
            p_class0 = class0_num / total_num
            p_class1 = class1_num / total_num
            if p_class0 != 0 and p_class1 != 0:         # 防止log0的报错
                entropy_class = -p_class0 * math.log(p_class0, 2) - p_class1 * math.log(p_class1, 2)
            elif p_class0 == 0 and p_class1 != 0:
                entropy_class = - p_class1 * math.log(p_class1, 2)
            else:
                entropy_class = -p_class0 * math.log(p_class0, 2)
            charac_entropy = charac_entropy - total_num / melons_num * entropy_class
            entropy_gain = Entropy(melons) + charac_entropy

    # 其他特征有三种情况
    else:
        class0_melons = []
        class1_melons = []
        class2_melons = []
        class_melons = [[], [], []]
        for i in range(melons_num):
            if melons[i][charac] == 0:
                class0_melons.append(melons[i][7])
            elif melons[i][charac] == 1:
                class1_melons.append(melons[i][7])
            else:
                class2_melons.append(melons[i][7])
        class_melons[0] = class0_melons
        class_melons[1] = class1_melons
        class_melons[2] = class2_melons
        #print(class_melons)

        for i in range(3):
            class0_num = 0
            class1_num = 0
            total_num = len(class_melons[i])

            # 避免除零报错
            if total_num != 0:
                for j in range(total_num):
                    if class_melons[i][j] == 0:
                        class0_num = class0_num + 1
                    else:
                        class1_num = class1_num + 1
                p_class0 = class0_num / total_num
                p_class1 = class1_num / total_num
                if p_class0 != 0 and p_class1 != 0:             # 防止log0的报错
                    entropy_class = -p_class0 * math.log(p_class0, 2) - p_class1 * math.log(p_class1, 2)
                elif p_class0 == 0 and p_class1 != 0:
                    entropy_class = - p_class1 * math.log(p_class1, 2)
                else:
                    entropy_class = -p_class0 * math.log(p_class0, 2)
                charac_entropy = charac_entropy - total_num / melons_num * entropy_class
                entropy_gain = Entropy(melons) + charac_entropy
            else:
                entropy_gain = 0
    return [entropy_gain, charac]

# 输出：[信息增益,第几个特征]
def select_best_feature(melons, features):
    best_feature = 0
    max_entropy = Entropy_Gain(melons, features[0])
    for i in range(len(features)):
        entropy = Entropy_Gain(melons, features[i])
        if entropy[0] > max_entropy[0]:
            max_entropy = entropy
    return max_entropy
