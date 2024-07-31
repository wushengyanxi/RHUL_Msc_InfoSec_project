import sys
import random
from random import shuffle

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
from Linear_Regression_1_1 import train_linear_regression
from Linear_Regression_1_1 import LR_preprocess_data
from Linear_Regression_1_1 import linear_regression_predict
from Linear_Regression_1_1 import one_step_LR

def linear_regression_predict1(features, weights, sample, scale_factors):
    # sample = preprocess_data(sample) 这些数据需要被初始化
    predict_score = 0
    predict_label = 0
    for n in range(0, len(features)):
        index = next((i for i, sublist in enumerate(scale_factors) if sublist[0] == features[n]), -1)
        if index != -1:
            sample[n] = sample[n] * scale_factors[index][1]
            predict_score += sample[n] * weights[features[n]]

    predict_score += 0.92
    
    if predict_score >= 0.5:
        predict_label = 0

    return predict_label, predict_score

average_predict_score_atall = 0

for x in range(0,1):
    
    random.seed(65)
    Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(3000,3000,1500,1500,1500,1500)

    s, w = one_step_LR(Training_Data_Set, Features_name,35000)



    '''
    # print(s)
    print(w)
    print(type(w))
    print()

    top_ten_items = sorted(w.items(), key=lambda item: item[1], reverse=True)[:10]
    top_ten_keys = [item[0] for item in top_ten_items]
    top_ten_values = [item[1] for item in top_ten_items]

    print("前十个键:", top_ten_keys)
    print("前十个值:", top_ten_values)
    '''



    all_test_sample = []

    all_test_sample = Testing_Data_Set[0][:3000] + Testing_Data_Set[1][:3000] + Testing_Data_Set[2][:1500] + Testing_Data_Set[3][:1500] + Testing_Data_Set[4][:1500] + Testing_Data_Set[5][:1500]

    random.shuffle(all_test_sample)

    all_test_sample = all_test_sample[:30000]
    # all_test_sample = Training_Data_Set[:3000]

    #print(all_test_sample[0])
    #print()
    #print(all_test_sample[0][:-1])

    count = 0
    score = 0
    correct_0 = 0
    correct_1 = 0
    for i in all_test_sample:
        predict, predict_score = linear_regression_predict1(Features_name,w,i[:-1],s)
        score += predict_score
        if predict == i[-1]:
            count += 1

        if predict == 0 and i[-1] == 0:
            correct_0 += 1
        
        if predict == 1 and i[-1] == 1:
            correct_1 += 1
        
    accurate_rate = count/len(all_test_sample)
    average_predict_score = score/len(all_test_sample)

    print("round ",x+1," result: ")
    print("original accuracy rate:",accurate_rate)
    print("accurate rate for benign traffic:",correct_0/15000)
    print("accurate rate for attack traffic:",correct_1/15000)
    print("average predict score:",average_predict_score)
    print()
    
    average_predict_score_atall += average_predict_score

print("average predict score at all:", average_predict_score_atall/100)












'''
attack_traffic = 0

for i in all_test_sample:
    predict = linear_regression_predict(Features_name,w,i[:-1],s)
    if predict == 1:
        attack_traffic += 1

print("amount of detected attack traffic without causative integrity attack:",attack_traffic)

attack_traffic = 0

for i in all_test_sample:
    top1_index = Features_name.index(top_ten_keys[0])
    top2_index = Features_name.index(top_ten_keys[1])
    top3_index = Features_name.index(top_ten_keys[2])
    
    if i[top1_index] != 0:
        i[top1_index] = i[top1_index] * 0.2
    if i[top2_index] != 0:
        i[top2_index] = i[top2_index] * 0.2
    if i[top3_index] != 0:
        i[top3_index] = i[top3_index] * 0.2
    
    predict = linear_regression_predict(Features_name,w,i[:-1],s)
    if predict == 1:
        attack_traffic += 1

print("amount of detected attack traffic with causative integrity attack:",attack_traffic)
    

首先，手法肯定是对的

在找出10个最大比重的特征的过程中每次找出来的结果都不太一样

时常能看到理应较大的特征但并不稳定

可以通过若干次循环，找出最常出现的“大比重特征”，再去选择其中比较合理的，攻击者能够操作的特征去编辑

但不知道为什么线性回归模型本身对攻击流量的检测率能低到不足2%


--------------------------

后续是随着不断增加epoch,CICIDS-2017中声称的大权重参数逐渐出现在了对应的结果中,证明了模型本身的可靠性

然而,Unnamed:0.1逐渐开始成为了权重最大的特征,而对于所有的恶意样本而言,平均预测分接近0分

绝大多数良性样本都有相当大的unnamed:0.1,这是受数据库排列决定的。因此，在该特征占主导的情况下，容易出现误差

极少数正确被预判的恶意样本，应该就是放在最后,index为50w+的几个样本

要在训练之前的初始化步骤,强行把训练样本的unnamed:0.1归零，然后再进行训练，测试样本同理

'''