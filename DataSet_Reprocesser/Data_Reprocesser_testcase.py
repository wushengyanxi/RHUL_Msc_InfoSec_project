import sys
sys.path.append(r'C:\Users\Razer\Desktop\workingCopy\RHUL_Msc_InfoSec_project')
from Data_Reporcesser import K_fold
from Data_Reporcesser import Read_HIKARI2021_File
from Data_Reporcesser import Standard_Scalar
from Data_Reporcesser import Normalization
         
def Read_HIKARI2021_File_testcase():
    '''
    test case of Read_HIKARI2021
    '''
    Sample = Read_HIKARI2021_File("ALLFLOWMETER_HIKARI2021_simple_version.csv")
    return Sample[0]

def K_fold_testcase():
    a = [[1],[2],[3],[4],[5],[6],[7],[8],[9]]
    b = K_fold(5,a)
    c = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15]]
    d = K_fold(4,c)
    print(b == [[[9],[8]],[[7],[6]],[[5],[4]],[[3],[2]],[[1]]])
    print(d == [[[15],[14],[13],[12]],[[11],[10],[9],[8]],[[7],[6],[5],[4]],[[3],[2],[1]]])

def Standard_Scalar_testcase():
    Train_X = [[[10,2,32,14,5],[0]],[[26,75,18,29,10],[0]],[[11,12,33,14,55],[0]]]
    Test_y = [[[6,4,12,24,8],[0]],[[13,55,38,19,1],[0]],[[0,32,33,44,25],[0]]]
    a,b = Standard_Scalar(Train_X,Test_y)
    print(a)
    print("----")
    print(b)
    # for a, the first feature for these three sample should be -0.7743, 1.41, -0.63768
    # for b, the first feature for these three sample should be -1.32, 0.36, -2.14

def Normalization_testcase():
    Train_X1 = [[[180,100,17],[0]],[[190,105,22],[0]],[[173,72,31],[0]]]
    Test_y1 = [[[182,80,16],[0]],[[176,75,18],[0]],[[173,72,24],[0]]]
    a,b = Normalization(Train_X1,Test_y1)
    Train_X2 = [[[190,100,17],[0]],[[190,105,22],[0]],[[190,72,31],[0]]]
    Test_y2 = [[[182,80,16],[0]],[[176,75,18],[0]],[[173,72,24],[0]]]
    c,d = Normalization(Train_X2,Test_y2)
    
    print(a)
    print(b)
    print("----")
    print(c)
    print(d)

