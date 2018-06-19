
''' one-hot编码 '''
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

testdata = pd.DataFrame({'pet': ['chinese', 'english', 'english', 'math'],
                         'age': [6 , 5, 2, 2],
                         'salary':[7, 5, 2, 5]})
 
''' OneHotEncoder无法直接对字符串型的类别变量编码 '''
 result:
 >>> testdata
   age      pet  salary
0    6  chinese       7
1    5  english       5
2    2  english       2
3    2     math       5


OneHotEncoder(sparse = False).fit_transform(testdata[['age']])
result;
>>> OneHotEncoder(sparse = False).fit_transform(testdata[['age']])
array([[ 0.,  0.,  1.],
       [ 0.,  1.,  0.],
       [ 1.,  0.,  0.],
       [ 1.,  0.,  0.]])
 
 '''对两列同时做one-hot编码'''
 >>> OneHotEncoder(sparse = False).fit_transform( testdata[['age', 'salary']])
array([[ 0.,  0.,  1.,  0.,  0.,  1.],
       [ 0.,  1.,  0.,  0.,  1.,  0.],
       [ 1.,  0.,  0.,  1.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  1.,  0.]])
       

''' 对字符串进行one-hot处理 '''
>>> LabelEncoder().fit_transform(testdata['pet'])
array([0, 1, 1, 2], dtype=int64)
>>> testdata
   age      pet  salary
0    6  chinese       7
1    5  english       5
2    2  english       2
3    2     math       5
>>> LabelEncoder().fit_transform(testdata['pet'])
array([0, 1, 1, 2], dtype=int64)
>>> a = LabelEncoder().fit_transform(testdata['pet'])
>>> OneHotEncoder( sparse=False ).fit_transform(a.reshape(-1,1))
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])

########### 或者 ############

>>> LabelBinarizer().fit_transform(testdata['pet'])
array([[1, 0, 0],
       [0, 1, 0],
       [0, 1, 0],
       [0, 0, 1]])







'''pandas'''
>>> testdata = pd.DataFrame({'pet': ['chinese', 'english', 'english', 'math'],
...                          'age': [6 , 5, 2, 2],
...                          'salary':[7, 5, 2, 5]})
>>> testdata
   age      pet  salary
0    6  chinese       7
1    5  english       5
2    2  english       2
3    2     math       5
>>> a = pd.get_dummies(testdata,columns=['pet'])
>>> a
   age  salary  pet_chinese  pet_english  pet_math
0    6       7            1            0         0
1    5       5            0            1         0
2    2       2            0            1         0
3    2       5            0            0         1

>>> a = pd.get_dummies(testdata,columns=['pet', 'age'])
>>> a
   salary  pet_chinese  pet_english  pet_math  age_2  age_5  age_6
0       7            1            0         0      0      0      1
1       5            0            1         0      0      1      0
2       2            0            1         0      1      0      0
3       5            0            0         1      1      0      0
