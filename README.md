# 基于PCA降维和KNN分类的人脸识别
本次大作业采用两种算法来完成人脸识别的过程:
- PCA 提取图片的主元素特征
- KNN 加速分类的过程

## 1. 任务流程讲解
- 从本地读取图片到内存当中 [120*14张图片]
- 采用 cross-validation 二八法则分出训练集和测试集
- 对训练集进行PCA降维
- 采用简单的cosine 相似法来做分类
- 采用KNN 来做分类
- 图片压缩有重建

## 2. 源代码
### 2.1 导入图片入内存
```python
img_dir = '/Users/risan/Desktop/homework/2019机器学习大作业/AR/AR' # 存放图片的文件夹

inputs, tags = get_inputs_tags(img_dir) # 将图片导入内存
num_tags = len(set(tags)) 
train_x, test_x, train_y, test_y = train_test_split(inputs, tags, test_size=0.2)
```

### 2.2 PCA降维，同时观察不同主成分对正确率的影响
这里使用简单的cosine相似来做正确率的，虽然速度慢但是不会引入其他超参数
```python
n_sample, h, w = train_x.shape
X = np.reshape(train_x, [n_sample, -1]) # convert img_mat => img_array
X_ = np.reshape(test_x, [test_x.shape[0], -1])

n_components = 2000 # dimention after pca
component_acc_dict = {}
for cmpnt in range(1, n_components+1):
    pca = PCA(n_components=cmpnt, svd_solver='randomized', whiten=True, copy=True)
    pca.fit(X)
    X_pca = pca.transform(X)     # 执行pca变换得到的特征 
    X_test_pca = pca.transform(X_)     
    X_pca_norm = normalize(X_pca)     # 归一化
    X_test_pca_norm = normalize(X_test_pca)    # 归一化
    ans = np.matmul(X_pca_norm, X_test_pca_norm.T)    # cosine 相似度
    ans = np.argmax(ans, axis=0)
    pred = []
    for a in ans:
        pred.append(train_y[a])
    acc = np.mean(np.equal(pred, test_y).astype(np.float32))
    # print(acc)
    component_acc_dict[cmpnt] = acc

fig, axes = plt.subplots(figsize=(16,5))

axes.set_xlabel('decomposition dim')
axes.set_ylabel('recog acc')
dims = component_acc_dict.keys()
acc = component_acc_dict.values()
axes.scatter(dims, acc,1) # 1代表scatter 点的大小 default=50
```
由此可以得到在 指定[1,2000]不同主元素时准确率的走向


<img src="PCA.png"  align=center />
可以看出来准确率提高的拐点是在100d左右，准确率下降的拐点是1000d以后，在1300d左右准确率有断崖式骤减。

### 2.3 使用KNN分类
- 主成分选择为 100, 256, 1250
- KNN 近邻选择为[1,10]
```python
n_components = [100, 256, 1250]
fig, axes = plt.subplots()
legend = []
for cmpnt in n_components:
    pca = PCA(n_components=cmpnt, svd_solver='randomized', whiten=True, copy=True)
    pca.fit(X)
    X_pca = pca.transform(X)     # 执行pca变换得到的特征 
    X_test_pca = pca.transform(X_)
    n_neighbors=10

    nbrs_acc = []
    for nbr in range(1, n_neighbors+1):
        knn_clf = KNeighborsClassifier(n_neighbors=nbr)
        knn_clf.fit(X_pca, train_y)
        pred = knn_clf.predict(X_test_pca)
        acc = np.mean(np.equal(pred, test_y).astype(np.float32))
        nbrs_acc.append(acc)

    axes.set_xlabel('n_neighbors')
    axes.set_ylabel('recog acc')
    axes.plot(range(1, n_neighbors+1), nbrs_acc,'x-')
    legend.append('component = ' + str(cmpnt))
axes.legend(legend, loc=4)
```
此时得到在*3个*具有代表性的主元素的超参数下，采取不同的近邻数的效果如图

<img src="KNN.png" width="400" hegiht="500" align=center />

从图片中可以看出来 <b>最近邻</b> 效果是最优的且随着近邻数的增加对图片的类别判断也会带来更大的误差


## 3. 图片压缩后重建
图片压缩后重建即是对PCA 恢复的过程，由于采取不同主元素数的PCA 会对非主元素成分丢弃，因此如果主元素数量过少会导致重建困难。我这里采用对比实验:
- 对256个主元素压缩后重建
- 对8个主元素压缩后重建

### 3.1 对256个主元素压缩重建
```python
n_components = 256
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True, copy=True)
pca.fit(X)
X_pca = pca.transform(X)     # 执行pca变换得到的特征 np.mat(eignmatrix, X)
X_rec = pca.inverse_transform(X_pca)
x_rec = np.reshape(X_rec[1],(50,40))

img_rec = Image.fromarray(x_rec)
img_ori = Image.fromarray(inputs[1])


plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_ori)
plt.subplot(1,2,2)
plt.imshow(img_rec)
plt.show()
```
其结果是

<img src="PCA-256.png" width="400" hegiht="500" align=center />

左边是原始图片，右边是压缩重建图片， 可以看出来有部分特征被遗弃

### 3.2 对8个主元素压缩重建
```python
n_components = 8
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True, copy=True)
pca.fit(X)                   #PCA 的 copy 务必为True,否则pca.transform(X)中代码中会修改X
X_pca = pca.transform(X)     # 执行pca变换得到的特征 np.mat(eignmatrix, X)
X_rec = pca.inverse_transform(X_pca)
x_rec = np.reshape(X_rec[1],(50,40))

img_rec = Image.fromarray(x_rec)
img_ori = Image.fromarray(inputs[1])


plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_ori)
plt.subplot(1,2,2)
plt.imshow(img_rec)
plt.show()
```

<img src="PCA-8.png" width="400" hegiht="500" align=center />

左边是原始图片，右边是压缩重建图片， 可以看出来重建图片已经几乎完全丢失原图片特征

我们同时我们再PCA不同主元素的准确率对应表中可以看到8d的准确率甚至不到10%，完全属于盲猜

## 4. 代码实现说明
为方便可视化，本代码采用jupyter notebook工具
代码中采用的预处理工具放入utils.py文件中

## 5. Requirements
- python 3.6
- pillow 6.2.1
- scikit-learn 0.21.3
- matplotlib 3.1.1
