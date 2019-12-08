from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.inf)
import os

def image2mat(filepath: str) -> np.array:
    # convert image to matrix naively
    # filepath : abs path
    p = Image.open(filepath)
    mat = np.asarray(p, dtype='float32')
    
    return mat

# mat = image2mat('/Users/risan/Desktop/homework/2019机器学习大作业/AR/AR/001-01.bmp')
# im = Image.fromarray(mat)
# im.show()

def get_label(filepath:str) -> int:
    # get images label from filepath: abs path
    relative_path = filepath
    relative_path = filepath.split('/')[-1]
    label = relative_path.split('-')[0]
    label = int(label)

    return label

# print(get_label('/Users/risan/Desktop/homework/2019机器学习大作业/AR/AR/001-01.bmp'))



def get_inputs_tags(dir_path:str) -> [list, list]:
    # dir_path is an abs path
    img_list = os.listdir(dir_path)
    img_list.sort()
    img_list = [x for x in img_list if '-' in x]
    imgs_mat, tags = [], []
    for img in img_list:
        img = os.path.join(dir_path, img)
        imgs_mat.append(image2mat(img))
        tags.append(get_label(img))

    return np.asarray(imgs_mat), np.asarray(tags)

def train_test_split(inputs, tags, test_size=0.2):
    aspects = len(set(tags))
    num_per_class = len(inputs) // aspects
    train_span = int(num_per_class * ( 1 - test_size ))
    test_span = num_per_class - train_span

    train_x = []
    test_x = []
    train_y = []
    test_y = []
    for i in range(0, len(inputs), num_per_class):
        train_x.extend(inputs[i: i+train_span])
        test_x.extend(inputs[i+train_span: i+num_per_class])
        train_y.extend(tags[i: i+train_span])
        test_y.extend(tags[i+train_span: i+num_per_class])
    
    return np.asarray(train_x), np.asarray(test_x), np.asarray(train_y), np.asarray(test_y)

def normalize(matrix):
    """
    对行向量进行 正则化
    """
    sqr_matrix = matrix ** 2
    norm_len = np.sqrt(np.sum(sqr_matrix, axis=-1))
    # print(norm_len)
    # print((matrix.T / norm_len).T)
    return (matrix.T / norm_len).T

