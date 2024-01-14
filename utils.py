import os
import torch
import numpy as np
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from torch import Tensor
import torch.nn as nn
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import scipy.optimize as opt
import torch.distributions as dist
from torch.utils.data import Dataset
from skmultiflow.drift_detection import ADWIN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import IsolationForest
from scipy.stats import ks_2samp
from imblearn.under_sampling import RandomUnderSampler

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

class SplitData(BaseEstimator, TransformerMixin):
    
    def __init__(self, dataset):
        super(SplitData, self).__init__()
        self.dataset = dataset
    
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, labels, one_hot_label=True):
        if self.dataset == 'nsl':
            # Preparing the labels
            y = X[labels]
            X_ = X.drop(['labels5', 'labels2'], axis=1)

            # abnormal data is labeled as 1, normal data 0
            y = (y != 'normal')
            y_ = np.asarray(y).astype('float32')

        elif self.dataset == 'unsw':
            # UNSW dataset processing
            y_ = X[labels]
            X_ = X.drop('label', axis=1)

        else:
            raise ValueError("Unsupported dataset type")

        # Normalization
        normalize = MinMaxScaler().fit(X_)
        x_ = normalize.transform(X_)

        return x_, y_

def description(data):
    print("Number of samples(examples) ",data.shape[0]," Number of features",data.shape[1])
    print("Dimension of data set ",data.shape)

class AE(nn.Module):
    def __init__(self, input_dim):
        super(AE, self).__init__()

        # 找到最接近 input_dim 的 2 的幂次
        nearest_power_of_2 = 2 ** round(math.log2(input_dim))

        # 计算第 2/4 层和第 3 层的尺寸
        second_fourth_layer_size = nearest_power_of_2 // 2  # 二分之一
        third_layer_size = nearest_power_of_2 // 4         # 四分之一

        # # 创建编码器和解码器层
        # encoder_layers = [input_dim, second_fourth_layer_size, third_layer_size]
        # decoder_layers = [second_fourth_layer_size, input_dim]

        # 创建编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )

        # 创建解码器
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(third_layer_size, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, input_dim),
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

class CRCLoss(nn.Module):
    def __init__(self, device, temperature=0.1, scale_by_temperature=True):
        super(CRCLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):        
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None: # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32)
        elif labels is not None: # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float()
        else:
            mask = mask.float()
        # compute logits
        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # 构建mask 
        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size).to(self.device)  
        logits_without_ii = logits * logits_mask
        
        logits_normal = logits_without_ii[(labels == 0).squeeze()]
        logits_normal_normal = logits_normal[:,(labels == 0).squeeze()]
        logits_normal_abnormal = logits_normal[:,(labels > 0).squeeze()]

        sum_of_vium = torch.sum(torch.exp(logits_normal_abnormal), axis=1, keepdims=True)
        denominator = torch.exp(logits_normal_normal) + sum_of_vium
        log_probs = logits_normal_normal - torch.log(denominator)
  
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
    
def score_detail(y_test,y_test_pred,if_print=False):
    # Confusion matrix
    if if_print == True:
        print("Confusion matrix")
        print(confusion_matrix(y_test, y_test_pred))
        # Accuracy 
        print('Accuracy ',accuracy_score(y_test, y_test_pred))
        # Precision 
        print('Precision ',precision_score(y_test, y_test_pred))
        # Recall
        print('Recall ',recall_score(y_test, y_test_pred))
        # F1 score
        print('F1 score ',f1_score(y_test,y_test_pred))

    return accuracy_score(y_test, y_test_pred), precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), f1_score(y_test,y_test_pred)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# 定义对数似然函数
def log_likelihood(params, data):
    mu1, sigma1, mu2, sigma2 = params
    pdf1 = gaussian_pdf(data, mu1, sigma1)
    pdf2 = gaussian_pdf(data, mu2, sigma2)
    return -np.sum(np.log(0.5 * pdf1 + 0.5 * pdf2))

def evaluate(normal_temp, normal_recon_temp, x_train, y_train, x_test, y_test, model, get_confidence=False, en_or_de=False):
    num_of_layer = 0

    x_train_normal = x_train[(y_train == 0).squeeze()]
    x_train_abnormal = x_train[(y_train == 1).squeeze()]

    train_features = F.normalize(model(x_train)[num_of_layer], p=2, dim=1)
    train_features_normal = F.normalize(model(x_train_normal)[num_of_layer], p=2, dim=1)
    train_features_abnormal = F.normalize(model(x_train_abnormal)[num_of_layer], p=2, dim=1)
    # normal_temp = torch.mean(train_features_normal, dim=0)
    test_features = F.normalize(model(x_test)[num_of_layer], p=2, dim=1)

    values_features_all, indcies = torch.sort(F.cosine_similarity(train_features, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
    values_features_normal, indcies = torch.sort(F.cosine_similarity(train_features_normal, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
    values_features_abnormal, indcies = torch.sort(F.cosine_similarity(train_features_abnormal, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))

    values_features_all = values_features_all.cpu().detach().numpy()

    values_features_test = F.cosine_similarity(test_features, normal_temp.reshape([-1, normal_temp.shape[0]]))

    num_of_output = 1
    train_recon = F.normalize(model(x_train)[num_of_output], p=2, dim=1)
    train_recon_normal = F.normalize(model(x_train_normal)[num_of_output], p=2, dim=1)
    train_recon_abnormal = F.normalize(model(x_train_abnormal)[num_of_output], p=2, dim=1)
    # normal_recon_temp = torch.mean(train_recon_normal, dim=0)
    test_recon = F.normalize(model(x_test)[num_of_output], p=2, dim=1)

    values_recon_all, indcies = torch.sort(F.cosine_similarity(train_recon, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))
    values_recon_normal, indcies = torch.sort(F.cosine_similarity(train_recon_normal, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))
    values_recon_abnormal, indcies = torch.sort(F.cosine_similarity(train_recon_abnormal, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))

    values_recon_all = values_recon_all.cpu().detach().numpy()

    values_recon_test = F.cosine_similarity(test_recon, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1)

    mu1_initial = np.mean(values_features_normal.cpu().detach().numpy())
    sigma1_initial = np.std(values_features_normal.cpu().detach().numpy())

    mu2_initial = np.mean(values_features_abnormal.cpu().detach().numpy())
    sigma2_initial = np.std(values_features_abnormal.cpu().detach().numpy())

    # 使用最大似然估计拟合数据到两个高斯分布
    initial_params = np.array([mu1_initial, sigma1_initial, mu2_initial, sigma2_initial]) # 初始参数 
    result = opt.minimize(log_likelihood, initial_params, args=(values_features_all,), method='Nelder-Mead')
    mu1_fit, sigma1_fit, mu2_fit, sigma2_fit = result.x # 估计得到的参数值

    # # 输出估计得到的高斯分布参数
    # print("Gaussian 1: mu = {:.2f}, sigma = {:.2f}".format(mu1_fit, sigma1_fit))
    # print("Gaussian 2: mu = {:.2f}, sigma = {:.2f}".format(mu2_fit, sigma2_fit))


    if mu1_fit > mu2_fit:
        gaussian1 = dist.Normal(mu1_fit, sigma1_fit)
        gaussian2 = dist.Normal(mu2_fit, sigma2_fit)
    else:
        gaussian2 = dist.Normal(mu1_fit, sigma1_fit)
        gaussian1 = dist.Normal(mu2_fit, sigma2_fit)

    pdf1 = gaussian1.log_prob(values_features_test).exp()

    pdf2 = gaussian2.log_prob(values_features_test).exp()
    y_test_pred_2 = (pdf2 > pdf1).cpu().numpy().astype("int32")
    y_test_pro_en = (torch.abs(pdf2-pdf1)).cpu().detach().numpy().astype("float32")

    if isinstance(y_test, int) == False:
        if y_test.device != torch.device("cpu"):
            y_test = y_test.cpu().numpy()

    if isinstance(y_test, int) == False:
        result_encoder = score_detail(y_test,y_test_pred_2)

    # result_before_encoder.append(result_encoder)

    mu3_initial = np.mean(values_recon_normal.cpu().detach().numpy())
    sigma3_initial = np.std(values_recon_normal.cpu().detach().numpy())

    mu4_initial = np.mean(values_recon_abnormal.cpu().detach().numpy())
    sigma4_initial = np.std(values_recon_abnormal.cpu().detach().numpy())

    # 使用最大似然估计拟合数据到两个高斯分布
    initial_params = np.array([mu3_initial, sigma3_initial, mu4_initial, sigma4_initial]) # 初始参数 
    result = opt.minimize(log_likelihood, initial_params, args=(values_recon_all,), method='Nelder-Mead')
    mu3_fit, sigma3_fit, mu4_fit, sigma4_fit = result.x # 估计得到的参数值

    # # 输出估计得到的高斯分布参数
    # print("Gaussian 3: mu = {:.2f}, sigma = {:.2f}".format(mu3_fit, sigma3_fit))
    # print("Gaussian 4: mu = {:.2f}, sigma = {:.2f}".format(mu4_fit, sigma4_fit))


    if mu3_fit > mu4_fit:
        gaussian3 = dist.Normal(mu3_fit, sigma3_fit)
        gaussian4 = dist.Normal(mu4_fit, sigma4_fit)
    else:
        gaussian4 = dist.Normal(mu3_fit, sigma3_fit)
        gaussian3 = dist.Normal(mu4_fit, sigma4_fit)

    pdf3 = gaussian3.log_prob(values_recon_test).exp()

    pdf4 = gaussian4.log_prob(values_recon_test).exp()
    y_test_pred_4 = (pdf4 > pdf3).cpu().numpy().astype("int32")
    y_test_pro_de = (torch.abs(pdf4-pdf3)).cpu().detach().numpy().astype("float32")

    if isinstance(y_test, int) == False:
        result_decoder = score_detail(y_test,y_test_pred_4)

    # result_before_decoder.append(result_decoder)

    # y_test_pred = np.round(np.add(y_test_pred_2, y_test_pred_4).T/2) 
    # result_before_vote.append(score_detail(y_test,y_test_pred))

    y_test_pred_no_vote = torch.where(torch.from_numpy(y_test_pro_en) > torch.from_numpy(y_test_pro_de), torch.from_numpy(y_test_pred_2), torch.from_numpy(y_test_pred_4))
    if en_or_de == True:
        index_output = torch.where(torch.from_numpy(y_test_pro_en) > torch.from_numpy(y_test_pro_de), 0, 1)
    # # confidence of the prediction
    # confidence_vote = torch.where(torch.from_numpy(y_test_pro_en) > torch.from_numpy(y_test_pro_de), torch.from_numpy(y_test_pro_en), torch.from_numpy(y_test_pro_de))
    if get_confidence == True:
        mu5_initial = np.mean(values_recon_normal.cpu().detach().numpy())
        sigma5_initial = np.std(values_recon_normal.cpu().detach().numpy())

        mu6_initial = np.mean(values_recon_abnormal.cpu().detach().numpy())
        sigma6_initial = np.std(values_recon_abnormal.cpu().detach().numpy())

        # 使用最大似然估计拟合数据到两个高斯分布
        initial_params = np.array([mu5_initial, sigma5_initial, mu6_initial, sigma6_initial]) # 初始参数 
        confidence_params = opt.minimize(log_likelihood, initial_params, args=(values_recon_all,), method='Nelder-Mead')
        mu5_fit, sigma5_fit, mu6_fit, sigma6_fit = confidence_params.x # 估计得到的参数值

        # 输出估计得到的高斯分布参数
        print("Gaussian 5: mu = {:.2f}, sigma = {:.2f}".format(mu5_fit, sigma5_fit))
        print("Gaussian 6: mu = {:.2f}, sigma = {:.2f}".format(mu6_fit, sigma6_fit))


    if isinstance(y_test, int) == False:
        result_final = score_detail(y_test,y_test_pred_no_vote,if_print=True)
    # result_no_vote.append(result_final)
    # x_train_this_epoch, x_test_left_epoch, y_train_this_epoch, y_test_left_epoch = online_x_tr
    if get_confidence == False:
        if en_or_de == False:
            if isinstance(y_test, int) == False:
                return result_encoder, result_decoder, result_final
            else:
                return y_test_pred_no_vote
        else:
            return y_test_pred_no_vote, index_output
    else:
        if isinstance(y_test, int) == False:
            return result_encoder, result_decoder, result_final, confidence_params.x
        else:
            return y_test_pred_no_vote, confidence_params.x


