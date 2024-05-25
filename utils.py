import torch
import numpy as np
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
import scipy.optimize as opt
import torch.distributions as dist
from sklearn.metrics import accuracy_score
from torch.distributions import Normal

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

        # Find the nearest power of 2 to input_dim
        nearest_power_of_2 = 2 ** round(math.log2(input_dim))

        # Calculate the dimensions of the 2nd/4th layer and the 3rd layer.
        second_fourth_layer_size = nearest_power_of_2 // 2  # A half
        third_layer_size = nearest_power_of_2 // 4         # A quarter

        # Create encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )

        # Create decoder
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
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float()
        # compute logits
        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # Calculate the dot product similarity between pairwise samples
        # create mask 
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

def log_likelihood(params, data):
    mu1, sigma1, mu2, sigma2 = params
    pdf1 = gaussian_pdf(data, mu1, sigma1)
    pdf2 = gaussian_pdf(data, mu2, sigma2)
    return -np.sum(np.log(0.5 * pdf1 + 0.5 * pdf2))

def extract_features(x, model, layer_index):
    return F.normalize(model(x)[layer_index], p=2, dim=1)

def calculate_cosine_similarity(features, temp):
    return F.cosine_similarity(features, temp.reshape([-1, temp.shape[0]]), dim=1)

def fit_gaussian(values):
    initial_params = np.array([np.mean(values), np.std(values), np.mean(values), np.std(values)])
    result = opt.minimize(log_likelihood, initial_params, args=(values,), method='Nelder-Mead')
    mu1, sigma1, mu2, sigma2 = result.x
    if mu1 > mu2:
        gaussian1, gaussian2 = Normal(mu1, sigma1), Normal(mu2, sigma2)
    else:
        gaussian1, gaussian2 = Normal(mu2, sigma2), Normal(mu1, sigma1)
    return gaussian1, gaussian2

def process_all_data(x_train, y_train, x_test, temp, model, layer_index):
    x_train_normal = x_train[(y_train == 0).squeeze()]
    x_train_abnormal = x_train[(y_train == 1).squeeze()]
    
    train_features_normal = extract_features(x_train_normal, model, layer_index)
    train_features_abnormal = extract_features(x_train_abnormal, model, layer_index)
    test_features = extract_features(x_test, model, layer_index)

    values_normal = calculate_cosine_similarity(train_features_normal, temp)
    values_abnormal = calculate_cosine_similarity(train_features_abnormal, temp)
    values_test = calculate_cosine_similarity(test_features, temp)
    
    return values_normal, values_abnormal, values_test

def predict_with_gaussian(values_normal, values_test):
    gaussian1, gaussian2 = fit_gaussian(values_normal.cpu().detach().numpy())
    pdf1 = gaussian1.log_prob(values_test).exp()
    pdf2 = gaussian2.log_prob(values_test).exp()
    predictions = (pdf2 > pdf1).cpu().numpy().astype("int32")
    confidence = (torch.abs(pdf2 - pdf1)).cpu().detach().numpy().astype("float32")
    return predictions, confidence

def evaluate(normal_temp, normal_recon_temp, x_train, y_train, x_test, y_test, model):
    num_of_layer = 0
    num_of_output = 1

    values_features_normal, values_features_abnormal, values_features_test = process_all_data(x_train, y_train, x_test, normal_temp, model, num_of_layer)
    values_recon_normal, values_recon_abnormal, values_recon_test = process_all_data(x_train, y_train, x_test, normal_recon_temp, model, num_of_output)

    y_test_pred_2, y_test_pro_en = predict_with_gaussian(values_features_normal, values_features_test)
    y_test_pred_4, y_test_pro_de = predict_with_gaussian(values_recon_normal, values_recon_test)

    if not isinstance(y_test, int):
        if y_test.device != torch.device("cpu"):
            y_test = y_test.cpu().numpy()
        result_encoder = score_detail(y_test, y_test_pred_2)
        result_decoder = score_detail(y_test, y_test_pred_4)

    y_test_pred_no_vote = torch.where(torch.from_numpy(y_test_pro_en) > torch.from_numpy(y_test_pro_de), torch.from_numpy(y_test_pred_2), torch.from_numpy(y_test_pred_4))

    if not isinstance(y_test, int):
        result_final = score_detail(y_test, y_test_pred_no_vote, if_print=True)
        return result_encoder, result_decoder, result_final
    else:
        return y_test_pred_no_vote