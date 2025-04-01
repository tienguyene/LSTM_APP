from vnstock import *
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import joblib

def read_data(file_path, vn100_list):
    stock_df = pd.read_csv(file_path) # Đọc dữ liệu từ file csv
    stock_df['time'] = pd.to_datetime(stock_df['time']) # Chuyển cột date thành kiểu dữ liệu datetime

    temp = stock_df.groupby('stock_id')
    date_list = temp.get_group(vn100_list[0])['time'].tolist() # Lấy danh sách ngày từ dữ liệu của một mã chứng khoán bất kỳ

    stock_dict = {}
    for stock, stock_data in temp: # Gom nhóm dữ liệu theo mã chứng khoán, stock là tên các group, stock_data là dữ liệu của group đó
        # Kiểm tra xem dữ liệu của các mã chứng khoán có đồng nhất về số lượng ngày không
        if stock_data.shape[0] == len(date_list):
            stock_dict[stock] = stock_data.drop(columns= ['stock_id', 'time'])
        else:
            print(f"Dữ liệu của mã {stock} không đồng nhất với các mã chứng khoán khác")
            return None, None    

    return stock_dict, date_list
