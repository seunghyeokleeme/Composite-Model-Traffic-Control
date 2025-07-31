import pandas as pd
import numpy as np

# 1. 원본 CSV 파일 읽기
df = pd.read_csv('./data/traffic_speed/dnn_train.csv')

# 2. 특성(X)과 타겟(y) 데이터 정의
feature_cols = [
    '교통량1', '교통량2', '교통량3', '교통량4', '교통량5', '교통량6', '교통량7', 
    '교통량8', '교통량9', '교통량10', '교통량11', '교통량12', '교통량13', 
    '교통량14', '교통량15', '교통량16', '공휴일', '신호1', '신호2', '신호3', 
    '신호4', '신호5', '신호6', '방향'
]
target_col = ['속도']

X = df[feature_cols]
Y = df[target_col]

n_samples = len(df)
print(f"Total samples: {n_samples}")
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# 4. 인덱스를 사용하여 데이터 순차적으로 분할
# data 0501 - 0930
X_train = X.iloc[:22031+1]
Y_train = Y.iloc[:22031+1]

# data 1001 - 1015
X_val = X.iloc[22031+1:24191+1]
Y_val = Y.iloc[22031+1:24191+1]

# data 1016 - 1031
X_test = X.iloc[24191+1:]
Y_test = Y.iloc[24191+1:]

# 5. 분할된 데이터를 새로운 pkl 파일로 저장 (파일명에 _seq 추가)
X_train.to_pickle('./data/traffic_speed/train_X.pkl')
Y_train.to_pickle('./data/traffic_speed/train_Y.pkl')
X_val.to_pickle('./data/traffic_speed/val_X.pkl')
Y_val.to_pickle('./data/traffic_speed/val_Y.pkl')
X_test.to_pickle('./data/traffic_speed/test_X.pkl')
Y_test.to_pickle('./data/traffic_speed/test_Y.pkl')

# 6. 분할 결과 shape 출력
print("데이터 순차 분할이 완료되었습니다!")
print(f'X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}')
print(f'X_val shape: {X_val.shape},   Y_val shape: {Y_val.shape}')
print(f'X_test shape: {X_test.shape},  Y_test shape: {Y_test.shape}')