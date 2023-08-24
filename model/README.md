## Models

아기의 울음소리 분류를 수행하는 신경망 모델들을 다룬다.

### DNN

- dnn.ipynb : 전처리되지 않은 원본 데이터를 일정한 길이로 잘라 사용된 MLP 모델.

### LSTM

- lstm_2_sec.ipynb : 원본 데이터를 2초 단위로 잘라 사용된 모델. 78%의 정확도를 보인다.

- lstm_5_sec.ipynb : 원본 데이터를 5초 단위로 잘라 사용된 모델. 77%의 정확도를 보인다.

- lstm2_epochs_400.ipynb : trans_data 과정을 통해 전처리된 데이터를 2초 단위로 잘라 사용된 모델. 76%의 정확도를 보인다.

### CRNN

- crnn_v1.ipynb : trans_data 과정을 통해 전처리된 데이터를 사용한 모델. 84%의 정확도를 보인다.

- crnn_v2.ipynb : trans_data2 과정을 통해 전처리된 데이터를 사용한 모델. 84%의 정확도를 보인다.

### ResNet

- resnet50.ipynb : batch_size, epochs이 각각 32, 25인 resNet 전의학습 모델

- resnet_v2.ipynb : batch_size, steps_per_epoch,epochs이 각각 32, 130, 50인 resNet 전의학습 모델

- resnet_v3.ipynb : batch_size, steps_per_epoch,epochs이 각각 32, 152, 83 인 resNet 전의학습 모델. 89%의 정확도를 보인다.

- resnet_v4.ipynb : batch_size, steps_per_epoch,epochs이 각각 32, 160, 85 인 resNet 전의학습 모델. 93%의 정확도를 보인다.

- resnet_v3.h5 : resnet_v3.ipynb 를 통해 생성된 모델 파일

- resnet_model_load_test.ipynb : resnet_v3.h5 파일을 불러와 예측을 수행.
