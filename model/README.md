## Models

아기의 울음소리 분류를 수행하는 신경망 모델들을 다룬다.

<br>

### LSTM

- lstm.ipynb : trans_data 과정을 통해 전처리된 데이터를 2초 단위로 잘라 사용된 모델. 76%의 정확도를 보인다.

<br>

### CRNN

- crnn.ipynb : trans_data2 과정을 통해 전처리된 데이터를 사용한 모델. 84%의 정확도를 보인다.

<br>

### ResNet

- resnet.ipynb : batch_size, steps_per_epoch,epochs이 각각 32, 160, 85 인 resNet 전의학습 모델. 93%의 정확도를 보인다.

- resnet.h5 : resnet.ipynb 를 통해 생성된 모델 파일

- resnet_model_load_test.ipynb : resnet.h5 파일을 불러와 예측을 수행하는 코드.
