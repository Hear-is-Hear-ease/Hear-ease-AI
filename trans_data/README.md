## trans_data 설명

본 폴더는 sampling rate 통일, 음성 길이 통일과 같이 음성 데이터를 직간접적으로 변형하는 내용을 담고 있다.

각각의 파일에 대한 설명은 아래와 같다.

<br>

### create_state_folder.py

원천 데이터를 처리하여 가용한 데이터로 변환한다. 처리 내용은 아래와 같다.

1. 분석할 수 없는 파일을 제거한다.

2. sampling rate를 통일한다.

3. state 별 파일을 폴더 이름을 통해 구분한다. 이때 상위 7개의 state에 대해서만 구분하며 나머지는 etc 파일에 csv와 함께 저장한다.

위 코드를 수행한 다음 data 폴더를 삭제하고 temp 폴더의 이름을 'data'로 변경하는 것은 사용자에게 맡긴다. 예기치 못하게 데이터가 삭제되는 것을 미연에 방지하는 것이다.

<br>

### bit_sampling.py

음성의 sampling rate를 통일하는 resampling 함수와 파일 리스트가 모두 동일한 sampling rate를 가지는지 확인하는 is_same_sample_rate 함수를 제공한다.

<br>

### rename_by_state.py

state 이름을 통해 파일명을 hungry_1.wav, hungry_2.wav와 같이 변경한다.

<br>

### get_sample_data.py

라벨에 따라 균등하게 맞추어 n개의 샘플을 추출한다.

<br>

### split_audio.py

오디오를 지정된 시간만큼 자른다.

<br>

### get_state_list.py

폴더의 이름으로부터 state 이름을 가져온다.
