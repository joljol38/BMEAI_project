# 2023-1 BME AI projct <도마뱀 MBD 판별기>

### 1. Data
1) data1 : 도마뱀 중 정상개체와 MBD개체를 나누어서 train, validation, test set에 8:1:1로 분할
2) data2 : 도마뱀의 부위를 크게 몸통과 꼬리로 나누어서 train, validation, test set에 8:1:1로 분할

### 2. Model
1) Binary classification 
    
    총 세가지 모델로 구성
    1. base : CNN layer 직접 쌓음
    2. ResNet50 : pre-trained된 ResNet50모델을 사용해서 fine-tuning 진행
    3. ViT : pre-trained된 Vision Transformer모델을 사용해서 fine-tuning 진행

2) Multi-class classification

    binary classification과 동일한 방법으로 진행<br>
    이때 총 class 4개 : MBD body / MBD tail / Normal body / Normal tail
    
3) Video binary classification

    input값에 이미지와 동영상 추가
