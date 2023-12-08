## 2023-1 딥러닝기초 기말프로젝트: Image classification

##### link: [https://www.kaggle.com/competitions/whale-categorization-playground/submissions](https://www.kaggle.com/competitions/2023-final)

#### Problem

한림대학교에 재학 중인 철순이는 사람의 나이를 가늠하지 못해 실수를 하게 되었습니다.
친한 친구인 성민의 집에 놀러갔던날, 성민이의 여동생에게 "어머님"이라고 인사를 했던 것이 문제였습니다.
화가 난 성민이의 여동생은 철순이를 내쫓았고, 당황한 철순이는 가방도 챙기지 못한 채 밖으로 나오게 되었습니다.
성민이를 통해 가방을 받으려고 하자, 여동생이 이를 가로막고서

"지금 보여주는 사진에서 사람들의 연령대를 맞추지 못하면 가방을 돌려주지 않을 거야."

라며 문제를 내기 시작했습니다. 평소에도 사람들의 연령대를 잘 맞추지 못하는 철순은,
최근에 '딥러닝 기초' 수업을 통해 배운 CNN을 통해 이 문제를 해결하려고 합니다.

Data는 얼굴 이미지(200x200)와 각 이미지에 대한 Class로 구성이 되어 있습니다.
Class는 다음과 같습니다.

#### Dataset

0 : 1세 ~ 10세
1 : 11세 ~ 20세
2 : 21세 ~ 30세
3 : 31세 ~ 40세
4 : 41세 ~ 50세

추가적으로, Class를 제외한, 나이(Age), 성별(Gender), 인종(Race)에 대한 정보가 이미지의 이름으로 주어집니다.

#### 최종점수
![image](https://github.com/LeeGaHyeon/Deep-Learning-Project/assets/50908451/0e2d38bb-cce7-4e91-bde0-03148b164cb2)

