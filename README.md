# TinyViT from Scratch

[TinyViT (ECCV 2022)](https://arxiv.org/abs/2207.10666)의 **5M 모델 구조를 PyTorch로 직접 구현**하기

## Objective

- 20 클래스 이미지 분류 파인튜닝
- `timm`의 모델을 그대로 쓰지 않고, 모든 모듈(Patch Embedding, MBConv, Patch Merging, Window Attention 등)을 직접 구현

## Result

| 항목 | 값 |
| --- | --- |
| 모델 | TinyViT-5M (`tiny_vit_5m_224.dist_in22k_ft_in1k`) |
| 최고 검증 Accuracy | **92.50%** (epoch 23) |
| 학습 시간 | 732.4초 (25 epochs, Tesla T4) |
| 데이터 분할 | Train 9,600 / Val 2,400 / Test 3,408 (8:2 stratified split) |
| 클래스 수 | 20 |

## Model

### Modules

| 모듈 | 설명 |
| --- | --- |
| `ConvNorm` | Conv2d + BatchNorm2d 기본 블록 |
| `PatchEmbed` | Conv 2개(stride 2)로 입력을 특징 맵으로 변환 (해상도 1/4) |
| `MBConv` / `ConvStage` | Stage 0: 채널 확장 → depthwise conv → 채널 축소 (residual) |
| `PatchMerging` | 해상도를 절반으로 줄이고 다음 stage 채널 수로 변환 |
| `WindowAttention` | window 내부 multi-head self-attention + 학습 가능한 상대 위치 bias |
| `TinyViTBlock` | Window Attention → depthwise local conv → MLP |
| `TinyViTStage` | Patch Merging + TinyViTBlock 반복 |

### Stage Configuration (5M)

> 이미지 그리기

| Stage | 구성 | 채널 | depth | heads | window |
| --- | --- | --- | --- | --- | --- |
| 0 | MBConv | 64 | 2 | - | - |
| 1 | TinyViT | 128 | 2 | 4 | 3 |
| 2 | TinyViT | 160 | 6 | 5 | 5 |
| 3 | TinyViT | 320 | 2 | 10 | 3 |

### Implementation Notes

- **입력 해상도 변경**: 사전학습은 224×224, 이 프로젝트의 입력은 훨씬 작은 이미지입니다. window가 해상도의 배수가 아닐 때를 위해 오른쪽·아래쪽 padding 후 attention을 수행하고, 이후 원래 크기로 자릅니다.
- **상대 위치 bias 보간**: window 크기가 달라지면 `attention_biases`의 shape가 달라지므로, 가중치를 불러올 때 bicubic 보간으로 새 window 크기에 맞춥니다.
- **가중치 로딩**: 공개 safetensors 가중치를 `strict=True`로 불러와 직접 구현한 구조가 원본과 일치하는지 검증합니다. 이후 분류 head만 20클래스로 교체합니다.

> 논문의 핵심 기법인 대규모 사전학습 지식 증류(distillation)는 직접 수행하지 않았고, 증류로 학습된 공개 가중치를 사용했습니다.

## Training Configuration

| 항목 | 값 |
| --- | --- |
| Optimizer | AdamW (lr 1e-4, weight decay 1e-4) |
| Scheduler | CosineAnnealingLR (T_max = 25) |
| Loss | CrossEntropyLoss |
| Batch size / Epochs | 64 / 25 |
| Seed | 42 |

## Environment

Google Colab(Tesla T4) 환경에서 작성·실행했습니다. CUDA GPU가 필요합니다.

```bash
pip install torch torchvision torchaudio timm safetensors scikit-learn pandas matplotlib pillow
```

## Project Structure

```
.
├── TinyViT.ipynb      # 모델 구현, 학습, 평가, 추론 전체 코드
├── outputs/
│   ├── model.pt           # best 검증 정확도 체크포인트
│   └── model_history.csv  # epoch별 train/val loss, val accuracy
└── submission.csv     # 테스트 셋 예측 결과
```

## Future Work

- **출력 일치 검증**: 노트북에서 timm 모델을 이미 불러오고 있으니, 직접 구현한 모델과 같은 입력에서 logits 차이를 비교합니다. 구현이 정확하다는 가장 직접적인 증거가 됩니다.
- **간단한 ablation**: 사전학습 가중치 사용 여부, bias 보간 여부에 따른 검증 정확도를 비교합니다.
- **baseline 비교**: MobileNet 같은 비슷한 크기 모델과 정확도, 파라미터 수, 추론 속도를 비교합니다.

## Reference

- [TinyViT: Fast Pretraining Distillation for Small Vision Transformers](https://arxiv.org/abs/2207.10666)
- [microsoft/Cream (TinyViT)](https://github.com/microsoft/Cream/tree/main/TinyViT)
- [timm/tiny_vit_5m_224.dist_in22k_ft_in1k](https://huggingface.co/timm/tiny_vit_5m_224.dist_in22k_ft_in1k)