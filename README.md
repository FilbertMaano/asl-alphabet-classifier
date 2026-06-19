# ASL Alphabet Classifier

Real-time American Sign Language alphabet recognition using two CNN models built and compared in PyTorch.

## Model
- **AslNet** (custom CNN)
- **ResNet18** (fine-tuned pre-trained model)

| Model    | Real-World Accuracy |
|----------|-------------------|
| AslNet   | 16.78%            |
| ResNet18 | 54.25%            |

Full analysis in the [report](report/ASL_Classifier_Report.pdf).

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```

Run real-time inference:
```bash
python inference/aslnet_real_time.py
python inference/resnet18_real_time.py
```

Position your hand in the green square on screen. Press `q` to quit.

## Dataset
- Training: [ASL Alphabet - Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (87,000 images)
- Test: [ASL Alphabet Test - Kaggle](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test) (870 images)