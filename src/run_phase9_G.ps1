python d:\ALK\src\train.py --exp G1_NoWeights --aug --resnet --epochs 15 --img_size 128
python d:\ALK\src\train.py --exp G2_Cap5 --weights --cap_weight 5 --aug --resnet --epochs 15 --img_size 128
python d:\ALK\src\train.py --exp G3_Cap10 --weights --cap_weight 10 --aug --resnet --epochs 15 --img_size 128
python d:\ALK\src\train.py --exp G4_Cap20 --weights --cap_weight 20 --aug --resnet --epochs 15 --img_size 128
python d:\ALK\src\train.py --exp G5_FocalLoss --focal_loss --aug --resnet --epochs 15 --img_size 128
python d:\ALK\src\evaluate_phase9.py
