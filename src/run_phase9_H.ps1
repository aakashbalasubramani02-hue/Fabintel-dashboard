python d:\ALK\src\train.py --exp H1_RotOnly --focal_loss --aug --aug_rotation --resnet --epochs 15 --img_size 128
python d:\ALK\src\train.py --exp H2_FlipOnly --focal_loss --aug --aug_flip --resnet --epochs 15 --img_size 128
python d:\ALK\src\train.py --exp H3_RotFlip --focal_loss --aug --aug_rotation --aug_flip --resnet --epochs 15 --img_size 128
python d:\ALK\src\train.py --exp H4_ReducedAug --focal_loss --aug --aug_rotation --aug_flip --aug_prob 0.5 --resnet --epochs 15 --img_size 128
python d:\ALK\src\evaluate_phase9_H.py
