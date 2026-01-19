import torchvision.transforms as transforms
import random
from torchvision.transforms import v2





# def strong_augmentation():
   
    
#     # 定义可用的图像变换
#     transforms_list = [
#         transforms.RandomHorizontalFlip(p=0.5),
#         v2.RandomEqualize(),
#         transforms.RandomInvert(),
#         transforms.ColorJitter(brightness=(0.05,0.95)),
#         transforms.ColorJitter(contrast=(0.05,0.95)),
#         transforms.RandomSolarize(threshold=random.uniform(0, 1)),
#         v2.RandomAdjustSharpness(sharpness_factor=random.uniform(0.05,0.95)),
#         v2.RandomErasing(scale=(0.02,0.4), ratio=(0.3,0.3)),
#         transforms.RandomAffine(degrees=(-30,30),shear=(-0.3, 0.3,-0.3,0.3))
#     ]

#     # 随机选择几个变换（这里选择1到3个）
#     # num_transforms = 1
#     num_transforms = random.randint(1,3)
#     selected_transforms = random.sample(transforms_list, num_transforms)

#     # 将变换列表转换为transforms.Compose对象
#     strong_aug = transforms.Compose(selected_transforms)

#     return strong_aug
def augmentation():
    """返回优化后的数据增强管道"""
    return transforms.Compose([
        # 几何变换 (60-80%概率应用其中1-2个)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.RandomRotation(degrees=(-30, 30)),
                transforms.RandomAffine(degrees=(-30, 30), shear=(-0.2, 0.2, -0.2, 0.2)),
            ])
        ], p=0.7),
        
        # 色彩变换 (40-60%概率应用)
        transforms.RandomApply([
            transforms.ColorJitter(brightness=(0.05,0.95), contrast=(0.05,0.95), saturation=0.3, hue=0.05)
        ], p=0.6),
        
        # 特效变换 (20-40%概率应用其中1个)
        transforms.RandomApply([
            transforms.RandomChoice([
                v2.RandomEqualize(),
                transforms.RandomInvert(),
                transforms.RandomSolarize(threshold=0.5),
                v2.RandomAdjustSharpness(sharpness_factor=2),
            ])
        ], p=0.3),
    ])