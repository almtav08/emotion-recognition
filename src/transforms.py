"""
Transformaciones y Aumento de Datos Seguro para TEA (TEA-Safe Augmentations)
Preserva la micro-geometría facial de las Unidades de Acción Facial (AUs).
"""

from torchvision import transforms


def get_mobile_transforms(stage="train"):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if stage == "eval":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # Transformación de entrenamiento con aumentos controlados
    return transforms.Compose([
        transforms.Resize((236, 236)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # Oclusión puntual pequeña para evitar dependencia única en la boca
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random')
    ])
