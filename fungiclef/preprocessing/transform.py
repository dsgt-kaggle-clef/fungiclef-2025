from torchvision import transforms


def get_transform_pipeline():
    """Get a transformation pipeline based on intensity level for augmentation only"""
    transform = transforms.Compose(
        [
            transforms.RandomApply([transforms.RandomRotation(30)], p=0.3),
            transforms.RandomResizedCrop(
                (224, 224), scale=(0.9, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

    return transform
