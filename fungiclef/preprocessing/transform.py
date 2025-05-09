from torchvision import transforms


def get_transform_pipeline():
    """Get a transformation pipeline based on intensity level for augmentation only"""
    transform = transforms.Compose(
        [
            # Rotations (don't affect aspect ratio in multiples of 90°)
            transforms.RandomApply(
                [
                    transforms.RandomRotation(180, expand=False)  # Don't expand canvas
                ],
                p=0.8,
            ),
            # Crop with aspect ratio preservation
            transforms.RandomApply(
                [
                    # Crop to a random size and aspect ratio
                    transforms.RandomResizedCrop(
                        size=(
                            224,
                            224,
                        ),  # Temporary size - will be overridden by processor
                        scale=(0.8, 1.0),  # Size range
                        ratio=(0.75, 1.33),  # Aspect ratio range (3/4 to 4/3)
                    )
                ],
                p=0.7,
            ),
            # Color adjustments
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    )
                ],
                p=0.7,
            ),
            # Flips
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            # Blur
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        ]
    )
    return transform
