"""
data_two_view.py -- CIFAR-10 loading plus the two-augmented-views dataset wrapper
shared by both train_contrastive.py and train_noncontrastive.py.

This is the same construction used in Project D's cifar_demo.py (two independent
random augmentations of the same photo, standing in for "two views x, y of the same
thing"). Kept as a local copy rather than importing Project D's script directly --
see PLAN.md's infrastructure table for why. Points at Project A's already-downloaded
CIFAR-10 folder to avoid a third redundant ~170MB download.
"""

from torch.utils.data import Dataset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR_ROOT = "../project-a-ijepa-cifar10/cifar_data"

AUGMENT_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.4, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


class TwoViewCIFAR10(Dataset):
    """Returns two independently-augmented views of the same image; label discarded
    (self-supervised) -- see cifar_demo.py in Project D for the identical pattern."""

    def __init__(self, base_dataset_no_transform):
        self.base = base_dataset_no_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, _label = self.base[idx]
        return AUGMENT_TRANSFORM(image), AUGMENT_TRANSFORM(image)


def load_cifar10_raw(train: bool):
    """No transform applied -- used as the base dataset for TwoViewCIFAR10, which
    applies its own augmentation per call."""
    return datasets.CIFAR10(root=CIFAR_ROOT, train=train, download=True)


def load_cifar10_eval(train: bool):
    """Fixed (non-augmented) transform -- used for linear-probe feature extraction,
    where a stable, repeatable representation per image is wanted, not a random crop."""
    return datasets.CIFAR10(root=CIFAR_ROOT, train=train, download=True, transform=EVAL_TRANSFORM)
