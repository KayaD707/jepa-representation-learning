"""
data.py -- CIFAR-10 loading and I-JEPA-style context/target masking.

THE CORE IDEA THIS FILE IMPLEMENTS:

I-JEPA trains by showing the model most of an image (the "context") and asking it to
predict the REPRESENTATION of a hidden region (the "target") -- never the raw pixels of
that region. This is the whole point of the architecture: comparison happens in
representation space, not pixel space (see PLAN.md and the repo README for why that
distinction matters).

The original I-JEPA paper does this masking at the level of patch TOKENS, because it uses
a Vision Transformer, which naturally operates on a sequence of patches. This project
uses a small CNN encoder instead (a deliberate simplification stated in PLAN.md, to keep
training feasible on a CPU). A CNN doesn't take a sequence of patch tokens -- it takes a
full image grid. So instead of removing target patches from a sequence, this file does the
*spatial* equivalent: it zeroes out the target region directly in pixel space before
encoding, and separately zeroes out everything EXCEPT the target region for the target
view. The underlying idea (predict a hidden region's representation from the visible
rest) is preserved; only the mechanics of "hiding" something change to fit a CNN.

Picture it like this, for one image:

    original image          context view              target view
    +-------------+        +-------------+          +-------------+
    |             |        |             |          |             |
    |    [###]    |   -->  |    [   ]    |    and   |    [###]    |
    |             |        |             |          |             |
    +-------------+        +-------------+          +-------------+
                             (###  region          (everything EXCEPT
                              zeroed out --          the ### region
                              this is what the       zeroed out --
                              context encoder sees)   this is what the
                                                       target encoder sees)
"""

import random

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# CIFAR-10 images are 32x32 pixels, 3 color channels (RGB).
IMAGE_SIZE = 32

# We divide the image into a grid of small square patches, purely to make it easy to
# pick a rectangular BLOCK of patches as the target region. The patches themselves are
# never turned into tokens (that would be the ViT approach) -- they're just a convenient
# coordinate system for "which rectangle of the image do we hide."
PATCH_SIZE = 4
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE  # 32 / 4 = 8 -> an 8x8 grid of patches

# These are the well-known, pre-computed per-channel mean/std of the CIFAR-10 training
# set. Normalizing with them (rather than a generic 0.5/0.5) is standard practice -- it
# centers each color channel's pixel values around 0 with unit variance, which makes
# gradient-based training noticeably more stable than feeding in raw 0-1 pixel values.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(root: str = "./cifar_data", train: bool = True, download: bool = True):
    """Loads CIFAR-10 as a torchvision dataset of (image_tensor, label) pairs.

    Note: for I-JEPA pretraining, the label is simply never used (see
    JEPAMaskedCIFAR10 below) -- this function loads the full dataset either way, since
    the same download is reused later for the linear-probe evaluation, which DOES need
    the labels.
    """
    transform = transforms.Compose([
        # PIL image (0-255 integers) -> float tensor scaled to [0, 1], shape (3, 32, 32).
        transforms.ToTensor(),
        # Shift/scale each channel to roughly zero-mean, unit-variance, using CIFAR-10's
        # own known statistics -- see the comment on CIFAR10_MEAN/STD above.
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
    return datasets.CIFAR10(root=root, train=train, download=download, transform=transform)


def sample_target_block(
    grid_size: int = GRID_SIZE,
    scale_range: tuple = (0.15, 0.25),
    aspect_range: tuple = (0.75, 1.5),
):
    """Randomly chooses ONE rectangular block of patches to act as the JEPA target region.

    Follows the spirit of the original I-JEPA paper's target-block sampling: pick a
    block covering a random FRACTION of the image area (scale_range), with a random
    width-to-height ratio (aspect_range), placed at a random position -- rather than
    always masking, say, a fixed-size square in a fixed location, which would make the
    prediction task too easy and too repetitive for the model to learn anything general.

    Returns a tuple (row_start, row_end, col_start, col_end) in PATCH-grid coordinates
    (i.e. units of one patch, not one pixel) -- row_end/col_end are exclusive, Python-
    slice style.
    """
    total_patches = grid_size * grid_size

    # Pick how many patches the target block should cover, as a fraction of the image.
    target_scale = random.uniform(*scale_range)
    target_area_in_patches = max(1, round(total_patches * target_scale))

    # Pick an aspect ratio, then derive a (height, width) in patches that multiplies
    # out to roughly the target area -- clamped so the block can never be larger than
    # the grid itself in either dimension.
    aspect_ratio = random.uniform(*aspect_range)
    block_h = max(1, min(grid_size, round((target_area_in_patches / aspect_ratio) ** 0.5)))
    block_w = max(1, min(grid_size, round(target_area_in_patches / block_h)))

    # Pick a random top-left corner such that the block fits entirely inside the grid.
    row_start = random.randint(0, grid_size - block_h)
    col_start = random.randint(0, grid_size - block_w)

    return row_start, row_start + block_h, col_start, col_start + block_w


def make_context_target_masks(grid_size: int = GRID_SIZE, patch_size: int = PATCH_SIZE):
    """Builds two PIXEL-space boolean masks (context_mask, target_mask), each shaped
    (1, IMAGE_SIZE, IMAGE_SIZE) so they broadcast cleanly against a (3, H, W) image
    tensor via elementwise multiplication.

    context_mask is True everywhere EXCEPT the sampled target block (the context view
    hides the target region). target_mask is True ONLY inside the target block (the
    target view hides everything else). The two masks are exact complements of each
    other by construction.
    """
    row_start, row_end, col_start, col_end = sample_target_block(grid_size)

    # Build the mask at patch-grid resolution first (an 8x8 grid of True/False), since
    # that's the natural resolution the target block was chosen in.
    target_mask_grid = torch.zeros((grid_size, grid_size), dtype=torch.bool)
    target_mask_grid[row_start:row_end, col_start:col_end] = True

    # Upscale the 8x8 boolean grid to full 32x32 pixel resolution. Each grid cell was
    # defined to correspond to exactly one patch_size x patch_size block of pixels, so
    # repeat_interleave along both spatial axes does exactly that -- no interpolation,
    # no blurring, just "stamp this patch's True/False value onto all its pixels."
    target_mask = target_mask_grid.repeat_interleave(patch_size, dim=0) \
                                   .repeat_interleave(patch_size, dim=1)

    context_mask = ~target_mask  # exact complement

    # Add a leading channel dimension of size 1 so these masks broadcast against a
    # (3, H, W) image tensor: (1, H, W) * (3, H, W) -> (3, H, W), same mask applied to
    # all three color channels.
    return context_mask.unsqueeze(0), target_mask.unsqueeze(0)


class JEPAMaskedCIFAR10(Dataset):
    """Wraps a plain CIFAR-10 dataset to return (context_image, target_image) pairs for
    self-supervised pretraining.

    The label is deliberately read and then thrown away (see __getitem__) -- this is
    not a bug or an oversight. JEPA pretraining is self-supervised specifically because
    it never looks at the label; the "supervision" signal is the image predicting part
    of itself, not a human-provided class name. The labels only get used later, in the
    separate linear-probe evaluation, on a FROZEN encoder.
    """

    def __init__(self, base_dataset: Dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, _label = self.base[idx]  # label discarded on purpose -- see class docstring

        context_mask, target_mask = make_context_target_masks()

        # Elementwise multiply: wherever the mask is False, the pixel value becomes 0.
        # Both masks are boolean tensors here; PyTorch automatically treats True/False
        # as 1.0/0.0 for this multiplication.
        context_image = image * context_mask
        target_image = image * target_mask

        return context_image, target_image


if __name__ == "__main__":
    # A quick, visual sanity check -- the single best way to catch a masking bug is to
    # actually look at the masked images, not just trust that the shapes line up.
    import matplotlib
    matplotlib.use("Agg")  # write straight to a file, no interactive display needed
    import matplotlib.pyplot as plt

    raw_dataset = load_cifar10(train=True, download=True)
    masked_dataset = JEPAMaskedCIFAR10(raw_dataset)

    # Undo the normalization just for display purposes, so the images look like normal
    # photos instead of washed-out/shifted colors.
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    def to_displayable(img_tensor):
        img = img_tensor * std + mean          # undo normalization
        img = img.clamp(0, 1)                  # guard against tiny float overshoot
        return img.permute(1, 2, 0).numpy()    # (3,H,W) -> (H,W,3) for matplotlib

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for col in range(4):
        original_image, _ = raw_dataset[col]
        context_image, target_image = masked_dataset[col]

        axes[0, col].imshow(to_displayable(original_image))
        axes[1, col].imshow(to_displayable(context_image))
        axes[2, col].imshow(to_displayable(target_image))

    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("context (target hidden)")
    axes[2, 0].set_ylabel("target (only hidden region)")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("masking_sanity_check.png", dpi=120)
    print("Saved masking_sanity_check.png -- open it and confirm the masking looks right:")
    print("  - row 2 (context) should have a solid block missing")
    print("  - row 3 (target) should show ONLY that missing block, rest flat-colored")
    print("  - rows 2 and 3 should together reconstruct row 1 exactly, with no overlap")
    print()
    print("Note on color: the masked-out area appears as a muted grey/beige, not pure")
    print("black. That's expected, not a bug -- masking sets pixels to 0 in NORMALIZED")
    print("space (mean ~0), and this display function un-normalizes for viewing, so a")
    print("normalized 0 maps back to CIFAR-10's average pixel color, not black. The")
    print("model itself only ever sees the normalized tensor, where the masked region")
    print("really is a clean, consistent 0 -- the display is just reversing that step")
    print("to look like a photo again.")
