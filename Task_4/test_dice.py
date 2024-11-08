class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # Flatten tensors to calculate Dice coefficient
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice  # Since we want it as a loss (1 - Dice coefficient)

def dice_loss(orig_mask, seg_mask, epsilon = 1e-6):
    # convert to float32 numpy array
    orig_mask = orig_mask.astype(np.float32)/255.
    seg_mask = seg_mask.astype(np.float32)/255.
    # Compute the intersection that is the sum of element wise multiplication
    intersection = np.sum(orig_mask *  seg_mask)
    # Compute the union between the two masks that is sum of element wise addition
    union = np.sum(orig_mask) + np.sum(seg_mask) + epsilon
    # Compute the dice coefficient
    dice_coef = (2 * intersection)/union
    # Calculate the dice loss
    dice_loss = 1 - dice_coef
    # return the dice loss
    return np.round(dice_loss, 3)
