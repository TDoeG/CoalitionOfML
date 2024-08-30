import torch
import torchvision

class PerceptualLoss(torch.nn.Module):
    def __init__(self, feature_extractor):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.criterion = torch.nn.MSELoss()
    
    def forward(self, pred, target):
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        loss = self.criterion(pred_features, target_features)
        return loss

class CombinedLoss(torch.nn.Module):
    def __init__(self, perceptual_loss, pixel_loss, perceptual_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss = perceptual_loss
        self.pixel_loss = pixel_loss
        self.perceptual_weight = perceptual_weight
    
    def forward(self, pred, target):
        perceptual_loss = self.perceptual_loss(pred, target)
        pixel_loss = self.pixel_loss(pred, target)
        return pixel_loss + self.perceptual_weight * perceptual_loss
