import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, 
                 dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16,
                 norm_rate=0.25, dropoutType='Dropout', feature_dim=None):
        """
        EEGNet: Compact CNN for EEG-based BCIs.
        
        Parameters
        ----------
        nb_classes : int
            Number of output classes.
        Chans : int
            Number of EEG channels.
        Samples : int
            Number of time samples per trial.
        dropoutRate : float
            Dropout probability.
        kernLength : int
            Length of temporal convolution kernel (half sampling rate recommended).
        F1 : int
            Number of temporal filters.
        D : int
            Depth multiplier for depthwise convolution.
        F2 : int
            Number of pointwise filters (usually F1 * D).
        feature_dim : int, optional
            If provided, adds a projection layer to reduce features to this dimension.
            Useful for BOTDA-GL compatibility with CSP (e.g., feature_dim=6).
        """
        super(EEGNet, self).__init__()
        
        self.D = D
        self.F1 = F1
        self.F2 = F2
        self.Chans = Chans
        self.Samples = Samples
        
        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise - Spatial filtering (análogo a CSP)
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(p=dropoutRate)
        
        # Block 2 - Separable Conv
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same', groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(p=dropoutRate)
        
        # Feature dimension after flatten
        self.flatten = nn.Flatten()
        self._flat_features = F2 * (Samples // 32)
        
        # Optional projection layer for BOTDA-GL compatibility
        self.feature_dim = feature_dim
        if feature_dim is not None:
            self.projection = nn.Linear(self._flat_features, feature_dim)
            self.classifier = nn.Linear(feature_dim, nb_classes)
        else:
            self.projection = None
            self.classifier = nn.Linear(self._flat_features, nb_classes)
        
        # Constraint value for depthwise conv
        self.max_norm = 1.0
    
    def _apply_max_norm(self):
        """Aplica constraint max_norm a depthwiseConv (como en el paper original)"""
        with torch.no_grad():
            w = self.depthwiseConv.weight
            norms = w.view(w.size(0), -1).norm(dim=1, keepdim=True)
            desired = torch.clamp(norms, max=self.max_norm)
            w *= (desired / (norms + 1e-8)).view(-1, 1, 1, 1)
    
    def extract_features(self, x):
        """
        Extract features before the classifier (for BOTDA-GL).
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, Chans, Samples).
        
        Returns
        -------
        features : torch.Tensor
            Feature tensor. Shape depends on feature_dim:
            - If feature_dim is set: (batch, feature_dim)
            - Otherwise: (batch, F2 * Samples // 32)
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwiseConv(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.separableConv(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = self.flatten(x)
        
        # Project if projection layer exists
        if self.projection is not None:
            x = self.projection(x)
        
        return x
    
    def forward(self, x):
        """Forward pass through the network."""
        features = self.extract_features(x)
        out = self.classifier(features)
        return out
    
    def get_feature_dim(self):
        """Return the dimension of extracted features."""
        if self.feature_dim is not None:
            return self.feature_dim
        return self._flat_features
    
    def freeze_temporal_layers(self):
        """Freeze temporal filtering layers for fine-tuning."""
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.parameters():
            param.requires_grad = True

# Default configuration for BCI Competition IV 2a (BNCI2014_001)
BNCI2014_001_CONFIG = {
    "nb_classes": 2,
    "Chans": 22,           # 22 EEG electrodes
    "Samples": 500,        # 2 seconds at 250Hz
    "dropoutRate": 0.5,
    "kernLength": 125,     # Half sampling rate
    "F1": 8,               # Temporal filters
    "D": 2,                # Spatial filters per temporal
    "F2": 16,              # F1 * D
    "feature_dim": None    # Set to 6 for CSP compatibility with BOTDA-GL
}


def main():
    """Test EEGNet instantiation and feature extraction."""
    
    # Standard EEGNet
    model = EEGNet(**BNCI2014_001_CONFIG)
    print(f"EEGNet feature dimension: {model.get_feature_dim()}")
    
    # EEGNet with projection for BOTDA-GL (CSP compatible)
    config_botda = BNCI2014_001_CONFIG.copy()
    config_botda["feature_dim"] = 6  # Match CSP n_components
    model_botda = EEGNet(**config_botda)
    print(f"EEGNet (BOTDA-GL) feature dimension: {model_botda.get_feature_dim()}")
    
    # Test forward pass
    x = torch.randn(4, 1, 22, 500)  # Batch of 4 trials
    
    out = model(x)
    print(f"Output shape: {out.shape}")
    
    features = model_botda.extract_features(x)
    print(f"Features shape (for BOTDA-GL): {features.shape}")


if __name__ == "__main__":
    main()