import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # Encoder path (Downsampling)
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1] * 2)
        
        # Decoder path (Upsampling)
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._block(feature * 2, feature))
        
        # Final Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            # If shapes are not equal due to pooling, adjust by cropping
            if x.shape != skip_connection.shape:
                skip_connection = self._center_crop(skip_connection, x.shape[2], x.shape[3])
            
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)
        
        return self.final_conv(x)
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def _center_crop(self, layer, target_height, target_width):
        _, _, height, width = layer.size()
        crop_y = (height - target_height) // 2
        crop_x = (width - target_width) // 2
        return layer[:, :, crop_y:crop_y + target_height, crop_x:crop_x + target_width]

