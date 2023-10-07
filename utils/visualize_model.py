import torch
import torch.nn as nn
from torchviz import make_dot

# Define your PyTorch model (replace this with your actual model)
# Define the UNet architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()


        self.use_xavier_init = True
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3 ,padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder11 = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3 ,padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder21 = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3 ,padding=1),
            nn.ReLU(inplace=True)
        )
        #self.encoder31 = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

        '''self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3 ,padding=1),
            nn.ReLU(inplace=True)
        )'''

        self.up31 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up32 = nn.Conv2d(512, 256, kernel_size=3,padding=1)
        self.decoder3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up21 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up22 = nn.Conv2d(256, 128, kernel_size=3,padding=1)
        self.decoder2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up11 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up12 = nn.Conv2d(128, 64, kernel_size=3,padding=1)
        self.last = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )


    def forward(self, x):
        # Encoder

        # encoder part
        x1 = self.encoder1(x)
        x11 = self.encoder11(x1)
        x2 = self.encoder2(x11)
        x21 = self.encoder21(x2)
        x3 = self.encoder3(x21)
        #x31 = self.encoder21(x3)
        #x4 = self.encoder4(x31)

        #y3 = self.decoder3(self.up32(torch.cat([self.up31(x4),x3], dim=1)))
        y2 = self.decoder2(self.up22(torch.cat([self.up21(x3),x2], dim=1)))
        y1 = self.last(self.up12(torch.cat([self.up11(y2),x1], dim=1)))

        return y1
    

# Create an instance of your model
model = UNet(1,5)

# Define a random input tensor for visualization (686, 990)
x = torch.randn(1, 1, 680, 984)  # Batch size of 1, 3 channels, 32x32 image

# Forward pass
output = model(x)

# Visualize the model architecture
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render("model_architecture", format="png")  # Saves the image as "model_architecture.png"
