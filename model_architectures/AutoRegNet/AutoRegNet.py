import torch
import torch.nn as nn

def ResNetBlock(num_channels, num_groups = 8, kernel_size = 3, **conv_kwargs):

  return nn.Sequential(
                       nn.GroupNorm(num_groups = num_groups, num_channels = num_channels),
                       nn.LeakyReLU(0.01),
                       nn.Conv3d(in_channels = num_channels , out_channels = num_channels, kernel_size = kernel_size , **conv_kwargs),
                       nn.GroupNorm(num_groups = num_groups, num_channels = num_channels),
                       nn.LeakyReLU(0.01),
                       nn.Conv3d(in_channels = num_channels , out_channels = num_channels, kernel_size = kernel_size , **conv_kwargs)
                      )


def UpsizeLayer(c_in, c_out, ks=1, scale=2):

 return nn.Sequential(nn.Conv3d(c_in, c_out, ks),
                      nn.Upsample(scale_factor=scale, mode='trilinear', align_corners = False))

class AutoEncoderRegularizationModel(nn.Module):

  def __init__(self, input_channels = 32, output_channels = 256, kernel_size = 3, **conv_kwargs):
    super().__init__()
    # encoder part
    self.conv0 = nn.Conv3d(in_channels = 4, out_channels = 32, kernel_size = 1)    
    
    self.drop = nn.Dropout3d(p=0.2)

    self.res_block1 = ResNetBlock(num_channels = 32, num_groups=8, kernel_size = 3, padding = 1)

    self.conv1 = nn.Conv3d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)    

    self.res_block2 = ResNetBlock(num_channels = 64, num_groups=8, kernel_size = 3, padding = 1)
    self.res_block3 = ResNetBlock(num_channels = 64, num_groups=8, kernel_size = 3, padding = 1)

    self.conv2 = nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1)    
  
    self.res_block4 = ResNetBlock(num_channels = 128, num_groups=8, kernel_size = 3, padding = 1)
    self.res_block5 = ResNetBlock(num_channels = 128, num_groups=8, kernel_size = 3, padding = 1)

    self.conv3 = nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding = 1)    
  
    self.res_block6 = ResNetBlock(num_channels = 256, num_groups=8, kernel_size = 3, padding = 1)
    self.res_block7 = ResNetBlock(num_channels = 256, num_groups=8, kernel_size = 3, padding = 1)
    self.res_block8 = ResNetBlock(num_channels = 256, num_groups=8, kernel_size = 3, padding = 1)
    self.res_block9 = ResNetBlock(num_channels = 256, num_groups=8, kernel_size = 3, padding = 1)   
    
    # decoder part
    self.up1 = UpsizeLayer(c_in = 256, c_out = 128, ks =1 , scale =2)  

    self.res_block10 = ResNetBlock(num_channels = 128, num_groups=8, kernel_size = 3, padding = 1)

    self. up2 = UpsizeLayer(c_in = 128, c_out = 64, ks =1 , scale =2)  

    self.res_block11 = ResNetBlock(num_channels = 64, num_groups=8, kernel_size = 3, padding = 1)

    self. up3 = UpsizeLayer(c_in = 64, c_out = 32, ks =1 , scale =2)  

    self.res_block12 = ResNetBlock(num_channels = 32, num_groups=8, kernel_size = 3, padding = 1)

    self.convf = nn.Conv3d(in_channels = 32, out_channels = 3, kernel_size =1)

    self.sig = torch.nn.Sigmoid()


  # def add_layer(layer_a, layer_b):

  def forward(self, x):

    x = self.conv0(x)
    x = self.drop(x)

    r1 = self.res_block1(x)
    r1 = r1 + x

    c1 = self.conv1(r1)

    r2 = self.res_block2(c1)
    r2 = r2 + c1
    r3 = self.res_block3(r2)
    r3 = r3 + r2

    c2 = self.conv2(r3)

    r4 = self.res_block4(c2)
    r4 = r4 + c2
    r5 = self.res_block5(r4)
    r5 = r5 + r4

    c3 = self.conv3(r5)

    r6 = self.res_block6(c3)
    r6 = r6 + c3
    r7 = self.res_block7(r6)
    r7 = r7 + r6
    r8 = self.res_block8(r7)
    r8 = r8 + r7 
    r9 = self.res_block9(r8)
    r9 = r9 + r8

    u1 = self.up1(r9)
    u1 = u1 + r5

    r10 = self.res_block10(u1)
    r10 = r10 + u1

    u2 = self.up2(r10)
    u2 = u2 + r3

    r11 = self.res_block11(u2)
    r11 = r11 + u2

    u3 = self.up3(r11)
    u3 = u3 + r1

    r12 = self.res_block12(u3)
    r12 = r12 + u3

    cf = self.convf(r12)
    prob = self.sig(cf)

    return(prob)