import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_SIZE = (32,64,128,256,320)

class ResNetBlock(nn.Module):
  
    def __init__(self, input_channels,output_channels = None, kernel_size = 3, stride = 1, padding = 1, **conv_kwargs):
        
        super().__init__()

        self.input_channels = input_channels
        if output_channels is None:
            self.output_channels = input_channels
        else:
            self.output_channels = output_channels
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv3d (
                                in_channels = self.input_channels,
                                out_channels = self.output_channels,
                                kernel_size = self.ks ,
                                stride = self.stride,
                                padding = self.padding, 
                                **conv_kwargs
                                )
        self.in1 = nn.InstanceNorm3d(num_features = self.output_channels)

    def forward(self, x):

        #res = x
        out = self.conv(x)
        out = F.relu(self.in1(out))

        #return (out + res)
        return out

class nnUNetModel(nn.Module):

    def __init__(self, input_channels = 32, output_channels = 256, kernel_size = 3, **conv_kwargs):
        super().__init__()
        # encoder part
        self.conv0 = nn.Conv3d(in_channels = 4, out_channels = 32, kernel_size = 1)    

        self.drop = nn.Dropout3d(p=0.1)

        self.res_block1 = ResNetBlock(input_channels = 32)
        self.res_block2 = ResNetBlock(input_channels = 32)

        #self.conv1 = nn.Conv3d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)   

        self.res_block3 = ResNetBlock(input_channels = 32, output_channels = 64, stride = 2)
        self.res_block4 = ResNetBlock(input_channels = 64)

        #self.conv2 = nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1)    
            
        self.res_block5 = ResNetBlock(input_channels = 64, output_channels = 128, stride = 2)
        self.res_block6 = ResNetBlock(input_channels = 128)

        #self.conv3 = nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding = 1)    

        self.res_block7 = ResNetBlock(input_channels = 128, output_channels = 256, stride = 2)
        self.res_block8 = ResNetBlock(input_channels = 256)

        #self.conv4 = nn.Conv3d(in_channels = 256, out_channels = 320, kernel_size = 3, stride = 2, padding = 1)    

        self.res_block9 = ResNetBlock(input_channels = 256, output_channels = 320, stride = 2)
        self.res_block10 = ResNetBlock(input_channels = 320)

        self.res_block11 = ResNetBlock(input_channels = 320, stride = 2)
        self.res_block12 = ResNetBlock(input_channels = 320)

        self.upconv1 = nn.ConvTranspose3d (
                                            in_channels = 320,
                                            out_channels = 320,
                                            kernel_size = 3,
                                            stride = 2,
                                            padding = 1,
                                            output_padding = 1
                                            )

        self.res_block13 = ResNetBlock(input_channels = 320)
        self.res_block14 = ResNetBlock(input_channels = 320)

        self.upconv2 = nn.ConvTranspose3d (
                                            in_channels = 320,
                                            out_channels = 256,
                                            kernel_size = 3,
                                            stride = 2,
                                            padding = 1,
                                            output_padding = 1
                                            )

        self.res_block15 = ResNetBlock(input_channels = 256)
        self.res_block16 = ResNetBlock(input_channels = 256)

        self.upconv3 = nn.ConvTranspose3d(                                  
                                            in_channels = 256,
                                            out_channels = 128,
                                            kernel_size = 3,
                                            stride = 2,
                                            padding = 1,
                                            output_padding = 1
                                            )

        self.res_block17 = ResNetBlock(input_channels = 128)
        self.res_block18 = ResNetBlock(input_channels = 128)

        self.upconv4 = nn.ConvTranspose3d(
                                            in_channels = 128,
                                            out_channels = 64,
                                            kernel_size = 3,
                                            stride = 2,
                                            padding = 1,
                                            output_padding = 1
                                            )

        self.res_block19 = ResNetBlock(input_channels = 64)
        self.res_block20 = ResNetBlock(input_channels = 64)

        self.upconv5 = nn.ConvTranspose3d(
                                            in_channels = 64,
                                            out_channels = 32,
                                            kernel_size = 3,
                                            stride = 2,
                                            padding = 1,
                                            output_padding = 1
                                            )

        self.res_block21 = ResNetBlock(input_channels = 32)
        self.res_block22 = ResNetBlock(input_channels = 32)

        self.fconv = nn.Conv3d(in_channels = 32, out_channels = 3, kernel_size = 1)
        #self.sig = torch.nn.Sigmoid()

    def forward(self, x):

        x = self.conv0(x)
        #x = self.drop(x)

        r1 = self.res_block1(x)
        r2 = self.res_block2(r1)

        r3 = self.res_block3(r2)
        r4 = self.res_block4(r3)

        r5 = self.res_block5(r4)
        r6 = self.res_block6(r5)

        r7 = self.res_block7(r6)
        r8 = self.res_block8(r7)

        r9 = self.res_block9(r8)
        r10 = self.res_block10(r9)

        r11 = self.res_block11(r10)
        r12 = self.res_block12(r11)
       

        # here decoder part starts

        u1 = self.upconv1(r12)
        u1 = u1 + r10

        r13 = self.res_block13(u1)
        r14 = self.res_block14(r13)

        u2 = self.upconv2(r14)
        u2 = u2 + r8

        r15 = self.res_block15(u2)
        r16 = self.res_block16(r15)

        u3 = self.upconv3(r16)
        u3 = u3 + r6

        r17 = self.res_block17(u3)
        r18 = self.res_block18(r17)

        u4 = self.upconv4(r18)
        u4 = u4 + r4

        r19 = self.res_block19(u4)
        r20 = self.res_block20(r19)

        u5 = self.upconv5(r20)
        u5 = u5 + r2

        r21 = self.res_block21(u5)
        r22 = self.res_block22(r21)

        cf = self.fconv(r22)
        #prob = self.sig(cf)

        return(cf)