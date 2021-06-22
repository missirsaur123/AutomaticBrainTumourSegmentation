import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
  
    def __init__(self, input_channels,output_channels = None, kernel_size = 3, stride = 1, padding = 1, **conv_kwargs):
        
        super().__init__()

        self.input_channels = input_channels
        """if output_channels is None:
            self.output_channels = input_channels
        else:
            self.output_channels = output_channels"""
        self.output_channels = input_channels if output_channels is None else output_channels
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv3d(
                              in_channels = self.input_channels,
                              out_channels = self.output_channels,
                              kernel_size = self.ks ,
                              stride = self.stride,
                              padding = self.padding, 
                              **conv_kwargs
                              )
        self.in1 = nn.InstanceNorm3d(num_features = self.output_channels)
        #self.use_res_connect = True
        #self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):

        res = x
        out = self.conv(x)
        out = F.relu(self.in1(out))
        """if self.use_res_connect:
            return self.skip_add.add(res, out)
        else:
            return out
        #return out"""

        return (out + res)
        

class ResUNetModel(nn.Module):

    def __init__(self, input_channels = 32, output_channels = 256, kernel_size = 3, **conv_kwargs):
        super().__init__()
        # encoder part
        self.conv0 = nn.Conv3d(in_channels = 4, out_channels = 32, kernel_size = 1)    

        self.drop0 = nn.Dropout3d(p=0.2)

        self.res_block1 = ResNetBlock(input_channels = 32)
        self.res_block2 = ResNetBlock(input_channels = 32)
        
        self.downconv1 = nn.Sequential(
                                       nn.Conv3d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
                                       nn.Dropout3d(p=0.2)
                                  
                                      )

        self.res_block3 = ResNetBlock(input_channels = 64)
        self.res_block4 = ResNetBlock(input_channels = 64)

        self.downconv2 = nn.Sequential(
                                       nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
                                       nn.Dropout3d(p=0.2)
                                  
                                      )
            
        self.res_block5 = ResNetBlock(input_channels = 128)
        self.res_block6 = ResNetBlock(input_channels = 128)

        self.downconv3 = nn.Sequential(
                                       nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding = 1),
                                       nn.Dropout3d(p=0.2)
                                  
                                      )

        self.res_block7 = ResNetBlock(input_channels = 256)
        self.res_block8 = ResNetBlock(input_channels = 256)

        self.downconv4 = nn.Sequential(
                                       nn.Conv3d(in_channels = 256, out_channels = 320, kernel_size = 3, stride = 2, padding = 1),
                                       nn.Dropout3d(p=0.2)
                                  
                                      )

        self.res_block9 =  ResNetBlock(input_channels = 320)
        self.res_block10 = ResNetBlock(input_channels = 320)

        self.downconv5 = nn.Sequential(
                                       nn.Conv3d(in_channels = 320, out_channels = 320, kernel_size = 3, stride = 2, padding = 1),
                                       nn.Dropout3d(p=0.2)
                                  
                                      )

        self.res_block11 = ResNetBlock(input_channels = 320)
        self.res_block12 = ResNetBlock(input_channels = 320)

        self.upconv1 = nn.Sequential(
                                     nn.ConvTranspose3d(
                                          in_channels = 320,
                                          out_channels = 320,
                                          kernel_size = 3,
                                          stride = 2,
                                          padding = 1,
                                          output_padding = 1
                                      ),
                                      nn.Dropout3d(p = 0.2)
        )


        self.res_block13 = ResNetBlock(input_channels = 320)
        self.res_block14 = ResNetBlock(input_channels = 320)

        self.upconv2 = nn.Sequential(
                                     nn.ConvTranspose3d(
                                          in_channels = 320,
                                          out_channels = 256,
                                          kernel_size = 3,
                                          stride = 2,
                                          padding = 1,
                                          output_padding = 1
                                      ),
                                      nn.Dropout3d(p = 0.2)
        )

        self.res_block15 = ResNetBlock(input_channels = 256)
        self.res_block16 = ResNetBlock(input_channels = 256)

        self.cs_block1 = nn.Sequential(
                                      nn.Conv3d(256,256,1),
                                      nn.Softmax(dim = 1)
        )

        self.upconv3 = nn.Sequential(
                                     nn.ConvTranspose3d(
                                          in_channels = 256,
                                          out_channels = 128,
                                          kernel_size = 3,
                                          stride = 2,
                                          padding = 1,
                                          output_padding = 1
                                      ),
                                      nn.Dropout3d(p = 0.2)
        )

        self.res_block17 = ResNetBlock(input_channels = 128)
        self.res_block18 = ResNetBlock(input_channels = 128)

        self.cs_block2 = nn.Sequential(
                                      nn.Conv3d(128,128,1),
                                      nn.Softmax(dim = 1)
        )

        self.upconv4 = nn.Sequential(
                                     nn.ConvTranspose3d(
                                          in_channels = 128,
                                          out_channels = 64,
                                          kernel_size = 3,
                                          stride = 2,
                                          padding = 1,
                                          output_padding = 1
                                      ),
                                      nn.Dropout3d(p = 0.2)
        )


        self.res_block19 = ResNetBlock(input_channels = 64)
        self.res_block20 = ResNetBlock(input_channels = 64)

        self.cs_block3 = nn.Sequential(
                                      nn.Conv3d(64,64,1),
                                      nn.Softmax(dim = 1)
        )

        self.upconv5 = nn.Sequential(
                                     nn.ConvTranspose3d(
                                          in_channels = 64,
                                          out_channels = 32,
                                          kernel_size = 3,
                                          stride = 2,
                                          padding = 1,
                                          output_padding = 1
                                      ),
                                      nn.Dropout3d(p = 0.2)
        )


        self.res_block21 = ResNetBlock(input_channels = 32)
        self.res_block22 = ResNetBlock(input_channels = 32)

        self.fconv = nn.Conv3d(in_channels = 32, out_channels = 4, kernel_size = 1)
        #self.sig = torch.nn.Sigmoid()
        #self.quant = QuantStub()
        #self.dequant = DeQuantStub()
        

    def forward(self, x):
        
        #x = self.quant(x)

        x = self.conv0(x)
        x = self.drop0(x)

        r1 = self.res_block1(x)
        r2 = self.res_block2(r1)

        c1 = self.downconv1(r2)

        r3 = self.res_block3(c1)
        r4 = self.res_block4(r3)

        c2 = self.downconv2(r4)

        r5 = self.res_block5(c2)
        r6 = self.res_block6(r5)

        c3 = self.downconv3(r6)

        r7 = self.res_block7(c3)
        r8 = self.res_block8(r7)

        c4 = self.downconv4(r8)

        r9 = self.res_block9(c4)
        r10 = self.res_block10(r9)

        c5 = self.downconv5(r10)

        r11 = self.res_block11(c5)
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

        cs1 = self.cs_block1(r16)

        u3 = self.upconv3(cs1)
        u3 = u3 + r6

        r17 = self.res_block17(u3)
        r18 = self.res_block18(r17)

        cs2 = self.cs_block2(r18)

        u4 = self.upconv4(cs2)
        u4 = u4 + r4

        r19 = self.res_block19(u4)
        r20 = self.res_block20(r19)

        cs3 = self.cs_block3(r20)

        u5 = self.upconv5(cs3)
        u5 = u5 + r2

        r21 = self.res_block21(u5)
        r22 = self.res_block22(r21)

        cf = self.fconv(r22)
        #prob = self.sig(cf)
        #cf = self.dequant(cf)

        return(cf)