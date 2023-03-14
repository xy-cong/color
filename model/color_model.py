import torch
import torch.nn as nn
from model.VGG19 import VGG19_pytorch
from model.WarpNet import WarpNet
from utils.util import *
from model.unet_model import UNet
from skimage import color


class Colorization(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.vggnet = VGG19_pytorch()
        self.vggnet.load_state_dict(torch.load("data/vgg19_conv.pth"))
        self.vggnet.eval()
        for param in self.vggnet.parameters():
            param.requires_grad = False
        self.WarpNet = WarpNet(conf.batch_size)
        self.UNet = UNet()
        # import ipdb; ipdb.set_trace()
        if conf.pre_trained_unet:
            unet_pth = torch.load("data/unet.pth")
            unet_pth['outc.conv.weight'] = torch.randn((3,64,1,1))
            unet_pth['outc.conv.bias'] = torch.randn((3))
            self.UNet.load_state_dict(unet_pth)
        
    def warp_color(self, IA_l, IB_lab, temperature=0.01):
        IA_rgb_from_gray = gray2rgb_batch(IA_l)
        with torch.no_grad():
            A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1 = self.vggnet(
                IA_rgb_from_gray, ["r12", "r22", "r32", "r42", "r52"], preprocess=True
            )
            B_relu1_1, B_relu2_1, B_relu3_1, B_relu4_1, B_relu5_1 = self.vggnet(
                IB_lab, ["r12", "r22", "r32", "r42", "r52"], preprocess=True
            )

        # NOTE: output the feature before normalization
        # features_A = [A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1]
        # features_B = [B_relu1_1, B_relu2_1, B_relu3_1, B_relu4_1, B_relu5_1]
        # feature = {
        #     'A': features_A,
        #     'B': features_B
        # }
        
        A_relu2_1 = feature_normalize(A_relu2_1)
        A_relu3_1 = feature_normalize(A_relu3_1)
        A_relu4_1 = feature_normalize(A_relu4_1)
        A_relu5_1 = feature_normalize(A_relu5_1)
        B_relu2_1 = feature_normalize(B_relu2_1)
        B_relu3_1 = feature_normalize(B_relu3_1)
        B_relu4_1 = feature_normalize(B_relu4_1)
        B_relu5_1 = feature_normalize(B_relu5_1)

        nonlocal_BA_lab, similarity_map = self.WarpNet(
            IB_lab,
            A_relu2_1,
            A_relu3_1,
            A_relu4_1,
            A_relu5_1,
            B_relu2_1,
            B_relu3_1,
            B_relu4_1,
            B_relu5_1,
            temperature=temperature,
        )

        # return nonlocal_BA_lab, similarity_map, features_A # get coarse C
        return nonlocal_BA_lab # get coarse C(LAB)
    
    def forward(self, imgs_Input):
        # import ipdb; ipdb.set_trace()
        """
        Img_RGB: GT (0,1) 除了255
        Img_GREY: (0,255)
        Img_FAKE: (0,1) test中一张irrelavant的RGB
        """
        Img_RGB = imgs_Input['img_RGB'].cuda() # torch.Size([1, 256, 256, 3])

        Img_FAKE_RGB = imgs_Input['img_FAKE'].cuda() # torch.Size([1, 256, 256, 3])
        
        Img_LAB = torch.from_numpy(color.rgb2lab(Img_RGB[0].cpu().detach().numpy())).cuda() # torch.Size([256, 256, 3])
        Img_L = Img_LAB[:, :, 0:1].permute(2,0,1) # torch.Size([1, 256, 256])
        Img_AB = Img_LAB[:, :, 1:3].permute(2,0,1) # torch.Size([2, 256, 256])
        
        Img_FAKE_LAB = torch.from_numpy(color.rgb2lab(Img_FAKE_RGB[0].cpu().detach().numpy())).cuda() # torch.Size([256, 256, 3])
        Img_FAKE_L = Img_FAKE_LAB[:, :, 0:1].permute(2,0,1)
        Img_FAKE_AB = Img_FAKE_LAB[:, :, 1:3].permute(2,0,1) # torch.Size([2, 256, 256])
        
        Mask = torch.randn(Img_L.shape).cuda() # torch.Size([1, 256, 256])
        thresh = 0.2
        Mask[Mask >= thresh] = 1.0
        Mask[Mask < thresh] = 0.0
        """
        distortion!
        腐蚀: 
        kernel = np.ones((3, 3), dtype=np.uint8)
        erosion = cv2.erode(img, kernel, iterations=1)
        """
        Img_Distortion_L = Img_L*(1-Mask) + Img_FAKE_L * Mask
        Img_Distortion_AB = Img_AB*(1-Mask) + Img_FAKE_AB*Mask # torch.Size([2, 256, 256])
        Img_Distortion_LAB = torch.cat([Img_Distortion_L, Img_Distortion_AB],0) # torch.Size([3, 256, 256])
        # kernel = np.ones((3, 3), dtype=np.uint8)
        # Img_Distortion_LAB = torch.from_numpy(cv2.erode(Img_Distortion_LAB.cpu().detach().numpy(), kernel, iterations=1)).cuda()
        pred_lab = np.concatenate((Img_L.cpu().detach().numpy(), Img_Distortion_AB.cpu().detach().numpy()), axis=0).transpose((1, 2, 0))
        Img_Ref_RGB = color.lab2rgb(pred_lab)
        # mg_Ref_RGB = lab2rgb_transpose(Img_L.cpu().detach().numpy(), Img_Distortion_AB.cpu().detach().numpy())
        # import ipdb; ipdb.set_trace()
        
        Img_Coarse_LAB = self.warp_color(Img_L.unsqueeze(0), Img_Distortion_LAB.unsqueeze(0))
        # Img_Coarse_LAB = Img_Distortion_LAB.unsqueeze(0)  # 简单版
        Img_Coarse_L = Img_Coarse_LAB[:, 0:1, :, :]
        Img_Coarse_AB = Img_Coarse_LAB[:, 1:3, :, :]
        Img_Coarse_L = Img_Coarse_L[0]
        Img_Coarse_AB = Img_Coarse_AB[0]
        Img_Coarse_L = torch.clamp(Img_Coarse_L, 0.0, 100.0) # stable? for lab2rgb_transpose
        Img_Coarse_AB = torch.clamp(Img_Coarse_AB, -100.0, 100.0)
        
        # Img_Coarse_RGB = lab2rgb_transpose(Img_Coarse_L.cpu().detach().numpy(), Img_Coarse_AB.cpu().detach().numpy()) # (x, x, 3)
        Img_Coarse_RGB = np.concatenate((Img_L.cpu().detach().numpy(), Img_Coarse_AB.cpu().detach().numpy()), axis=0).transpose((1, 2, 0))
        Img_Coarse_RGB = np.clip(color.lab2rgb(Img_Coarse_RGB), 0, 1)
        Img_Coarse_RGB = torch.from_numpy(Img_Coarse_RGB).float().cuda()
        Img_Fine_RGB = self.UNet(Img_Coarse_RGB.unsqueeze(0).permute(0,3,1,2)).permute(0,2,3,1) # fine: (0,255) (batch_size(1), img_size, 3)
        # Img_Fine_RGB = Img_Fine_RGB / 255.0
        model_output = {
            'Img_Fine_RGB': Img_Fine_RGB, # torch.Size([1, 256, 256, 3])  (0,1)
            'Img_RGB': Img_RGB, # torch.Size([1, 256, 256, 3]) (0,1)
            'Img_FAKE_RGB': Img_FAKE_RGB, # torch.Size([1, 256, 256, 3]) (0,1)
            'Img_L': Img_L, # (0,100)
            'Img_Ref': Img_Ref_RGB, # (0,1)
            'Img_Coarse_RGB': Img_Coarse_RGB, # (0,1)
            'VGG19': self.vggnet
        }
        # import ipdb; ipdb.set_trace()
        return model_output
        
        