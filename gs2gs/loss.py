# Adapted from style-nerf2nerf (MIT License)
# https://github.com/haruolabs/style-nerf2nerf


import torch
import torch.nn as nn

class VGG19(torch.nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()

        self.block1_conv1 = nn.Conv2d(3, 64, (3,3), padding=(1,1), padding_mode='reflect')
        self.block1_conv2 = nn.Conv2d(64, 64, (3,3), padding=(1,1), padding_mode='reflect')

        self.block2_conv1 = nn.Conv2d(64, 128, (3,3), padding=(1,1), padding_mode='reflect')
        self.block2_conv2 = nn.Conv2d(128, 128, (3,3), padding=(1,1), padding_mode='reflect')

        self.block3_conv1 = nn.Conv2d(128, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv2 = nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv3 = nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv4 = nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')

        self.block4_conv1 = nn.Conv2d(256, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv2 = nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv3 = nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv4 = nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')

        self.relu = nn.ReLU(inplace=True)
        self.downsampling = nn.AvgPool2d((2,2))

    def forward(self, image):
        
        # RGB to BGR
        image = image[:, [2,1,0], :, :]

        # [0, 1] --> [0, 255]
        image = 255 * image

        # remove average color
        image[:,0,:,:] -= 103.939
        image[:,1,:,:] -= 116.779
        image[:,2,:,:] -= 123.68

        # block1
        block1_conv1 = self.relu(self.block1_conv1(image))
        block1_conv2 = self.relu(self.block1_conv2(block1_conv1))
        block1_pool = self.downsampling(block1_conv2)

        # block2
        block2_conv1 = self.relu(self.block2_conv1(block1_pool))
        block2_conv2 = self.relu(self.block2_conv2(block2_conv1))
        block2_pool = self.downsampling(block2_conv2)

        # block3
        block3_conv1 = self.relu(self.block3_conv1(block2_pool))
        block3_conv2 = self.relu(self.block3_conv2(block3_conv1))
        block3_conv3 = self.relu(self.block3_conv3(block3_conv2))
        block3_conv4 = self.relu(self.block3_conv4(block3_conv3))
        block3_pool = self.downsampling(block3_conv4)

        # block4
        block4_conv1 = self.relu(self.block4_conv1(block3_pool))
        block4_conv2 = self.relu(self.block4_conv2(block4_conv1))
        block4_conv3 = self.relu(self.block4_conv3(block4_conv2))
        block4_conv4 = self.relu(self.block4_conv4(block4_conv3))

        layer_list = [block1_conv1, block1_conv2, block2_conv1, block2_conv2, block3_conv1, block3_conv2, block3_conv3, block3_conv4, block4_conv1, block4_conv2, block4_conv3, block4_conv4]

        return [torch.var_mean(t, dim=(2,3), unbiased=False) for t in layer_list],  layer_list
    
    def vgg_loss(self, gt_images, pred_images, alpha = 1.0):
        # pred)images : predicted images by the model 
        # gt_images : ground truth images
        # Both tensor should be the same shape in the form of [B, C, H, W]
        # Pixel values should be normalized to [0,1]

        gt_stats, gt_features = self.forward(gt_images)
        pred_stats, pred_features = self.forward(pred_images)

        # Matching Feature
        content_loss = 0.0
        for gt_feature, pred_feature in zip(gt_features, pred_features):
            content_loss = torch.sum((gt_feature - pred_feature) ** 2)

        # Matching Feature Statistics
        style_loss = 0.0
        for gt_stat, pred_stat in zip(gt_stats, pred_stats):
            style_loss += torch.sum((gt_stat[0] - pred_stat[0]) ** 2)
            style_loss += torch.sum((gt_stat[1] - pred_stat[1]) ** 2)


        return content_loss + alpha * style_loss
