# sliced wasserstein texture loss

#######################################################################
# Implementation of 
# A Sliced Wasserstein Loss for Neural Texture Synthesis
# Heitz et al., CVPR 2021
#######################################################################

import numpy as np
import torch

#######################################################################
# Load pretrained VGG19
#######################################################################

SCALING_FACTOR = 1

class VGG19(torch.nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()

        self.block1_conv1 = torch.nn.Conv2d(3, 64, (3,3), padding=(1,1), padding_mode='reflect')
        self.block1_conv2 = torch.nn.Conv2d(64, 64, (3,3), padding=(1,1), padding_mode='reflect')

        self.block2_conv1 = torch.nn.Conv2d(64, 128, (3,3), padding=(1,1), padding_mode='reflect')
        self.block2_conv2 = torch.nn.Conv2d(128, 128, (3,3), padding=(1,1), padding_mode='reflect')

        self.block3_conv1 = torch.nn.Conv2d(128, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv2 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv3 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv4 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')

        self.block4_conv1 = torch.nn.Conv2d(256, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv2 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv3 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv4 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsampling = torch.nn.AvgPool2d((2,2))

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

        return [block1_conv1, block1_conv2, block2_conv1, block2_conv2, block3_conv1, block3_conv2, block3_conv3, block3_conv4, block4_conv1, block4_conv2, block4_conv3, block4_conv4]

    def set_gt_image(self, image_gt):
        self.list_activations_gt = self.forward(image_gt)
        return self.list_activations_gt

    def slicing_loss(self, image_generated, image_example, l1=False):

        # generate VGG19 activations
        list_activations_generated = self.forward(image_generated)#vgg(image_generated)
        list_activations_example   = self.forward(image_example)#vgg(image_example)
        
        # iterate over layers
        loss = 0
        for l in range(len(list_activations_example)):
            # get dimensions
            b = list_activations_example[l].shape[0]
            dim = list_activations_example[l].shape[1]
            n = list_activations_example[l].shape[2]*list_activations_example[l].shape[3]
            # linearize layer activations and duplicate example activations according to scaling factor
            activations_example = list_activations_example[l].view(b, dim, n).repeat(1, 1, SCALING_FACTOR*SCALING_FACTOR)
            activations_generated = list_activations_generated[l].view(b, dim, n*SCALING_FACTOR*SCALING_FACTOR)
            # sample random directions
            Ndirection = dim
            directions = torch.randn(Ndirection, dim).to(torch.device("cuda:0")) # [Ndir, dim]
            directions = directions / torch.sqrt(torch.sum(directions**2, dim=1, keepdim=True))
            # project activations over random directions
            projected_activations_example = torch.einsum('bdn,md->bmn', activations_example, directions)
            projected_activations_generated = torch.einsum('bdn,md->bmn', activations_generated, directions)
            # sort the projections
            sorted_activations_example = torch.sort(projected_activations_example, dim=2)[0]
            sorted_activations_generated = torch.sort(projected_activations_generated, dim=2)[0]
            # L2 over sorted lists
            if l1:
                loss += torch.mean( torch.abs(sorted_activations_example-sorted_activations_generated) ) 
            else:
                loss += torch.mean( (sorted_activations_example-sorted_activations_generated)**2 ) 
        return loss

    def ebsw_loss(self, image_generated, image_example, l1=False, mask=None, sample_pixels=False): # mask [b, 1, H, W]

        # 0,1 / 2,3 / 4,5,6,7 / 8,9,10,11
        if mask is not None:
            mask_tensor = mask.float()#torch.tensor(mask*1.).float().to(torch.device("cuda:0")) # reshape to [b, 1, H , W]
            if mask_tensor.shape == 3:
                mask_tensor = mask_tensor.unsqueeze(dim=0)
            #torch.nn.functional.interpolate(mask_tensor, scale_factor=(0.5, 0.5), mode='nearest')
            # 1,3 H,W
            #image_generated = image_generated*mask_tensor
            #image_example = image_example*mask_tensor
        
        # generate VGG19 activations
        list_activations_generated = self.forward(image_generated)#vgg(image_generated)
        list_activations_example   = self.forward(image_example)#vgg(image_example)
        
        # iterate over layers
        loss = 0
        
        for l in range(len(list_activations_example)):
            # get dimensions
            b = list_activations_example[l].shape[0]
            dim = list_activations_example[l].shape[1]
            n = list_activations_example[l].shape[2]*list_activations_example[l].shape[3]
            n_target = list_activations_generated[l].shape[2]*list_activations_generated[l].shape[3]            

            if mask is not None:
                if l in [2, 4, 8]: # downscale mask
                    mask_tensor = torch.nn.functional.interpolate(mask_tensor, scale_factor=(0.5, 0.5), mode='nearest') # 'nearest-exact'
                mask_ = mask_tensor.view(b, 1, -1) # [b, 1, pixels]
                #activations_exmaple

            # linearize layer activations and duplicate example activations according to scaling factor
            activations_example = list_activations_example[l].view(b, dim, n).repeat(1, 1, SCALING_FACTOR*SCALING_FACTOR) #
            activations_generated = list_activations_generated[l].view(b, dim, n_target*SCALING_FACTOR*SCALING_FACTOR)

            n_sample = min(n, n_target)
            if n != n_target:
                perm1 = torch.randperm(n_sample)
                if n > n_target:
                    activations_example = activations_example[:,:,perm1[:n_sample]]
                    #mask_ = mask_[:,:,perm1[:n_sample]]
                else:
                    activations_generated = activations_generated[:,:,perm1[:n_sample]]
                    #mask_ = mask_[:,:,perm1[:n_sample]]
                if mask is not None:
                    mask_ = mask_[:,:,perm1[:n_sample]]
                    
            # If sample pixels
            if sample_pixels:
                r = 1
                n = activations_example.shape[2]
                perm = torch.randperm(n)
                activations_example = activations_example[:,:,perm[:(n_sample//r)]]
                activations_generated = activations_generated[:,:,perm[:(n_sample//r)]]
                if mask_ is not None:
                    mask_ = mask_[:,:,perm[:n//r]]
            
            # sample random directions
            Ndirection = dim
            directions = torch.randn(Ndirection, dim).to(torch.device("cuda:0")) # [Ndir, dim]
            directions = directions / torch.sqrt(torch.sum(directions**2, dim=1, keepdim=True))
            
            if mask is not None: # and l < 2:
                # activation: [b, dim, pixels], directions: [ndir, dim+1]
                max_val = torch.max( torch.cat( (activations_example, activations_generated), dim=0 ) ).item()
                directions = torch.cat((directions, torch.ones(Ndirection, 1).to(torch.device("cuda:0"))), dim=1) # [ndir, dim+1]
                #mask_ = mask_tensor.view(b, 1, -1)*max_val # [b, 1, pixels]
                mask_ = mask_*max_val
                activations_example = torch.cat((activations_example, mask_), dim=1)
                activations_generated = torch.cat((activations_generated, mask_), dim=1)
            
            # project activations over random directions
            projected_activations_example = torch.einsum('bdn,md->bmn', activations_example, directions)
            projected_activations_generated = torch.einsum('bdn,md->bmn', activations_generated, directions) # [b, num_of_dirs, num_of_pixels]
            
            # sort the projections
            sorted_activations_example = torch.sort(projected_activations_example, dim=2)[0]
            sorted_activations_generated = torch.sort(projected_activations_generated, dim=2)[0]

            SW = torch.mean((sorted_activations_example-sorted_activations_generated)**2, dim=2) # L2 distance [b, num_of_dirs]
            
            # L2 over sorted lists
            if l1:
                loss += torch.mean( torch.abs(sorted_activations_example-sorted_activations_generated) ) 
            elif False:
                weights = torch.nn.functional.softmax(SW, dim=1) # [b, num_of_dirs]
                loss += torch.sum(weights*SW, dim=1).mean()
            else:
                loss += SW.mean() #torch.mean( (sorted_activations_example-sorted_activations_generated)**2 ) 
                
        return loss #, loss_content

    def content_loss(self, image_generated, image_content, l1=False, mask=None): # mask [H, W]
            # https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
            # 0,1 / 2,3 / 4,5,6,7 / 8,9,10,11
            if mask is not None:
                mask_tensor = torch.tensor(mask*1.).unsqueeze(dim=0).unsqueeze(dim=0).float().to(torch.device("cuda:0")) # reshape to [1, 1, H , W]
                #torch.nn.functional.interpolate(mask_tensor, scale_factor=(0.5, 0.5), mode='nearest')
                # 1,3 H,W
                image_generated = image_generated*mask_tensor
                image_example = image_example*mask_tensor
            
            # generate VGG19 activations
            list_activations_generated = self.forward(image_generated)#vgg(image_generated)
            list_activations_content   = self.forward(image_content)#vgg(image_example)
            
            # iterate over layers
            loss = 0
            
            #for l in range(len(list_activations_content)):
            for l in range(8,12):
                loss += torch.nn.functional.mse_loss(list_activations_content[l], list_activations_generated[l])
                
            return loss #, loss_content

    def gram_loss(self, image_generated, image_example, l1=False): # [b, c, h, w]

        # generate VGG19 activations
        list_activations_generated = self.forward(image_generated)#vgg(image_generated)
        list_activations_example   = self.forward(image_example)#vgg(image_example)

        
        # iterate over layers
        loss = 0
        for l in range(len(list_activations_example)):
        #for l in [0,1,2,3,4]:
            # get dimensions
            b = list_activations_example[l].shape[0]
            dim = list_activations_example[l].shape[1]
            n = list_activations_example[l].shape[2]*list_activations_example[l].shape[3]

            feature_generated = list_activations_generated[l].view(b*dim, n)
            feature_example = list_activations_example[l].view(b*dim, n)

            G_gen = torch.mm(feature_generated, feature_generated.t()).div(b*dim*n).clamp(-1,1)
            G_example = torch.mm(feature_example, feature_example.t()).div(b*dim*n).clamp(-1,1)
            #print('DEBUG gram', G_gen.mean().item(), G_example.mean().item())
            dG = G_gen - G_example
            #if dG.isnan().any():
            if torch.any(G_example.isinf()):
                print('nan g_example', G_example.max())
                print('nan g_gen', G_gen.max())
                print('nan dG', dG.max())
            #dG = torch.nan_to_num(dG)
            loss += torch.mean( dG**2 )
            #loss += torch.mean( torch.abs(dG) )

        return loss