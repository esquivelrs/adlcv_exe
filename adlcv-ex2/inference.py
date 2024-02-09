import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from vit import ViT
from imageclassification import prepare_dataloaders, set_seed
import cv2
import matplotlib.pyplot as plt


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(image_size=(32,32), patch_size=(4,4), channels=3, 
         embed_dim=128, num_heads=4, num_layers=4, num_classes=2,
         pos_enc='learnable', pool='cls', dropout=0.3, fc_dim=None, 
         num_epochs=20, batch_size=16, lr=1e-4, warmup_steps=625,
         weight_decay=1e-3, gradient_clipping=1):
    

    _, _, dataset, _ = prepare_dataloaders(batch_size=batch_size)

    model = ViT(image_size=image_size, patch_size=patch_size, channels=channels, 
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim, 
                num_classes=num_classes
    )

    model.load_state_dict(torch.load('model.pth'))

    #print(model)


    if torch.cuda.is_available():
        model = model
    
    
    # idx = np.random.randint(len(dataset))
    idx = 1
    image, label = dataset[idx]
    print(label)

    print(image.shape)
    image_visual = image.permute(1, 2, 0)  # Transpose to (H, W, C)
    image_visual = image_visual.numpy()  # Convert to numpy array
    
    image = image.unsqueeze(0)
    print(image.shape)

    if torch.cuda.is_available():
        image = image


    with torch.no_grad():
        model.eval()
        out,attention = model(image)
    
        out  = out.argmax(dim=1)
        print(out)
        for att in attention:
            print(att.shape)

        mask = rollout(attention, discard_ratio=0.9, head_fusion="mean")
        print(mask.shape)
        
        # plt.imshow(mask)
        # plt.show()
    # print(attention[0].shape)
        # print(attention[].shape)

    # plt.imshow(mask)
    # plt.show()

    mask = cv2.resize(mask, (image_visual.shape[1], image_visual.shape[0]))
    mask = show_mask_on_image(image_visual, mask)
    
    # place one image at the side of the other
    plt.subplot(1, 2, 1)
    plt.imshow(image_visual)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title('Attention Mask')
    plt.axis('off')
    plt.show()


    # if torch.cuda.is_available():
    #     image, label = image.to('cuda'), label.to('cuda')



def show_mask_on_image(img, mask):
    #img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam =  0.8*heatmap +  np.float32(img)
    #cam = cam / np.max(cam)
    return cam





def rollout(attentions, discard_ratio=0.9, head_fusion="mean"):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=0)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=0)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=0)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask  


set_seed(seed=1)
main()