import torch.nn as nn
import torch


class CTE_3D(nn.Module):
    def __init__(self, time_length):
        super(CTE_3D, self).__init__()
        self.norm0 = nn.LayerNorm(time_length)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * time_length//2 * 1 * 1, 64)

    def forward(self, x):

        x = self.norm0(x)
        x = x.unsqueeze(1).permute(0, 1, 4, 2, 3)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        # There are out_channels filters; each filter corresponds to in_channels weight matrices of size kernel_size
        # Each filter has in_channels 3x3 (or other size) kernels; each filter generates one feature matrix
        # padding default value is 0
        # stride default value is 1
        nn.Conv2d(in_channels, out_channels, 3, padding=1),  # Since the kernel size is 3 and padding=1, the convolution operation will not change the height and width of the input image.
        # Does this need a BN (Batch Normalization) here?
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class HSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x1 = self.maxpool(conv1)
        conv2 = self.dconv_down2(x1)
        x2 = self.maxpool(conv2)
        conv3 = self.dconv_down3(x2)
        x3 = self.maxpool(conv3)
        conv4 = self.dconv_down4(x3)
        return {'conv1': conv1, 'conv2': conv2, 'conv3': conv3, 'conv4': conv4}

class SiameseNet(nn.Module):
    def __init__(self, encoder, tnet, patch_size, hidden_dim=512, img_dims=None, img_features_sizes=None):
        super().__init__()

        if img_dims is None:
            img_dims = [64, 128, 256, 512]
        if img_features_sizes is None:
            img_features_sizes = [patch_size, patch_size // 2, patch_size // 4, patch_size // 8]
        
        self.encoder = encoder
        self.tnet = tnet

        self.fc = nn.Linear(64 * 2, 64)  # Add fully connected layer
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(64)

        self.conv1 = nn.Conv2d(img_dims[0]*2, img_dims[0], kernel_size=1)
        self.conv2 = nn.Conv2d(img_dims[1]*2, img_dims[1], kernel_size=1)
        self.conv3 = nn.Conv2d(img_dims[2]*2, img_dims[2], kernel_size=1)
        self.conv4 = nn.Conv2d(img_dims[3]*2, img_dims[3], kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(img_dims[2] + img_dims[3], img_dims[2])
        self.dconv_up2 = double_conv(img_dims[1] + img_dims[2], img_dims[1])
        self.dconv_up1 = double_conv(img_dims[1] + img_dims[0], img_dims[0])


        self.pool = nn.AdaptiveAvgPool2d(1)  # Global pooling

        self.relu_patch = nn.ReLU()
        self.norm_patch = nn.LayerNorm(64)

        # Final fusion fully connected layer
        self.fc_last = nn.Sequential(
            nn.Linear(64*2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 2),
            
        )

    def forward(self, batches):
        data_imgs = batches['imgs']
        data_seqs = batches['seqs']
        batch_size = data_imgs.shape[0]

        data_left_img_batch = data_imgs[:, 0].unsqueeze(1).float().cuda()
        data_right_img_batch = data_imgs[:, 1].unsqueeze(1).float().cuda()
        data_left_seq_batch = data_seqs[:, 0].float().cuda()
        data_right_seq_batch = data_seqs[:, 1].float().cuda()

        left_ts_features = self.tnet(data_left_seq_batch)
        right_ts_features = self.tnet(data_right_seq_batch)
        left_img_features = self.encoder(data_left_img_batch)
        right_img_features = self.encoder(data_right_img_batch)


        combine_data = torch.cat([left_ts_features, right_ts_features], dim=1)
        combine_data = self.fc(combine_data)
        combine_data = self.relu(combine_data)
        combine_data_ts = self.norm1(combine_data)

        combine_data_img1=torch.cat([left_img_features['conv1'], right_img_features['conv1']], dim=1)
        combine_data_img1=self.conv1(combine_data_img1)

        combine_data_img2=torch.cat([left_img_features['conv2'], right_img_features['conv2']], dim=1)
        combine_data_img2=self.conv2(combine_data_img2)

        combine_data_img3=torch.cat([left_img_features['conv3'], right_img_features['conv3']], dim=1)
        combine_data_img3=self.conv3(combine_data_img3)

        combine_data_img4=torch.cat([left_img_features['conv4'], right_img_features['conv4']], dim=1)
        combine_data_img4=self.conv4(combine_data_img4)


        x = self.upsample(combine_data_img4)
        x = torch.cat([x, combine_data_img3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, combine_data_img2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, combine_data_img1], dim=1)
        x = self.dconv_up1(x)
        
        pooled = self.pool(x).view(batch_size, 64)  # [512, 64]
        patch_data = self.relu_patch(pooled)
        patch_data = self.norm_patch(patch_data)

        out=self.fc_last(torch.cat([patch_data, combine_data_ts], dim=1))

        return out,(left_ts_features,right_ts_features,left_img_features,right_img_features)