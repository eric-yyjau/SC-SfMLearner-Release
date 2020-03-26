import torch
import torch.nn as nn
from torch import sigmoid
from torch.nn.init import xavier_uniform_, zeros_

# from .Convolutional_LSTM_PyTorch.convolution_lstm import ConvLSTMCell # from https://github.com/automan000/Convolutional_LSTM_PyTorch
from .Convolutional_LSTM_PyTorch.convolution_lstm import ConvLSTMCell # from https://github.com/automan000/Convolutional_LSTM_PyTorch
import math

def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )

def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )

class PoseLstmNet(nn.Module):

    def __init__(self, channel=3):
        super(PoseLstmNet, self).__init__()

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.hidden_channels = []
        self.lstm_width_ratio = []
        self.conv1 = conv(channel, conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        # self.convLstm2 = ConvLSTMCell(conv_planes[0], conv_planes[1], kernel_size=5)
        # self.hidden_channels.append(conv_planes[1])
        # self.lstm_width_ratio.append(2)

        self.conv3 = conv(conv_planes[1], conv_planes[2]) # stride = 2
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        # self.convLstm4 = ConvLSTMCell(conv_planes[2], conv_planes[3], kernel_size=3)
        # self.hidden_channels.append(conv_planes[3])
        # self.lstm_width_ratio.append(4)

        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        # self.convLstm6 = ConvLSTMCell(conv_planes[4], conv_planes[5], kernel_size=3)
        # self.hidden_channels.append(conv_planes[5])
        # self.lstm_width_ratio.append(8)
        self.conv7 = conv(conv_planes[5], conv_planes[6])
        self.convLstm7 = ConvLSTMCell(conv_planes[6], conv_planes[6], kernel_size=1)
        self.hidden_channels.append(conv_planes[6])
        self.lstm_width_ratio.append(128)

        self.pose_pred = nn.Conv2d(conv_planes[6], 6, kernel_size=1, padding=0)
        # lstm layers
        # self.lstmLayers = [self.convLstm2, self.convLstm4, self.convLstm6]
        self.lstmLayers = [self.convLstm7]
        # self.convLayers = [self.conv1, self.conv3, self.conv5, self.conv7]
        self.lstm_internal_states = []

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def init_lstm_states(self, x):
        self.lstm_internal_states.clear()
        bsize, _, height, width = x.size()
        for i, layer in enumerate(self.lstmLayers):
            ratio = self.lstm_width_ratio[i]
            (h, c) = layer.init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                            shape=(math.ceil(height/ratio), math.ceil(width/ratio)) )
            # print(f"h: {h.shape}")
            self.lstm_internal_states.append((h, c))


    def forward(self, target_image, ref_img, init_states=False):
        if len(self.lstm_internal_states) == 0 or init_states:
            # self.init_lstm_states(target_image)
            self.lstm_internal_states.clear()
            bsize, _, height, width = target_image.size()
            for i, layer in enumerate(self.lstmLayers):
                ratio = self.lstm_width_ratio[i]
                (h, c) = layer.init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                shape=(math.ceil(height/ratio), math.ceil(width/ratio)) )
                # print(f"h: {h.shape}")
                self.lstm_internal_states.append((h, c))            
            # print(f"lstm states initializing...")
            # raise
        # input = [target_image, ref_img]
        # input = torch.cat(input, 1)
        # input = target_image
        input = [target_image, ref_img]
        input = torch.cat(input, 1)

        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        # lstm layer forward
        i = 0
        out_conv = out_conv7
        (h, c) = self.lstm_internal_states[i]
        out_conv, new_c = self.lstmLayers[i](out_conv, h, c)
        self.lstm_internal_states[i] = (out_conv, new_c)
        out_conv7 = out_conv
        
        # pose layer
        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), 6)
 
        return pose

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PoseLstmNet(channel=6)
    model = model.to(device)


    # check keras-like model summary using torchsummary
    from torchsummary import summary
    # width, height = 320, 240 # (256, 832, 3)
    height, width = 256, 832 # (256, 832, 3)
    # model.init_lstm_states(torch.zeros((1,3, height, width)) )
    # summary(model, input_size=[(3, 240, 320), (3, 240, 320)])
    summary(model, input_size=[(3, height, width), (3, height, width)] )

    ## test
    image = torch.zeros((1,3, height, width))
    model = torch.nn.DataParallel(model)
    outs = model(image.to(device), image.to(device))
    print("outs: ", list(outs))


if __name__ == "__main__":
    main()
    pass