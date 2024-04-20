import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
		super(ConvBNReLU, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
		self.bn = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x
	

class Net(nn.Module):
	def __init__(self, vgg16):
		super(Net, self).__init__()
		self.vgg16 = vgg16
		self.vgg16 = list(self.vgg16.children())[0]
		self.layers = list(self.vgg16.children())
		self.conv0 = self.layers[0]
		self.conv1 = self.layers[2]
		self.conv2 = self.layers[5]
		self.conv3 = self.layers[7]
		self.conv4 = self.layers[10]
		self.conv5 = self.layers[12]
		self.conv6 = self.layers[14]
		self.conv7 = self.layers[17]
		self.conv8 = self.layers[19]
		self.conv9 = self.layers[21]
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
		self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
		self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
		self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear')
		self.conv10 = ConvBNReLU(512, 256, kernel_size=1, stride=1, padding=0)
		self.conv11 = ConvBNReLU(256, 128, kernel_size=3, stride=1, padding=1)
		self.conv12 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
		self.conv13 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
		self.conv14 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
		self.conv15 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
		self.conv16 = nn.Conv2d(64, 50, kernel_size=1, stride=1, padding=0)
		self.conv17 = nn.Conv2d(64, 50, kernel_size=1, stride=1, padding=0)

		self.bn0 = nn.BatchNorm2d(64)
		self.bn1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(128)
		self.bn3 = nn.BatchNorm2d(128)
		self.bn4 = nn.BatchNorm2d(256)
		self.bn5 = nn.BatchNorm2d(256)
		self.bn6 = nn.BatchNorm2d(256)
		self.bn7 = nn.BatchNorm2d(512)
		self.bn8 = nn.BatchNorm2d(512)
		self.bn9 = nn.BatchNorm2d(512)
		self.bn10 = nn.BatchNorm2d(256)
		self.bn11 = nn.BatchNorm2d(128)
		self.bn12 = nn.BatchNorm2d(64)

	def forward(self, x):
		skip0 = F.relu(self.bn1(self.conv1(F.relu(self.bn0(self.conv0(x))))))
		x = self.maxpool(skip0)
		skip1 = F.relu(self.bn3(self.conv3(F.relu(self.bn2(self.conv2(x))))))
		x = self.maxpool(skip1)
		skip2 = F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(F.relu(self.bn4(self.conv4(x)))))))))
		x = self.maxpool(skip2)
		skip3 = F.relu(self.bn9(self.conv9(F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(x)))))))))
		out0 = F.relu(self.bn10(self.conv10(skip3)))
		x = self.upsample2(out0)
		out1 = F.relu(self.bn11(self.conv11(x + skip2)))
		x = self.upsample2(out1)
		out2 = F.relu(self.bn12(self.conv12(x + skip1)))
		out0 = self.upsample8(out0)
		out1 = self.upsample4(out1)
		out2 = self.upsample2(out2)
		if len(out0.shape) == 4:
			out0 = torch.cat((out0, out1, out2, skip0), 1)
		elif len(out0.shape) == 3:
			out0 = torch.cat((out0, out1, out2, skip0), 0)
		else: 
			raise NotImplementedError
		
		out0 = F.relu(self.conv15(F.relu(self.conv14(F.relu(self.conv13(out0))))))

		out1 = self.conv16(out0)
		out1 = F.softmax(out1, dim=1)
		out0 = self.conv17(out0)
		out0 = F.softmax(out0, dim=1)

		return out0, out1

if __name__ == "__main__":
	vgg_model = torchvision.models.vgg16(weights='DEFAULT')
	net = Net(vgg_model)

	img = torch.randn(10, 3, 224, 224)
	print(net(img)[0].shape)