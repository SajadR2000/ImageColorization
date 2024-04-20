import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


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

		for i, child in enumerate(self.layers):
			print(i, child)

	def forward(self, x):
		skip0 = F.relu(self.conv1(F.relu(self.conv0(x))))
		x = self.maxpool(skip0)
		skip1 = F.relu(self.conv3(F.relu(self.conv2(x))))
		x = self.maxpool(skip1)
		skip2 = F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(x))))))
		x = self.maxpool(skip2)
		skip3 = F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(x))))))
		return skip3

if __name__ == "__main__":
	vgg_model = torchvision.models.vgg16(weights='DEFAULT')
	net = Net(vgg_model)

	img = torch.randn(3, 3, 224, 224)
	print(net(img).shape)