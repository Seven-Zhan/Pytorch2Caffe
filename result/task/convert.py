import sys
sys.path.append('D:\\DevCodes\\pytorch_caffe')
import torch
import pytorch_to_caffe

from model import Net



if __name__=='__main__':

	name = 'model'
	ckpt = 'D:\\DevSnaps\\vehicle_face\\2021-01-18-17-18-52\\epoch=25.pth'
	model = Net()
	model.load_state_dict(torch.load(ckpt, map_location='cpu'))
	model.eval()

	dummy_inputs = torch.randn(1, 3, 128, 256)

	pytorch_to_caffe.trans_net(model, dummy_inputs, name)
	pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
	pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
