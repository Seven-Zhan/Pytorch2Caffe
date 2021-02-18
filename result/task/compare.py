import os
os.environ['GLOG_minloglevel'] = '3'
import numpy as np

import torch
import caffe
caffe.set_mode_cpu()

from model import Net



SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

if __name__ == '__main__':

    ckpt = 'pytorch model file .pth'
    torch_model = Net()
    torch_model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    torch_model.eval()

    dummy_inputs = torch.randn(1, 3, 128, 256) # input shape
    torch_ouputs = torch_model(dummy_inputs).data.numpy()
    print(torch_ouputs.shape)

    numpy_inputs = dummy_inputs.numpy()
    prototxt_path = 'caffe model prototxt'
    caffemodel_path = 'caffe model caffemodel'
    caffe_model = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

    input_name = 'input blob name'
    ouput_name = 'output blob name'

    caffe_model.blobs[input_name].data[...] = numpy_inputs
    caffe_ouputs = caffe_model.forward()[ouput_name]
    print(caffe_ouputs.shape)

    diff = np.abs(torch_ouputs - caffe_ouputs).max()

    print('===== torch =====', torch_ouputs[0][:5])
    print('===== caffe =====', caffe_ouputs[0][:5])
    print('max diff between torch and caffe model outputs: {}'.format(diff))
