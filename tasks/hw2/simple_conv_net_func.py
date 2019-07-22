from __future__ import print_function
import torch


def diff_mse(x, y):

    x_vec = x.view(1, -1).squeeze()
    y_vec = y.view(1, -1).squeeze()
    
    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()
  

def conv2d_scalar(x_in, conv_weight, conv_bias, device):
    
    #read dimentionas of input tensor and weights
    batch_size, n_channels_in, height_in, width_in = x_in.shape
    n_channels_out, n_channels_in, kernel_size, kernel_size = conv_weight.shape
    
    #calculate the dimentions of output tensor
    height_out = height_in - kernel_size + 1
    width_out = width_in - kernel_size + 1
    
    #move to device
    x_in = x_in.to(device)
    conv_weight = conv_weight.to(device)
    conv_bias = conv_bias.to(device)
    
    #intiale output tensor of the correct size
    z = torch.zeros([batch_size,n_channels_out,height_out,width_out]).to(device)
    
    #fulfill z based on scalar representation
    for n in range(batch_size):
        for c_out in range(n_channels_out):
            for c_in in range(n_channels_in):
                for m in range(height_out):
                    for l in range(width_out):
                        z[n,c_out,m,l] = (x_in[n,c_in,m:m+kernel_size,l:l+kernel_size]*conv_weight[c_out,c_in]).sum() + conv_bias[c_out]
                                                                                                                                                                                                                                                                                                                                                          
    return z

  
def im2col(X, kernel_size, device, stride = 1):
  
    #read dimentions of input tensor - 3-dimentional
    C_in, S_in, S_in = X.shape

    #calculate size_out
    S_out = (S_in - kernel_size)//stride + 1
    
    #move to device
    X = X.to(device)
    
    #intiale output tensor of the correct size
    X_cols = torch.zeros([S_out*S_out, kernel_size*kernel_size]).to(device)
    
    for i in range(S_out):
        for j in range(S_out):
            X_cols[i*S_out+j] = X[0][i: i + kernel_size, j: j + kernel_size].contiguous().view(1, -1)
    
    return X_cols.t() # [K*K x S_out*S_out]

  
def conv_weight2rows(conv_weight):
    
    ##read dimentions of input tensor
    C_out = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    
    #resize 
    conv_weight_rows = conv_weight.view(C_out,kernel_size*kernel_size).contiguous()
    
    return conv_weight_rows # [C_out x K*K]
  

def conv2d_vector(x_in, conv_weight, conv_bias, device):

    #read dimentionas of input tensor and weights
    batch_size, C_in, S_in, S_in = x_in.shape
    C_out, C_in, kernel_size, kernel_size = conv_weight.shape
    
    #calculate the dimentions of output tensor
    S_out = S_in - kernel_size + 1
    
    #move to device
    x_in = x_in.to(device)
    conv_weight = conv_weight.to(device)
    conv_bias = conv_bias.to(device)
    
    #intiale output tensor of the correct size
    z = torch.zeros([batch_size,C_out,S_out,S_out]).to(device)
    
    #transformation of conv_weight
    conv_weight_rows = conv_weight2rows(conv_weight)
    
    for n in range(batch_size):
        #WconvX+b, dim(WconvX+b)=[C_out x S_out*S_out], reshape [C_out x S_out x S_out]
        z[n] = (conv_weight_rows.matmul(im2col(x_in[n], kernel_size, device, stride=1)) + conv_bias.view(-1,1)).view(C_out,S_out,S_out)
    
    return z

  
def pool2d_scalar(a, device, stride = 2):
    
    #read dimentionas of input tensor
    batch_size, n_channels_in, height_in, width_in = a.shape
    pooling_size = 2
    
    #calculate the dimentions of output tensor
    height_out = (height_in-pooling_size)//stride + 1
    width_out = (width_in-pooling_size)//stride + 1
    n_channels_out = n_channels_in
    
    #move to device
    a = a.to(device)
    
    #intiale an output tensor of the correct size
    z = torch.zeros([batch_size,n_channels_out,height_out,width_out]).to(device)
    
    #fulfill z based on scalar representation
    for n in range(batch_size):
        for c_out in range(n_channels_out):
            for i in range(height_out):
                for j in range(width_out):
                    z[n,c_out,i,j] = a[n,c_out,2*i:2*i+2,2*j:2*j+2].max()
    
    return z
 

def pool2d_vector(a, device, stride = 2):
    
    #read dimentionas of input tensor
    batch_size, C_in, S_in, S_in = a.shape
    pooling_size = 2
    stride = 2
    
    #calculate the dimentions of output tensor
    S_out = (S_in - pooling_size)//stride + 1 
    C_out = C_in
    
    #move to device
    a = a.to(device)
    
    #intiale an output tensor of the correct size
    z = torch.zeros([batch_size,C_out,S_out,S_out]).to(device)
    
    for n in range(batch_size):
        z[n] = im2col(a[n], pooling_size, device, stride=2).max(dim=0).values.view(-1, S_out, S_out)
    return z 
  

def relu_scalar(a, device):
  
    #read dimentionas of input matrix
    batch_size, n_inputs = a.shape
    
    #move to device
    a = a.to(device)
    
    #intiale an output matrix of the correct size
    z = torch.zeros([batch_size, n_inputs]).to(device)
    
    for n in range(batch_size):
        for i in range(n_inputs):
            if a[n,i]<0:
                z[n,i]=0
            else:
                z[n,i]=a[n,i]
                
    return z
  

def relu_vector(a, device):
    
    #move to device
    a = a.to(device)
    
    #clone input tensor
    z = a.clone().to(device)
    
    #elements < 0 replace with 0
    z[z<0] = 0
    
    return z

  
def reshape_scalar(a, device):
    
    #read dimentionas of input tensor
    batch_size, n_channels_in, height_in, width_in = a.shape
    
    #calculate the dimentions of output tensor
    n_outputs = n_channels_in * height_in * width_in
    
    #move to device
    a = a.to(device)
    
    #intiale an output matrix of the correct size
    z = torch.zeros([batch_size, n_outputs]).to(device)
    
    for n in range(batch_size):
        for c_in in range(n_channels_in):
            for m in range(height_in):
                for l in range(width_in):
                    z[n,c_in*height_in*width_in+m*height_in+l] = a[n,c_in,m,l]
    
    return z

  
def reshape_vector(a, device):
    
    batch_size = a.shape[0]
    
    #move to device
    a = a.to(device)
    
    z = a.clone().view(batch_size,-1)
    
    return z
 

def fc_layer_scalar(a, weight, bias, device):
    
    #read dimentionas of input matrix
    batch_size, n_inputs = a.shape
    n_outputs = bias.shape[0]
    
    #move to device
    a = a.to(device)
    weight = weight.to(device)
    bias = bias.to(device)
    
    #intiale an output matrix of the correct size
    z = torch.zeros([batch_size, n_outputs]).to(device)
    
    for n in range(batch_size):
        for j in range(n_outputs):
            z[n,j] = bias[j]
            for i in range(n_inputs):
                z[n,j] += weight[j,i]*a[n,i]
                
    return z
  
    
def fc_layer_vector(a, weight, bias, device):
    
    #move to device
    a = a.to(device)
    weight = weight.to(device)
    bias = bias.to(device)
    
    z = (a.matmul(weight.t())+ bias).clone()
    
    return z