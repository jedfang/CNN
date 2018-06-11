
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0)  
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[ ]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



# In[ ]:


x=mnist.train.images.reshape(55000,28,28,1)
y=mnist.train.labels.reshape(55000,10,1)
x=x[:10,:,:,:]
y=y[:10,:,:]
print(y)
print(x.shape)
y=y.T
m=y.shape[1]
print(m)
print(y)


# In[ ]:


def zero_pad(X, pad):
    X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant')
    return X_pad


# In[ ]:


def conv_single_step(a_slice_prev, W, b):
    s =a_slice_prev*W
    Z = np.sum(s)
    Z = Z+np.float(b)
    return Z


# In[ ]:


def conv_forward(A_prev, W, b, hparameters):
     
    (m, n_H_prev, n_W_prev, n_C_prev) =A_prev.shape 
    (f, f, n_C_prev, n_C) = W.shape
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]
     
    n_H = int((n_H_prev-f+2*pad)/stride) +1
    n_W = int((n_W_prev-f+2*pad)/stride) +1
    Z = np.zeros((m,n_H,n_W,n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):                                 
        a_prev_pad = A_prev_pad[i]                    
        for h in range(n_H):                           
            for w in range(n_W):                      
                for c in range(n_C):                   
                     
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride 
                    horiz_end = horiz_start+f
                     
                    a_slice_prev =a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:] 
                    Z[i, h, w, c] =conv_single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])
                                        
    assert(Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)
    return Z, cache


# In[ ]:


def pool_forward(A_prev, hparameters, mode = "max"):
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))              
    
    for i in range(m):                         
        for h in range(n_H):                     
            for w in range(n_W):                 
                for c in range (n_C):            
                    
                    vert_start =h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f
            
                    a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    cache = (A_prev, hparameters)
    assert(A.shape == (m, n_H, n_W, n_C))
    return A, cache


# In[ ]:


def fc(A_prev,w,b):
    (m,hw) = A_prev.shape
    A=A_prev 
    Z=np.dot(w,A)+b
    cache_fc=(A_prev,w,b)
    return Z,cache_fc


# In[ ]:


def softmax(Z):
    Z_exp=np.exp(Z)
    Z_sum=np.sum(Z_exp,axis=0)
    print(Z_sum.shape)
    Z_sf=Z_exp/Z_sum
    return Z_sf


# In[ ]:


def loss(predicted,y):
       m=y.shape[1]
       loss=-np.log(predicted)*y
       loss=(np.sum(loss))  
       loss=loss/m
       return loss


# In[ ]:


def fc_backward(dZ,cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1.0/m*(np.dot(dZ,A_prev.T))
    db = 1.0/m*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev,dW,db


# In[ ]:


def create_mask_from_window(x):
   
    mask = (x==np.max(x))
    return mask

def distribute_value(dz, shape):
  
    (n_H, n_W) = shape
    average = n_H*n_W
    a = np.ones(shape)*dz/average
    return a

def pool_backward(dA, cache, mode = "max"):
   
    (A_prev, hparameters) = cache
    stride = hparameters["stride"]
    f = hparameters["f"]
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    dA_prev = np.zeros(A_prev.shape)
    
    for i in range(m):                       
        a_prev = A_prev[i]
        
        for h in range(n_H):                   
            for w in range(n_W):               
                for c in range(n_C):          
                    
                    vert_start = h
                    vert_end = h+f
                    horiz_start = w
                    horiz_end = w+f
                    
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask*dA[i,h,w, c]
                        
                    elif mode == "average":
                        da = dA[i,h,w,c]
                        shape = (f,f)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)
                        
    assert(dA_prev.shape == A_prev.shape)
    return dA_prev


# In[ ]:


def conv_backward(dZ, cache):
  
   (A_prev, W, b, hparameters) = cache
   (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
   (f, f, n_C_prev, n_C) = W.shape
   stride = hparameters["stride"]
   pad = hparameters["pad"]
   (m, n_H, n_W, n_C) = dZ.shape
   dA_prev = np.zeros(A_prev.shape)                           
   dW = np.zeros(W.shape)
   db = np.zeros(b.shape)
   A_prev_pad = zero_pad(A_prev, pad)
   dA_prev_pad = zero_pad(dA_prev, pad)
   
   for i in range(m):                       
       
       a_prev_pad = A_prev_pad[i]
       da_prev_pad = dA_prev_pad[i]
       
       for h in range(n_H):                  
           for w in range(n_W):               
               for c in range(n_C):           
                   
                   vert_start = h*pad
                   vert_end = h*pad+f
                   horiz_start = w*pad
                   horiz_end = w*pad+f
                   
                   a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                   da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                   dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                   db[:,:,:,c] += dZ[i, h, w, c]
                   
       dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
  
   assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
   return dA_prev, dW, db


# In[ ]:


A_0=x
W_1=np.random.randn(3,3,1,5)*0.01
b_1=np.random.randn(1,1,1,5)
A_1,cache_1=conv_forward(A_0, W_1, b_1, hparameters={"stride":1,"pad":1})


# In[ ]:


A_2, cache_2= pool_forward(A_1, hparameters={"f":2,"stride":2}, mode = "max")
A_2=(A_2.reshape(10,980)).T


# In[ ]:


W_3=np.random.randn(10,980)*0.01
b_3=np.random.randn(10,1)
Z_3,cache_3=fc(A_2, W_3, b_3)

 
A_3=softmax(Z_3)
dZ_3=A_3-y


# In[ ]:


costs=[]


# In[ ]:


costs.append(loss(A_3, y))
temp_cost=loss(A_3, y)
change=1


# In[ ]:


for i in range(10):
    dA_2,dW_3,db_3=fc_backward(dZ_3, cache_3)
    
    dA_2=dA_2.T
    dA_2=dA_2.reshape(10,14,14,5)
    dA_1=pool_backward(dA_2, cache_2, mode = "max")
    dZ_1=dA_1*(1-np.power(A_1,2))
    dA_1,dW_1,db_1=conv_backward(dZ_1, cache_1)
    learning_rate=0.0001
    W_1=W_1-learning_rate*dW_1
    W_3=W_3-learning_rate*dW_3
    
    b_1=b_1-learning_rate*db_1
    b_3=b_3-learning_rate*db_3
    A_1,cache_1=conv_forward(A_0, W_1, b_1, hparameters={"stride":1,"pad":1})
    A_2, cache_2= pool_forward(A_1, hparameters={"f":2,"stride":2}, mode = "max")
    A_2=(A_2.reshape(10,980)).T
    W_3=np.random.randn(10,980)*0.01
    b_3=np.random.randn(10,1)
    Z_3,cache_3=fc(A_2, W_3, b_3)

 
    A_3=softmax(Z_3)
    cost=loss(A_3, y)
    costs.append(cost)
    change=temp_cost-cost
    temp_cost=cost
    
    print("{} epoch completed and cost is {}".format(i+1,cost))


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(costs)
plt.show()

