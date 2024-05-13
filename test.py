# import cytnx

# # tensors
# A = cytnx.UniTensor.zeros(shape = [2,3,4], labels = ["a","b","c"],dtype = 3, device = -1, name = "zero")+9
# print(cytnx.UniTensor.uniform(shape = [2,3,4],low = 0, high = 1, in_labels = ["a","b","c"], seed = -1, dtype = 3, device = -1, name = "random")-100)
# print(cytnx.UniTensor.zeros(shape = [2,3,4], labels = ["a","b","c"],dtype = 3, device = -1, name = "zero"))
# print(cytnx.UniTensor.eye(dim = 3, labels = ["a","b"], is_diag = False, dtype = 3, device = -1, name = "zero"))
# print(cytnx.UniTensor.zeros(shape = [1], labels = ["a"],dtype = 3, device = -1, name = "zero"))

# print((cytnx.UniTensor.zeros(shape = [1], labels = ["a"],dtype = cytnx.Type.Bool, device = -1, name = ""),))
# print((cytnx.UniTensor.ones(shape = [1], labels = ["a"],dtype = cytnx.Type.Bool, device = -1, name = ""),))    

# ### tensor dots
# A = cytnx.UniTensor.ones(shape = [6,4], labels = ["a","b"],dtype = cytnx.Type.Bool, device = -1, name = "")
# B = cytnx.UniTensor.ones(shape = [3,4,5], labels = ["a","b","c"],dtype = cytnx.Type.Bool, device = -1, name = "")
# C =  cytnx.linalg.Tensordot(A.get_block(),B.get_block(),[1],[1],False, False)
# print(C.shape())

# ### transpose
# A = cytnx.UniTensor.ones(shape = [6,4], labels = ["a","b"],dtype = cytnx.Type.Bool, device = -1, name = "")
# print(A.Transpose().shape())

# ### conj
# A = cytnx.UniTensor.ones(shape = [6,4], labels = ["a","b"],dtype = cytnx.Type.Bool, device = -1, name = "")
# print(A.Conj())

# ### contiguous
# A = cytnx.UniTensor.ones(shape = [6,4], labels = ["a","b"],dtype = cytnx.Type.Bool, device = -1, name = "")
# print(A.contiguous())

# ### mm
# A = cytnx.UniTensor.ones(shape = [6,4], labels = ["a","b"],dtype = cytnx.Type.Bool, device = -1, name = "")
# B = cytnx.UniTensor.ones(shape = [4,6], labels = ["a","b"],dtype = cytnx.Type.Bool, device = -1, name = "")
# print(cytnx.UniTensor(cytnx.linalg.Matmul(A.get_block(),B.get_block())))

# ### permute
# A = cytnx.UniTensor.ones(shape = [6,4,3], labels = ["a","b","C"],dtype = cytnx.Type.Bool, device = -1, name = "")
# print(A.permute([0,2,1]))
    
# ### view
# A = cytnx.UniTensor.ones(shape = [6,4,3], labels = ["a","b","C"],dtype = cytnx.Type.Bool, device = -1, name = "")
# print(A.reshape([6,12],0).is_contiguous())
# print(A.is_contiguous())

# cytnx::linalg::Gesvd_truncate	(	const cytnx::UniTensor & 	Tin,
# const cytnx_uint64 & 	keepdim,
# const double & 	err = 0,
# const bool & 	is_U = true,
# const bool & 	is_vT = true,
# const unsigned int & 	return_err = 0 
# )	
# svdval
# A = cytnx.UniTensor.uniform(shape = [30,30],low = 0, high = 1, in_labels = ["a","b"], seed = -1, dtype = 3, device = -1, name = "random")
# print(cytnx.linalg.Gesvd_truncate(A,20,1e-10,True,True,0)[0])

# #abs max
# print(cytnx.UniTensor.ones(shape = [6,4,3], labels = ["a","b","C"],dtype = cytnx.Type.Double, device = -1, name = "").get_block().Abs().Max())

# # tensor to UniTensor 
# print(cytnx.UniTensor(cytnx.zeros([2,2] ,dtype=cytnx.Type.Double, device=-1)))

### cytnx type
# print(int(cytnx.Type.Float))
# print(int(cytnx.Type.ComplexDouble))
# print(cytnx.__version__)

# net = cytnx.Network()
# net.FromString(["c0: t0-c0, t3-c0",\
#                 "c1: t1-c1, t0-c1",\
#                 "c2: t2-c2, t1-c2",\
#                 "c3: t3-c3, t2-c3",\
#                 "t0: t0-c1, w-t0, t0-c0",\
#                 "t1: t1-c2, w-t1, t1-c1",\
#                 "t2: t2-c3, w-t2, t2-c2",\
#                 "t3: t3-c0, w-t3, t3-c3",\
#                 "w: w-t0, w-t1, w-t2, w-t3",\
#                 "TOUT:",\
#                 "ORDER: ((((((((c0,t0),c1),t3),w),t1),c3),t2),c2)"])

## none indexing

# import numpy as np
# a = np.ones([100])
# b = a[None,:10]
# print(b.shape)
# import torch


import numpy as np
import time
import cytnx
import cProfile
import re
# def contract():

d = 2
D = 2
chi = 64

T = cytnx.UniTensor(cytnx.zeros([chi,D,D,chi])).set_labels(["0","1","2","3"])
Pt2 = cytnx.UniTensor(cytnx.zeros([chi,D,D,chi])).set_labels(["0","8","9","4"])
P1 = cytnx.UniTensor(cytnx.zeros([chi,D,D,chi])).set_labels(["3","10","11","7"])
A = cytnx.UniTensor(cytnx.zeros([d,D,D,D,D])).set_labels(["12","1","8","5","10"])
Aconj = cytnx.UniTensor(cytnx.zeros([d,D,D,D,D])).set_labels(["12","2","9","6","11"])


# net = cytnx.Network()
# net.FromString(["T:0,1,2,3","Pt2:0,8,9,4","A:12,1,8,5,10","Aconj:12,2,9,6,11","P1:3,10,11,7","TOUT:4,5,6,7","ORDER: (T,(P1,(Pt2,(A,Aconj))))"])
# net.PutUniTensors(["T","Pt2","A","Aconj","P1"],[T,Pt2,A,A.Conj(),P1])

# t0_net= time.perf_counter()
# nT = net.Launch()
# t1_net= time.perf_counter()
# print(t1_net-t0_net)

t0_net= time.perf_counter()
res = cytnx.Contract(T,cytnx.Contract(P1,cytnx.Contract(Pt2,cytnx.Contract(A,Aconj))))
# net.setOrder(optimal = True)
# print(net.getOrder())
t1_net= time.perf_counter()

print(t1_net-t0_net)

import numpy as np
d = 2
D = 2
chi = 64
T = np.ones([chi,D,D,chi])
Pt2 =   np.ones([chi,D,D,chi])
P1 =   np.ones([chi,D,D,chi])
A =  np.ones([d,D,D,D,D])

t0_net= time.perf_counter()
nT= np.einsum_path(T,[0,1,2,3],Pt2,[0,8,9,4],A,[12,1,8,5,10],A.conj(),[12,2,9,6,11],P1,[3,10,11,7],[4,5,6,7])[0]
t1_net= time.perf_counter()

print(t1_net-t0_net)

    
# cProfile.run('net.Launch()')
# T = torch.zeros([chi,D,D,chi])
# Pt2 =  torch.zeros([chi,D,D,chi])
# P1 =  torch.zeros([chi,D,D,chi])
# A = torch.zeros([d,D,D,D,D])

