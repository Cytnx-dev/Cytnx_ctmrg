# import torch
import cytnx 

# def contract(t1, t2, *args):
#     return torch.tensordot(t1, t2, *args)

# def mm(m1, m2):
#     return torch.mm(m1, m2)

# def einsum(op, *ts):
#     return torch.einsum(op, *ts)

# def view(t, *args):
#     return t.view(*args)

# def permute(t, *args):
#     return t.permute(*args)

# def contiguous(t):
#     return t.contiguous()

# def transpose(t):
#     return torch.transpose(t, 0, 1)

# def conj(t):
#     return t.conj()

def contract(t1, t2, axis,*args):
    return cytnx.UniTensor(cytnx.linalg.Tensordot(t1.get_block(), t2.get_block(), *(axis),*args))

def mm(m1, m2):
    return cytnx.UniTensor(cytnx.linalg.Matmul(m1.get_block(), m2.get_block()))

# def einsum(op, *ts):
#     return torch.einsum(op, *ts)

def view(t, shape):
    return t.reshape(*(shape))

def permute(t, *args):
    return t.permute(*args)

def contiguous(t):
    return t.contiguous()

def transpose(t):
    return t.Transpose()

def conj(t):
    return t.Conj()

def rsqrt(S_nz):
    return S_nz**(-0.5)

def svdvals(T):
    return cytnx.linalg.Svd(T,False)[0]