import cytnx

# tensors
A = cytnx.UniTensor.zeros(shape = [2,3,4], labels = ["a","b","c"],dtype = 3, device = -1, name = "zero")+9
print(cytnx.UniTensor.uniform(shape = [2,3,4],low = 0, high = 1, in_labels = ["a","b","c"], seed = -1, dtype = 3, device = -1, name = "random")-100)
print(cytnx.UniTensor.zeros(shape = [2,3,4], labels = ["a","b","c"],dtype = 3, device = -1, name = "zero"))
print(cytnx.UniTensor.eye(dim = 3, labels = ["a","b"], is_diag = False, dtype = 3, device = -1, name = "zero"))
print(cytnx.UniTensor.zeros(shape = [1], labels = ["a"],dtype = 3, device = -1, name = "zero"))

print((cytnx.UniTensor.zeros(shape = [1], labels = ["a"],dtype = cytnx.Type.Bool, device = -1, name = ""),))
print((cytnx.UniTensor.ones(shape = [1], labels = ["a"],dtype = cytnx.Type.Bool, device = -1, name = ""),))    

### tensor dots
A = cytnx.UniTensor.ones(shape = [6,4], labels = ["a","b"],dtype = cytnx.Type.Bool, device = -1, name = "")
B = cytnx.UniTensor.ones(shape = [3,4,5], labels = ["a","b","c"],dtype = cytnx.Type.Bool, device = -1, name = "")
C =  cytnx.linalg.Tensordot(A.get_block(),B.get_block(),[1],[1],False, False)
print(C.shape())

### transpose
A = cytnx.UniTensor.ones(shape = [6,4], labels = ["a","b"],dtype = cytnx.Type.Bool, device = -1, name = "")
print(A.Transpose().shape())

### conj
A = cytnx.UniTensor.ones(shape = [6,4], labels = ["a","b"],dtype = cytnx.Type.Bool, device = -1, name = "")
print(A.Conj())

### contiguous
A = cytnx.UniTensor.ones(shape = [6,4], labels = ["a","b"],dtype = cytnx.Type.Bool, device = -1, name = "")
print(A.contiguous())

### mm
A = cytnx.UniTensor.ones(shape = [6,4], labels = ["a","b"],dtype = cytnx.Type.Bool, device = -1, name = "")
B = cytnx.UniTensor.ones(shape = [4,6], labels = ["a","b"],dtype = cytnx.Type.Bool, device = -1, name = "")
print(cytnx.UniTensor(cytnx.linalg.Matmul(A.get_block(),B.get_block())))

### permute
A = cytnx.UniTensor.ones(shape = [6,4,3], labels = ["a","b","C"],dtype = cytnx.Type.Bool, device = -1, name = "")
print(A.permute([0,2,1]))
    
### view
A = cytnx.UniTensor.ones(shape = [6,4,3], labels = ["a","b","C"],dtype = cytnx.Type.Bool, device = -1, name = "")
print(A.reshape([6,12],0).is_contiguous())
print(A.is_contiguous())

# # svdval
# A = cytnx.UniTensor.ones(shape = [6,4,3], labels = ["a","b","C"],dtype = cytnx.Type.Bool, device = -1, name = "")
# print(cytnx.linalg.Svd(A,False)[0])
    
# #abs max
# print(cytnx.UniTensor.ones(shape = [6,4,3], labels = ["a","b","C"],dtype = cytnx.Type.Double, device = -1, name = "").get_block().Abs().Max())

# # tensor to UniTensor 
# print(cytnx.UniTensor(cytnx.zeros([2,2] ,dtype=cytnx.Type.Double, device=-1)))

### cytnx type
# print(int(cytnx.Type.Float))
# print(int(cytnx.Type.ComplexDouble))
# print(cytnx.__version__)