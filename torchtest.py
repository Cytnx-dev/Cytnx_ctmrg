from opt_einsum import contract, contract_path
import time
import torch
d = 2
D = 2
chi = 128
T = torch.ones([chi,D,D,chi])
Pt2 =   torch.ones([chi,D,D,chi])
P1 =   torch.ones([chi,D,D,chi])
A =  torch.ones([d,D,D,D,D])


# nT = np.einsum('abcd,aije,mbifk,mcjgl,dklh->efgh',T,Pt2,A,A.conj(),P1, optimize=['einsum_path', (2, 3), (0, 1), (1, 2), (0, 1)])
# nT= np.einsum(T,[0,1,2,3],Pt2,[0,8,9,4],A,[12,1,8,5,10],A.conj(),[12,2,9,6,11],P1,[3,10,11,7],[4,5,6,7])
for kw in ['branch-all','dp','optimal']:
    nT= contract_path(T,[0,1,2,3],Pt2,[0,8,9,4],A,[12,1,8,5,10],A.conj(),[12,2,9,6,11],P1,[3,10,11,7],[4,5,6,7], optimize = kw)
    print(nT)
    
    t0_net= time.perf_counter()
    # nT= contract(T,[0,1,2,3],Pt2,[0,8,9,4],A,[12,1,8,5,10],A.conj(),[12,2,9,6,11],P1,[3,10,11,7],[4,5,6,7], optimize = kw)
    nT= contract(T,[0,1,2,3],Pt2,[0,8,9,4],A,[12,1,8,5,10],A.conj(),[12,2,9,6,11],P1,[3,10,11,7],[4,5,6,7], order = nT)
    t1_net= time.perf_counter()
    print(t1_net-t0_net)