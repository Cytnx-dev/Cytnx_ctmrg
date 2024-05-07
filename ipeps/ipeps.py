import warnings
# import torch
import cytnx
from collections import OrderedDict
import json
import itertools
import math
import config as cfg
# from ipeps.tensor_io import *
import logging
log = logging.getLogger(__name__)

# TODO drop constrain for aux bond dimension to be identical on 
# all bond indices

class IPEPS():
    def __init__(self, sites=None, vertexToSite=None, lX=None, lY=None, peps_args=cfg.peps_args,\
        global_args=cfg.global_args):
        r"""
        :param sites: map from elementary unit cell to on-site tensors
        :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                             into a vertex within elementary unit cell
        :param lX: length of the elementary unit cell in X direction
        :param lY: length of the elementary unit cell in Y direction
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type sites: dict[tuple(int,int) : cytnx.tensor]
        :type vertexToSite: function(tuple(int,int))->tuple(int,int)
        :type lX: int
        :type lY: int
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Member ``sites`` is a dictionary of non-equivalent on-site tensors
        indexed by tuple of coordinates (x,y) within the elementary unit cell.
        The index-position convetion for on-site tensors is defined as follows::

               u s 
               |/ 
            l--a--r  <=> a[s,u,l,d,r]
               |
               d
        
        where s denotes physical index, and u,l,d,r label four principal directions
        up, left, down, right in anti-clockwise order starting from up.
        Member ``vertexToSite`` is a mapping function from any vertex (x,y) on a square lattice
        passed in as tuple(int,int) to a corresponding vertex within elementary unit cell.
        
        On-site tensor of an IPEPS object ``wfc`` at vertex (x,y) is conveniently accessed 
        through the member function ``site``, which internally uses ``vertexToSite`` mapping::
            
            coord= (0,0)
            a_00= wfc.site(coord)

        By combining the appropriate ``vertexToSite`` mapping function with elementary unit 
        cell specified through ``sites``, various tilings of a square lattice can be achieved:: 
            
            # Example 1: 1-site translational iPEPS
            
            sites={(0,0): a}
            def vertexToSite(coord):
                return (0,0)
            wfc= IPEPS(sites,vertexToSite)
        
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   a  a a a a
            # -1   a  a a a a
            #  0   a  a a a a
            #  1   a  a a a a
            # Example 2: 2-site bipartite iPEPS
            
            sites={(0,0): a, (1,0): b}
            def vertexToSite(coord):
                x = (coord[0] + abs(coord[0]) * 2) % 2
                y = abs(coord[1])
                return ((x + y) % 2, 0)
            wfc= IPEPS(sites,vertexToSite)
        
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   A  b a b a
            # -1   B  a b a b
            #  0   A  b a b a
            #  1   B  a b a b
        
            # Example 3: iPEPS with 3x2 unit cell with PBC 
            
            sites={(0,0): a, (1,0): b, (2,0): c, (0,1): d, (1,1): e, (2,1): f}
            wfc= IPEPS(sites,lX=3,lY=2)
            
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   b  c a b c
            # -1   e  f d e f
            #  0   b  c a b c
            #  1   e  f d e f

        where in the last example a default setting for ``vertexToSite`` is used, which
        maps square lattice into elementary unit cell of size ``lX`` x ``lY`` assuming 
        periodic boundary conditions (PBC) along both X and Y directions.
        """
        if not sites:
            self.dtype= global_args.cytnx_dtype
            self.device= global_args.device
        else:
            assert len(set( tuple( site.dtype for site in sites.values() ) ))==1,"Mixed dtypes in sites"
            assert len(set( tuple( site.device for site in sites.values() ) ))==1,"Mixed devices in sites"
            # self.dtype= next(iter(sites.values())).dtype
            # self.device= next(iter(sites.values())).device
            self.dtype= next(iter(sites.values())).dtype()
            self.device= next(iter(sites.values())).device()
            self.sites= OrderedDict(sites)

        # TODO we infer the size of the cluster from the keys of sites. Is it OK?
        # infer the size of the cluster
        if (lX is None or lY is None) and sites:
            min_x = min([coord[0] for coord in sites.keys()])
            max_x = max([coord[0] for coord in sites.keys()])
            min_y = min([coord[1] for coord in sites.keys()])
            max_y = max([coord[1] for coord in sites.keys()])
            self.lX = max_x-min_x + 1
            self.lY = max_y-min_y + 1
        elif lX and lY:
            self.lX = lX
            self.lY = lY
        else:
            raise Exception("lX and lY has to set either directly or implicitly by sites")

        if vertexToSite is not None:
            self.vertexToSite = vertexToSite
        else:
            def vertexToSite(coord):
                x = coord[0]
                y = coord[1]
                return ( (x + abs(x)*self.lX)%self.lX, (y + abs(y)*self.lY)%self.lY )
            self.vertexToSite = vertexToSite

    def site(self, coord):
        """
        :param coord: tuple (x,y) specifying vertex on a square lattice
        :type coord: tuple(int,int)
        :return: on-site tensor corresponding to the vertex (x,y)
        :rtype: cytnx.tensor
        """
        return self.sites[self.vertexToSite(coord)]

    def get_parameters(self):
        r"""
        :return: variational parameters of iPEPS
        :rtype: iterable
        
        This function is called by optimizer to access variational parameters of the state.
        """
        return self.sites.values()

    def write_to_file(self,outputfile,aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
        """
        Writes state to file. See :meth:`write_ipeps`.
        """
        write_ipeps(self,outputfile,aux_seq=aux_seq, tol=tol, normalize=normalize)

    def add_noise(self,noise,noise_f=None):
        r"""
        :param noise: magnitude of the noise
        :type noise: float

        Take IPEPS and add random uniform noise with magnitude ``noise`` to all on-site tensors
        """
        for coord in self.sites.keys():
            if noise_f:
                rand_t = noise_f(self.sites[coord].size(), dtype=self.dtype, device=self.device)    
            else:
                # rand_t = torch.rand(self.sites[coord].size(), dtype=self.dtype, device=self.device)-0.5
                rand_t = cytnx.UniTensor.uniform(shape = self.sites[coord].size(),low = 0, high = 1, in_labels = ["a","b","c"], seed = -1, dtype = self.dtype, device = self.device, name = "")-0.5
            self.sites[coord] = self.sites[coord] + noise * rand_t

    def get_aux_bond_dims(self):
        return [d for key in self.sites.keys() for d in self.sites[key].size()[1:]]

    def __str__(self):
        print(f"lX x lY: {self.lX} x {self.lY}")
        for nid,coord,site in [(t[0], *t[1]) for t in enumerate(self.sites.items())]:
            print(f"a{nid} {coord}: {site.size()}")
        
        # show tiling of a square lattice
        coord_list = list(self.sites.keys())
        mx, my = 3*self.lX, 3*self.lY
        label_spacing = 1+int(math.log10(len(self.sites.keys())))
        for y in range(-my,my):
            if y == -my:
                print("y\\x ", end="")
                for x in range(-mx,mx):
                    print(str(x)+label_spacing*" "+" ", end="")
                print("")
            print(f"{y:+} ", end="")
            for x in range(-mx,mx):
                print(f"a{coord_list.index(self.vertexToSite((x,y)))} ", end="")
            print("")
        
        return ""

    def normalize_(self):
        for c in self.sites.keys():
            self.sites[c]= self.sites[c]/self.sites[c].abs().max()
