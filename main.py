# import context
# import torch
import cytnx
import argparse
import config as cfg
from ipeps.ipeps import *
# from ctm.generic.env import *
# from ctm.generic import ctmrg

# parse command line args and build necessary configuration objects
# parser= cfg.get_args_parser()
# additional model-dependent arguments
# parser.add_argument("--tensor", default="TFIM.cytnx", help="Input building block tensor for iPEPS.")
# args, unknown_args = parser.parse_known_args()

def main():
    # cfg.configure(args)
       
    # if args.tiling == "BIPARTITE":
    #     def lattice_to_site(coord):
    #         vx = (coord[0] + abs(coord[0]) * 2) % 2
    #         vy = abs(coord[1])
    #         return ((vx + vy) % 2, 0)
    # elif args.tiling == "1SITE":
    #     def lattice_to_site(coord):
    #         return (0, 0)
    # elif args.tiling == "2SITE":
    #     def lattice_to_site(coord):
    #         vx = (coord[0] + abs(coord[0]) * 2) % 2
    #         vy = (coord[1] + abs(coord[1]) * 1) % 1
    #         return (vx, vy)
    # elif args.tiling == "4SITE":
    #     def lattice_to_site(coord):
    #         vx = (coord[0] + abs(coord[0]) * 2) % 2
    #         vy = (coord[1] + abs(coord[1]) * 2) % 2
    #         return (vx, vy)
    # elif args.tiling == "8SITE":
    #     def lattice_to_site(coord):
    #         shift_x = coord[0] + 2*(coord[1] // 2)
    #         vx = shift_x % 4
    #         vy = coord[1] % 2
    #         return (vx, vy)
    # else:
    #     raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
    #         +"BIPARTITE, 2SITE, 4SITE, 8SITE")


    ### cytnx tensors
    A = cytnx.UniTensor.zeros(shape = [2,3,4], labels = ["a","b","c"],dtype = 3, device = -1, name = "zero")+9
    print(cytnx.UniTensor.uniform(shape = [2,3,4],low = 0, high = 1, in_labels = ["a","b","c"], seed = -1, dtype = 3, device = -1, name = "random")-100)
    print(cytnx.UniTensor.zeros(shape = [2,3,4], labels = ["a","b","c"],dtype = 3, device = -1, name = "zero"))
    print(cytnx.UniTensor.eye(dim = 3, labels = ["a","b"], is_diag = False, dtype = 3, device = -1, name = "zero"))
    print(cytnx.UniTensor.zeros(shape = [1], labels = ["a"],dtype = 3, device = -1, name = "zero"))
    
    print((cytnx.UniTensor.zeros(shape = [1], labels = ["a"],dtype = cytnx.Type.Bool, device = -1, name = ""),))
    print((cytnx.UniTensor.ones(shape = [1], labels = ["a"],dtype = cytnx.Type.Bool, device = -1, name = ""),))    
    ### cytnx type
    # print(int(cytnx.Type.Float))
    # print(int(cytnx.Type.ComplexDouble))
    # print(cytnx.__version__)
    # def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
    #     return False, history

    # ctm_env_init = ENV(args.chi, state)
    # init_env(state, ctm_env_init)
    
    # print(", ".join(["epoch","energy"]+obs_labels))
    # print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    # ctm_env_init, *ctm_log= ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

    # # 6) compute final observables
    # e_curr0 = energy_f(state, ctm_env_init)
    # obs_values0, obs_labels = eval_obs_f(state,ctm_env_init)
    # history, t_ctm, t_obs= ctm_log
    # print("\n")
    # print(", ".join(["epoch","energy"]+obs_labels))
    # print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    # print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # path = args.txt
    # with open(path, 'a') as f:
    #     f.write("  ".join([f"{args.h}"]+[f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    #     f.write("\n")
    #         print(f"{i} {l[i,0]} {l[i,1]}")

if __name__=='__main__':
    # if len(unknown_args)>0:
    #     print("args not recognized: "+str(unknown_args))
    #     raise Exception("Unknown command line arguments")
    main()
