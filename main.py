# import context
# import torch
import cytnx
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
parser.add_argument("--tensor", default="TFIM.cytnx", help="Input building block tensor for iPEPS.")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)

    tmp = cytnx.UniTensor(cytnx.ones([2,2,2,2,2],dtype=cfg.global_args.dtype, device=cfg.global_args.device))
    tmp= tmp/tmp.get_block().Abs().Max().item()
    sites = {(0,0): tmp}
    state = IPEPS(sites)
    
    # def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
    #     return False, history

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    
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
