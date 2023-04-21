import argparse
import window
import numpy as np
from locals import *



def main():
    parser = argparse.ArgumentParser(
                    prog = '',
                    description = '')
    parser.add_argument('task', type=str,
                        choices=["window"])
    
    parser.add_argument('--loss', default=False, action="store_true")
    parser.add_argument('--path', default=False, action="store_true")
    
    parser.add_argument('vars', nargs='*')
    args = parser.parse_args()

    if args.task == "window":
        if args.loss:
            Loss, L_loss_max_v, L_max_a, L_id_v_dist, L_loss_lenght = window.analyse_direction_constante(L_min, L_max, T_min, T_max, n_T, n_L, x0, v0, a0, a1, direction_win, v_win, norme_v1, conv_L_m, conv_T_s,  n_point)
            window.plot_analyse_direction_constante(Loss, L_loss_lenght, L_loss_max_v, L_id_v_dist, L_max_a, L_min, L_max, T_min, T_max, n_L, n_T)

        elif args.path:
            window.plot_path(L, T, x0, v0, a0, a1, norme_v1, direction_win, v_win,scale, theta, n_point)
    

if __name__ == "__main__":
    main()