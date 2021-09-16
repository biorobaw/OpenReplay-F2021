import scikit_posthocs as sp
from plotnine import *
import os
import numpy as np
from .data_loader import *
from .pythonUtils.BinaryFiles import *
from ..utils.MazeParser import parse_maze


def plot_replay_paths(title, maze_file, pc_file, replay_path_file, save_name):

    # first plot paths and then plot maze
    # create plot
    p = ggplot() + ggtitle(title)

    # # plot paths
    # num_rats = 100
    # for id in range(num_rats):
    #     # check if file exists
    #     file_name = os.path.join(config_folder, f'r{id}-paths.bin')
    #     if not os.path.exists(file_name):
    #         continue

    #     # print(file_name, save_name)
    #     # open file and plot each path
    #     with open( file_name, 'rb') as file:
    #         for i in range(int(config_df['numStartingPositions'])):
    #             xy_df = pd.DataFrame(data=load_float_vector(file).reshape((-1, 2)), columns=["x", "y"])
    #             p = p + geom_path(aes(x='x', y='y'), data=xy_df, color='blue', alpha=1.0/num_rats)


    # plot maze
    maze_file = os.path.join(config_folder, '../../mazes', config_df['mazeFile'])
    walls, feeders, start_positions = parse_maze(maze_file)
    p = p + geom_segment(aes(x='x1', y='y1', xend='x2', yend='y2'), data=walls, color='k')
    p = p + geom_point(aes(x='x', y='y'), data=feeders, color='r')
    p = p + coord_fixed(ratio = 1)

    # save plot
    ggsave(p, save_name, dpi=300)


if __name__ == '__main__':
    #folder_arg = os.path.join(sys.argv[1], '')
    #config_arg = None if len(sys.argv) < 3 else sys.argv[2]
    # folder_arg = 'D:/JavaWorkspaceSCS/Multiscale-F2019/experiments/BICY2020_modified/logs/experiment1-traces/'
    plot_replay_paths('test', '/Users/titonka/ReplayWS/OpenReplay-F2021/experiments/replayF2021_experiments', None, test.pdf)
