import scikit_posthocs as sp
from plotnine import *
import os
import numpy as np
from data_loader import *
from pythonUtils.BinaryFiles import *
import csv

import xml.etree.ElementTree as ET
from functools import *
import pandas as pd


def parse_feeder(xml_feeder):
    fid = int(xml_feeder.get('id'))
    x = float(xml_feeder.get('x'))
    y = float(xml_feeder.get('y'))
    return pd.DataFrame(data=[[fid, x, y]], columns=['fid', 'x', 'y'])


def parse_all_feeders(root):
    return pd.concat([pd.DataFrame(columns=['fid', 'x', 'y'])] + [parse_feeder(xml_feeder) for xml_feeder in root.findall('feeder')]).reset_index(drop=True)


def parse_wall(xml_wall):
    data = [[float(xml_wall.get(coord)) for coord in ['x1', 'y1', 'x2', 'y2']]]
    return pd.DataFrame(data=data, columns=['x1', 'y1', 'x2', 'y2'])


def parse_all_walls(xml_root):
    return pd.concat([pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])]  + [parse_wall(xml_wall) for xml_wall in xml_root.findall('wall')]).reset_index(drop=True)


def parse_all_generators(xml_root):
    wall_sets = [generator_parsers[g.get('class')](g) for g in xml_root.findall('generator')]
    return pd.concat([pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])] + wall_sets).reset_index(drop=True)


def parse_rectangle(xml_rectangle):
    x1 = float(xml_rectangle.get('x1'))
    y1 = float(xml_rectangle.get('y1'))
    x2 = float(xml_rectangle.get('x2'))
    y2 = float(xml_rectangle.get('y2'))

    data = [
        [x1, y1, x2, y1],
        [x2, y1, x2, y2],
        [x2, y2, x1, y2],
        [x1, y2, x1, y1]
    ]
    return pd.DataFrame(data=data, columns=['x1', 'y1', 'x2', 'y2'])


generator_parsers = {'$(SCS).maze.mazes.Rectangle': parse_rectangle}


def parse_position(xml_position):
    return pd.DataFrame(data=[[float(xml_position.get(p)) for p in ['x', 'y']]], columns=['x','y'])


def parse_start_positions(xml_positions):
    if xml_positions is None:
        return pd.DataFrame(columns=['x', 'y', 'w'])
    return pd.concat([pd.DataFrame(columns=['x', 'y', 'w'])] + [parse_position(xml_pos) for xml_pos in xml_positions.findall('pos')]).reset_index(drop=True)


def parse_maze(file):
    root = ET.parse(file).getroot()
    start_positions = parse_start_positions(root.find('startPositions'))
    walls = pd.concat([parse_all_walls(root), parse_all_generators(root)]).reset_index(drop=True)
    feeders = parse_all_feeders(root)

    return walls, feeders, start_positions

def plot_replay_paths(title, maze_file, replay_path_file, save_name):

    # first plot paths and then plot maze
    # create plot
    paths_test = [[16,24,32,100]] 



    p = ggplot() + ggtitle(title + '1')


     # plot maze
    maze_file = os.path.join(maze_file, 'mazes/M03.xml')
    walls, feeders, start_positions = parse_maze(maze_file)
    p = p + geom_segment(aes(x='x1', y='y1', xend='x2', yend='y2'), data=walls, color='k')
    p = p + geom_point(aes(x='x', y='y'), data=feeders, color='r')
    p = p + coord_fixed(ratio = 1)
    

    cells = pd.read_csv('/Users/titonka/ReplayWS/OpenReplay-F2021/experiments/pc_layers/test/u09_09.csv')
    with open(replay_path_file) as path_file:
        paths = csv.reader(path_file, delimiter = '\n')
        episode_counter = 0
        path_counter = 0
        for path in paths:
            path_counter = path_counter + 1
            replay_event = path[0].split(',')
            #print(replay_event)
            replay_event = map(int, replay_event)
            p = p + geom_path(aes(x='x', y='y'), data=cells.loc[replay_event], color='blue', alpha=1.0/10)
            if (path_counter%200 == 0) :
                episode_counter = episode_counter + 1
                # save plot
                ggsave(p, save_name + str(episode_counter) + '.pdf', path = '/Users/titonka/ReplayWS/OpenReplay-F2021/logs/development/replayf2021/experiments/ReplayPaths/', dpi=300)
                p = ggplot() + ggtitle(title + str(episode_counter))
                # plot maze
                #maze_file = os.path.join(maze_file, 'mazes/M03.xml')
                #walls, feeders, start_positions = parse_maze(maze_file)
                p = p + geom_segment(aes(x='x1', y='y1', xend='x2', yend='y2'), data=walls, color='k')
                p = p + geom_point(aes(x='x', y='y'), data=feeders, color='r')
                p = p + coord_fixed(ratio = 1)

    # for step in paths_test:
    #     print(cells.loc[step])
    #     p = p + geom_path(aes(x='x', y='y'), data=cells.loc[step], color='blue', alpha=1.0/2)
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


if __name__ == '__main__':
    #folder_arg = os.path.join(sys.argv[1], '')
    #config_arg = None if len(sys.argv) < 3 else sys.argv[2]
    # folder_arg = 'D:/JavaWorkspaceSCS/Multiscale-F2019/experiments/BICY2020_modified/logs/experiment1-traces/'
    plot_replay_paths('episode', '/Users/titonka/ReplayWS/OpenReplay-F2021/experiments', '/Users/titonka/ReplayWS/OpenReplay-F2021/logs/development/replayf2021/experiments/Replay_Paths.csv', 'episode')
