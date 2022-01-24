import scikit_posthocs as sp
from plotnine import *
import os
import numpy as np
from data_loader import *
from pythonUtils.BinaryFiles import *
import csv
import sys

import xml.etree.ElementTree as ET
from functools import *
import pandas as pd
from os import path
import glob
from matplotlib import pyplot as plt
import matplotlib.collections

# Gets the root and base of the project directory
basepath = path.dirname(__file__)
rootpath = path.abspath(path.join(basepath, "..", ".."))
experiment_path = path.abspath(path.join(rootpath,"experiments"))

# File containing the simulated replay paths
replay_file = path.abspath(path.join(rootpath,"logs/development/replayf2021/experiments/Replay_Paths.csv"))

# Directory to save the Replay matrices figures
replayMatrix_dir = path.abspath(path.join(rootpath,"logs/development/replayf2021/experiments/ReplayMatrices"))
# Gets all the csv files of the Replay Matrices
replay_matrices_files = glob.glob(os.path.join(replayMatrix_dir, "*.csv"))
# File containting the Replay matrices
replay_matrix_file = path.abspath(path.join(rootpath,"logs/development/replayf2021/experiments/Replay_matrix.csv"))
# File containing all the number of cycles count for all the trails
path_to_number_cycles = path.abspath((path.join(rootpath,"logs/development/replayf2021/experiments/")))

# Number of episodes
num_trial = 800

# Number of reply per
replay_per = 200



def setMaze(file):
    global maze_file
    maze_file = file

def setPC(file):
    global pc_file
    pc_file = file

def parse_all_cells(file):
    return pd.read_csv(file, usecols=['x','y','r'])

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

def plot_replay_paths(save_name, num_replay_to_print):

    p = ggplot() + ggtitle(save_name + '1')

    # plot maze
    walls, feeders, start_positions = parse_maze(maze_file)
    p = p + geom_segment(aes(x='x1', y='y1', xend='x2', yend='y2'), data=walls, color='k')
    p = p + geom_point(aes(x='x', y='y'), data=feeders, color='r')
    p = p + coord_fixed(ratio = 1)
    cells = pd.read_csv(pc_file)
    with open(replay_file) as path_file:
        paths = csv.reader(path_file, delimiter = '\n')
        episode_counter = num_trial + 1
        path_counter = 0
        for replay_path in reversed(list(paths)):
            path_counter = path_counter + 1
            replay_event = replay_path[0].split(',')

            if len(replay_event) > 3:
                replay_event = map(int, replay_event)
                p = p + geom_path(aes(x='x', y='y'), data=cells.loc[replay_event], color='blue', alpha=0.5)
            if (path_counter % replay_per == 0) :

                episode_counter = episode_counter - 1
                # save plot
                replay_path = path.join(rootpath,"logs/development/replayf2021/experiments/ReplayPaths/")
                ggsave(p, save_name + str(episode_counter) + '.png', path = replay_path, dpi=300)
                p = ggplot() + ggtitle(save_name + str(episode_counter))
                # plot maze
                p = p + geom_segment(aes(x='x1', y='y1', xend='x2', yend='y2'), data=walls, color='k')
                p = p + geom_point(aes(x='x', y='y'), data=feeders, color='r')
                p = p + coord_fixed(ratio = 1)
            if episode_counter <= num_trial - num_replay_to_print + 1 :
                break

def plot_replay_matrix(save_name, experiment_file, replay_matrix_file, root_path):

    # first plot paths and then plot maze
    # create plot
    replay_path = path.join(rootpath,"logs/development/replayf2021/experiments/ReplayMatrixPlots/")

    p = ggplot() + ggtitle(save_name)

    # plot maze
    maze_file = os.path.join(experiment_file, 'mazes/M8.xml')
    walls, feeders, start_positions = parse_maze(maze_file)
    p = p + geom_segment(aes(x='x1', y='y1', xend='x2', yend='y2'), data=walls, color='k',alpha=1.0/5)
    p = p + geom_point(aes(x='x', y='y'), data=feeders, color='r')
    p = p + coord_fixed(ratio = 1)
    pc_file = os.path.join(experiment_file, 'pc_layers/largeTest/u20_25.csv')
    cells = parse_all_cells(pc_file)
    p = p + geom_point(aes(x='x', y='y', size = 'r'), data=cells, color='b', alpha=1.0/10)

    with open(replay_matrix_file) as path_matrix:
        cell_weights = csv.reader(path_matrix, delimiter = '\n')
        cell_data = []
        for cell in cell_weights:
            cell_data.append(cell)
        for cell_index in range(len(cell_data)):
            weights = cell_data[cell_index][0].split(',')
            # Find strongest connection
            max_weight = 0
            max_weight_index = None
            for i in range(len(weights) - 1):
                if float(weights[i]) > max_weight:
                    max_weight = float(weights[i])
                    max_weight_index = i
            # Plot a connection if it exsists
            if max_weight_index != None:
                if cells.values[cell_index][0] != cells.values[max_weight_index][0] or cells.values[cell_index][1] != cells.values[max_weight_index][1] :
                    p = p + geom_segment(aes(x=cells.values[cell_index][0], y=cells.values[cell_index][1], xend=cells.values[max_weight_index][0], yend=cells.values[max_weight_index][1]), color='blue', alpha=1.0/2)
                else :
                    p = p + geom_point(aes(x=cells.values[cell_index][0], y=cells.values[cell_index][1], size = cells.values[cell_index][2]), color='g', alpha=1.0)
            else :
                p = p + geom_point(aes(x=cells.values[cell_index][0], y=cells.values[cell_index][1], size = cells.values[cell_index][2]), color='r', alpha=1.0)
        # Save Plot
        ggsave(p, save_name + '.png', path = replay_path, dpi=300)


def plot_replay_matrix2(save_name, replay_matrix_file):

    # Defines map for labels
    my_labels = {'sc' : 'Stongest Connection'}
    replay_path = path.join(rootpath,"logs/development/replayf2021/experiments/ReplayMatrixPlots/")

    # Plots the maze and Place Cells
    #maze_file = os.path.join(experiment_path, 'mazes/ReplayTest/M200.xml')
    walls, feeders, start_positions = parse_maze(maze_file)
    #pc_file = os.path.join(experiment_path, 'pc_layers/ReplayTest/u16_15.csv')
    cells = parse_all_cells(pc_file)
    fig, ax = plt.subplots()
    # print(feeders)
    x = walls.loc[:, walls.columns[::2]]
    y = walls.loc[:, walls.columns[1::2]]
    for i in range(len(walls)):
        ax.plot(x.iloc[i,:], y.iloc[i,:], color='black', alpha=1.0)
    x = feeders['x']
    y = feeders['y']
    for i in range(len(x)):
        ax.plot(x,y,'rx', label='Goal Location')
    x = start_positions['x']
    y = start_positions['y']
    for i in range(len(x)):
        ax.plot(x,y,'bo', label='Start Location')

    x = cells['x']
    y = cells['y']
    r = cells['r']
    xy = tuple(zip(x,y))
    coll = matplotlib.collections.EllipseCollection(r, r,
                                                    np.zeros_like(r),
                                                    offsets=xy, units='x',
                                                    transOffset=ax.transData,
                                                    color= 'b',
                                                    alpha= 0,
                                                    label = 'Place Cells')
    ax.add_collection(coll)

    # Plots the Replay Matrix
    self_connected_cells = []
    self_connected_cells_r = []
    nonconnected_cells = []
    nonconnected_cells_r = []
    with open(replay_matrix_file) as path_matrix:
        cell_weights = csv.reader(path_matrix, delimiter = '\n')
        cell_data = []
        for cell in cell_weights:
            cell_data.append(cell)
        for cell_index in range(len(cell_data)):
            weights = cell_data[cell_index][0].split(',')
            # Find strongest connection
            max_weight = 0
            max_weight_index = None
            for i in range(len(weights) - 1):
                if float(weights[i]) > max_weight:
                    max_weight = float(weights[i])
                    max_weight_index = i
            # Plot a connection if it exsists
            if max_weight_index != None:
                # Connected to another Place Cell
                if cells.values[cell_index][0] != cells.values[max_weight_index][0] or cells.values[cell_index][1] != cells.values[max_weight_index][1] :
                    x=cells.values[cell_index][0]
                    y=cells.values[cell_index][1]
                    dx=cells.values[max_weight_index][0] - x
                    dy=cells.values[max_weight_index][1] - y
                    ax.arrow(x,y,dx,dy,width=.01, length_includes_head = True, label= my_labels['sc'],color = 'b',alpha=.5)
                    my_labels['sc'] = '_nolegend_'

                # Connected to itself
                else:
                    self_connected_cells.append((cells.values[cell_index][0], cells.values[cell_index][1]))
                    self_connected_cells_r.append(cells.values[cell_index][2])

            # Has no connection
            else:
                nonconnected_cells.append((cells.values[cell_index][0], cells.values[cell_index][1]))
                nonconnected_cells_r.append(cells.values[cell_index][2])

        # Save Plot
        coll = matplotlib.collections.EllipseCollection(nonconnected_cells_r, nonconnected_cells_r,
                                                        np.zeros_like(nonconnected_cells_r),
                                                        offsets=nonconnected_cells, units='x',
                                                        transOffset=ax.transData,
                                                        color= 'r',
                                                        alpha= .5,
                                                        label = 'Place Cells')
        ax.add_collection(coll)

        coll = matplotlib.collections.EllipseCollection(self_connected_cells_r, self_connected_cells_r,
                                                        np.zeros_like(self_connected_cells_r),
                                                        offsets=self_connected_cells, units='x',
                                                        transOffset=ax.transData,
                                                        color= 'g',
                                                        alpha= .5,
                                                        label = 'Place Cells')
        ax.add_collection(coll)
        lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        tit = fig.suptitle(save_name, fontsize=14)
        # plt.show()
        fig.savefig(replay_path+save_name + '.png',
                    dpi=300,
                    format='png',
                    bbox_extra_artists=(lg,tit),
                    bbox_inches = 'tight')
        plt.close(fig)

def plot_paths(title, config_folder, trial_id,  save_name):

    # Directory of where we save the path plot figures
    trial_path_dir = path.abspath((path.join(rootpath,"logs/development/replayf2021/experiments/TrialPaths/")))

    # first plot paths and then plot maze
    # create plot
    p = ggplot() + ggtitle(title)

    # plot paths
    num_rats = 3
    num_starting_pos = 4
    for id in range(num_rats):
        # check if file exists
        file_name = os.path.join(config_folder, 'r' + str(trial_id) + '-paths.bin')
        print(file_name)
        if not os.path.exists(file_name):
            continue

        # print(file_name, save_name)
        # open file and plot each path
        with open( file_name, 'rb') as file:
            for i in range(num_starting_pos):
                xy_df = pd.DataFrame(data=load_float_vector(file).reshape((-1, 2)), columns=["x", "y"])
                p = p + geom_path(aes(x='x', y='y'), data=xy_df, color='blue', alpha=1.0/num_rats)


    # plot maze
    maze_file = os.path.join(experiment_path, 'mazes/ReplayTest/M200.xml')
    walls, feeders, start_positions = parse_maze(maze_file)
    p = p + geom_segment(aes(x='x1', y='y1', xend='x2', yend='y2'), data=walls, color='k')
    p = p + geom_point(aes(x='x', y='y'), data=feeders, color='r')
    p = p + coord_fixed(ratio = 1)

    save_file = os.path.join(trial_path_dir,save_name)
    # save plot
    ggsave(p, save_file, dpi=300)


def cycle_plotter(path_to_number_cycles):

    cycle_file = path.join(path_to_number_cycles,"Number_Cycles.csv")
    save_path =  path.join(path_to_number_cycles,"NumberCyclePlots/Number_Cycles_Plot.png")
    data = []
    with open(cycle_file) as path_file:
        cycles = csv.reader(path_file)
        data = next(cycles)
        data.pop()
        for i in range(len(data)):
            data[i]= int(data[i])
        Mean = np.mean(data)
        m = min(data)
        ma = max(data)
        s = np.std(data)
        fig = plt.figure()

        plt.boxplot(data,showfliers=False)
        plt.text(1.2, 60, 'Mean: %.2s cycles\n STD: %.2f \n Min: %s \n Max: %s \n' % (Mean,s,m,ma))
        #plt.show()
        plt.savefig(save_path, bbox_inches='tight')



if __name__ == '__main__':

    # Parameters to access the
    experiments_arg = os.path.join(sys.argv[1], '')
    experiment_num = int(sys.argv[2])


    experiment_file = path.abspath(path.join(rootpath,experiments_arg))
    print(experiment_file)
    data = pd.read_csv(experiment_file, sep='\t', index_col=False)

    trial_id = data.iloc[experiment_num]['run_id']

    # Path to the Config file to get the trial path bin file
    config_dir = path.abspath(path.join(rootpath,"logs/development/replayf2021/configs/" + data.iloc[experiment_num]['config']))

    # Gets the path to the the Maze file
    setMaze(path.abspath(path.join(rootpath,data.iloc[experiment_num]['mazeFile'])))

    # Gets the path to the the PC-layer file
    setPC(path.abspath(path.join(rootpath,data.iloc[experiment_num]['pc_files'])))

    # Plots boxplots for number of cycles per trail
    cycle_plotter(path_to_number_cycles)

    # Plots the last path taken from each starting position during a trail
    plot_paths("Last Trial Paths", config_dir, trial_id, "TrailPaths.png")

    # Plots the replay matix
    i = 0
    for matrix in replay_matrices_files:
        i+=1
        # if i%2 == 0 :
        #     plot_replay_matrix2('Marix' + str(i), experiment_path, f, rootpath)
        plot_replay_matrix2('Marix' + str(i), matrix)

    # Plots replay paths
    plot_replay_paths('episode', 10)

