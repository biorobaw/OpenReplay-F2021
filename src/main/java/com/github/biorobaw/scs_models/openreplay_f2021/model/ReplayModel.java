package com.github.biorobaw.scs_models.openreplay_f2021.model;


import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import com.github.biorobaw.scs.maze.Maze;

import com.github.biorobaw.scs.experiment.Experiment;
import com.github.biorobaw.scs.experiment.Subject;
import com.github.biorobaw.scs.robot.commands.TranslateXY;
import com.github.biorobaw.scs.robot.modules.FeederModule;
import com.github.biorobaw.scs.robot.modules.distance_sensing.DistanceSensingModule;
import com.github.biorobaw.scs.robot.modules.localization.SlamModule;
import com.github.biorobaw.scs.simulation.object.maze_elements.Feeder;
import com.github.biorobaw.scs.utils.Debug;
import com.github.biorobaw.scs.utils.files.BinaryFile;
import com.github.biorobaw.scs.utils.files.XML;
import com.github.biorobaw.scs.utils.math.DiscreteDistribution;
import com.github.biorobaw.scs.utils.math.Floats;
import com.github.biorobaw.scs_models.openreplay_f2021.gui.fx.GUI;
import com.github.biorobaw.scs_models.openreplay_f2021.model.modules.a_input.Affordances;
import com.github.biorobaw.scs_models.openreplay_f2021.model.modules.b_state.EligibilityTraces;
import com.github.biorobaw.scs_models.openreplay_f2021.model.modules.b_state.PlaceCellBins;
import com.github.biorobaw.scs_models.openreplay_f2021.model.modules.b_state.PlaceCells;
import com.github.biorobaw.scs_models.openreplay_f2021.model.modules.b_state.QTraces;
import com.github.biorobaw.scs_models.openreplay_f2021.model.modules.c_rl.ObstacleBiases;
import com.github.biorobaw.scs_models.openreplay_f2021.model.modules.d_action.MotionBias;
import com.github.biorobaw.scs_models.openreplay_f2021.model.modules.e_replay.ReplayMatrix;

public class ReplayModel extends Subject {

    // Model Parameters: RL
    public float[] v_traceDecay;
    public float[] q_traceDecay;

    public float[] v_learningRate;
    public float[] q_learningRate;
    public float[] replay_v_learningRate;
    public float[] replay_q_learningRate;

    public float discountFactor;
    public float foodReward;
    public float replay_discountFactor;
    public float replay_foodReward;

    public int num_layers;

    // Model Parameters: Action Space
    public int numActions;
    public float max_move_distance;
    public float certainty_threshold = 1;
    public float threshold_distance;

    // Model Parameters: Wall Bias option
    final int obstacle_bias_method; // 1 = wall reward, 2 = bias to closest elements


    // Model Variables: input
    public SlamModule slam;
    public FeederModule feederModule;
    public ObstacleBiases obstacle_biases;
    public DistanceSensingModule distance_sensors;


    // Model Variables: state
    public PlaceCells[] pcs;
    public PlaceCells[] pcs_replay;
    public PlaceCellBins[] pc_bins;
    public PlaceCellBins[] pc_bins_prime;

    public ReplayMatrix rmatrix;

    public EligibilityTraces[] vTraces;
    public QTraces[] qTraces;


    // Model Variables: RL
    public float[][] vTable;    // v[layer][pc]
    public float[][] vTableCopy; // a copy made to compare changes between start and end of episode
    public float episodeDeltaV; // max abs difference between vTable and vTableCopy
    public float[][][] qTable;  // q[layer][pc][action]
    public Float oldStateValue = null;
    public float[] qValues;
    public int num_episodes;

    // Model Variables: action selection
    public Affordances affordances;
    public MotionBias motionBias;   // Module to add bias to probabilities
    public float[] softmax;  // probability after applying softmax
    public float[] possible; // probability after applying affordances
    public float[] action_selection_probs;
    public int chosenAction;
    public boolean actionWasOptimal = false;
    public Random rn = new Random();
    public boolean replay_flag = true;
    int[] cell_activation_indexs;
    int replay_cycle = 0;
    public double theta = 0;
    public double dt = Math.PI / 4;
    public ArrayList<Feeder> feeders = new ArrayList<Feeder>();
    public double[] feeder_position = {.1, 1.2};

    // Replay Flags and Parameters
    ArrayList<Integer> replay_path_cells = new ArrayList<Integer>(); //Indexes of cells that are propagated during replay path
    public int replay_budget;
    public float replay_reward = 0;
    public int connection_threshold;
    public String propagation_type;
    public String starting_replay_location;
    public int max_replay_path;
    int freq_replay_matrix_clear = 1;
    public float replay_matrix_update_lr;

    public int num_writes = 0;
    int episode = 0;
    boolean append_flag = false;

    public long trial_cycle = 0;
    ArrayList<Long> num_trial_cycle = new ArrayList<Long>();

    // GUI
    GUI gui;


    public ReplayModel(XML xml) {
        super(xml);

        // ======== GENERAL PARAMETERS ===============
        numActions = xml.getIntAttribute("numActions");
        max_move_distance = xml.getFloatAttribute("robot_max_move_distance");
        num_episodes = xml.getIntAttribute("num_episodes");
        threshold_distance = xml.getFloatAttribute("threshold_distance");
        float mazeWidth = xml.getFloatAttribute("mazeWidth");
        float mazeHeight = xml.getFloatAttribute("mazeHeight");

        // ======== REPLAY PARAMETERS ===============
        replay_budget = xml.getIntAttribute("replay_budget");
        max_replay_path = xml.getIntAttribute("max_replay_path");
        connection_threshold = xml.getIntAttribute("connection_threshold");
        propagation_type = xml.getAttribute("propagation_type");
        starting_replay_location = xml.getAttribute("starting_replay_location");
        replay_matrix_update_lr = xml.getFloatAttribute("replay_matrix_update_lr");


        // ======== MODEL INPUT ======================
        // get robot modules
        slam = robot.getModule("slam");
        feederModule = robot.getModule("FeederModule");
        distance_sensors = robot.getModule("distance_sensors");
        //Create affordances / distance sensing module
        affordances = new Affordances(robot, numActions, threshold_distance);
        // Gets the Positions of feeders from the maze file

        Maze maze = Experiment.get().maze;
        var feeders_map = maze.feeders;
        feeders_map.forEach((k, v) -> feeders.add(v));
//		System.out.println(feeders.get(0).pos);


        // Joystick module for testing purposes:
        // joystick = new JoystickModule();

        // ======== MODEL STATE =======================

        // Initilizes Place Cells
        var pc_bin_size = xml.getFloatAttribute("pc_bin_size");
        pcs = PlaceCells.load(xml);
        num_layers = pcs.length;
        pcs_replay = new PlaceCells[num_layers];
        for (int i = 0; i < num_layers; i++) {
            pcs_replay[i] = pcs[i].copyPartial();
        }

        // Initilizes Place Cell Bins
        pc_bins = new PlaceCellBins[num_layers];
        pc_bins_prime = new PlaceCellBins[num_layers];
        for (int i = 0; i < num_layers; i++) {
            pc_bins[i] = new PlaceCellBins(pcs[i], pc_bin_size);
            pc_bins_prime[i] = new PlaceCellBins(pcs_replay[i], pc_bin_size);

        }
        // Initilizes Replay Matrix
        rmatrix = new ReplayMatrix(pcs);
        // ======== TRACES =============================

        v_traceDecay = xml.getFloatArrayAttribute("v_traces");
        q_traceDecay = xml.getFloatArrayAttribute("q_traces");

        vTraces = new EligibilityTraces[num_layers];
        qTraces = new QTraces[num_layers];

        // TODO : Why do we find the average?
        // need to find average number of active place cells:
        float average_active_pcs = 0;
        for (var bins : pc_bins) average_active_pcs += bins.averageBinSize;
        System.out.println("average active pcs: " + average_active_pcs);

        //  create traces
        for (int i = 0; i < num_layers; i++) {

            vTraces[i] = new EligibilityTraces(1, pcs[i].num_cells, v_traceDecay[i], 0.0001f);
            qTraces[i] = new QTraces(numActions, pcs[i].num_cells, q_traceDecay[i], 0.0001f);
        }

        // ======== REINFORCEMENT LEARNING ===========

        boolean load_model = xml.hasAttribute("load_model") && xml.getBooleanAttribute("load_model");
        // Parameters During Reward Trials
        v_learningRate = xml.getFloatArrayAttribute("v_learningRate");
        q_learningRate = xml.getFloatArrayAttribute("q_learningRate");
        discountFactor = xml.getFloatAttribute("discountFactor");
        foodReward = xml.getFloatAttribute("foodReward");

        // Parameters During Replay Propagations
        replay_v_learningRate = xml.getFloatArrayAttribute("replay_v_learningRate");
        replay_q_learningRate = xml.getFloatArrayAttribute("replay_q_learningRate");
        replay_discountFactor = xml.getFloatAttribute("replay_discountFactor");
        replay_foodReward = xml.getFloatAttribute("replay_foodReward");


        vTable = new float[num_layers][];
        vTableCopy = new float[num_layers][];
        qTable = new float[num_layers][][];
        qValues = new float[numActions];


        for (int i = 0; i < num_layers; i++) {

            if (!load_model) {
                vTable[i] = new float[pcs[i].num_cells];
                qTable[i] = new float[pcs[i].num_cells][numActions];

            } else {

                var e = Experiment.get();
                String logFolder = e.getGlobal("logPath").toString();
                String ratId = e.getGlobal("run_id").toString();
                String save_prefix = logFolder + "/r" + ratId + "-";
//				System.out.println(save_prefix);
                vTable[i] = BinaryFile.loadFloatVector(save_prefix + "V" + i + ".bin", true);
                qTable[i] = BinaryFile.loadMatrix(save_prefix + "Q" + i + ".bin", true);

            }

        }


        // ======== ACTION SELECTION =======================

        certainty_threshold = xml.getFloatAttribute("certainty_threshold");
        var wall_selection_weights = xml.getFloatArrayAttribute("wall_selection_weights");
        int numStartingPositions = Integer.parseInt(Experiment.get().getGlobal("numStartingPositions"));
        motionBias = new MotionBias(numActions, 2000 * numStartingPositions);
        softmax = new float[numActions];
        possible = new float[numActions];
        action_selection_probs = new float[numActions];


        obstacle_bias_method = xml.getIntAttribute("wall_bias_method");
        float wall_reached_distance = xml.getFloatAttribute("wall_reached_distance");
        float wall_reward = xml.getFloatAttribute("wall_reward");
        float bin_distance = obstacle_bias_method == 1 ? wall_reached_distance : xml.getFloatAttribute("wall_detection_distance");
        float bin_size = 0.1f;
        obstacle_biases = new ObstacleBiases(bin_size, bin_distance, wall_reached_distance, wall_reward, wall_selection_weights); // bin size, bin_distance, reward distance, reward value


        // ======== GUI ====================================
        gui = new GUI(this);

    }

    static public long cycles = 0;
    static final int num_tics = 10;
    static public long tics[] = new long[num_tics];
    static public float tocs[] = new float[num_tics];
    static public float averages[] = new float[num_tics];

    private void debug() {
//		var old_value = ((DisplaySwing)Experiment.get().display).setSync(true);
//		Experiment.get().display.updateData();
//		Experiment.get().display.repaint();
//		((DisplaySwing)Experiment.get().display).setSync(old_value);
//		SimulationControl.setPause(true);
    }

    @Override
    public long runModel() {
        cycles++;
        trial_cycle++;

        tics[tics.length - 1] = Debug.tic();

        tics[0] = Debug.tic();

        // get inputs
        var pos = slam.getPosition();
        var orientation = slam.getOrientation2D();
        float reward = feederModule.ate() ? foodReward : 0f;

        if (obstacle_bias_method == 1)
            if (reward == 0) {
                reward = obstacle_biases.getReward(pos);
                //			if (reward > 0) System.out.println(cycles + "Wall reward: " + reward);
            }

        // get alocentric distances from egocentric measures:
        float[] ego_distances = distance_sensors.getDistances();
        float[] distances = new float[numActions];
        int id0 = angle_to_index(orientation);
        for (int i = 0; i < numActions; i++) {
            distances[i] = ego_distances[(i + id0) % numActions];
        }

        tocs[0] = Debug.toc(tics[0]);

        tics[1] = Debug.tic();
        // calculate state
        float totalActivity = 0;
        for (int i = 0; i < num_layers; i++)
            totalActivity += pc_bins[i].activateBin((float) pos.getX(), (float) pos.getY());
        for (int i = 0; i < num_layers; i++) pc_bins[i].active_pcs.normalize(totalActivity);

        // Records Place Cell Activation
        for (int i = 0; i < num_layers; i++) pcs[i].activate((float) pos.getX(), (float) pos.getY());
        tocs[1] = Debug.toc(tics[1]);

        // Adds PC Bins to Replay Matrix
        rmatrix.setPcs_current(pcs);
        rmatrix.addPlaceCellBins(pc_bins[0].getActive_pcs((float) pos.getX(), (float) pos.getY()), 1);

        tics[2] = Debug.tic();
        // If not initial cycle, update state and action values
        if (oldStateValue != null) {

            // compute path matrix
            //rmatrix.setPcs_current(pcs);
            if (cycles > 2) {
                rmatrix.update(replay_matrix_update_lr);
            }
            rmatrix.addPlaceCellBins(pc_bins[0].getActive_pcs((float) pos.getX(), (float) pos.getY()), 0);


            // calculate bootstraps
            float bootstrap = reward;
            if (reward == 0) {
                // only calculate next state value if non terminal state
                float value = 0;
                for (int i = 0; i < num_layers; i++) {
                    var pcs = pc_bins[i].active_pcs;
                    for (int j = 0; j < pcs.num_cells; j++) {
                        value += vTable[i][pcs.ids[j]] * pcs.ns[j];
                    }
                }
                bootstrap += value * discountFactor;
            }

            // calculate rl error
            float error = bootstrap - oldStateValue;


            for (int i = 0; i < num_layers; i++) {
                // update V
                // v = v + error*learning_rate*trace
                if (actionWasOptimal || error > 0 || true) {
                    var traces = vTraces[i].traces[0];
                    for (var id : vTraces[i].non_zero[0]) {

                        vTable[i][id] += error * v_learningRate[i] * traces[id];
                    }
                }


                // update Q
                for (int j = 0; j < numActions; j++) {
                    var traces = qTraces[i].traces[j];
                    for (var id : qTraces[i].non_zero)
                        qTable[i][id][j] += error * q_learningRate[i] * traces[id];
                }
            }


        }
        tocs[2] = Debug.toc(tics[2]);

        // calculate V,Q
        tics[3] = Debug.tic();
        oldStateValue = 0f;
        qValues = new float[numActions];

        for (int i = 0; i < num_layers; i++) {
            var pcs = pc_bins[i].active_pcs;
            var ids = pcs.ids;

//			System.out.println(Arrays.toString(pcs.ns));
            for (int j = 0; j < pcs.num_cells; j++) {
                var activation = pcs.ns[j];
                oldStateValue += vTable[i][ids[j]] * activation;

                for (int k = 0; k < numActions; k++)
                    qValues[k] += qTable[i][ids[j]][k] * activation;
            }
        }

        tocs[3] = Debug.toc(tics[3]);

        // perform action selection
        tics[4] = Debug.tic();


        var aff_values = affordances.calculateAffordances(distances);
        float[] learning_dist;
        float[] optimal_action_dist;

        // METHOD 2, get soft max then calculate certainty:
        Floats.softmaxWithWeights(qValues, aff_values, softmax);
        float non_zero = Floats.sum(aff_values);
        float certainty = 1 - Floats.entropy(softmax, non_zero > 1 ? non_zero : 2f);

        // If Q policy is not certain enough, use bias, else, don't use it
        if (certainty < certainty_threshold) {

            // calculate motion bias
            var bias_motion = motionBias.calculateBias(chosenAction);

            // calculate obstacle bias if necessary
            if (obstacle_bias_method == 2) {
                var bias_obstacles = obstacle_biases.calculateBias(pos);
                Floats.mul(bias_motion, bias_obstacles, action_selection_probs);
            } else Floats.copy(bias_motion, action_selection_probs);


            // Combine bias, then add bias to softmax to get resulting probabilities
            addMultiplicativeBias(action_selection_probs, softmax, action_selection_probs);
        } else Floats.copy(softmax, action_selection_probs);

        learning_dist = softmax;
//		optimal_action_dist = softmax;
        optimal_action_dist = action_selection_probs;

        chosenAction = DiscreteDistribution.sample(action_selection_probs);
        actionWasOptimal = optimal_action_dist[chosenAction] == Floats.max(optimal_action_dist);

        tocs[4] = Debug.toc(tics[4]);

        // update traces
        tics[5] = Debug.tic();
        for (int i = 0; i < num_layers; i++) {
            var pcs = pc_bins[i].active_pcs;
            vTraces[i].update(pcs.ns, pcs.ids, 0);
            qTraces[i].update(pcs.ns, pcs.ids, chosenAction, learning_dist);

        }

        // perform action
        double tita = 2 * Math.PI / numActions * chosenAction;
        robot.getRobotProxy().send_command(new TranslateXY(max_move_distance * (float) Math.cos(tita), max_move_distance * (float) Math.sin(tita)));
        feederModule.eatAfterMotion();
        tocs[5] = Debug.toc(tics[5]);

        tocs[tocs.length - 1] = Debug.toc(tics[tocs.length - 1]);
        if (cycles == 1) {
            Floats.copy(tocs, averages);
        } else {
            Floats.mul(averages, 0.95f, averages);
            Floats.add(averages, Floats.mul(tocs, 0.05f), averages);
        }

        // Save the old state of the Place Cells
        rmatrix.setPcs_old(pcs);
        return 0;
    }

    @Override
    public void newEpisode() {
        episode += 1;
        super.newEpisode();
        motionBias.newEpisode();
        obstacle_biases.newEpisode();

        for (int i = 0; i < num_layers; i++) {
            vTraces[i].clear();
            qTraces[i].clear();
            pc_bins[i].clear();
        }

        oldStateValue = null;
        chosenAction = -1;
        actionWasOptimal = false;

        // copy state values:
        for (int i = 0; i < num_layers; i++)
            vTableCopy[i] = Floats.copy(vTable[i]);

    }

    @Override
    public void endEpisode() {
        super.endEpisode();

        episodeDeltaV = 0;
        for (int i = 0; i < num_layers; i++) {
            var dif = Floats.sub(vTable[i], vTableCopy[i]);
            episodeDeltaV = Math.max(episodeDeltaV, Floats.max(Floats.abs(dif, dif)));
        }

        // Sets flag for new replay path
        boolean new_replay_path = true;

        // Runs replay propagations
        int propagation_counter = 0;
        // Gets an array of indexes of PCs that have a number of connections higher than a threshold
        rmatrix.getConnectedPC(connection_threshold);

        // Preforms replay propigations for off line learning
        while (propagation_counter < replay_budget || !new_replay_path) {
            new_replay_path = replayPropagation(new_replay_path);
            propagation_counter++;
        }

        if (episode != 0) {
            append_flag = true;
        }

        // Clears Replay Matrix periodically
        if (episode % freq_replay_matrix_clear *4 == 0 && episode<num_episodes-1 ) {
            rmatrix.clearRMatrix();
        }

        num_trial_cycle.add(trial_cycle);
        trial_cycle = 0;

    }

    @Override
    public void newTrial() {
        super.newTrial();
        obstacle_biases.newTrial();
        motionBias.newTrial();

    }

    @Override
    public void endTrial() {
        super.endTrial();

    }

    @Override
    public void newExperiment() {
        super.newExperiment();

    }

    @Override
    public void endExperiment() {
        var ex = Experiment.get();
        String logFolder = ex.getGlobal("logPath").toString();
        String ratId = ex.getGlobal("run_id").toString();

        //create folders
        String prefix = logFolder + "/r" + ratId + "-";

        super.endExperiment();

        // Writes the connections recorded in the replay matrix for post processing
        rmatrix.writeRMatrix(episode, append_flag);

    }

    // Creates a single replay propigation event
    public boolean replayPropagation(boolean new_path_flag) {

        // If starting a new path
        if (new_path_flag) {

            // Clears PC_bins to record State Action sequence
            for (int j = 0; j < num_layers; j++) {
                pc_bins[j].clear();
            }
            // Clears array holding PC path
            replay_path_cells.clear();

            int start_pc_index = -1;
            // Randomly chooses a starting position for replay event
            if (starting_replay_location.equals("random")){
                start_pc_index = rmatrix.possible_starting_PCs.get(rn.nextInt(rmatrix.possible_starting_PCs.size()));
            }
            // TODO: add a stochastic sampling for selecting the starting PC for replay propagation
            else if(starting_replay_location.equals("stochastic")){
                start_pc_index = rmatrix.possible_starting_PCs.get(rn.nextInt(rmatrix.possible_starting_PCs.size()));
            }


            // Gets the next PC either through stochastic or Max connection
            cell_activation_indexs = rmatrix.replayEvent(start_pc_index, propagation_type);
            replay_path_cells.add(cell_activation_indexs[0]);

            // Sets flags to correct state to continue a propagation
            replay_flag = true;
            new_path_flag = false;
            replay_cycle = 0;
        }


        // Detects a Cycle formed in the Replay path or if exceeds max allotment
        if (replay_path_cells.contains(cell_activation_indexs[1]) ||
                cell_activation_indexs[1] == -1 ||
                replay_cycle > max_replay_path) {
            replay_flag = false;
            new_path_flag = true;
            writeReplayEvent(replay_path_cells);

        } else {
            replay_path_cells.add(cell_activation_indexs[1]);
        }

        // Replay RL Algorithm
        // If not at a terminal state
        if (replay_flag) {
            // calculates action and position
            var x1 = pcs[0].xs[cell_activation_indexs[0]];
            var y1 = pcs[0].ys[cell_activation_indexs[0]];
            var x2 = pcs[0].xs[cell_activation_indexs[1]];
            var y2 = pcs[0].ys[cell_activation_indexs[1]];

            theta = Math.atan2((y2 - y1), (x2 - x1));

            var action_selected = (int) Math.round((theta / dt)) % numActions;
            if (action_selected < 0) {
                action_selected += numActions;
            }

            // Calculate next postion after action is preformed
            double tita = 2 * Math.PI / numActions * action_selected;
            var x_prime = x1 + max_move_distance * Math.cos(tita);
            var y_prime = y1 + max_move_distance * Math.sin(tita);


            // TODO: Move into a function
            // calculate activity for current state
            float totalActivity = 0;
            for (int i = 0; i < num_layers; i++)
                totalActivity += pc_bins[i].activateBin((float) x1, (float) y1);
            for (int i = 0; i < num_layers; i++) pc_bins[i].active_pcs.normalize(totalActivity);

            // calculate activity for current state
            float totalActivity_prime = 0;
            for (int i = 0; i < num_layers; i++)
                totalActivity_prime += pc_bins_prime[i].activateBin((float) x_prime, (float) y_prime);
            for (int i = 0; i < num_layers; i++) pc_bins_prime[i].active_pcs.normalize(totalActivity_prime);


            //Calculates bootstrap
            replay_reward = getReward(x_prime, y_prime);
            float bootstrap = replay_reward != 0 ?
                    replay_reward : calculateValueFunction(pc_bins_prime) * discountFactor;

            if (replay_reward != 0) {
                new_path_flag = true;
                writeReplayEvent(replay_path_cells);
            }
            // Calculates the bootstrap error (or RL error)
            // TODO: consider recording error for Starting replay location selection (in model too)
            var rl_error = bootstrap - calculateValueFunction(pc_bins);

            // Calculates policy for current state
            var policy_gradient = calculateQValue(pc_bins);
            Floats.negate(Floats.softmax(policy_gradient, policy_gradient), policy_gradient);
            policy_gradient[action_selected]++;


            // Update V and Q
            // TODO: read importance sampling and see if applicable
            for (int i = 0; i < num_layers; i++) {
                var pcs = pc_bins[i].getActive_pcs();
                for (int j = 0; j < pcs.num_cells; j++) {
                    // update V
                    var pc_index = pcs.ids[j];
                    vTable[i][pc_index] += rl_error * replay_v_learningRate[i] * pcs.ns[j];

                    // Update Q
                    for (int k = 0; k < numActions; k++) {
                        // Zik
                        var replay_trace = pcs.ns[j] * policy_gradient[k];
                        qTable[i][pc_index][k] += rl_error * replay_q_learningRate[i] * replay_trace;
                    }
                }
            }

            replay_cycle++;

            //Shift indexes
            cell_activation_indexs = rmatrix.replayEvent(cell_activation_indexs[1],propagation_type);

            // If Replay Path leads to the goal
            if (replay_reward == 1) {
                replay_flag = false;
                new_path_flag = true;
                writeReplayEvent(replay_path_cells);
            }

        }
        // return flag for new path
        return new_path_flag;

    }

    // Converts Directional angle to nearest action direction
    int angle_to_index(float angle) {
        double dtita = Math.PI * 2 / numActions;
        var res = (int) Math.round(angle / dtita) % numActions;
        return res < 0 ? res + numActions : res;
    }

    /**
     * @param x x-coordinate of agent
     * @param y y-coordinate of agent
     * @return int the value of the reward if agent gets within range of feeder
     */
    public int getReward(double x, double y) {
        for (var f : feeders) {
            var diff_x = f.pos.getX() - x;
            var diff_y = f.pos.getY() - y;
            var dist_feeder = Math.sqrt(Math.pow((diff_x), 2) + Math.pow((diff_y), 2));
            if (dist_feeder <= threshold_distance) {
                //System.out.println("Replay Path found feeder");
                return 1;
            }
        }
        return 0;
    }

    /**
     * @param pc_bins
     * @return
     */
    private float calculateValueFunction(PlaceCellBins[] pc_bins) {
        float value = 0;
        for (int i = 0; i < num_layers; i++) {
            var pcs = pc_bins[i].getActive_pcs();
            for (int j = 0; j < pcs.num_cells; j++) {
                value += vTable[i][pcs.ids[j]] * pcs.ns[j];
            }
        }
        return value;
    }

    /**
     * @param pc_bins
     * @return
     */
    private float[] calculateQValue(PlaceCellBins[] pc_bins) {
        var qValues = new float[numActions];
        for (int i = 0; i < num_layers; i++) {
            var pcs = pc_bins[i].getActive_pcs();
            var ids = pcs.ids;

            for (int j = 0; j < pcs.num_cells; j++) {
                var activation = pcs.ns[j];
                for (int k = 0; k < numActions; k++)
                    qValues[k] += qTable[i][ids[j]][k] * activation;
            }
        }
        return qValues;
    }

    /**
     * @param bias
     * @param input
     * @param output
     */
    void addMultiplicativeBias(float[] bias, float[] input, float[] output) {
        output = Floats.mul(bias, input, output);
        var sum = Floats.sum(output);

        if (sum != 0) Floats.div(output, sum, output);
        else {
            System.err.println("WARNING: Probability sum is 0, setting uniform distribution (MulytiscaleModel.java)");
            for (int i = 0; i < numActions; i++) output[i] = 1 / numActions;
        }
    }

    /**
     * @param bias
     * @param input
     * @return
     */
    float[] addMultiplicativeBias(float[] bias, float[] input) {
        var output = Floats.mul(bias, input);
        var sum = Floats.sum(output);

        if (sum != 0) Floats.div(output, sum, output);
        else {
            System.err.println("WARNING: Probability sum is 0, setting uniform distribution (MulytiscaleModel.java)");
            Floats.uniform(output);
        }
        return output;
    }

    /**
     * Writes a path generated during replay to a txt file
     * @param path
     */
    // TODO: Convert to a Bin File
    public void writeReplayEvent(ArrayList<Integer> path) {
        var ex = Experiment.get();
        String logFolder = ex.getGlobal("logPath").toString();
        String ratId = ex.getGlobal("run_id").toString();

        //create folders
        String prefix = logFolder + "/r" + ratId + "-";

        // Create file name
        // save cycles per episode
        var file = prefix + "Replay_Paths.csv";


        FileWriter writer = null;
        if (num_writes != 0) {
            try {
                writer = new FileWriter(file, true);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            try {
                writer = new FileWriter(file);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        for (int i = 0; i < path.size(); i++) {
            try {
                if (i != path.size() - 1) {
                    writer.append(String.valueOf(path.get(i)) + ",");
                } else {
                    writer.append(String.valueOf(path.get(i)));
                }


            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {

            writer.append("\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        num_writes++;
    }


}
