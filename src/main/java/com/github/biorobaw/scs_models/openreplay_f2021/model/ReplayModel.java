package com.github.biorobaw.scs_models.openreplay_f2021.model;




import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.FileSystemNotFoundException;
import java.util.*;

import com.github.biorobaw.scs.maze.Maze;

import com.github.biorobaw.scs.experiment.Experiment;
import com.github.biorobaw.scs.experiment.Subject;
import com.github.biorobaw.scs.gui.Display;
import com.github.biorobaw.scs.robot.commands.TranslateXY;
import com.github.biorobaw.scs.robot.modules.FeederModule;
import com.github.biorobaw.scs.robot.modules.distance_sensing.DistanceSensingModule;
import com.github.biorobaw.scs.robot.modules.localization.SlamModule;
import com.github.biorobaw.scs.simulation.SimulationControl;
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
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;

public class ReplayModel extends Subject{
	
	// Model Parameters: RL
	public float[] v_traceDecay;
	public float[] q_traceDecay;
	public float[] v_learningRate;
	public float[] q_learningRate;
	public float discountFactor;
	public float foodReward;
	public int num_layers;

	// Model Parameters: Action Space
	public int numActions;
	public float certainty_threshold = 1;
	
	// Model Parameters: Wall Bias option
	final int obstacle_bias_method; // 1 = wall reward, 2 = bias to closest elements

	
	// Model Variables: input
	public SlamModule slam;
	public FeederModule feederModule;
	public ObstacleBiases obstacle_biases;
	public DistanceSensingModule distance_sensors;
	
	
	// Model Variables: state
	public PlaceCells[] pcs;
	public PlaceCellBins[] pc_bins;

	public ReplayMatrix rmatrix;

	public EligibilityTraces[] vTraces;
	public QTraces[] qTraces;



	// Model Variables: RL
	public float[][] vTable;	// v[layer][pc]
	public float[][] vTableCopy; // a copy made to compare changes between start and end of episode
	public float     episodeDeltaV; // max abs difference between vTable and vTableCopy
	public float[][][] qTable;  // q[layer][pc][action]
	public Float oldStateValue = null;
	public float[] qValues;
	
	// Model Variables: action selection
	public Affordances affordances;
	public MotionBias motionBias;   // Module to add bias to probabilities
	public float[] softmax;  // probability after applying softmax
	public float[] possible; // probability after applying affordances
	public float[] action_selection_probs;
	public int chosenAction;
	public boolean actionWasOptimal = false;
	public PlaceCells old_active_bin = null;
	public Random rn = new Random();
	public boolean replay_flag = true;
	int[] cell_activation_indexs;
	int replay_cycle = 0;
	public double theta=0;
	public double dt = Math.PI/4;
	public ArrayList<Feeder> feeders = new ArrayList<Feeder>();
	public double[] feeder_position = {.1,1.2};

	// Replay Flags and Parameters
	boolean record_trail_paths = false;
	int freq_replay_matrix = 100;
	int freq_replay_matrix_writes = 100;
	public int num_replay = 200;
	public int num_writes = 0;
	int episode = 0;

	public long trial_cycle = 0;
	ArrayList<Long> num_trial_cycle = new ArrayList<Long>();
	
	// GUI
	GUI gui;


	public ReplayModel(XML xml) {
		super(xml);
		
		// ======== GENERAL PARAMETERS ===============

		numActions = xml.getIntAttribute("numActions");
		float mazeWidth = xml.getFloatAttribute("mazeWidth");
		float mazeHeight = xml.getFloatAttribute("mazeHeight");


		// ======== MODEL INPUT ======================
		
		// get robot modules
		slam = robot.getModule("slam");
		feederModule = robot.getModule("FeederModule");
		distance_sensors = robot.getModule("distance_sensors");
		//Create affordances / distance sensing module
		affordances = new Affordances( robot, numActions, 0.1f);
		// Gets the Positions of feeders from the maze file

		Maze maze = Experiment.get().maze;
		var feeders_map = maze.feeders;
		feeders_map.forEach((k,v) -> feeders.add(v));
//		System.out.println(feeders.get(0).pos);


		// Joystick module for testing purposes:
		// joystick = new JoystickModule();
		
		// ======== MODEL STATE =======================
		
		// Initilizes Place Cells
		var pc_bin_size  = xml.getFloatAttribute("pc_bin_size");
		pcs = PlaceCells.load(xml);	
		num_layers = pcs.length;

		// Initilizes Place Cell Bins
		pc_bins = new PlaceCellBins[num_layers];
		for(int i=0; i<num_layers; i++)
			pc_bins[i] = new PlaceCellBins(pcs[i], pc_bin_size);
		
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
		for(int i=0; i<num_layers; i++) {
			
			vTraces[i] = new EligibilityTraces(1, pcs[i].num_cells, v_traceDecay[i], 0.0001f);
			qTraces[i] = new QTraces(numActions, pcs[i].num_cells, q_traceDecay[i], 0.0001f);
		}
		
		// ======== REINFORCEMENT LEARNING ===========
		
		boolean load_model	= xml.hasAttribute("load_model") && xml.getBooleanAttribute("load_model");
		v_learningRate 		= xml.getFloatArrayAttribute("v_learningRate");
		q_learningRate 		= xml.getFloatArrayAttribute("q_learningRate");
		discountFactor 		= xml.getFloatAttribute("discountFactor");
		foodReward 	   		= xml.getFloatAttribute("foodReward");
		
		vTable = new float[num_layers][];
		vTableCopy = new float[num_layers][];
		qTable = new float[num_layers][][];
		qValues = new float[numActions];
		
		
		for(int i=0; i<num_layers; i++) {
			
			if(!load_model) {
				vTable[i] = new float[pcs[i].num_cells];
				qTable[i] = new float[pcs[i].num_cells][numActions];
				
			} else {
				
				var e = Experiment.get();
				String logFolder   = e.getGlobal("logPath").toString();
				String ratId	   = e.getGlobal("run_id").toString();
				String save_prefix = logFolder  +"/r" + ratId + "-";
//				System.out.println(save_prefix);
				vTable[i] = BinaryFile.loadFloatVector(save_prefix + "V" + i+ ".bin", true);
				qTable[i] = BinaryFile.loadMatrix(save_prefix + "Q" + i+ ".bin", true);				
				
			}
			
		}
		
		
		// ======== ACTION SELECTION =======================

		certainty_threshold 		= xml.getFloatAttribute("certainty_threshold");
		var wall_selection_weights 	= xml.getFloatArrayAttribute("wall_selection_weights");
		int numStartingPositions 	= Integer.parseInt(Experiment.get().getGlobal("numStartingPositions"));
		motionBias 				= new MotionBias(numActions, 2000*numStartingPositions);
		softmax 				= new float[numActions];
		possible 				= new float[numActions];
		action_selection_probs 	= new float[numActions];

		
		
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
	static public long  tics[]= new long[num_tics];
	static public float  tocs[]= new float[num_tics];
	static public float averages[]=new float[num_tics];
	
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
		
		tics[tics.length-1] = Debug.tic();
		
		
		tics[0] = Debug.tic();
		// get inputs
		var pos = slam.getPosition();
		var orientation = slam.getOrientation2D();
		float reward = feederModule.ate() ? foodReward : 0f;


		
		if(obstacle_bias_method == 1)
			if(reward == 0) {
				reward = obstacle_biases.getReward(pos);
	//			if (reward > 0) System.out.println(cycles + "Wall reward: " + reward);
			}
		
		// get alocentric distances from egocentric measures:
		float[] ego_distances = distance_sensors.getDistances();
		float[] distances = new float[numActions];
		int id0 = angle_to_index(orientation);
		for(int i=0; i<numActions; i++) {
			distances[i] = ego_distances[(i + id0) % numActions];
		}
		
		tocs[0] = Debug.toc(tics[0]);
		
		tics[1] = Debug.tic();
		// calculate state
		float totalActivity =0;
		for(int i=0; i<num_layers; i++) 
			totalActivity+=pc_bins[i].activateBin((float)pos.getX(), (float)pos.getY());
		for(int i=0; i<num_layers; i++) pc_bins[i].active_pcs.normalize(totalActivity);

		// Records Place Cell Activation
		for(int i=0; i<num_layers; i++) pcs[i].activate((float)pos.getX(), (float)pos.getY());
		tocs[1] = Debug.toc(tics[1]);

		// Adds PC Bins to Replay Matrix
		rmatrix.setPcs_current(pcs);
		rmatrix.addPlaceCellBins(pc_bins[0].getActive_pcs((float)pos.getX(), (float)pos.getY()), 1);

		tics[2] = Debug.tic();
		// If not initial cycle, update state and action values
		if(oldStateValue!=null) {

			// compute path matrix
			//rmatrix.setPcs_current(pcs);
			if(cycles>2){
				rmatrix.update();

			}
			rmatrix.addPlaceCellBins(pc_bins[0].getActive_pcs((float)pos.getX(), (float)pos.getY()), 0);



			// calculate bootstraps
			float bootstrap = reward;
			if(reward==0 ) {
				// only calculate next state value if non terminal state
				float value = 0;
				for(int i=0; i<num_layers; i++) {
					var pcs = pc_bins[i].active_pcs;
					for(int j=0; j<pcs.num_cells; j++ ) {
						value+= vTable[i][pcs.ids[j]]*pcs.ns[j];
					}
				}
				bootstrap+= value*discountFactor;
			}
			
			// calculate rl error
			float error = bootstrap - oldStateValue;
			

			
			for(int i=0; i<num_layers; i++) {
				// update V
				// v = v + error*learning_rate*trace
				if(actionWasOptimal || error >0  || true) {
					var traces = vTraces[i].traces[0];
					for(var id : vTraces[i].non_zero[0]) {

						vTable[i][id]+=  error*v_learningRate[i]*traces[id];
					}
				}
			
				
				// update Q
				for(int j=0; j<numActions; j++) {
					var traces = qTraces[i].traces[j];
					for(var id : qTraces[i].non_zero)
						qTable[i][id][j] += error*q_learningRate[i]*traces[id];
				}
			}
	

			
		}
		tocs[2] = Debug.toc(tics[2]);
		
		
		// calculate V,Q
		tics[3] = Debug.tic();
		oldStateValue = 0f;
		qValues = new float[numActions];
		

		
		for(int i=0; i<num_layers; i++) {
			var pcs = pc_bins[i].active_pcs;
			var ids = pcs.ids;
			
//			System.out.println(Arrays.toString(pcs.ns));
			for(int j=0; j<pcs.num_cells; j++) {
				var activation = pcs.ns[j];
				oldStateValue+= vTable[i][ids[j]]*activation;

				for(int k=0; k<numActions; k++)
					qValues[k]+= qTable[i][ids[j]][k]*activation;	
			}
		}
		

		
		tocs[3] = Debug.toc(tics[3]);
		
		// perform action selection
		tics[4] = Debug.tic();
//		System.out.println(Arrays.toString(qValues));
		

		var aff_values = affordances.calculateAffordances(distances);
		float[] learning_dist;
		float[] optimal_action_dist;
		

			
		// METHOD 2, get soft max then calculate certainty:
		Floats.softmaxWithWeights(qValues, aff_values, softmax);
		float non_zero = Floats.sum(aff_values);
		float certainty = 1 - Floats.entropy(softmax, non_zero > 1 ? non_zero : 2f);


		// If Q policy is not certain enough, use bias, else, don't use it
//		System.out.prin tln("Certainty: " + certainty );
		if(certainty < certainty_threshold ) {

			// calculate motion bias
			var bias_motion = motionBias.calculateBias(chosenAction);

			// calculate obstacle bias if necessary
			if(obstacle_bias_method==2) {
				var bias_obstacles = obstacle_biases.calculateBias(pos);
				Floats.mul(bias_motion, bias_obstacles, action_selection_probs);
			} else Floats.copy(bias_motion, action_selection_probs);


			// Combine bias, then add bias to softmax to get resulting probabilities
			addMultiplicativeBias(action_selection_probs, softmax, action_selection_probs);
		} else Floats.copy(softmax,action_selection_probs);
		
				
		
		learning_dist = softmax;
//		optimal_action_dist = softmax;
		optimal_action_dist = action_selection_probs;
		
		
//		Floats.softmaxWithWeights(qValues, biased, biased);
		
		
		chosenAction = DiscreteDistribution.sample(action_selection_probs);
		actionWasOptimal = optimal_action_dist[chosenAction] == Floats.max(optimal_action_dist);
		
		
		
		tocs[4] = Debug.toc(tics[4]);

		
		// update traces
		tics[5] = Debug.tic();
		for(int i=0; i<num_layers; i++) {
			var pcs = pc_bins[i].active_pcs;
			vTraces[i].update(pcs.ns, pcs.ids, 0);
			qTraces[i].update(pcs.ns, pcs.ids, chosenAction, learning_dist);

		}
		
		// perform action
		double tita = 2*Math.PI/numActions*chosenAction;
		robot.getRobotProxy().send_command(new TranslateXY(0.08f*(float)Math.cos(tita), 0.08f*(float)Math.sin(tita)));
		feederModule.eatAfterMotion();
		tocs[5] = Debug.toc(tics[5]);
		
		tocs[tocs.length-1] = Debug.toc(tics[tocs.length-1]);
		if(cycles==1) {
			Floats.copy(tocs,averages);
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

		for(int i=0; i<num_layers; i++) {
			vTraces[i].clear();
			qTraces[i].clear();
			pc_bins[i].clear();
		}

		oldStateValue = null;
		chosenAction = -1;
		actionWasOptimal = false;
		
		
		// copy state values:
		for(int i=0; i < num_layers; i++)
			vTableCopy[i] = Floats.copy(vTable[i]);
				
	}
	
	@Override
	public void endEpisode() {
		super.endEpisode();
		// TODO Find out what this does
		episodeDeltaV = 0;
		for(int i=0; i < num_layers; i++) {
			var dif = Floats.sub(vTable[i], vTableCopy[i]);
			episodeDeltaV = Math.max(episodeDeltaV, Floats.max(Floats.abs(dif,dif)));

		}
		// Runs replay events
		for(int i = 0; i<num_replay;i++){
			// TODO: ask why we need to clear these, Just like if they were a new episode?
			for(int j=0; j<num_layers; j++) {
				vTraces[j].clear();
				qTraces[j].clear();
				pc_bins[j].clear();
			}
			oldStateValue = null;
			replayEvent();
		}
		// TODO: write the current ReplayMatrix
		if (episode%freq_replay_matrix_writes == 0){
			rmatrix.writeRMatrix(episode);
		}

		//Updated Post Processing of Replay Matrix
		// TODO: clear ReplayMatrix
		if (episode%freq_replay_matrix == 0){
			rmatrix.clearRMatrix();
		}


		num_trial_cycle.add(trial_cycle);
		trial_cycle = 0;
		//rmatrix = new ReplayMatrix(pcs);
		
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
		super.endExperiment();
//		rmatrix.writeRMatrix(episode);
		FileWriter writer = null;
		try {
			writer = new FileWriter("./logs/development/replayf2021/experiments/Number_Cycles.csv",false);
		} catch (IOException e) {
			e.printStackTrace();
		}

		for(int i = 0; i < num_trial_cycle.size() ; i++){
			try {
				writer.append(String.valueOf(num_trial_cycle.get(i))+",");
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		try {
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
	
	int angle_to_index(float angle) {
		double dtita = Math.PI*2 / numActions;
		var res = (int)Math.round(angle/dtita) % numActions;
		return res < 0 ? res + numActions : res;
	}
	
	void addMultiplicativeBias(float[] bias, float[] input, float[] output) {
		output = Floats.mul(bias, input, output);
		var sum = Floats.sum(output);
		
		if(sum!=0) Floats.div(output, sum, output);
		else {
			System.err.println("WARNING: Probability sum is 0, setting uniform distribution (MulytiscaleModel.java)");
			for(int i=0; i<numActions; i++) output[i] = 1/numActions;
		}
	}
	
	float[] addMultiplicativeBias(float[] bias, float[] input) {
		var output = Floats.mul(bias, input);
		var sum = Floats.sum(output);
		
		if(sum!=0) Floats.div(output, sum, output);
		else {
			System.err.println("WARNING: Probability sum is 0, setting uniform distribution (MulytiscaleModel.java)");
			Floats.uniform(output);			
		}
		return output;
	}
	
	
	public void replayEvent(){
		replay_cycle = 0;
		replay_flag = true;
		ArrayList<Integer> cells_vist = new ArrayList<Integer>();

		var start_pc_index = rn.nextInt(pcs[0].num_cells);
		//System.out.println(start_pc_index);
		cell_activation_indexs=rmatrix.replayEvent(start_pc_index);
		cells_vist.add(cell_activation_indexs[0]);


		while(replay_cycle < num_replay && replay_flag){
			replay_cycle++;
			var old_position = cell_activation_indexs[0];
			// Detects a Cycle formed in the Replay event
			if(cells_vist.contains(cell_activation_indexs[1])|| cell_activation_indexs[1]==-1){
				replay_flag = false;

			}else{
				cells_vist.add(cell_activation_indexs[1]);
			}



			// If not at a terminal state
			if (replay_flag){
				// calculates action and position
				// TODO add a distrabution of actions to sample from
				//System.out.println("Replay event");
				var xo = pcs[0].xs[old_position];
				var yo = pcs[0].ys[old_position];
				var x1 = pcs[0].xs[cell_activation_indexs[0]];
				var y1 = pcs[0].ys[cell_activation_indexs[0]];
				var x2 = pcs[0].xs[cell_activation_indexs[1]];
				var y2 = pcs[0].ys[cell_activation_indexs[1]];

				theta = Math.atan2((y2-y1),(x2-x1));

				var action_selected = Math.round((theta/dt)) % numActions;
				if (action_selected < 0){
					action_selected+=numActions;
				}

				// Calculates Reward
				var replay_reward = 0;
				for (var f: feeders){
					var diff_x = f.pos.getX() - x1;
					var diff_y = f.pos.getY() - y1;
					var dist_feeder = Math.sqrt(Math.pow((diff_x),2)+Math.pow((diff_y),2));
					if (dist_feeder <= .1){
						//System.out.println("Replay Path found feeder");
						replay_reward = 1;
					}
				}


				//System.out.println("Replay Action Selected:"+ action_selected);


				// TODO: Confirm that this is the correct formulas applied for RL

				// Calculates Active Place Cells
				// TODO: Figure out if this is needed
//				float totalActivity =0;
//				for(int i=0; i<num_layers; i++)
//					totalActivity+=pc_bins[i].activateBin(x1, y1);
//				for(int i=0; i<num_layers; i++) pc_bins[i].active_pcs.normalize(totalActivity);

				//Calculates V' or bootstrap
				float bootstrap = replay_reward;
				if(replay_reward==0 ) {
					// only calculate next state value if non terminal state
					float value = 0;
					for(int i=0; i<num_layers; i++) {
						var pcs = pc_bins[i].getActive_pcs(x1,y1);
						for(int j=0; j<pcs.num_cells; j++ ) {
							value+= vTable[i][pcs.ids[j]]*pcs.ns[j];
						}
					}
					bootstrap+= value*discountFactor;
				}

				// TODO Find out what this does
				oldStateValue = 0f;
				qValues = new float[numActions];
				for(int i=0; i<num_layers; i++) {
					var pcs = pc_bins[i].getActive_pcs(xo,yo);
					var ids = pcs.ids;

//					System.out.println(Arrays.toString(pcs.ns));
					for(int j=0; j<pcs.num_cells; j++) {
						var activation = pcs.ns[j];
						oldStateValue+= vTable[i][ids[j]]*activation;

						for(int k=0; k<numActions; k++)
							qValues[k]+= qTable[i][ids[j]][k]*activation;
					}
				}

				// Calculates Error
				float error = bootstrap - oldStateValue;

				// Update V and Q
				// TODO find out what traces do
				for(int i=0; i<num_layers; i++) {
					// update V
					// v = v + error*learning_rate*trace
					var traces = vTraces[i].traces[0];
					for(var id : vTraces[i].non_zero[0]) {
						vTable[i][id]+=  error*v_learningRate[i]*traces[id];
					}

					// update Q
					for(int j=0; j<numActions; j++) {
						traces = qTraces[i].traces[j];
						for(var id : qTraces[i].non_zero)
							qTable[i][id][j] += error*q_learningRate[i]*traces[id];
					}
				}
				Floats.softmax(qValues, softmax);
				if (replay_reward == 1){
					replay_flag = false;
				}
				//Shift indexes
				cell_activation_indexs=rmatrix.replayEvent(cell_activation_indexs[1]);

			}

		}
		writeReplayEvent(cells_vist);
	}


	public void writeReplayEvent(ArrayList<Integer> path){
		FileWriter writer = null;
		if (num_writes !=0){
			try {
				writer = new FileWriter("./logs/development/replayf2021/experiments/Replay_Paths.csv",true);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}else {
			try {
				writer = new FileWriter("./logs/development/replayf2021/experiments/Replay_Paths.csv");
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		for(int i = 0; i < path.size() ; i++){
			try {
				if (i!= path.size() -1){
					writer.append(String.valueOf(path.get(i))+",");
				}
				else{
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
