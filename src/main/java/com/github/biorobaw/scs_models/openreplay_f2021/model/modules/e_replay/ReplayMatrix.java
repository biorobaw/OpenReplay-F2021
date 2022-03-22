package com.github.biorobaw.scs_models.openreplay_f2021.model.modules.e_replay;

import com.github.biorobaw.scs.experiment.Experiment;
import com.github.biorobaw.scs.utils.math.DiscreteDistribution;
import com.github.biorobaw.scs.utils.math.Floats;
import com.github.biorobaw.scs_models.openreplay_f2021.model.modules.b_state.PlaceCells;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Class to store and hold the ReplayMatrix
 * This will be used to generate Replay Events to train the RL model
 * @author ChanceHamilton59
 *
 */
public class ReplayMatrix {

	private float[] pcs_current;
	private float[] pcs_old;
	private double[][] replay_matrix;
	private double[][] delta;
	private int length;
	private PlaceCells[] placeCellBins = new PlaceCells[2];
	private float connection_threshold = .0001f;
	private int[] cell_propigation = {0, 0};
	private int previous_index = 0;
	public ArrayList<Integer> possible_starting_PCs = new ArrayList<Integer>();
	public ArrayList<double[]>[] strongly_connected_PCs;

	/**
	 * Creates an inital Replay matrix that contain the activation for all Place Cells
	 *
	 * @param initial_pcs array containing the min x value for each layer
	 */
	public ReplayMatrix(PlaceCells[] initial_pcs) {
		this.length = initial_pcs[0].num_cells;
		this.pcs_current = new float[length];
		this.replay_matrix = new double[length][length];
		this.delta = new double[length][length];
		this.strongly_connected_PCs = (ArrayList<double[]>[]) new ArrayList[length];


	}

	//TODO: Add learning rate to update equation
	public void update(float replay_matrix_update_lr) {
		int num_current_cells = placeCellBins[1].num_cells;
		int num_old_cells = placeCellBins[0].num_cells;
		for (int i = 1; i < num_current_cells; i++) {
			var current_pc_index = placeCellBins[1].ids[i];
			double arg2 = pcs_current[current_pc_index] - pcs_old[current_pc_index];
			for (int j = 1; j < num_old_cells; j++) {
				var old_pc_index = placeCellBins[0].ids[j];
				double arg1 = (pcs_current[old_pc_index] + pcs_old[old_pc_index]) / (2);

				delta[old_pc_index][current_pc_index] = Math.atan(arg1 * arg2);
				replay_matrix[old_pc_index][current_pc_index] += replay_matrix_update_lr * delta[old_pc_index][current_pc_index];
			}
		}

	}

	public void clear() {
		int num_current_cells = placeCellBins[1].num_cells;
		int num_old_cells = placeCellBins[0].num_cells;
		for (int i = 1; i < num_current_cells; i++) {
			var current_pc_index = placeCellBins[1].ids[i];
			double arg2 = pcs_current[current_pc_index] - pcs_old[current_pc_index];
			for (int j = 1; j < num_old_cells; j++) {
				var old_pc_index = placeCellBins[0].ids[j];
				double arg1 = (pcs_current[old_pc_index] + pcs_old[old_pc_index]) / (2);

				delta[old_pc_index][current_pc_index] = Math.atan(arg1 * arg2);
				replay_matrix[old_pc_index][current_pc_index] += delta[old_pc_index][current_pc_index];
			}
		}

	}

	public void setPcs_current(PlaceCells[] current_pcs) {
		this.pcs_current = new float[length];

		for (int i = 0; i < length; i++) {
			pcs_current[i] = current_pcs[0].as[i];
		}

	}

	public void setPcs_old(PlaceCells[] old_pcs) {
		this.pcs_old = new float[length];

		for (int i = 0; i < length; i++) {
			pcs_old[i] = old_pcs[0].as[i];
		}


	}

	public void addPlaceCellBins(PlaceCells bin, int index) {
		placeCellBins[index] = bin;

	}

	// This functions takes the current Place cell index and returns the index of the next place cell for replay propagation
	public int[] replayEvent(int pc_index, String propagation_type) {
		previous_index = cell_propigation[0];
		int current_pc_index = pc_index;
		int next_pc_index = -1;
		double max_pc_weight = connection_threshold;

		// Selects the PC index with the strongest connection
		if (propagation_type.equals("max_connection")){
			for (double[] connections : strongly_connected_PCs[pc_index]){
				if (connections[1] >= max_pc_weight && connections[0] != pc_index && connections[0] != previous_index){
					next_pc_index = (int)connections[0];
					max_pc_weight = connections[1];
				}
			}
		}

		// Uses stochastic sampling to select the next PC index
		if (propagation_type.equals("stochastic")){
			float[] connection_weights = new float[strongly_connected_PCs[pc_index].size()];
			int iterator = 0;
			for (double[] connections : strongly_connected_PCs[pc_index]){
				connection_weights[iterator++] = (float)connections[1];
			}

			float[] connection_probs = Floats.softmax(connection_weights);
			if (connection_probs.length >0){
				var next_cell = DiscreteDistribution.sample(connection_probs);
				next_pc_index = (int) strongly_connected_PCs[pc_index].get(next_cell)[0];
			}

		}

		// Shifts indexes for propagation
		cell_propigation[0] = current_pc_index;
		cell_propigation[1] = next_pc_index;
		return cell_propigation;
	}

	public void writeRMatrix(int episode, boolean append) {

		// Gets log path and Rat Id to create a log file
		var ex = Experiment.get();
		String logFolder = ex.getGlobal("logPath").toString();
		String ratId	 = ex.getGlobal("run_id").toString();
		String dir =  logFolder;

		File directory = new File(dir);

		if (!directory.exists()){
			directory.mkdir();
		}

		//create folders
		String prefix = dir  +"/r" + ratId + "-";

		// Creates the file name structure
		var file = prefix + "Replay_Matrix.csv";

		FileWriter writer = null;
		try {
			writer = new FileWriter(file, append);
		} catch (IOException e) {
			e.printStackTrace();
		}

//		try {
//			writer.append("# \n");
//
//		} catch (IOException e) {
//			e.printStackTrace();
//		}

		for (int i = 0; i < length; i++) {
			for (int j = 0; j < length; j++)
				try {
					writer.append(String.valueOf(replay_matrix[i][j]) + ", ");

				} catch (IOException e) {
					e.printStackTrace();
				}
			try {
				writer.append("\n");
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


	public void clearRMatrix() {
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < length; j++) {
				replay_matrix[i][j] = 0.0;
			}
		}

	}

	public void getConnectedPC(int num_connection_threshold) {

		possible_starting_PCs.clear();
		double[] pos_val = new double[2];
		ArrayList<double[]> index_and_connections = new ArrayList<double[]>();
		for (int i = 0; i < length; i++) {
			int connection_count = 0;

			for (int j = 0; j < length; j++) {
				if (replay_matrix[i][j] >= this.connection_threshold){
					connection_count++;
					pos_val[0] = j;
					pos_val[1] = replay_matrix[i][j];
					index_and_connections.add(Arrays.copyOf(pos_val,2));
				}
				if (connection_count >= num_connection_threshold){
					possible_starting_PCs.add(i);
					//break;
				}
			}
			strongly_connected_PCs[i] = new ArrayList<>(index_and_connections);
			index_and_connections.clear();

		}

	}


}
