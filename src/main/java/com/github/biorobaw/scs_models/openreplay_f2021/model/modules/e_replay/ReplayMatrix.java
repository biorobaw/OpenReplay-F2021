package com.github.biorobaw.scs_models.openreplay_f2021.model.modules.e_replay;

import com.github.biorobaw.scs.utils.math.Floats;
import com.github.biorobaw.scs_models.openreplay_f2021.model.modules.b_state.PlaceCellBins;
import com.github.biorobaw.scs_models.openreplay_f2021.model.modules.b_state.PlaceCells;

import java.io.FileWriter;
import java.io.IOException;
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
	private float threshold = .0001f;
	private int[] cell_propigation = {0, 0};
	private int previous_index = 0;


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

		System.out.println("Place Cell 0 id: " + initial_pcs[0].ids[0]);
		System.out.println("Place Cell 3 id: " + initial_pcs[0].ids[3]);

	}

	public void update() {
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

	// This function will update the replay matrix using the brute force methode?
	public void updateAll() {
		int num_current_cells = placeCellBins[1].num_cells;
		int num_old_cells = placeCellBins[0].num_cells;
		// TODO: change to a brute force why of updating replay matrix
		for (int i = 1; i < replay_matrix[0].length; i++) {
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

	public int[] replayEvent(int pc_index) {
		previous_index = cell_propigation[0];
		int current_pc_index = pc_index;
		int next_pc_index = -1;
		double max_pc_wieght = 0;
		for (int i = 0; i < replay_matrix[current_pc_index].length; i++) {
			// TODO: Add possible condition to ensure connection does not cross obstical
			if (replay_matrix[current_pc_index][i] >= threshold &&
					replay_matrix[current_pc_index][i] >= max_pc_wieght &&
					current_pc_index != i &&
					i != previous_index) {
				next_pc_index = i;
			}

		}
		cell_propigation[0] = current_pc_index;
		cell_propigation[1] = next_pc_index;
		return cell_propigation;
	}

	public void writeRMatrix(int episode) {
		FileWriter writer = null;
		try {
			writer = new FileWriter("./logs/development/replayf2021/experiments/ReplayMatrices/Replay_matrix"+ String.valueOf(episode)+".csv");
		} catch (IOException e) {
			e.printStackTrace();
		}

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


}
