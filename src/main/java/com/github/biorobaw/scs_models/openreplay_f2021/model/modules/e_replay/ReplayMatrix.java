package com.github.biorobaw.scs_models.openreplay_f2021.model.modules.e_replay;

import com.github.biorobaw.scs.utils.math.Floats;
import com.github.biorobaw.scs_models.openreplay_f2021.model.modules.b_state.PlaceCellBins;
import com.github.biorobaw.scs_models.openreplay_f2021.model.modules.b_state.PlaceCells;

import java.util.Arrays;

/**
 * Class to store and hold the ReplayMatrix
 * This will be used to generate Replay Events to train the RL model
 * @author ChanceHamilton59
 *
 */
public class ReplayMatrix {

	private Float[] pcs_current;
	private Float[] pcs_old;
	private Double[][] replay_matix;
	private Double[][] delta;
	private int length;
	private PlaceCells[] placeCellBins = new PlaceCells[2];


	/**
	 * Creates an inital Replay matrix that contain the activation for all Place Cells
	 * @param initial_pcs	array containing the min x value for each layer
	 */
	public ReplayMatrix(PlaceCells[] initial_pcs){
		this.length = initial_pcs[0].num_cells;
		this.pcs_current = new Float[length];
		this.replay_matix = new Double[length][length];
		this.delta = new Double[length][length];

		for (int i = 1; i < length; i++){
			for (int j = 1; j < length; j++) {
				replay_matix[i][j] = Double.valueOf(0);
			}
		}
		for(int i = 0; i < length; i++){
			pcs_current[i]=initial_pcs[0].as[i];
		}
		System.out.println("Place Cell 0 id: "+initial_pcs[0].ids[0]);
		System.out.println("Place Cell 3 id: "+initial_pcs[0].ids[3]);

	}

	public void update(){
		int num_current_cells = placeCellBins[1].num_cells;
		int num_old_cells = placeCellBins[0].num_cells;
		for (int i = 1; i < num_current_cells; i++){
			var current_pc_index = placeCellBins[1].ids[i];
			for (int j = 1; j < num_old_cells; j++) {
				var old_pc_index = placeCellBins[0].ids[j];
				double arg1 = (pcs_current[current_pc_index]+pcs_old[old_pc_index])/(2);
				double arg2 = pcs_current[old_pc_index]-pcs_old[old_pc_index];
				delta[current_pc_index][old_pc_index] = Math.atan(arg1*arg2);
				replay_matix[current_pc_index][old_pc_index] += delta[current_pc_index][old_pc_index];
			}
		}

	}

	public void setPcs_current(PlaceCells[] current_pcs){
		this.pcs_current = new Float[length];

		for(int i = 0; i < length; i++){
			pcs_current[i]=current_pcs[0].as[i];
		}




	}

	public void setPcs_old(PlaceCells[] old_pcs){
		this.pcs_old = new Float[length];

		for(int i = 0; i < length; i++){
			pcs_old[i]=old_pcs[0].as[i];
		}


	}

	public void addPlaceCellBins(PlaceCells bin, int index){
		placeCellBins[index] = bin;

	}








}