package com.github.biorobaw.scs_models.openreplay_f2021.model.modules.e_replay;

import com.github.biorobaw.scs.utils.math.Floats;
import com.github.biorobaw.scs_models.openreplay_f2021.model.modules.b_state.PlaceCells;

/**
 * Class to store and hold the ReplayMatrix
 * This will be used to generate Replay Events to train the RL model
 * @author ChanceHamilton59
 *
 */
public class ReplayMatrix {

	public PlaceCells[] pcs_current;
	public PlaceCells[] pcs_old;
	public int length;


	/**
	 * Creates an inital Replay matrix that contain the activation for all Place Cells
	 * @param initial_pcs	array containing the min x value for each layer
	 */
	public ReplayMatrix(PlaceCells[] initial_pcs){
		this.pcs_current = initial_pcs;
		this.length = initial_pcs[0].num_cells;

		System.out.println("Here is some shit "+ initial_pcs);
		System.out.println("Here is some more shit "+ length);

	}

	public PlaceCells[] update(PlaceCells[] current_pcs){
		pcs_old = pcs_current;
		pcs_current = current_pcs;
		int largest_i = 0;
		for (int i = 1; i < length; i++){
			if(pcs_current[0].as[i]>pcs_current[0].as[largest_i]) largest_i = i;
		}

		System.out.println("Place Cell Current "+ pcs_current[0].as[largest_i]);
		System.out.println("Place Cell Old "+ pcs_old[0].as[largest_i]);
		return pcs_old;


	}








}