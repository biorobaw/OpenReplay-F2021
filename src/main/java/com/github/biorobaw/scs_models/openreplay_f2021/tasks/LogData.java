package com.github.biorobaw.scs_models.openreplay_f2021.tasks;

import com.github.biorobaw.scs.experiment.Experiment;
import com.github.biorobaw.scs.simulation.scripts.Script;
import com.github.biorobaw.scs.utils.files.BinaryFile;
import com.github.biorobaw.scs.utils.files.XML;
import com.github.biorobaw.scs_models.openreplay_f2021.model.ReplayModel;
import com.github.biorobaw.scs_models.openreplay_f2021.tasks.SetInitialPosition;


public class LogData implements Script {
	
	
	int numEpisodes;
	int startPositions[];
	int stepsTaken[];
	float maxDeltaV[];
	String subject_id = "";
	
	public LogData(XML xml) {
		subject_id =xml.getAttribute("subject_id");
	}
	
	@Override
	public void newTrial() {
		numEpisodes = Experiment.get().getGlobal("trial_episodes");
		startPositions = new int[numEpisodes];
		stepsTaken = new int[numEpisodes];
		maxDeltaV  = new float[numEpisodes];
	}
	
	@Override
	public void endEpisode() {
		// Log steps taken at the end of the episode
//		System.out.println("Logging data...");
		
		var e = Experiment.get();
		int episode = e.getGlobal("episode");
		startPositions[episode] = SetInitialPosition.getStartIndex();
		stepsTaken[episode] = (int)(long)e.getGlobal("cycle");
		
		var model = (ReplayModel)e.getSubject(subject_id);
		maxDeltaV[episode] = model.episodeDeltaV;

		// Write Replay Matrix

		
	}
	
	
	@Override
	public void endTrial() {

		// save logs: cycles per episode, state and action values during last episode
		
		
//		System.out.println("Saving log...");
		var e = Experiment.get();
		String logFolder = e.getGlobal("logPath").toString();
		String ratId	 = e.getGlobal("run_id").toString();
		
		//create folders
		String prefix = logFolder  +"/r" + ratId + "-";  
		
		// save cycles per episode
		var file = BinaryFile.openFileToWrite(prefix + "steps.bin");
		BinaryFile.writeArray(file, startPositions,true);
		BinaryFile.writeArray(file, stepsTaken,true);
		BinaryFile.close(file);
		
		// save state and action values of last episode
		var model = (ReplayModel)e.getSubject(subject_id);
		var numActions = model.numActions;
		for(int i=0; i<model.num_layers; i++) {
			int num_pcs = model.vTable[i].length;
			BinaryFile.saveBinaryVector(model.vTable[i], prefix + "V" + i+ ".bin", true);
			BinaryFile.saveBinaryMatrix(model.qTable[i], num_pcs, numActions , prefix + "Q" + i+ ".bin", true);
		}
		
		// save deltaV
		BinaryFile.saveBinaryVector(maxDeltaV, prefix + "deltaV.bin", true);
		
		// save seed
		BinaryFile.saveBinaryVector(new long[] {e.getGlobal("seed")}, prefix + "seed.bin", true);
		
	}
	
	
	
}
