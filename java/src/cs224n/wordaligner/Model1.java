package cs224n.wordaligner;

import cs224n.util.*;
import java.lang.Math;
import java.util.*;

public class Model1 implements WordAligner {

	private CounterMap<String, String> efCounter;
    private Counter<String> eCounter;
    private CounterMap<String, String> feProb;
    
    private final double THRESHOLD = 0.05; 
    
	public Model1() {
    	feProb = new CounterMap<String,String>();
	}
	
	public Alignment align(SentencePair sentencePair) {
		Alignment alignment = new Alignment();
		
		List<String> sourceSentence = sentencePair.getSourceWords();
		List<String> targetSentence = sentencePair.getTargetWords();
		
		// Since it seems like these are references, adding NULL_WORD 
		// once causes it to be added permanently in future reads.
		if(!sourceSentence.get(0).equals(NULL_WORD))
			sourceSentence.add(0,NULL_WORD);
		
		double prob = 0;
		double maxProb = 0;
		int maxIndex = 0;
		int targetIndex = 1;
		int sourceIndex = 0;

		for(String target : targetSentence) {
			maxIndex = 0;
			maxProb = 0;
			for (String source : sourceSentence) {
				prob = feProb.getCount(source, target);
				if (prob > maxProb) {
					maxProb = prob;
					maxIndex = sourceIndex;
				}
				sourceIndex++;
			}
			if (maxIndex != 0)
				alignment.addPredictedAlignment(targetIndex-1, maxIndex-1);
			targetIndex++;
			sourceIndex = 0;
		}
		return alignment;
	}

	public void train(List<SentencePair> trainingData) {
	
		long startTime = System.currentTimeMillis();
		
		// Initialize uniform probabilities
		for (SentencePair pair : trainingData) {
			List<String> eWords = pair.getTargetWords();
			List<String> fWords = pair.getSourceWords();
			fWords.add(0,NULL_WORD);
			
			for (String f : fWords) {
				for (String e : eWords) {
					feProb.setCount(f,e,0.2);
				}
			}
		}
		
		double delta = 0;
		double likelihood = 0;
		double oldlikelihood = 0;
		boolean converge = false;
		double sum = 0;
		
		// For t = 1 ... T
		for (int i = 0; i < 10; i++) {//while (!converge) {
			oldlikelihood = likelihood;
			likelihood = 0;
			
			// Set all counts c(...) = 0
			efCounter = new CounterMap<String,String>();
			eCounter = new Counter<String>();
			
			// For k = 1 ... n
			for (SentencePair pair : trainingData) {
				List<String> eWords = pair.getTargetWords();
			    List<String> fWords = pair.getSourceWords();
			    
			    // For i = 1 ... m
			    for (String f : fWords) {			    	
			    	sum = 0;
			    	
			    	// For j = 0 ... lk
			    	for (String e : eWords) {
			    		sum+=feProb.getCount(f,e);
			    	}
			    	for (String e : eWords) {
			    		delta = feProb.getCount(f,e)/sum;
			    		eCounter.incrementCount(e, delta);
			    		efCounter.incrementCount(e,f, delta);
			    	}
				}
			}
			
			// Set t
			for (SentencePair pair : trainingData) {
				List<String> eWords = pair.getTargetWords();
			    List<String> fWords = pair.getSourceWords();
			    
			    for (String f : fWords) { 			    	
			    	for (String e : eWords) {
			    		feProb.setCount(f,e,((double)efCounter.getCount(e,f)/eCounter.getCount(e)));
			    	}
			    }
			}
			
			
//			System.out.println(Math.abs(oldlikelihood-likelihood));
//			if (Math.abs(oldlikelihood-likelihood) < THRESHOLD)
//				converge = true;
		}
		long endTime = System.currentTimeMillis();
		System.out.println("Execution Time: " + ((endTime-startTime)/1000.0)+"s");
	}
}
