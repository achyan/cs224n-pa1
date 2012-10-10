package cs224n.wordaligner;

import cs224n.util.*;
import java.lang.Math;
import java.util.*;

public class Model1 implements WordAligner {

	private CounterMap<String, String> efCounter;
    private Counter<String> eCounter;
    private CounterMap<String, String> feProb;
    private ArrayList<String> pairs;
    
    private final double THRESHOLD = 0.05; 
    
	public Model1() {
    	feProb = new CounterMap<String,String>();
    	pairs = new ArrayList<String>();
	}
	
	public Alignment align(SentencePair sentencePair) {
		Alignment alignment = new Alignment();
		
		List<String> sourceSentence = sentencePair.getSourceWords();
		List<String> targetSentence = sentencePair.getTargetWords();
		
		double prob = 0;
		double maxProb = 0;
		int maxIndex = 0;
		int targetIndex = 0;
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
				alignment.addPredictedAlignment(targetIndex, maxIndex-1);
			targetIndex++;
			sourceIndex = 0;
		}
		
		return alignment;
	}

	public void train(List<SentencePair> trainingData) {
	
		// Initialize uniform probabilities
		for (SentencePair pair : trainingData) {
			List<String> eWords = pair.getTargetWords();
			List<String> fWords = pair.getSourceWords();
			fWords.add(0,NULL_WORD);
			
			for (String f : fWords) {
				for (String e : eWords) {
					pairs.add(f+" "+e);
					feProb.setCount(f,e,0.5);
				}
			}
		}
		
		double delta = 0;
		double likelihood = 0;
		double oldlikelihood = 0;
		boolean converge = false;
		double sum = 0;
		
		while (!converge) {
			oldlikelihood = likelihood;
			likelihood = 0;
			
			efCounter = new CounterMap<String,String>();
			eCounter = new Counter<String>();
			
			for (SentencePair pair : trainingData) { // For k = 1 ... n
				List<String> eWords = pair.getTargetWords();
			    List<String> fWords = pair.getSourceWords();
			    
			    for (String f : fWords) { // For i = 1 ... mk			    	
			    	sum = 0;
			    	for (String e : eWords) {
			    		sum+=feProb.getCount(f,e);
			    	}
			    	for (String e : eWords) { // For j = 0 ... lk
			    		delta = feProb.getCount(f,e)/sum;
			    		eCounter.incrementCount(e, delta);
			    		efCounter.incrementCount(e,f, delta);
			    	}
				}
			}
			
			for (String p : pairs) {
				String[]fe = p.split(" ");
	    		feProb.setCount(fe[0],fe[1],((double)efCounter.getCount(fe[1],fe[0])/eCounter.getCount(fe[1])));
	    		likelihood+=feProb.getCount(fe[0],fe[1]);
			}
//			
//			for (SentencePair pair : trainingData) { // For k = 1 ... n
//				List<String> eWords = pair.getTargetWords();
//			    List<String> fWords = pair.getSourceWords();
//			    
//			    for (String f : fWords) { // For i = 1 ... mk			    	
//			    	for (String e : eWords) { // For j = 0 ... lk
//			    		feProb.setCount(f,e,((double)efCounter.getCount(e,f)/eCounter.getCount(e)));
//			    		likelihood+=feProb.getCount(f,e);
//			    		System.out.println(f+","+e+": "+feProb.getCount(f,e));
//			    	}
//			    }
//			}
			System.out.println(Math.abs(oldlikelihood-likelihood));
			if (Math.abs(oldlikelihood-likelihood) < THRESHOLD)
				converge = true;
		}
	}
}
