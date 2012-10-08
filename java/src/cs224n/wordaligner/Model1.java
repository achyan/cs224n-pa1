package cs224n.wordaligner;

import cs224n.util.*;
import java.util.*;

public class Model1 implements WordAligner {

	private CounterMap<String, String> sourceTargetCounter;
    private Counter<String> sourceCounter;
    //private Counter<String> targetCounter;	
	
    private Map<String, List<Double>> sourceTargetProb;
    private Set<String> sourceVocab;
    private Set<String> targetVocab;
    
	public Model1() {
		sourceTargetCounter = new CounterMap<String,String>();
    	sourceCounter = new Counter<String>();
    	//targetCounter = new Counter<String>();
    	
    	sourceVocab = new HashSet<String>();
    	targetVocab = new HashSet<String>();
    	sourceTargetProb = new HashMap<String, List<Double>>();
	}
	
	public Alignment align(SentencePair sentencePair) {
		Alignment alignment = new Alignment();
		
		List<String> sourceSentence = sentencePair.getSourceWords();
		List<String> targetSentence = sentencePair.getTargetWords();
		sourceSentence.add(0, NULL_WORD);

		double prob = 0;
		double maxProb = 0;
		int maxIndex = 0;
		int targetIndex = 1;
		int sourceIndex = 0;

		for(String target : targetSentence) {
			maxIndex = 0;
			maxProb = 0;
			for (String source : sourceSentence) {
				prob = sourceTargetProb.getCount(source, target);
				if (prob > maxProb) {
					maxProb = prob;
					maxIndex = sourceIndex;
				}
				sourceIndex++;
			}
			alignment.addPredictedAlignment(targetIndex, maxIndex);
			targetIndex++;
			sourceIndex = 0;
		}
		return alignment;
	}

	public void train(List<SentencePair> trainingData) {
	
		// Initialize uniform probabilities
		for (SentencePair pair : trainingData) {
			List<String> targetWords = pair.getTargetWords();
			List<String> sourceWords = pair.getSourceWords();
			
			for (String word : targetWords) {
				targetVocab.add(word);
			}
			
			for (String word : sourceWords) {
				sourceVocab.add(word);
			}
		}
		
		for (String source : sourceVocab) {
			sourceTargetProb.setCount(source,target,1.0/sourceVocab.size());
		}
				
		//sourceVocab.add(NULL_WORD);
		
		// English = SOURCE	; French = TARGET
		
		double delta = 0;
		double sum = 0;
		
		for (SentencePair pair : trainingData) { // For k = 1 ... n
			List<String> targetWords = pair.getTargetWords();
		    List<String> sourceWords = pair.getSourceWords();
		
		    for (String target : targetWords) { // For i = 1 ... mk
		    	
		    	sum = 0;
		    	
		    	for (String source : sourceWords) {
		    		sum+=sourceTargetProb.getCount(source,target);
		    	}
		    	
		    	for (String source : sourceWords) { // For j = 0 ... lk
		    		delta = sourceTargetProb.getCount(source,target)/sum;
		    		sourceCounter.incrementCount(source, delta);
		    		sourceTargetCounter.incrementCount(source, target, delta);
		    	  	sourceTargetProb.setCount(source, target, 
		    	  			((double)sourceTargetCounter.getCount(source,target)/sourceCounter.getCount(source)));
		    	}
			}
		}
	}
}
