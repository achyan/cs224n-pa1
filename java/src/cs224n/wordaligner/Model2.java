package cs224n.wordaligner;

import java.util.HashMap;
import java.util.List;

import cs224n.util.Counter;
import cs224n.util.CounterMap;
import cs224n.util.Pair;

public class Model2 implements WordAligner {
    private static final double THRESHOLD = 0.05; 

	private CounterMap<String, String> eAndFCounts = new CounterMap<String, String>();
	private Counter<String> eCounts = new Counter<String>();
	private CounterMap<Pair<Integer, Integer>, Pair<Integer, Integer>> jGivenIlmCounts = new CounterMap<Pair<Integer, Integer>, Pair<Integer, Integer>>();
	private Counter<Pair<Integer, Pair<Integer, Integer>>> ilmCounts = new Counter<Pair<Integer, Pair<Integer, Integer>>>();

	private double lastAlignLikelihood = 0;
	
	
//	private HashMap<String, Integer> targetDict = new HashMap<String,Integer>();
//	private HashMap<String, Integer> sourceDict = new HashMap<String, Integer>();
//	private int numSource = 0;
//	private int numTarget = 0;
	
	@Override
	public Alignment align(SentencePair sentencePair) {
		lastAlignLikelihood = 0;
		Alignment alignment = new Alignment();
		
		List<String> sourceSentence = sentencePair.getSourceWords(); //sourceSentence already has the null added during training
		if(!sourceSentence.get(0).equals(NULL_WORD)) {
			sourceSentence.add(0,NULL_WORD);
		}
		
		List<String> targetSentence = sentencePair.getTargetWords();
		int m_k = sourceSentence.size() - 1;  //don't count null
		int l_k = targetSentence.size();

		double prob = 0;
		double maxProb = 0;
		int maxIndex = 0;
		int targetIndex = 1;
		int sourceIndex = 0;

		for(String target : targetSentence) {
			maxIndex = 0;
			maxProb = 0;
			for (String source : sourceSentence) {
				prob = ((double) jGivenIlmCounts.getCount(new Pair<Integer,Integer>(targetIndex,sourceIndex), new Pair<Integer,Integer>(l_k,m_k))) 
						/ ilmCounts.getCount(new Pair<Integer,Pair<Integer,Integer>>(sourceIndex,new Pair<Integer,Integer>(l_k,m_k)))
						* eAndFCounts.getCount(target, source)
						/ eCounts.getCount(target);
				if (prob > maxProb) {
					maxProb = prob;
					maxIndex = sourceIndex;
				}
				sourceIndex++;
			}
			lastAlignLikelihood += Math.log(maxProb);
			if (maxIndex != 0) {
				alignment.addPredictedAlignment(targetIndex-1, maxIndex-1); //0 is <NULL>, which shifts everything down one; targetIndex was originally 1-indexed	
			}
			
			targetIndex++;
			sourceIndex = 0;
		}
		return alignment;
	}

	//TODO: smoothing?
	
	@Override
	public void train(List<SentencePair> trainingData) {
		//int k = trainingData.size();
		int sourceIndex, targetIndex;
		double delta;
		boolean initIter = true;
		
		//preprocess
		for (SentencePair sentencePair: trainingData) {
			List<String> sourceWords = sentencePair.getSourceWords();
			if(!sourceWords.get(0).equals(NULL_WORD)) {
				sourceWords.add(0,NULL_WORD);
			}
			
//			for(String sourceWord : sourceWords) {
//				if(!sourceDict.containsKey(sourceWord)) {
//					sourceDict.put(sourceWord, numSource);
//					numSource++;
//				}
//			}
//			
//			List<String> targetWords = sentencePair.getTargetWords();
//
//			for(String targetWord : targetWords) {
//				if(!targetDict.containsKey(targetWord)) {
//					targetDict.put(targetWord, numTarget);
//					numTarget++;
//				}
//			}
					
		}
		
		//initialize to results of Model1
		CounterMap<String, String> prevEAndFCounts = null;
		Counter<String> prevECounts = null;
		CounterMap<Pair<Integer, Integer>, Pair<Integer, Integer>> prevJGivenIlmCounts = null;
		Counter<Pair<Integer, Pair<Integer, Integer>>> prevIlmCounts = null;
		
		double likelihood = 0;
		double oldlikelihood = 0;
		boolean isConverged = false;
		double sum = 0;
		int iter = 0;
		while (!isConverged) {
			iter++;
			oldlikelihood = likelihood;
			likelihood = 0;
			
		//for (int iter = 0; iter < 10; iter++) {
			eAndFCounts = new CounterMap<String, String>();
			eCounts = new Counter<String>();
			jGivenIlmCounts = new CounterMap<Pair<Integer, Integer>, Pair<Integer, Integer>>();
			ilmCounts = new Counter<Pair<Integer, Pair<Integer, Integer>>>();
			
			for (SentencePair sentencePair: trainingData) { //for all k
				List<String> sourceWords = sentencePair.getSourceWords();
				List<String> targetWords = sentencePair.getTargetWords();
				int m_k = sourceWords.size() - 1 ; //null not counted in size
				int l_k = targetWords.size();
				
				sourceIndex = 0;
				
				for (String sourceWord : sourceWords) {
					
					double normalizer = 0;
					double qt = 0;
					
					//sum up all the qt values to get normalization factor
					targetIndex = 1; 
					for(String targetWord : targetWords) {
						if (prevJGivenIlmCounts == null) {
							qt = 1;
						}
						else {
							qt = ((double) prevJGivenIlmCounts.getCount(new Pair<Integer,Integer>(targetIndex,sourceIndex), new Pair<Integer,Integer>(l_k,m_k))) 
									/ prevIlmCounts.getCount(new Pair<Integer,Pair<Integer,Integer>>(sourceIndex,new Pair<Integer,Integer>(l_k,m_k)))
									* prevEAndFCounts.getCount(targetWord, sourceWord)
									/ prevECounts.getCount(targetWord);
						}
											
						normalizer += qt;
						targetIndex++;
					}
					
					targetIndex = 1; 
					for(String targetWord : targetWords) {
						if (prevJGivenIlmCounts == null) {  //first time running.  need to initialize qt
							qt = 1;
						}
						else {
							qt = ((double) prevJGivenIlmCounts.getCount(new Pair<Integer,Integer>(targetIndex,sourceIndex), new Pair<Integer,Integer>(l_k,m_k))) 
									/ prevIlmCounts.getCount(new Pair<Integer,Pair<Integer,Integer>>(sourceIndex,new Pair<Integer,Integer>(l_k,m_k)))
									* prevEAndFCounts.getCount(targetWord, sourceWord)
									/ prevECounts.getCount(targetWord);
						}
						
						delta = qt/normalizer;
						
						eAndFCounts.incrementCount(targetWord, sourceWord, delta);
						eCounts.incrementCount(targetWord, delta);  //for normalization purposes
						jGivenIlmCounts.incrementCount(new Pair<Integer,Integer>(targetIndex,sourceIndex),
								new Pair<Integer,Integer>(l_k,m_k), delta);
						ilmCounts.incrementCount(new Pair<Integer,Pair<Integer,Integer>>(sourceIndex,new Pair<Integer,Integer>(l_k,m_k)),delta);
						
						
						targetIndex++;
					}
					
					
					sourceIndex++;
				}			
			}
			
			prevEAndFCounts = eAndFCounts;
			prevECounts = eCounts;
			prevJGivenIlmCounts = jGivenIlmCounts;
			prevIlmCounts = ilmCounts;
			
			
			//test to see if converged
			int sentencePairIndex = 0;
			for (SentencePair sentencePair : trainingData) {
				align(sentencePair);
				likelihood += this.lastAlignLikelihood;
				sentencePairIndex++;

			}
			System.out.println(iter + ":" + likelihood);
			if(oldlikelihood != 0) {
				if(likelihood < oldlikelihood || Math.abs(likelihood - oldlikelihood) < trainingData.size() * .05) {
					isConverged = true;
				}
			}
		}
		
	}

}
