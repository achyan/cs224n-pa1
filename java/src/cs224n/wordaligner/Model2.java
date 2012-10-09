package cs224n.wordaligner;

import java.util.List;

import cs224n.util.Counter;
import cs224n.util.CounterMap;
import cs224n.util.Pair;

public class Model2 implements WordAligner {

	private CounterMap<String, String> eAndFCounts = new CounterMap<String, String>();
	private Counter<String> eCounts = new Counter<String>();
	private CounterMap<Pair<Integer, Integer>, Pair<Integer, Integer>> jGivenIlmCounts = new CounterMap<Pair<Integer, Integer>, Pair<Integer, Integer>>();
	private Counter<Pair<Integer, Pair<Integer, Integer>>> ilmCounts = new Counter<Pair<Integer, Pair<Integer, Integer>>>();

	
	@Override
	public Alignment align(SentencePair sentencePair) {
		Alignment alignment = new Alignment();
		
		List<String> sourceSentence = sentencePair.getSourceWords();
		sourceSentence.add(0,NULL_WORD);
		List<String> targetSentence = sentencePair.getTargetWords();
		int m_k = sourceSentence.size();
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
				prob = ((double) jGivenIlmCounts.getCount(new Pair<Integer,Integer>(sourceIndex,targetIndex), new Pair<Integer,Integer>(l_k,m_k))) 
						/ ilmCounts.getCount(new Pair<Integer,Pair<Integer,Integer>>(targetIndex,new Pair<Integer,Integer>(l_k,m_k)))
						* eAndFCounts.getCount(target, source)
						/ eCounts.getCount(target);
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

	//TODO: smoothing?
	
	@Override
	public void train(List<SentencePair> trainingData) {
		//int k = trainingData.size();
		int i = 1;
		int j = 0;
		double delta;
		boolean initIter = true;
		
		//initialize to results of Model1
		CounterMap<String, String> prevEAndFCounts = null;
		Counter<String> prevECounts = null;
		CounterMap<Pair<Integer, Integer>, Pair<Integer, Integer>> prevJGivenIlmCounts = null;
		Counter<Pair<Integer, Pair<Integer, Integer>>> prevIlmCounts = null;
		
		for (int iter = 0; iter < 10; iter++) {
			eAndFCounts = new CounterMap<String, String>();
			eCounts = new Counter<String>();
			jGivenIlmCounts = new CounterMap<Pair<Integer, Integer>, Pair<Integer, Integer>>();
			ilmCounts = new Counter<Pair<Integer, Pair<Integer, Integer>>>();
			
	
			
			
			for (SentencePair sentencePair: trainingData) { //for all k
				List<String> sourceWords = sentencePair.getSourceWords();
				sourceWords.add(0,NULL_WORD);
				List<String> targetWords = sentencePair.getTargetWords();
				int m_k = sourceWords.size();
				int l_k = targetWords.size();
				
				i = 1;
				
				for (String sourceWord : sourceWords) {
					
					double normalizer = 0;
					double qt = 0;
					
					//sum up all the qt values to get normalization factor
					j = 0; //TODO: why target has null? shouldn't source have null?
					for(String targetWord : targetWords) {
						if (prevJGivenIlmCounts == null) {
							qt = 1;
						}
						else {
							qt = ((double) prevJGivenIlmCounts.getCount(new Pair<Integer,Integer>(j,i), new Pair<Integer,Integer>(l_k,m_k))) 
									/ prevIlmCounts.getCount(new Pair<Integer,Pair<Integer,Integer>>(i,new Pair<Integer,Integer>(l_k,m_k)))
									* prevEAndFCounts.getCount(targetWord, sourceWord)
									/ prevECounts.getCount(targetWord);
						}
											
						normalizer += qt;
						j++;
					}
					
					j = 0; //TODO: why target has null? shouldn't source have null?
					for(String targetWord : targetWords) {
						if (prevJGivenIlmCounts == null) {
							qt = 1;
						}
						else {
							qt = ((double) prevJGivenIlmCounts.getCount(new Pair<Integer,Integer>(j,i), new Pair<Integer,Integer>(l_k,m_k))) 
									/ prevIlmCounts.getCount(new Pair<Integer,Pair<Integer,Integer>>(i,new Pair<Integer,Integer>(l_k,m_k)))
									* prevEAndFCounts.getCount(targetWord, sourceWord)
									/ prevECounts.getCount(targetWord);
						}
						
						delta = qt/normalizer;
						
						eAndFCounts.incrementCount(targetWord, sourceWord, delta);
						eCounts.incrementCount(targetWord, delta); //TODO: why is this in inner for loop
						jGivenIlmCounts.incrementCount(new Pair<Integer,Integer>(j,i),
								new Pair<Integer,Integer>(l_k,m_k), delta);
						ilmCounts.incrementCount(new Pair<Integer,Pair<Integer,Integer>>(i,new Pair<Integer,Integer>(l_k,m_k)),delta);
						
						
						j++;
					}
					
					
					i++;
				}			
			}
			
			prevEAndFCounts = eAndFCounts;
			prevECounts = eCounts;
			prevJGivenIlmCounts = jGivenIlmCounts;
			prevIlmCounts = ilmCounts;
		}
		
	}

}
