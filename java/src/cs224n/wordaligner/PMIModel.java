package cs224n.wordaligner;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import cs224n.util.Counter;
import cs224n.util.CounterMap;
import cs224n.util.Pair;

public class PMIModel implements WordAligner {
	
	private CounterMap<String, String> counterMap;
    private Counter<String> sourceCounter;
    private Counter<String> targetCounter;
    private int numSentencePairs = 0;
    
    public PMIModel() {
    	counterMap = new CounterMap<String,String>();
    	sourceCounter = new Counter<String>();
    	targetCounter = new Counter<String>();
    }
    
	@Override
	public Alignment align(SentencePair sentencePair) {
		Alignment alignment = new Alignment();
		
		List<String> sourceSentence = sentencePair.getSourceWords();
		List<String> targetSentence = sentencePair.getTargetWords();
		//sourceSentence.add(0, NULL_WORD);

		double pmi = 0;
		double maxPmi = 0;
		int maxIndex = 0;
		int targetIndex = 0;
		int sourceIndex = 0;

		for(String targetWord : targetSentence) {
			maxIndex = 0;
			maxPmi = 0;
			for (String sourceWord : sourceSentence) {
				pmi = counterMap.getCount(sourceWord,  targetWord) /
						((double) (sourceCounter.getCount(sourceWord) * targetCounter.getCount(targetWord)));
				if(pmi > maxPmi) {
					maxPmi = pmi;
					maxIndex = sourceIndex;
				}
				
				sourceIndex++;
				
			}
			if (!sourceSentence.get(maxIndex).equals(NULL_WORD)) {
				alignment.addPredictedAlignment(targetIndex, maxIndex); //0 is <NULL>, which shifts everything down one
			}
			targetIndex++;
			sourceIndex = 0;
		}
		return alignment;
	}

	@Override
	public void train(List<SentencePair> trainingData) {
		//use sets to keep track of unique pairs
		Set<Pair<String, String>> pairSet = new HashSet<Pair<String,String>>();
		
		for (SentencePair sentencePair : trainingData) {
			numSentencePairs++;
			
			List<String> sourceSentence = sentencePair.getSourceWords();
			List<String> targetSentence = sentencePair.getTargetWords();
			sourceSentence.add(NULL_WORD);
			
			for (String targetWord : targetSentence) {
				targetCounter.incrementCount(targetWord, 1);
			}
			
			for (String sourceWord : sourceSentence) {
				sourceCounter.incrementCount(sourceWord, 1);
				pairSet.clear();
				for (String targetWord : targetSentence) {
					Pair<String,String> pair = new Pair<String, String>(sourceWord, targetWord);
					if (!pairSet.contains(pair)) {
						counterMap.incrementCount(sourceWord, targetWord, 1);
						pairSet.add(pair);
					}
				}
			}
		}
		
	}
  
	
  
}
