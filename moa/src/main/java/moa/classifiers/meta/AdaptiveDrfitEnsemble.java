package moa.classifiers.meta;

import weka.core.Instance;
import weka.core.Utils;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.core.driftdetection.DriftDetectionMethod;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;



public class AdaptiveDrfitEnsemble extends AbstractClassifier{
	private static final long serialVersionUID = 1L;

	public String getPurposeString() {
        return "This algorithm improved Accuracy Update Ensemble by add a Drift Detect Method leaner as chunk handler. Hence, chunk size can be adjusted adaptively. Adaptive Drift Ensemble was proposed by Hong-Che Lin, and it can be found in thesis \"An Application of Streaming Data Analysis on TAIEX Futures\"";
    }
	/**
	 * Type of classifier to use as a component classifier.
	 */
	public ClassOption learnerOption = new ClassOption("learner", 'l', "Classifier to train.", Classifier.class,"bayes.NaiveBayes");
	/**
	 * DDM method Option
	 */
	public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',"Drift detection method to use.", DriftDetectionMethod.class, "DDM");
	/**
	 * control learners of ensemble
	 */
	protected LearnersHandler learnersHandler;
	/**
	 * chunkhandler for handling chunk with DDM
	 */
	protected ChunkHandler chunkHandler;
	/**
	 * 
	 */
	public boolean isRandomizable() {
		return false;
	}
	/**
	 * voteChoice = 0, vote combination
	 * voteChoice = 1, vote select best
	 */
	protected int voteChoice=0;

	public double[] getVotesForInstance(Instance inst) {
		switch(this.voteChoice){
		case 1: // select best one to return 
			double highest=0;
			int tmpindex=0;
			for(int i=0;i<this.learnersHandler.learnersSize();i++){
				if(this.learnersHandler.getWeight(i)>highest){
					highest =this.learnersHandler.getWeight(i);
					tmpindex = i;
				}
			}
			return this.learnersHandler.getVotesForInstance(tmpindex, inst).getArrayRef();
			
		case 0:	default: // return combination result
			DoubleVector combinedVote = new DoubleVector();
		
			if (this.trainingWeightSeenByModel > 0.0) {
				for (int i = 0; i < this.learnersHandler.learnersSize(); i++) {
					if (this.learnersHandler.getWeight(i) > 0.0) {
						DoubleVector vote = this.learnersHandler.getVotesForInstance(i, inst);

						if (vote.sumOfValues() > 0.0) {
							vote.normalize();
						// 	scale weight and prevent overflow
							vote.scaleValues(this.learnersHandler.getWeight(i) / (1.0 * this.learnersHandler.getTotalWeight() + 1.0));
							combinedVote.addValues(vote);
						}
					}
				}
			}
			combinedVote.normalize();
			return combinedVote.getArrayRef();
		}
		
		
	}
	public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		super.prepareForUseImpl(monitor, repository);
	}
	public void resetLearningImpl() {
		learnersHandler = new LearnersHandler(this.learnerOption);
		chunkHandler = new ChunkHandler(((DriftDetectionMethod) getPreparedClassOption(this.driftDetectionMethodOption)).copy(),learnersHandler.createlearner());
	}

	public void trainOnInstanceImpl(Instance inst) {
		/**
		 * 1.handle chunk & adjust learners
		 * 2.train learner
		 * 3.update weight
		 */
		chunkHandler.processNewInstance(inst,this.learnersHandler);
		
		if(!chunkHandler.isWarning())
			learnersHandler.trainNewestLearner(inst);
		
		if(chunkHandler.isOutOfControl())
			learnersHandler.resetWeights();
		learnersHandler.updateWeights(inst);
		
		
	}

	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	public void getModelDescription(StringBuilder out, int indent) {
	}
	
	private class ChunkHandler{
		protected DriftDetectionMethod driftDetectionMethod;
		protected final DriftDetectionMethod initialDriftDetectionMethod;
		protected int ddmLevel;
		protected boolean newClassifierReset;
		private Classifier ddmLearner,candidate;
		/**
		 * 
		 * @param method: use DDM or EDDM
		 * @param givenLearner: a classifier
		 */
		public ChunkHandler(DriftDetectionMethod method,Classifier givenLearner) {
			this.driftDetectionMethod = method;
			this.initialDriftDetectionMethod = method.copy();
			this.ddmLearner = givenLearner;
			this.candidate = ddmLearner.copy();
		}
		/**
		 * 
		 * @param inst
		 * @param lrsHandler: learnersHandler
		 * 1.update ddm
		 * 2.react for new ddm level
		 * 	i. 	incontrol level: do nothing
		 * 	ii.	warning level: train candidate classifier (newClassifier), reset candidate if necessary
		 * 	iii.outOfcontrol level: instead of current hold classifier by candidate, reset ddm
		 */
		public void processNewInstance(Instance inst,LearnersHandler lrsHandler){
			boolean predition = (Utils.maxIndex(this.ddmLearner.getVotesForInstance(inst)) == (int) inst.classValue());
			this.ddmLevel = this.driftDetectionMethod.computeNextVal(predition);
			switch(ddmLevel){
			case DriftDetectionMethod.DDM_INCONTROL_LEVEL:
				newClassifierReset = true;
				break;
			case DriftDetectionMethod.DDM_WARNING_LEVEL:
				if (newClassifierReset == true) {
                    this.candidate.resetLearning();
                    newClassifierReset = false;
                }
				this.candidate.trainOnInstance(inst);
				break;
			case DriftDetectionMethod.DDM_OUTCONTROL_LEVEL:
				this.ddmLearner = this.candidate.copy();
				lrsHandler.addLearnerIntoEnsemble(ddmLearner);
				newClassifierReset = true;
				initializeDDM();
				break;
			default:
				break;
			}
		}
		private void initializeDDM(){
			this.driftDetectionMethod = this.initialDriftDetectionMethod.copy();
		}
		public boolean isOutOfControl(){
			if(this.ddmLevel == DriftDetectionMethod.DDM_OUTCONTROL_LEVEL)
				return true;
			else
				return false;
		}
		public boolean isWarning(){
			if(this.ddmLevel == DriftDetectionMethod.DDM_WARNING_LEVEL)
				return true;
			else
				return false;
		}
	    
	}
	private class LearnersHandler{
		protected AutoExpandVector<Classifier> learners;
		protected AutoExpandVector<Double> weights,mse_i;
		protected AutoExpandVector<Long> classDistributions;
		protected AutoExpandVector<Boolean> isEnsembleOfIndex;
		protected int newestLearnerIndex=-1;
		private final Classifier initialClassifier;
		protected Classifier newestClassifier;
		protected long numberOfProcessedInstances;
		protected long numOfTotalProcessedInstances;
		
		
		public LearnersHandler(ClassOption preparedClassOption) {
			this.learners = new AutoExpandVector<Classifier>();
			this.initialClassifier = (Classifier) getPreparedClassOption(preparedClassOption);
			this.initialClassifier.resetLearning();
			
			this.weights = new AutoExpandVector<Double>();
			this.mse_i = new AutoExpandVector<Double>();
			this.classDistributions = new AutoExpandVector<Long>();
			this.isEnsembleOfIndex = new AutoExpandVector<Boolean>();
			this.numberOfProcessedInstances = 0;
			this.numOfTotalProcessedInstances=0;
			
		}

		public void updateWeights(Instance inst) {
			this.numberOfProcessedInstances++;
			this.numOfTotalProcessedInstances++;
			double mse_r = this.computeMseR(inst);
			for(int i=0;i<this.weights.size();i++){
				weights.set(i, 1.0 / (mse_r + this.computeMse(this.learners, i,inst) + Double.MIN_VALUE)); //mse_r should not be added by original paper
			}
			
		}

		protected double computeMse(AutoExpandVector<Classifier> learners,int index, Instance inst) {
			/**
			 * @param f_ci means probability the learner give the true class 
			 */
			double f_ci, voteSum=0;
			Classifier learner = learners.get(index);
			for (double element : learner.getVotesForInstance(inst)) {
				voteSum += element;
			}
			if (voteSum > 0) {
				double[] tmp = learner.getVotesForInstance(inst);
				if(tmp.length<(int)inst.classValue()+1)
					f_ci =0;
				else
					f_ci = tmp[(int) inst.classValue()]/ voteSum;

			}else{
				f_ci=0;
			}
			
			this.mse_i.set(index, this.mse_i.get(index)+((1-f_ci)*(1-f_ci)-this.mse_i.get(index))/(double)this.numberOfProcessedInstances);
			
			return mse_i.get(index);
		}

		protected double computeMseR(Instance inst) {
			double p_c;
			double mse_r = 0;
			if(this.classDistributions.get((int)inst.classValue())==null){
				this.classDistributions.set((int)inst.classValue(),1L);
			}else{
				this.classDistributions.set((int)inst.classValue(), this.classDistributions.get((int)inst.classValue())+1);
			}
			for (int i = 0; i < classDistributions.size(); i++) {
				if(this.classDistributions.get(i)!=null){
					p_c = (double) this.classDistributions.get(i) / (double) this.numberOfProcessedInstances;
					mse_r += p_c * ((1 - p_c) * (1 - p_c));
				}
			}

			return mse_r;
		}

		public void trainNewestLearner(Instance inst) {
			this.newestClassifier.trainOnInstance(inst);
		}

		public Classifier createlearner(){
			this.newestClassifier = this.initialClassifier.copy();
			addLearnerIntoEnsemble(this.newestClassifier);
			return newestClassifier;
		}
		public void addLearnerIntoEnsemble(Classifier learner){
			attemptMerge();
			this.learners.set(this.newestLearnerIndex, learner);
			this.weights.set(this.newestLearnerIndex, 0.0);
			this.mse_i.set(this.newestLearnerIndex, 0.0);
			this.newestClassifier = this.learners.get(this.newestLearnerIndex);
		}
		protected boolean attemptMerge(){
			if(this.newestLearnerIndex<0){
				this.newestLearnerIndex = this.learnersSize();
				return false;
			}
			boolean mergeTest=false;
			int bigIndex=-1,lowIndex=-1;
			for(int i=0;i<this.learnersSize();i++){
				if(i == this.newestLearnerIndex)
					continue;
				if(this.weights.get(i)>this.weights.get(this.newestLearnerIndex)){
					if(this.weights.get(bigIndex)==null)
						bigIndex = i;
					else if(this.weights.get(bigIndex)>this.weights.get(i)){
						bigIndex = i;
					}
				}else{
					if(this.weights.get(bigIndex)==null)
						lowIndex=i;
					else if(this.weights.get(lowIndex)<this.weights.get(i))
						lowIndex=i;
				}
			}
			double totalWeight = this.getTotalWeight();
			double newestRelatedWeight= this.weights.get(newestLearnerIndex)/totalWeight;
			boolean isBigBoundPassed = true,isLowBoundPassed = true;
			double lowRelatedWeight=0,bigRelatedWeight=0;
			if(bigIndex>0){ // test hoeffding bound
				bigRelatedWeight = this.weights.get(bigIndex) / totalWeight;
				if((bigRelatedWeight - newestRelatedWeight) >(computeHoeffdingBound(1,0.001,this.numOfTotalProcessedInstances)))
					isBigBoundPassed = true;
				else
					isBigBoundPassed = false;
			}
			if(lowIndex>0){ // test hoeffding bound
				lowRelatedWeight = this.weights.get(lowIndex) / totalWeight;
				if((newestRelatedWeight-lowRelatedWeight) >(computeHoeffdingBound(1,0.001,this.numOfTotalProcessedInstances)))
					isLowBoundPassed = true;
				else
					isLowBoundPassed = false;
			}
			int mergeIndex=-1;
			if(isBigBoundPassed && isLowBoundPassed){
				mergeTest = false;
			}
			else if(!(isBigBoundPassed || isLowBoundPassed)){
				mergeTest = true;
				double lowD = newestRelatedWeight - lowRelatedWeight;
				double bigD = bigRelatedWeight - newestRelatedWeight;
				mergeIndex = lowD-bigD<0? lowIndex:bigIndex; 
			} else if(isBigBoundPassed){
				if(bigIndex>0){
					mergeTest = true;
					mergeIndex = lowIndex;
				}else
					mergeTest = false;
			} else if(isLowBoundPassed){
				if(lowIndex>0){
					mergeTest = true;
					mergeIndex = bigIndex;
				} else
					mergeTest = false;
			}else{
				System.out.println("Error");
			}
				
			if(mergeTest) {// merge success
				merge(this.newestClassifier,mergeIndex);
				this.newestClassifier = null;
			}
			else{// merge failed
				this.newestLearnerIndex = this.learners.size();
			}
			return mergeTest;
		}
		private void merge(Classifier classifier,int index){
			if(this.isEnsembleOfIndex.get(index)!=null && this.isEnsembleOfIndex.get(index)){
				InnerMajorityEnsemble tmp = (InnerMajorityEnsemble)this.learners.get(index);
				tmp.addLearner(classifier);
			}else{
				InnerMajorityEnsemble newEnsemble = new InnerMajorityEnsemble();
				newEnsemble.addLearner(this.learners.get(index));
				newEnsemble.addLearner(classifier);
				this.learners.set(index, newEnsemble);
				this.isEnsembleOfIndex.set(index, true);
			}
		}
		protected double computeHoeffdingBound(double range, double confidence, double n) {
			return Math.sqrt(((range * range) * Math.log(1.0 / confidence)) / (2.0 * n));
		}
		public void resetWeights() {
			this.numberOfProcessedInstances = 0;
			this.classDistributions = new AutoExpandVector<Long>();
			for(int i=0;i<this.weights.size();i++){
				this.weights.set(i, 0.0);
				this.mse_i.set(i, 0.0);
			}
			
		}
		public int learnersSize(){
			return this.learners.size();
		}
		public double getWeight(int index){
			return this.weights.get(index);
		}
		public double getTotalWeight(){
			double sum=0;
			for(int i=0;i<this.learnersSize();i++){
				 
				if(this.learners.get(i)!=null){
					sum+=this.weights.get(i);
				}
					
			}
			return sum;
		}
		public DoubleVector getVotesForInstance(int index, Instance inst){
			return new DoubleVector(this.learners.get(index).getVotesForInstance(inst));
		}
	}
	
	protected class InnerMajorityEnsemble extends AbstractClassifier{
		private static final long serialVersionUID = -1L;
		protected AutoExpandVector<Classifier> learners;
		
		public InnerMajorityEnsemble(){
			this.learners = new AutoExpandVector<Classifier>();
		}
		public void addLearner(Classifier learner){
			this.learners.add(learner);
		}
		@Override
		public boolean isRandomizable() {
			return false;
		}

		@Override
		public double[] getVotesForInstance(Instance inst) {
			DoubleVector combinedVote = new DoubleVector();
				for (int i = 0; i < this.learners.size(); i++) {
					DoubleVector vote = new DoubleVector(this.learners.get(i).getVotesForInstance(inst));
						if (vote.sumOfValues() > 0.0) {
							vote.normalize();
							combinedVote.addValues(vote);
						}
				}
			combinedVote.normalize();
			return combinedVote.getArrayRef();
		}

		@Override
		public void resetLearningImpl() {
			this.learners = new AutoExpandVector<Classifier>();
		}

		@Override
		public void trainOnInstanceImpl(Instance inst) {
		}

		@Override
		protected Measurement[] getModelMeasurementsImpl() {
			return null;
		}

		@Override
		public void getModelDescription(StringBuilder out, int indent) {
			
		}
		
	}

}
