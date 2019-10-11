public class ClusteringKMeans {
    public static void main(String[] args) {
        KMeansClustering xyClustering = new KMeansClustering("src/training_location.arff", "src/xyTesting.arff", "src/xyPredict.arff");
        xyClustering.trainAndTestSimpleKMean(15);
        xyClustering.predictInstances();
    }
}
