import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class KMeansClustering {
    String training_filename;
    String testing_filename;
    String predict_filename;

    SimpleKMeans clusteringModel;

    public KMeansClustering(String training_filename, String testing_filename, String predict_filename) {
        this.training_filename = training_filename;
        this.testing_filename = testing_filename;
        this.predict_filename = predict_filename;
    }

    public Instances getDataSet(String filename) {
        ArffLoader loader = new ArffLoader();
        try {
            loader.setFile(new File(filename));
            Instances dataSet = loader.getDataSet();
            return dataSet;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public void trainAndTestSimpleKMean(int numberOfClusters) {
        System.out.println("Training");
        Instances trainingDataSet = getDataSet(training_filename);
        clusteringModel = new SimpleKMeans();
        try {
            clusteringModel.setNumClusters(numberOfClusters);
            clusteringModel.buildClusterer(trainingDataSet);
            System.out.println(clusteringModel);

            Instances centroids = clusteringModel.getClusterCentroids();
            for (int i = 0; i < clusteringModel.getNumClusters(); i++) {
                System.out.println("Centroid " + i + ":" + centroids.instance(i));
                System.out.println(". Centroid size" + clusteringModel.getClusterSizes()[i]);
            }

            System.out.println("Testing with Class Clustering");
            Instances testDataSet = getDataSet(testing_filename);
            testDataSet.setClassIndex(testDataSet.numAttributes() - 1);

            ClusterEvaluation evaluation = new ClusterEvaluation();
            evaluation.setClusterer(clusteringModel);
            evaluation.evaluateClusterer(testDataSet);
            System.out.println(evaluation.clusterResultsToString());

            System.out.println("Cross Validation");
            EM expectMaximizedModel = new EM();
            expectMaximizedModel.setMaxIterations(15);
            expectMaximizedModel.buildClusterer(trainingDataSet);
            System.out.println(expectMaximizedModel);

            double logLikelyhood = ClusterEvaluation.crossValidateModel(expectMaximizedModel, trainingDataSet, 3, new Random(1));
            System.out.println("Logarithm of likelyhood is " + logLikelyhood);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void predictInstances() {
        System.out.println("Clustering Prediction");
        Instances predictDataSet = getDataSet(predict_filename);

        double clusterTH;
        Instance predictData;

        for (int i = 0; i < predictDataSet.numInstances(); i++) {
            predictData = predictDataSet.instance(i);
            try {
                clusterTH = clusteringModel.clusterInstance(predictData);
                System.out.println(clusterTH);
                System.out.println(predictData.toString() + "is in the cluster number " + clusterTH);
            } catch (Exception e) {
                e.printStackTrace();
            }

        }

    }
}
