import org.apache.commons.lang.ArrayUtils;
import org.apache.log4j.BasicConfigurator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.CachingRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.*;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

public class Mahout_DT {
    public static void main(String[] args) throws Exception {
//        关闭日志
//        BasicConfigurator.configure();
        FileReader in = new FileReader("data/mahout_userid.txt");
        BufferedReader br = new BufferedReader(in);
        String line;
        int num = 0;
        String fileName = "data/python_file.txt";
        PrintWriter outputStream = new PrintWriter(fileName);
        while ((line = br.readLine()) != null) {
//            System.out.println(num);
            num++;
            if (num % 100 == 0) {
                System.out.println(num);
            }
            DataModel model = new FileDataModel(new File("data/mahout_all_data.csv"));
            UserSimilarity similarity = new LogLikelihoodSimilarity(model);
            UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.85, similarity, model);
            long userID = Long.parseLong(line);
            UserBasedRecommender userBasedRecommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
            long[] useridarr = userBasedRecommender.mostSimilarUserIDs(userID, 30);
            Long[] longObjects = ArrayUtils.toObject(useridarr);
            List<Long> longList = java.util.Arrays.asList(longObjects);
            outputStream.print(userID);
            for (Long value : longList) {
                outputStream.print(",");
                outputStream.print(value);
            }
            outputStream.println();
            outputStream.flush();
        }
        outputStream.close();

    }
}