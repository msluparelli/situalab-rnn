package com.situalab.dlab;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.util.DoubleAccumulator;

import java.io.BufferedWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import static org.apache.spark.api.java.StorageLevels.MEMORY_ONLY_SER;

public class rnn {

    public static String localdirectory = "/Users/mike/GoogleDrive/situa/operaciones/deeplab/";

    public static void main(String[] args) throws Exception {

        //get params
        String[] dlabargs = getdlabRNNparams(args);
        Map<String, String> RNNparams = getRNNparams(dlabargs);
        for (int m = 0; m < dlabargs.length; m++) {
            String key = dlabargs[m].split(":")[0];
            String val = dlabargs[m].split(":")[1];
            RNNparams.replace(key, val);
        }

        //save output
        Charset charset = Charset.forName("US-ASCII");
        String deeplabmode = RNNparams.get("deeplabmode").toString();
        String directory = "";
        if (deeplabmode.equals("debug")){
            directory = localdirectory;
        }
        String modelgen = directory+"trainingoutput/modelGeneralisation/RNNmodelOutput.csv";
        Path modelgenpath = Paths.get(modelgen); //path to save thetas
        BufferedWriter genwriter = Files.newBufferedWriter(modelgenpath, charset); // thetas writter
        genwriter.close();

        //rnn classes
        rnn_trainVRNNthetas VRNNthetas = new rnn_trainVRNNthetas();
        rnn_trainLSTMthetas LSTMthetas = new rnn_trainLSTMthetas();



        //gradient check
        if (RNNparams.get("checkGradients").equals("yes")) {
            if (RNNparams.get("RNN_mode").equals("VRNN")) {

                rnn_gradientcheckVRNN VRNNgc = new rnn_gradientcheckVRNN();
                VRNNgc.checkGradient();

            } else if (RNNparams.get("RNN_mode").equals("LSTM")) {

                rnn_gradientcheckLSTM LSTMgc = new rnn_gradientcheckLSTM();
                LSTMgc.checkGradient();

            } else if (RNNparams.get("RNN_mode").equals("GRU")) {

                rnn_gradientcheckGRU GRUgc = new rnn_gradientcheckGRU();
                GRUgc.checkGradient();

            } else {
                System.out.println("not a valid RNN mode");
            }

        }



        //multi training params
        String nninputList = RNNparams.get("hidden_states");
        String[] nnV = nninputList.split("-");
        List<String> nnlayersL = new ArrayList<>();
        for (int L = 0; L < nnV.length; L++) {
            nnlayersL.add(nnV[L]);
        }

        String learninginputList = RNNparams.get("learning");
        String[] learningV = learninginputList.split(",");
        List<String> learningL = new ArrayList<>();
        for (int L = 0; L < learningV.length; L++) {
            learningL.add(learningV[L]);
        }


        //training
        for (int nnt = 0; nnt < nnlayersL.size(); nnt++) {

            String layers = nnlayersL.get(nnt);
            RNNparams.replace("hidden_states", layers);

            for (int lt = 0; lt < learningL.size(); lt++) {

                //spark hadoop context
                String datafile = RNNparams.get("filepath");
                SparkConf conf = new SparkConf().setAppName("situalab");
                conf.setMaster("local");
                JavaSparkContext jsc = new JavaSparkContext(conf);
                String macpath = "situalab";
                String validationData = RNNparams.get("validationdata");


                //training data
                String dlab = RNNparams.get("filepath").toString();
                //double[] dataSplit = {0.8, 0.2}; //train / val split
                //System.out.println("local "+RNNparams.get("local"));
                String trainfilePath = "hdfs://localhost:9000/user/" + RNNparams.get("local").toString() + "/input/" + dlab + "train"; //train
                JavaRDD<String> traindata = jsc.textFile(trainfilePath);

                String valfilePath = "hdfs://localhost:9000/user/" + RNNparams.get("local").toString() + "/input/" + dlab + "val"; //train
                JavaRDD<String> valdata = jsc.textFile(valfilePath);


                //persist data
                traindata.persist(MEMORY_ONLY_SER);
                valdata.persist(MEMORY_ONLY_SER);


                //rnn params
                String learningR = learningL.get(lt);
                RNNparams.replace("learning", learningR);
                RNNparams.replace("train", String.valueOf(traindata.count()));
                RNNparams.replace("val", String.valueOf(valdata.count()));

                //output RNN params
                printRNNparams(RNNparams);
                if (RNNparams.get("RNN_mode").equals("VRNN")){

                    VRNNthetas.trainVRNNthetas(jsc, RNNparams, traindata, valdata);

                } else if (RNNparams.get("RNN_mode").equals("LSTM")){

                    LSTMthetas.trainLSTMthetas(jsc, RNNparams, traindata, valdata);

                } else {

                    System.out.println("not a valid RNN mode");
                }

                jsc.stop();

            }
        }







    }








    //preprocess input data
    public static List<double[]> getinputcat (String inputx){
        List<double[]> seqXinput = new ArrayList<>();
        String[] seqX = inputx.split(";");

        double[] cat = new double[seqX.length];
        seqXinput.add(cat);
        for (int s=0; s<seqX.length; s++){
            String[] seqXscat = seqX[s].split("cat"); //split input data from classification
            seqXinput.get(0)[s] = Double.parseDouble(seqXscat[1]); //classification array

            String[] seqXs = seqXscat[0].split(","); //split input data
            double[] seqXss = Arrays.stream(seqXs).mapToDouble(Double::parseDouble).toArray();
            seqXinput.add(seqXss);
        }
        return seqXinput;
    }


    //init Accumulators
    public static List<DoubleAccumulator> initAccumulators(JavaSparkContext jsc) {

        List<DoubleAccumulator> AccumList = new ArrayList<>();

        DoubleAccumulator JthetasCost = new DoubleAccumulator(); //J Accumulator
        DoubleAccumulator mEvents = new DoubleAccumulator(); //events Accumulator
        DoubleAccumulator binA = new DoubleAccumulator(); //binary Accumulator

        JthetasCost = jsc.sc().doubleAccumulator("cost function"); //init at zero
        mEvents = jsc.sc().doubleAccumulator("events"); //init at zero
        binA = jsc.sc().doubleAccumulator("binA"); //init at zero

        AccumList.add(JthetasCost);
        AccumList.add(mEvents);
        AccumList.add(binA);

        return AccumList;

    }

    //print RNN params
    public static void printRNNparams (Map<String,String> RNNparams){
        System.out.print("RNN_mode:"+RNNparams.get("RNN_mode")+"["+RNNparams.get("hidden_states")+"]"+RNNparams.get("learning")+RNNparams.get("update")+"["+RNNparams.get("maxmu")+"]"+RNNparams.get("RNN_architecture")+":"+RNNparams.get("filepath"));
        System.out.println("[triain:"+RNNparams.get("train")+"-val:"+RNNparams.get("val")+"]");
    }


    //get params
    private static String[] getdlabRNNparams(String[] args){
        String deeplabDNNparams;
        try{
            deeplabDNNparams = args[0];
        } catch (java.lang.ArrayIndexOutOfBoundsException e) {
            deeplabDNNparams = "RNN_mode:LSTM;filepath:DH_;epochs:490;output:2;learning:7.e-1;hidden_states:12;deeplabmode:debug;checkGradients:no;local:mike";
        }
        return deeplabDNNparams.split(";");
    }
    //update RNN params
    public static Map<String,String> getRNNparams(String[] args){

        Map<String,String> nnparams = new HashMap<>();
        nnparams.put("filepath", "");
        nnparams.put("seed", "4237842");
        nnparams.put("RNN_mode", "LSTM"); //LSTM or VRNN or GRU[not yet available]
        nnparams.put("hidden_states", "4");
        nnparams.put("output", "2"); //softmax classification
        nnparams.put("learning", "1.e-1");
        nnparams.put("update", "momentum"); //vanilla; momentum
        nnparams.put("RNN_architecture", "M2O"); //M2O:manyTOone; M2M:manyTOmany
        nnparams.put("checkGradients", "no");
        nnparams.put("local", "mike");
        nnparams.put("epochs", "200");
        nnparams.put("itert", "1");
        nnparams.put("lmbda", "1.e-10");
        nnparams.put("maxmu", "0.95");
        nnparams.put("gradient", "SGD");
        nnparams.put("learning_mod", ""); //ADA
        nnparams.put("train", "0");
        nnparams.put("val", "0");
        nnparams.put("test", "0");
        nnparams.put("seq_analytics", "no"); //yes,no

        nnparams.put("deeplabmode", "");
        //save: save cost and thetas
        //debug: to develop
        //"": train
        //append: save in modelPerformance file


        nnparams.put("validationdata", "no"); //yes or no; deprecated


        return nnparams;
    }

}
