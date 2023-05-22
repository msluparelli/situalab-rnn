package com.situalab.dlab;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.util.DoubleAccumulator;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;

import static com.situalab.dlab.rnn.initAccumulators;
import static com.situalab.dlab.rnn.localdirectory;
import static java.lang.Math.min;


public class rnn_trainVRNNthetas {

    //train VRNN
    public static void trainVRNNthetas (JavaSparkContext jsc,
                                        Map<String,String> RNNparams,
                                        JavaRDD<String> traindata,
                                        JavaRDD<String> valdata) throws IOException {

        //deeplab RNN classes
        rnn_mops matrixOps = new rnn_mops(); //matrix operations
        rnn_Weights initW = new rnn_Weights(); //weights
        rnn_optimization optMethod = new rnn_optimization();
        rnn_feedbackpropVRNN VRNN = new rnn_feedbackpropVRNN(); //VRNNfeedpropagation


        //RNN params
        int epochsN = Integer.parseInt(RNNparams.get("epochs"));
        double learning = Double.parseDouble(RNNparams.get("learning"));
        String optimization = RNNparams.get("update"); //vanilla; momentum
        String learning_mod = RNNparams.get("learning_mod");
        double lmbda = Double.parseDouble(RNNparams.get("lmbda")); //regularisation term
        int inputxdim = traindata.first().split(";")[1].split(",").length; //input dimension
        int hiddencells = Integer.parseInt(RNNparams.get("hidden_states"));; //hidden states dimension
        int outputcat = Integer.parseInt(RNNparams.get("output")); //output classification dimension
        double maxmu = Double.parseDouble(RNNparams.get("maxmu")); double itert = Double.parseDouble(RNNparams.get("itert")); //mu threshold
        String RNNmode = RNNparams.get("RNN_mode");
        String dlab = RNNparams.get("filepath");
        String deeplabmode = RNNparams.get("deeplabmode");
        String RNNhead = inputxdim+"_"+hiddencells+"_"+outputcat+"_"+RNNmode;
        String seqAnalytics = RNNparams.get("seq_analytics");



        //outside training loop
        List<double[][]> WeigthsL = initW.initWeights(inputxdim,hiddencells,outputcat,564323); //Weights; Wxh, Whh, Why
        List<double[][]> gthetas = new ArrayList<double[][]>(); //gradient thetas; outside loop
        List<double[][]> gthetasAcc = initW.getThetasLZeros(WeigthsL); //gradiente accumulator; outside loop
        List<double[][]> vWA = initW.getThetasLZeros(WeigthsL); //velocity accumulator; outside loop


        //flaten thetas to save
        List<double[]> thetastrainedL = new ArrayList<>();
        double[] thetastrained = initW.getflatThetas(WeigthsL); //to flaten thetasL


        //sgd random seed
        Random sgdSEED = new Random();
        sgdSEED.setSeed(2332);
        sgdSEED.ints(epochsN);
        int[] sgsSEEDvalues = new int[epochsN];
        for (int s=0; s<epochsN; s++){sgsSEEDvalues[s] = sgdSEED.nextInt(9999);}
        double sgd = min(512./512., 512./traindata.count()); double gross = 1.0-sgd; double[] wtSplit = {sgd, gross};


        //evaluation params
        double best_cost_val = 2.0; //best cost validation
        int best_cost_val_epoch = 0; //best cost validation epoch



        //save cost and thetas
        //save cost and thetas
        Charset charset = Charset.forName("US-ASCII");
        String directory = "";
        if (deeplabmode.equals("debug")){
            directory = localdirectory;
        }
        String modelgen = directory+"trainingoutput/modelGeneralisation/RNNmodelOutput.csv";
        Path modelgenpath = Paths.get(modelgen); //path to save thetas
        BufferedWriter genwriter = Files.newBufferedWriter(modelgenpath, charset); // thetas writter
        genwriter.close();


        //save cost and thetas
        String costfilepath = directory+"trainingoutput/trainingCost/"+dlab+RNNmode+optimization+"["+hiddencells+"]"+learning+".csv";
        if (deeplabmode.equals("save")){
            Path RNNcost = Paths.get(costfilepath); //path to save costs
            BufferedWriter costwriter = Files.newBufferedWriter(RNNcost, charset); //cost file writter
            String filelabel = "epoch,cost_train,cost_val,acc_val\n"; //encabezado cost file
            costwriter.write(filelabel); //save header
            costwriter.close();
        }
        String thetasfilepath = directory+"weights/"+dlab+"_"+RNNmode+optimization+"["+hiddencells+"]"+learning+"_"+epochsN+".csv";
        String seqcostfilepath = directory+"trainingoutput/trainingCost/"+dlab+"_"+RNNmode+optimization+"["+hiddencells+"]"+learning+"_"+epochsN+".csv";
        if (seqAnalytics.equals("yes")){
            Path RNNcost = Paths.get(seqcostfilepath); //path to save costs
            BufferedWriter costwriter = Files.newBufferedWriter(RNNcost, charset); //remove old file version if exists
            costwriter.close();
        }


        //Jout accumulator
        int hslayers = traindata.first().split(";").length; //hs layers
        rnn_thetasAccumV Jouthslayer = new rnn_thetasAccumV(hslayers);
        jsc.sc().register(Jouthslayer, "Joutlayers");




        System.out.println("\nVRNN training iterations:"+epochsN+"\n");

        //inside training loop
        for (int epoch=0; epoch<epochsN; epoch++) {

            //VanillaRNN Weights
            Broadcast<List<double[][]>> weigthsLbd = jsc.broadcast(WeigthsL); //broadcast deeplab_thetas
            List<rnn_thetasAccumM> dthetasAccL = initW.getAccThetasL(WeigthsL); //derivative [gradients] thetas, Accumulator
            for (int a=0; a<dthetasAccL.size(); a++){
                String wName = "dW".concat(Integer.toString(a));
                jsc.sc().register(dthetasAccL.get(a), wName); //register accumulators List
            }




            //final variables for iteration
            Jouthslayer.reset();
            List<DoubleAccumulator> AccumListFeed = initAccumulators(jsc); //init accumulators, feedpropagation
            List<DoubleAccumulator> AccumListBack = initAccumulators(jsc); //init accumulators, backpropagation
            List<double[][]> velocity = vWA;
            double e = -1-(Math.log(((epoch/itert)+1))/Math.log(2));
            double mu = Math.min(1-Math.pow(2,e), maxmu); //inspired by nesterov momentum value (in Sutskever)
            NumberFormat formatter = new DecimalFormat("#0.000");


            //VanillaRNN feed propagation algorithm
            traindata.foreach(xrdd -> VRNN.VRNNfeedprop("feed", xrdd, weigthsLbd.value(), dthetasAccL, AccumListFeed, Jouthslayer));


            //VanillaRNN SGD back propagation algorithm
            JavaRDD<String> sgddata = traindata.randomSplit(wtSplit, sgsSEEDvalues[epoch])[0]; //split and select data
            sgddata.foreach(xrdd -> VRNN.VRNNfeedprop("back", xrdd, weigthsLbd.value(), dthetasAccL, AccumListBack, Jouthslayer));



            //optimizationUpdate[vanilla|momentum]
            double n = traindata.count(); //used for <<weightDecay = (1-(ADAlearning*lmbda/n))>>
            if (optimization.equals("vanilla")){
                gthetas = optMethod.getderivativeBackProp(weigthsLbd.value(), dthetasAccL, lmbda, AccumListBack.get(1).value()); //derivative, gradients
                gthetasAcc = optMethod.accumGradients(gthetas, gthetasAcc); //accumulate gradients for ADAGRAD
                WeigthsL = optMethod.getthetasVanillaUpdated(weigthsLbd.value(), gthetas, learning, gthetasAcc, AccumListFeed.get(1).value(), lmbda, n); //vanillaUpdate, update
            } else if (optimization.equals("momentum")){
                gthetas = optMethod.getderivativeBackProp(weigthsLbd.value(), dthetasAccL, lmbda, AccumListBack.get(1).value()); //derivative, gradients
                gthetasAcc = optMethod.accumGradients(gthetas, gthetasAcc); //accumulate gradients for ADAGRAD
                vWA = optMethod.getmuVelocity(gthetas, vWA, mu, learning, gthetasAcc, learning_mod); //velocity
                WeigthsL = optMethod.getthetasMuUpdated(weigthsLbd.value(), vWA); //update
            } else {
                System.out.println("not a valid optimization method");
            }



            //validate and test with updated weightsL
            Broadcast<List<double[][]>> weightsLbdVAL = jsc.broadcast(WeigthsL); //broadcast updated deeplab_thetas
            List<DoubleAccumulator> AccumListVali = initAccumulators(jsc); //accumulators
            valdata.foreach(xrdd -> VRNN.VRNNfeedprop("val", xrdd, weightsLbdVAL.value(), dthetasAccL, AccumListVali, Jouthslayer));


            //regularised cost
            double regterm = lmbda/2;
            double sqwt = initW.getthetassqw(WeigthsL); //W squared

            //cost train
            double cost = 1./AccumListFeed.get(1).value() * AccumListFeed.get(0).value(); //train cost
            double regCostTrain = cost + (regterm*sqwt); //train reg cost

            //cost val
            double costVal = 1./AccumListVali.get(1).value() * AccumListVali.get(0).value(); //val cost
            double regCostVAL = costVal + (regterm*sqwt); //VAL reg cost
            double bnAccuracyval = AccumListVali.get(2).value() / valdata.count();

            double costdiff = Math.abs(regCostVAL-best_cost_val);
            if (regCostVAL<=best_cost_val) {
                best_cost_val = regCostVAL;
                best_cost_val_epoch = epoch+1;
            }

            String regCostTrain_ = formatter.format(regCostTrain);
            String regCostVAL_ = formatter.format(regCostVAL);
            String bnAccuracyval_ = formatter.format(bnAccuracyval);
            String bestregCostVal = formatter.format(best_cost_val);
            System.out.print("\repoch:"+(epoch+1)+" train cost:"+regCostTrain_+" val cost:"+regCostVAL_+"[best:"+bestregCostVal+":"+best_cost_val_epoch+"] val acc:"+bnAccuracyval_);

            //save cost train val
            String regcost = Double.toString(epoch)+","+Double.toString(regCostTrain)+","+Double.toString(regCostVAL)+","+Double.toString(bnAccuracyval)+"\n";
            if (deeplabmode.equals("save")){
                Writer costoutput = new BufferedWriter(new FileWriter(costfilepath, true));
                costoutput.append(regcost);
                costoutput.close();
            }

            //sequential cost analysys
            double[] seqvalcost = Arrays.stream(Jouthslayer.value()).map(c -> 1./AccumListVali.get(1).value() *c).toArray();
            double[] seqvalcostreg = Arrays.stream(seqvalcost).map(c -> c + (regterm*sqwt)).toArray();
            String seqregcostS = epoch+","+Arrays.toString(seqvalcostreg).replace("[","").replace("]","").replace(" ","")+"\n";
            if (seqAnalytics.equals("yes")){
                Writer seqcostoutput = new BufferedWriter(new FileWriter(seqcostfilepath, true));
                //System.out.println(seqregcostS);
                seqcostoutput.append(seqregcostS);
                seqcostoutput.close();
            }



            //testdata evaluation & save thetas
            if (epoch==(epochsN-1)*1) {

                System.out.println("\n\nend training, saving thetas... \n");

                //flaten thetas to save
                thetastrained = initW.getflatThetas(WeigthsL); //last thetas trained
                thetastrainedL.add(thetastrained); //add thetastrained to List


                //RNN trained thetas params
                String datos = RNNparams.get("filepath").toString();
                String optim = RNNparams.get("update").toString();
                String learn = RNNparams.get("learning").toString();
                String nnlay = RNNparams.get("hidden_states").toString().replace(",", "-");
                String muval = RNNparams.get("maxmu").toString();
                String muada = RNNparams.get("itert").toString();
                String EPoch = RNNparams.get("epochs").toString();
                String RNNmd = RNNparams.get("RNN_mode").toString();

                //save thetas
                if (deeplabmode.equals("save")){
                    saveThetas(thetastrainedL, thetasfilepath, RNNhead);
                }

                //RNN output
                String modelgeneralisation = datos+";"+RNNmd+"["+nnlay+"]"+EPoch+";"+best_cost_val_epoch+";"+bestregCostVal.replace(".",",")+";"+optim+"[mu:"+muval+":"+muada+"]"+learn+"\n";

                try (Writer modelWriter = new FileWriter(modelgen, true)){
                    modelWriter.write(modelgeneralisation);
                } catch (IOException error) {
                    System.out.println("Problem occurs when deleting the directory:" + modelgen);
                    error.printStackTrace();
                }





                /*

                //get trained thetasList
                List<double[][]> trainedWheightsL = initW.getVRNNtrainedThetasL(thetastrained, RNNhead);


                //predict test data
                Broadcast<List<double[][]>> weigthsLbdtest = jsc.broadcast(trainedWheightsL); //broadcast deeplab_thetas
                List<rnn_thetasAccumM> dthetasAccLtest = initW.getAccThetasL(trainedWheightsL); //derivative [gradients] thetas, Accumulator
                for (int a=0; a<dthetasAccLtest.size(); a++){
                    String wName = "dW".concat(Integer.toString(a));
                    jsc.sc().register(dthetasAccLtest.get(a), wName); //register accumulators List
                }

                List<DoubleAccumulator> AccumListTest = initAccumulators(jsc); //init Test accumulators
                testdata.foreach(xrdd -> VRNN.VRNNfeedprop("test", xrdd, weigthsLbdtest.value(), dthetasAccLtest, AccumListTest, Jouthslayer));

                //test data performance
                double sqwtfinal = initW.getthetassqw(trainedWheightsL); //W squared
                double costfinal = 1./AccumListTest.get(1).value() * AccumListTest.get(0).value(); //test cost
                double regTermfinal = lmbda/2; //Ng lambda/2m, stanford UFLDL lambda/2
                double regCostfinal = costfinal + (regTermfinal*sqwtfinal); //train reg cost
                double testbnAccuracy = AccumListTest.get(2).value() / testdata.count();

                //test data output
                String regcostfinalTest = formatter.format(regCostfinal);
                String testAccuracy = formatter.format(testbnAccuracy);

                System.out.println("test accuracy:"+testAccuracy+" test cost:"+regcostfinalTest);







                weigthsLbdtest.destroy(); //destroy broadcast at the end of test
                */

            }

            weigthsLbd.destroy(); //destroy broadcast at the end of loop
            weightsLbdVAL.destroy(); //destroy broadcast at the end of loop

        } //end training loop

    } //end trainVRNNthetas class



    //save thetas to file csv
    public static void saveThetas(List<double[]> thetastrainedL,
                                  String thetasfilepath,
                                  String RNNhead) throws IOException {

        Charset charset = Charset.forName("US-ASCII");
        Path nnthetas = Paths.get(thetasfilepath); //path to save thetas
        BufferedWriter thetaswriter = Files.newBufferedWriter(nnthetas, charset); // thetas writter
        for (int w=0; w<thetastrainedL.size(); w++){
            String thetastosave = RNNhead+"_"+Arrays.toString(thetastrainedL.get(w))+"\n";
            String thetastosave_ = thetastosave.replace("[","").replace("]","").replace(" ","");
            thetaswriter.write(thetastosave_);
        }
        thetaswriter.close();

    }


}
