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


public class rnn_trainLSTMthetas {

    //Output
    public static List<double[]> model_output = new ArrayList<double[]>();

    public static void trainLSTMthetas(JavaSparkContext jsc,
                                       Map<String,String> RNNparams,
                                       JavaRDD<String> traindata,
                                       JavaRDD<String> valdata) throws IOException {

        //deeplab RNN classes
        rnn_mops matrixOps = new rnn_mops(); //matrix operations
        rnn_Weights initW = new rnn_Weights(); //weights
        rnn_optimization optMethod = new rnn_optimization();
        rnn_feedbackpropLSTM LSTM = new rnn_feedbackpropLSTM(); //VRNNfeedpropagation


        //RNN params
        int epochsN = Integer.parseInt(RNNparams.get("epochs"));
        double learning = Double.parseDouble(RNNparams.get("learning"));
        String optimization = RNNparams.get("update"); //momentum
        if (optimization.equals("vanilla")) {
            System.out.println("vanilla update not available for LSTM. Changed to momentum update");
            optimization = "momentum";
        }
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
        String RNNArchitecture = RNNparams.get("RNN_architecture"); //RNN_architecture M2O | M2M


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
        Charset charset = Charset.forName("US-ASCII");
        String directory = "";
        if (deeplabmode.equals("debug")){
            directory = localdirectory;
        }
        String modelgen = directory+"trainingoutput/modelGeneralisation/RNNmodelOutput.csv";
        Path modelgenpath = Paths.get(modelgen); //path to save thetas
        BufferedWriter genwriter = Files.newBufferedWriter(modelgenpath, charset); // thetas writter
        genwriter.close();




        String costfilepath = directory+"trainingoutput/modelCosts/"+dlab+RNNmode+optimization+"["+hiddencells+"]"+learning+".csv";
        if (deeplabmode.equals("save")){
            Path RNNcost = Paths.get(costfilepath); //path to save costs
            BufferedWriter costwriter = Files.newBufferedWriter(RNNcost, charset); //cost file writter
            String filelabel = "epoch,cost_train,cost_val,acc_val\n"; //encabezado cost file
            costwriter.write(filelabel); //save header
            costwriter.close();
        }
        String thetasfilepath = directory+"weights/"+dlab+"_"+RNNmode+optimization+"["+hiddencells+"]"+learning+"_"+epochsN;
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


        //Wi
        List<double[][]> WiL = initW.initWeightsGates(inputxdim,hiddencells,234819);
        List<double[][]> gWi = new ArrayList<double[][]>(); //gradient thetas; outside loop
        List<double[][]> gWiAcc = initW.getThetasLZeros(WiL); //gradiente accumulator; outside loop
        List<double[][]> vWiA = initW.getThetasLZeros(WiL); //velocity accumulator; outside loop

        //Wf
        List<double[][]> WfL = initW.initWeightsGates(inputxdim,hiddencells,109274);
        List<double[][]> gWf = new ArrayList<double[][]>(); //gradient thetas; outside loop
        List<double[][]> gWfAcc = initW.getThetasLZeros(WfL); //gradiente accumulator; outside loop
        List<double[][]> vWfA = initW.getThetasLZeros(WfL); //velocity accumulator; outside loop

        //Wo
        List<double[][]> WoL = initW.initWeightsGates(inputxdim,hiddencells,554637);
        List<double[][]> gWo = new ArrayList<double[][]>(); //gradient thetas; outside loop
        List<double[][]> gWoAcc = initW.getThetasLZeros(WoL); //gradiente accumulator; outside loop
        List<double[][]> vWoA = initW.getThetasLZeros(WoL); //velocity accumulator; outside loop

        //Wg
        List<double[][]> WgL = initW.initWeights(inputxdim,hiddencells,outputcat,718274); //contains Why
        List<double[][]> gWg = new ArrayList<double[][]>(); //gradient thetas; outside loop
        List<double[][]> gWgAcc = initW.getThetasLZeros(WgL); //gradiente accumulator; outside loop
        List<double[][]> vWgA = initW.getThetasLZeros(WgL); //velocity accumulator; outside loop

        //flaten Weights
        double[] Witrained = initW.getflatThetas(WiL); //to flaten thetasL
        double[] Wftrained = initW.getflatThetas(WfL); //to flaten thetasL
        double[] Wotrained = initW.getflatThetas(WoL); //to flaten thetasL;
        double[] Wgtrained = initW.getflatThetas(WgL); //to flaten thetas
        List<double[]> WtrainedL = new ArrayList<>();




        System.out.println("\nVRNN training iterations:"+epochsN+"\n");
        for (int epoch=0; epoch<epochsN; epoch++) {

            //broadcast Wi
            Broadcast<List<double[][]>> WiLbd = jsc.broadcast(WiL); //broadcast deeplab_thetas
            List<rnn_thetasAccumM> dWiAccL = initW.getAccThetasL(WiL); //derivative [gradients] thetas, Accumulator
            for (int a=0; a<dWiAccL.size(); a++){
                String wName = "dW".concat(Integer.toString(a));
                jsc.sc().register(dWiAccL.get(a), wName); //register accumulators List
            }

            //broadcast Wf
            Broadcast<List<double[][]>> WfLbd = jsc.broadcast(WfL); //broadcast deeplab_thetas
            List<rnn_thetasAccumM> dWfAccL = initW.getAccThetasL(WfL); //derivative [gradients] thetas, Accumulator
            for (int a=0; a<dWfAccL.size(); a++){
                String wName = "dWf".concat(Integer.toString(a));
                jsc.sc().register(dWfAccL.get(a), wName); //register accumulators List
            }

            //broadcast Wo
            Broadcast<List<double[][]>> WoLbd = jsc.broadcast(WoL); //broadcast deeplab_thetas
            List<rnn_thetasAccumM> dWoAccL = initW.getAccThetasL(WoL); //derivative [gradients] thetas, Accumulator
            for (int a=0; a<dWoAccL.size(); a++){
                String wName = "dWo".concat(Integer.toString(a));
                jsc.sc().register(dWoAccL.get(a), wName); //register accumulators List
            }

            //broadcast Wg
            Broadcast<List<double[][]>> WgLbd = jsc.broadcast(WgL); //broadcast deeplab_thetas
            List<rnn_thetasAccumM> dWgAccL = initW.getAccThetasL(WgL); //derivative [gradients] thetas, Accumulator
            for (int a=0; a<dWgAccL.size(); a++){
                String wName = "dWg".concat(Integer.toString(a));
                jsc.sc().register(dWgAccL.get(a), wName); //register accumulators List
            }


            //final variables for iteration
            Jouthslayer.reset();
            List<DoubleAccumulator> AccumListFeed = initAccumulators(jsc); //init accumulators, feedpropagation
            List<DoubleAccumulator> AccumListBack = initAccumulators(jsc); //init accumulators, backpropagation
            List<double[][]> vi = vWiA; //velocity i
            List<double[][]> vf = vWfA; //velocity f
            List<double[][]> vo = vWoA; //velocity o
            List<double[][]> vg = vWgA; //velocity g
            double e = -1-(Math.log(((epoch/itert)+1))/Math.log(2));
            double mu = Math.min(1-Math.pow(2,e), maxmu); //inspired by nesterov momentum value (in Sutskever)
            NumberFormat formatter = new DecimalFormat("#0.000");


            //LSTM feedpropagation
            traindata.foreach(xrdd -> LSTM.LSTMfeedprop("feed", xrdd, WiLbd.value(), WfLbd.value(), WoLbd.value(), WgLbd.value(), AccumListFeed, dWiAccL, dWfAccL, dWoAccL, dWgAccL, Jouthslayer, RNNArchitecture));

            //LSTM SGD back propagation algorithm
            JavaRDD<String> sgddata = traindata.randomSplit(wtSplit, sgsSEEDvalues[epoch])[0]; //split and select data
            sgddata.foreach(xrdd -> LSTM.LSTMfeedprop("back", xrdd, WiLbd.value(), WfLbd.value(), WoLbd.value(), WgLbd.value(), AccumListBack, dWiAccL, dWfAccL, dWoAccL, dWgAccL, Jouthslayer, RNNArchitecture));

            //optimizationUpdate[vanilla|momentum]
            double n = traindata.count(); //used for <<weightDecay = (1-(ADAlearning*lmbda/n))>>
            if (optimization.equals("momentum")) {

                //Wi
                gWi = optMethod.getderivativeBackProp(WiLbd.value(), dWiAccL, lmbda, AccumListBack.get(1).value()); //derivative, gradients
                gWiAcc = optMethod.accumGradients(gWi, gWiAcc); //accumulate gradients for ADAGRAD
                vWiA = optMethod.getmuVelocity(gWi, vWiA, mu, learning, gWiAcc, learning_mod); //velocity
                WiL = optMethod.getthetasMuUpdated(WiLbd.value(), vWiA); //update
                
                //Wf
                gWf = optMethod.getderivativeBackProp(WfLbd.value(), dWfAccL, lmbda, AccumListBack.get(1).value()); //derivative, gradients
                gWfAcc = optMethod.accumGradients(gWf, gWfAcc); //accumulate gradients for ADAGRAD
                vWfA = optMethod.getmuVelocity(gWf, vWfA, mu, learning, gWfAcc, learning_mod); //velocity
                WfL = optMethod.getthetasMuUpdated(WfLbd.value(), vWfA); //update

                //Wo
                gWo = optMethod.getderivativeBackProp(WoLbd.value(), dWoAccL, lmbda, AccumListBack.get(1).value()); //derivative, gradients
                gWoAcc = optMethod.accumGradients(gWo, gWoAcc); //accumulate gradients for ADAGRAD
                vWoA = optMethod.getmuVelocity(gWo, vWoA, mu, learning, gWoAcc, learning_mod); //velocity
                WoL = optMethod.getthetasMuUpdated(WoLbd.value(), vWoA); //update

                //Wg
                gWg = optMethod.getderivativeBackProp(WgLbd.value(), dWgAccL, lmbda, AccumListBack.get(1).value()); //derivative, gradients
                gWgAcc = optMethod.accumGradients(gWg, gWgAcc); //accumulate gradients for ADAGRAD
                vWgA = optMethod.getmuVelocity(gWg, vWgA, mu, learning, gWgAcc, learning_mod); //velocity
                WgL = optMethod.getthetasMuUpdated(WgLbd.value(), vWgA); //update

            } else {
                System.out.println("not a valid optimization method");
            }

            //validation
            Broadcast<List<double[][]>> WiLbdVAL = jsc.broadcast(WiL); //broadcast updated deeplab_thetas
            Broadcast<List<double[][]>> WfLbdVAL = jsc.broadcast(WfL); //broadcast updated deeplab_thetas
            Broadcast<List<double[][]>> WoLbdVAL = jsc.broadcast(WoL); //broadcast updated deeplab_thetas
            Broadcast<List<double[][]>> WgLbdVAL = jsc.broadcast(WgL); //broadcast updated deeplab_thetas
            List<DoubleAccumulator> AccumListVali = initAccumulators(jsc); //accumulators
            valdata.foreach(xrdd -> LSTM.LSTMfeedprop("val", xrdd, WiLbdVAL.value(), WfLbdVAL.value(), WoLbdVAL.value(), WgLbdVAL.value(), AccumListVali, dWiAccL, dWfAccL, dWoAccL, dWgAccL, Jouthslayer, RNNArchitecture));



            //regularised cost
            double regterm = lmbda/2;
            double sqWi = initW.getthetassqw(WiL); //Wi squared
            double sqWf = initW.getthetassqw(WfL); //Wf squared
            double sqWo = initW.getthetassqw(WoL); //Wo squared
            double sqWg = initW.getthetassqw(WgL); //Wg squared
            double regu = (regterm*(sqWi+sqWf+sqWo+sqWg));

            //cost train
            double costTrain = 1./AccumListFeed.get(1).value() * AccumListFeed.get(0).value(); //train cost
            double regCostTrain = costTrain + regu; //train reg cost

            //cost val
            double costVal = 1./AccumListVali.get(1).value() * AccumListVali.get(0).value(); //val cost
            double regCostVAL = costVal + regu; //VAL reg cost
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
            double[] seqvalcostreg = Arrays.stream(seqvalcost).map(c -> c + regu).toArray();
            String seqregcostS = epoch+","+Arrays.toString(seqvalcostreg).replace("[","").replace("]","").replace(" ","")+"\n";
            if (seqAnalytics.equals("yes")){
                Writer seqcostoutput = new BufferedWriter(new FileWriter(seqcostfilepath, true));
                //System.out.println(seqregcostS);
                seqcostoutput.append(seqregcostS);
                seqcostoutput.close();
            }


            //testdata evaluation & save thetas
            if (epoch==(epochsN-1)*1) {

                System.out.println("\n\nend training, saving weights...\n");

                //flaten thetas to save
                Witrained = initW.getflatThetas(WiL); //last thetas trained
                WtrainedL.add(Witrained); //add thetastrained to List

                Wftrained = initW.getflatThetas(WfL); //last thetas trained
                WtrainedL.add(Wftrained); //add thetastrained to List

                Wotrained = initW.getflatThetas(WoL); //last thetas trained
                WtrainedL.add(Wotrained); //add thetastrained to List

                Wgtrained = initW.getflatThetas(WgL); //last thetas trained
                WtrainedL.add(Wgtrained); //add thetastrained to List

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
                    saveThetas(WtrainedL, thetasfilepath, RNNhead);
                }

                //RNN output
                String modelgeneralisation = datos+";"+RNNmd+"["+nnlay+"]"+EPoch+";"+best_cost_val_epoch+";"+bestregCostVal.replace(".",",")+";"+optim+"[mu:"+muval+":"+muada+"]"+learn+"\n";

                try (Writer modelWriter = new FileWriter(modelgen, true)){
                    modelWriter.write(modelgeneralisation);
                } catch (IOException error) {
                    System.out.println("Problem occurs when deleting the directory:" + modelgen);
                    error.printStackTrace();
                }
                valdata.foreach(xrdd -> LSTM.LSTMfeedprop("test", xrdd, WiLbdVAL.value(), WfLbdVAL.value(), WoLbdVAL.value(), WgLbdVAL.value(), AccumListVali, dWiAccL, dWfAccL, dWoAccL, dWgAccL, Jouthslayer, RNNArchitecture));


            } //end test LSTM

            WiLbd.destroy(); //destroy broadcast at the end of loop
            WfLbd.destroy(); //destroy broadcast at the end of loop
            WoLbd.destroy(); //destroy broadcast at the end of loop
            WgLbd.destroy(); //destroy broadcast at the end of loop

            WiLbdVAL.destroy(); //destroy broadcast at the end of loop
            WfLbdVAL.destroy(); //destroy broadcast at the end of loop
            WoLbdVAL.destroy(); //destroy broadcast at the end of loop
            WgLbdVAL.destroy(); //destroy broadcast at the end of loop

        } //end training loop


    } //end trainLSMTthetas class


    //save thetas to file csv
    public static void saveThetas(List<double[]> thetastrainedL,
                                  String thetasfilepath,
                                  String RNNhead) throws IOException {

        rnn_Weights Weights = new rnn_Weights();
        Charset charset = Charset.forName("US-ASCII");

        String[] gates = {"Wi", "Wf", "Wo", "Wg"};

        for (int g=0; g<thetastrainedL.size(); g++){

            //save header
            String thetasGpath = thetasfilepath+"_"+gates[g]+".csv";
            String Gheader = RNNhead+"_"+gates[g]+"\n";
            Path nnthetas = Paths.get(thetasGpath); //path to save thetas
            BufferedWriter headerwriter = Files.newBufferedWriter(nnthetas, charset); // thetas writter
            headerwriter.write(Gheader);
            headerwriter.close();


            //save theta
            for (int t=0; t<thetastrainedL.get(g).length; t++){
                String theta = thetastrainedL.get(g)[t]+"\n";
                Writer thetaGwriter = new BufferedWriter(new FileWriter(thetasGpath, true));
                thetaGwriter.append(theta);
                thetaGwriter.close();
            }
        }
    }


}
