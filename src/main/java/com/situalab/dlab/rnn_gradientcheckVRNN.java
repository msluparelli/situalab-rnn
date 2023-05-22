package com.situalab.dlab;

import org.apache.spark.util.DoubleAccumulator;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.DoubleStream;


public class rnn_gradientcheckVRNN {

    public void checkGradient(){


        //skicarv Java Classes
        rnn_mops matrixOps = new rnn_mops(); //matrix operations
        rnn_Weights initW = new rnn_Weights(); //weights
        rnn_feedbackpropVRNN VRNN = new rnn_feedbackpropVRNN(); //feedpropagation

        String inputx = "2;0.882,0.791;0.834,0.715";
        inputx = "0,0.882,0.791-2;0.006,0.834,0.715-2;0.076,0,0-2;0.138,0.499,0.303-2;0.152,0.662,0.391-2;0.159,0.693,0.424-2";
        //inputx = "0,0.674,0.629-1;0.013,0.829,0.694-1;0.146,0.251,0.195-1;0.166,0.376,0.281-1;0.27,0.657,0.466-1;0.305,0.934,0.628-1";


        //RNN hyperparams
        double lmbda = 1.e-10;
        int inputxdim = inputx.split(";")[1].split(",").length; //input dimension
        int hiddencells = 4; //hidden states dimension
        int outputcat = 3; //output classification dimension
        double epsilon = 10.e-4; //this is OK



        List<double[][]> WeigthsL = initW.initWeights(inputxdim,hiddencells,outputcat,564323); //Weights; Wxh, Whh, Why
        List<rnn_thetasAccumM> dthetasAccL = initW.getAccThetasL(WeigthsL); //derivative [gradients] thetas, Accumulator
        DoubleAccumulator costA = new DoubleAccumulator(); //cost
        DoubleAccumulator mproA = new DoubleAccumulator(); //events
        List<DoubleAccumulator> AccumList = new ArrayList<>(); //init accumulators, backpropagation
        AccumList.add(costA); //add accumulator
        AccumList.add(mproA); //add accumulator
        rnn_thetasAccumV Jouthslayer = new rnn_thetasAccumV(3);
        VRNN.VRNNfeedprop("back", inputx, WeigthsL, dthetasAccL, AccumList, Jouthslayer); //Vanilla RNN feedpropagation algorithm
        double[] dthetasV = initW.getflatThetasAcc(dthetasAccL);




        String check = "yes"; //yes;no

        if (check.equals("yes")){

            //gradient checking equations
            List<double[][]> gAprox = initW.getThetasLZeros(WeigthsL);

            for (int layer=0; layer<WeigthsL.size(); layer++){

                for (int i=0; i<WeigthsL.get(layer).length; i++){
                    for (int j=0; j<WeigthsL.get(layer)[0].length; j++){

                        //thetaPlus
                        List<double[][]> thetaplus = initW.initWeights(inputxdim,hiddencells,outputcat,564323); //Weights; Wxh, Whh, Why
                        thetaplus.get(layer)[i][j] += epsilon; //10.e-4
                        AccumList.get(0).reset();
                        AccumList.get(1).reset();
                        VRNN.VRNNfeedprop("feed", inputx, thetaplus, dthetasAccL, AccumList, Jouthslayer); //Vanilla RNN feedpropagation algorithm
                        double sqwtplu = initW.getthetassqw(thetaplus); //thetas squared
                        double costplus = (1./AccumList.get(1).value() * AccumList.get(0).value()) + (lmbda/2*sqwtplu); //cost reg


                        //thetasMinus
                        List<double[][]> thetasmin = initW.initWeights(inputxdim,hiddencells,outputcat,564323); //Weights; Wxh, Whh, Why
                        thetasmin.get(layer)[i][j] -= epsilon; //10.e-4
                        AccumList.get(0).reset();
                        AccumList.get(1).reset();
                        VRNN.VRNNfeedprop("feed", inputx, thetasmin, dthetasAccL, AccumList, Jouthslayer); //Vanilla RNN feedpropagation algorithm
                        double sqwtmin = initW.getthetassqw(thetasmin); //thetas squared
                        double costmin = (1./AccumList.get(1).value() * AccumList.get(0).value()) + (lmbda/2*sqwtmin); //cost reg

                        //gradient check
                        gAprox.get(layer)[i][j] = (costplus-costmin) / (2*epsilon);

                    }
                }
            }
            double[] gAproxV = initW.getflatThetas(gAprox);

            double aproxgradient = DoubleStream.of(gAproxV).sum();
            double feedbackgradient = DoubleStream.of(dthetasV).sum();
            double gradientc = aproxgradient-feedbackgradient;


            NumberFormat formatter = new DecimalFormat("#0.0000");
            String feedbackgradient_ = formatter.format(feedbackgradient);
            String aproxgradient_ = formatter.format(aproxgradient);
            String gradientc_ = formatter.format(gradientc);

            System.out.println("\nVRNN Gradient Checking:"+gradientc_+"\n");
            //System.out.println("aproximation:"+aproxgradient_);
            //System.out.println("feedbackpropagation:"+feedbackgradient_);

        }

    }

}
