package com.situalab.dlab;

import org.apache.spark.util.DoubleAccumulator;

import java.util.ArrayList;
import java.util.List;



public class rnn_gradientcheckGRU {

    public void checkGradient(){

        //skicarv classes
        rnn_mops matrixOps = new rnn_mops(); //matrix operations
        rnn_Weights initW = new rnn_Weights(); //weights
        rnn_feedbackpropGRU GRU = new rnn_feedbackpropGRU(); //feedpropagation

        //input data
        String inputx = "2;0.882,0.791;0.834,0.715";
        //inputx = "2;0,0.882,0.791;0.006,0.834,0.715;0.076,0,0;0.138,0.499,0.303;0.152,0.662,0.391;0.159,0.693,0.424";
        //inputx = "2;0.882,0.791";
        inputx = "0,0.882,0.791-2;0.006,0.834,0.715-2";

        //RNN hyperparams
        double lmbda = 1.e-10;
        int inputxdim = inputx.split(";")[1].split(",").length; //input dimension
        int hiddencells = 4; //hidden states dimension
        int outputcat = 3; //output classification dimension
        double epsilon = 10.e-4; //this is OK

        //thetas
        List<double[][]> WrL = initW.initWeightsGates(inputxdim,hiddencells,234819);
        List<double[][]> WuL = initW.initWeightsGates(inputxdim,hiddencells,109274);
        List<double[][]> WcL = initW.initWeights(inputxdim,hiddencells,outputcat,718274); //contains Why

        List<rnn_thetasAccumM> dWrAccL = initW.getAccThetasL(WrL); //derivative [gradients] thetas, Accumulator
        List<rnn_thetasAccumM> dWuAccL = initW.getAccThetasL(WuL); //derivative [gradients] thetas, Accumulator
        List<rnn_thetasAccumM> dWcAccL = initW.getAccThetasL(WcL); //derivative [gradients] thetas, Accumulator


        //params
        DoubleAccumulator costA = new DoubleAccumulator(); //cost
        DoubleAccumulator mproA = new DoubleAccumulator(); //events
        List<DoubleAccumulator> AccumList = new ArrayList<>(); //init accumulators, backpropagation
        AccumList.add(costA); //add accumulator
        AccumList.add(mproA); //add accumulator

        GRU.GRUfeedprop("back", inputx, WrL, WuL, WcL, AccumList, dWrAccL, dWuAccL, dWcAccL);
        System.out.println("GC GRU Jout:"+AccumList.get(0).value());



    }

}
