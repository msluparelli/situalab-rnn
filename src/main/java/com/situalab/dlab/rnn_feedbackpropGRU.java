package com.situalab.dlab;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.util.DoubleAccumulator;


import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.lang.Math.log;

public class rnn_feedbackpropGRU implements Serializable {

    public static void GRUfeedprop(String propagation,
                                   String inputx,
                                   List<double[][]> WrL,
                                   List<double[][]> WuL,
                                   List<double[][]> WcL,
                                   List<DoubleAccumulator> AccumList,
                                   List<rnn_thetasAccumM> dWrAccL,
                                   List<rnn_thetasAccumM> dWuAccL,
                                   List<rnn_thetasAccumM> dWcAccL){


        //skicarv classes
        rnn_mops matrixOps = new rnn_mops();
        rnn_cellGRU GRUcell = new rnn_cellGRU();
        //rnn trainrnn = new rnn();
        rnn_optimization opt = new rnn_optimization();


        //hidden states
        int hsdim = WrL.get(0)[0].length; //hidden state dimension


        //input pre process
        List<double[]> X = rnn.getinputcat(inputx);
        double[] catArray = X.get(0);
        X.remove(0); //remove label from List


        //lists of hidden states and cells
        List<double[]> HS = new ArrayList<>();HS.add(new double[WrL.get(0)[0].length]); //initial hs; hs<t-1>
        List<double[]> C = new ArrayList<>();C.add(new double[WrL.get(0)[0].length]); //initial c; c<t-1>
        List<double[]> R = new ArrayList<>(); //reset gate
        List<double[]> U = new ArrayList<>(); //update gate
        List<double[]> Y = new ArrayList<>(); //output


        //output layer; softmax
        double[][] Why = WcL.get(2);
        double[] biasUnit = {1};
        double[] bias_hs_output = new double[hsdim];
        int outputdim = Why[0].length;
        double[] outputlayer = new double[outputdim];
        double[] ypredlayer = new double[outputdim];


        //GRU hidden states feedpropagation
        for (int t=0; t<X.size(); t++) {

            List<double[]> celloutput = GRUcell.cellGRUfeedprop(X.get(t), HS.get(t), WrL, WuL, WcL);
            C.add(celloutput.get(0));
            HS.add(celloutput.get(1));
            R.add(celloutput.get(2));
            U.add(celloutput.get(3));


            //output layer; [bias,last hidden state]
            bias_hs_output = ArrayUtils.addAll(biasUnit, HS.get(HS.size()-1)); //add bias unit = 1;

            //output = Why h(t); last hidden state; many to one architecture
            outputlayer = matrixOps.MatrixVectormultiplication(matrixOps.getTranspose(Why), bias_hs_output); //output unit

            //output; softmax
            ypredlayer = opt.getsoftmax(outputlayer);
            Y.add(ypredlayer);

            //loss function, default:multi classification
            int indxcat = (int) catArray[t]; //layer category
            double Joutlayer = -log(ypredlayer[indxcat]); //output; i.e. softmax(Why hs<t>); - y log(y_sm); this is OK:MIT


        }



        //loss function, default:multi classification
        double[] ypred = Y.get(Y.size()-1);
        int indxclass = (int) catArray[catArray.length-1]; //correct clasification or index of classifitation; (int)x.label()
        double Jout = -log(ypred[indxclass]); //output; i.e. softmax(Why hs<t>); - y log(y_sm); this is OK:MIT
        AccumList.get(0).add(Jout);


        //Output layer delta(error)
        double[] delta_out = ypred.clone(); delta_out[indxclass] -=1; //softmax output delta error, this is OK


        //Model performance
        double maxpred = Arrays.stream(ypred).max().getAsDouble();
        int indxpred = 0;
        for (int k=0; k<ypred.length; k++){
            if (ypred[k]==maxpred) {
                indxpred+=k;
                break;
            }
        }

        //prediction output, (prop=="test" | prop=="val")
        if (propagation.equals("test") | propagation=="val"){
            if (indxpred==indxclass) AccumList.get(2).add(1.0);
        }

        //backpropagation
        if (propagation.equals("back")){
            GRUbackprop(X, HS, C, R, U, WrL, WuL, WcL, delta_out, dWrAccL, dWuAccL, dWcAccL);
        }

        AccumList.get(1).add(1.0);


    }


    public static void GRUbackprop(List<double[]> X,
                                   List<double[]> HS,
                                   List<double[]> C,
                                   List<double[]> R,
                                   List<double[]> U,
                                   List<double[][]> WrL,
                                   List<double[][]> WuL,
                                   List<double[][]> WcL,
                                   double[] delta_out,
                                   List<rnn_thetasAccumM> dWrAccL,
                                   List<rnn_thetasAccumM> dWuAccL,
                                   List<rnn_thetasAccumM> dWcAccL){


        //skicarv class
        rnn_mops matrixOps = new rnn_mops();

        //weights; remove bias
        double[][] Wrhh = matrixOps.removeBias(WrL.get(1));
        double[][] Wuhh = matrixOps.removeBias(WuL.get(1));
        double[][] Wchh = matrixOps.removeBias(WcL.get(1));
        double[][] Why = matrixOps.removeBias(WcL.get(2));

        //hidden units
        int hu = Wrhh.length;

        //back prop arrays
        double[] dypred = new double[delta_out.length]; //dypred

        double[] hs_this = new double[hu]; //hs<t>
        double[] hs_next = new double[hu]; //hs<t+1>
        double[] hs_prev = new double[hu]; //hs<t-1>

        double[] c_this = new double[hu]; //c<t>
        double[] c_next = new double[hu]; //c<t+1>
        double[] c_prev = new double[hu]; //c<t-1>

        double[] hsr = new double[hu]; //hsr<t>
        double[] hsu = new double[hu]; //hsu<t>
        double[] hsc = new double[hu]; //hsc<t>

        double[] xl_this = new double[X.get(0).length]; //x<t>

        //derivatives, deltas arrays
        double[] delta_hs_next = new double[hs_next.length]; //delta_hs<t+1>
        double[] delta_c_next = new double[c_next.length]; //delta_c<t+1>
        double[] delta_hs = new double[hs_this.length]; //delta_hs<t>
        double[] delta_c = new double[c_this.length]; //delta_c<t>

        int RNNlayers = X.size();
        for (int RNNlayer=(RNNlayers-1); RNNlayer>=0; RNNlayer--){

            //x<t>, hs<t> and c<t> layer index
            int xl = RNNlayer;
            int cg = RNNlayer; //cel gate
            int hsl = RNNlayer+1;
            int cl = RNNlayer+1;


            if (xl==X.size()-1){
                dypred = delta_out;
            } else {
                dypred = new double[delta_out.length];
            }


            //cell params
            hs_this = HS.get(hsl); //hs<t>
            hs_prev = HS.get(hsl-1); //hs<t-1>
            xl_this = X.get(xl); //x<t>
            c_this = C.get(cl); //x<t>
            c_prev = C.get(cl-1); //c<t-1>

            hsc = C.get(cg); //input gate
            hsr = R.get(cg); //forget gate
            hsu = U.get(cg); //output gate


            //dWhy
            double[][] dWhy = new double[hs_this.length][dypred.length]; //dWhy
            getderivatevethetaslayer(dWhy, hs_this, dypred); //dWhy, this is OK
            dWhy = matrixOps.stackBias(dypred, dWhy); //stack bias
            dWcAccL.get(2).add(dWhy); //add dWhy to W Accumulator; 0:Wxh; 1:Whh; 2:Why


            //delta_hs
            delta_hs = matrixOps.MatrixVectormultiplication(Why, dypred);
            delta_hs = matrixOps.VectorSum(delta_hs,delta_hs_next); //backprop hs





        }




    }


    //get dthetaslayer
    public static void getderivatevethetaslayer(double[][] dthetas,
                                                double[] aj,
                                                double[] ej){
        for (int row = 0; row < aj.length; row++) {
            for (int col = 0; col < ej.length; col++) {
                dthetas[row][col] = aj[row]*ej[col];
            }
        }
    }

}
