package com.situalab.dlab;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.util.DoubleAccumulator;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.lang.Math.log;


public class rnn_feedbackpropVRNN implements Serializable {

    //VRNN
    public static void VRNNfeedprop (String propagation,
                                     String inputx,
                                     List<double[][]> weightsL,
                                     List<rnn_thetasAccumM> dthetasAccL,
                                     List<DoubleAccumulator> AccumList,
                                     rnn_thetasAccumV Jouthslayer) {


        //skicarv classes
        rnn_mops matrixOps = new rnn_mops();
        rnn_cellVRNN VRNNcell = new rnn_cellVRNN();
        rnn_optimization opt = new rnn_optimization();
        rnn rnn = new rnn();



        //VanillaRNN feed propagation algorithm
        double[][] Wxh = weightsL.get(0); //Wxh weights
        double[][] Whh = weightsL.get(1); //Whh weights


        //hidden states
        int hsdim = Whh[0].length; //hidden state dimension


        //input pre process
        List<double[]> X = rnn.getinputcat(inputx);
        double[] catArray = X.get(0);
        X.remove(0); //remove label from List


        //List of a and hs
        List<double[]> A = new ArrayList<>();
        List<double[]> HS = new ArrayList<>(); HS.add(new double[Wxh[0].length]); //initial hidden state(hs)

        //List of softmax(output)
        List<double[]> Y = new ArrayList<>();




        //output layer; softmax
        double[] biasUnit = {1};
        double[][] Why = weightsL.get(2);
        double[] bias_hs_output = new double[hsdim];
        int outputdim = Why[0].length;
        double[] outputlayer = new double[outputdim];
        double[] ypredlayer = new double[outputdim];

        String seqcost = "";

        for (int t=0; t<X.size(); t++){

            List<double[]> celloutput = VRNNcell.cellVRNNfeedprop(X.get(t),  HS.get(t), Wxh, Whh); //0:a; 1:hs
            A.add(celloutput.get(0));
            HS.add(celloutput.get(1));

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

            if (propagation.equals("val")){
                try {
                    seqcost += Joutlayer+",";
                    Jouthslayer.addV(Joutlayer, t);
                } catch (ArrayIndexOutOfBoundsException error){
                    //System.out.println(t+"out");
                }

            }

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
        if (propagation.equals("test") | propagation.equals("val")){
            if (indxpred==indxclass) AccumList.get(2).add(1.0);
        }


        //backpropagation
        if (propagation.equals("back")){
            VRNNbackprop(delta_out, Whh, Why, X, HS, dthetasAccL);
        }


        AccumList.get(1).add(1.0);




    }


    public static void VRNNbackprop (double[] delta_out,
                                     double[][] Whh,
                                     double[][] Why,
                                     List<double[]> X,
                                     List<double[]> HS,
                                     List<rnn_thetasAccumM> dthetasAccL){


        // https://gist.github.com/karpathy/d4dee566867f8291f086


        //rnn classes
        rnn_mops matrixOps = new rnn_mops();
        rnn_optimization opt = new rnn_optimization();



        //remove bias unit
        double[][] Why_ = matrixOps.removeBias(Why); //remove bias from Why
        double[][] Whh_ = matrixOps.removeBias(Whh); //remove bias from Whh


        //backpropdata
        double[] hs_this = new double[Whh_.length]; //hs<t>
        double[] hs_next = new double[Whh_.length]; //hs<t+1>
        double[] hs_prev = new double[Whh_.length]; //hs<t-1>
        double[] xl_this = new double[X.get(0).length]; //x<t>
        double[] delta_hs_next = new double[hs_next.length]; //delta_hs<t+1>
        double[] delta_hs = new double[hs_this.length]; //delta_hs<t>
        double[] dypred = new double[delta_out.length]; //dypred
        double[] dhs = new double[Whh_.length];


        //layers
        int RNNlayers = X.size();

        for (int RNNlayer=(RNNlayers-1); RNNlayer>=0; RNNlayer--){

            //x<t> and hs<t> layer index
            int xl = RNNlayer;
            int hsl = RNNlayer+1;



            if (xl==X.size()-1){
                dypred = delta_out;
            } else {
                dypred = new double[delta_out.length];
            }
            //System.out.println(RNNlayer+":"+xl+":"+hsl+":"+Arrays.toString(dypred));

            //cell params
            hs_this = HS.get(hsl); //hs<t>
            hs_prev = HS.get(hsl-1); //hs<t-1>
            xl_this = X.get(xl); //x<t>


            double[][] dWhy_rev = new double[hs_this.length][dypred.length]; //dWhy
            getderivatevethetaslayer(dWhy_rev, hs_this, dypred); //dWhy, this is OK
            dWhy_rev = matrixOps.stackBias(dypred, dWhy_rev); //stack bias
            dthetasAccL.get(2).add(dWhy_rev); //add dWhy to W Accumulator; 0:Wxh; 1:Whh; 2:Why

            delta_hs = matrixOps.MatrixVectormultiplication(Why_, dypred);
            delta_hs = matrixOps.VectorSum(delta_hs,delta_hs_next); //backprop hs
            dhs = Arrays.stream(hs_this).map(h -> 1 - Math.pow(h,2)).toArray(); //1 - (hs<t>)^2
            dhs = matrixOps.elementWiseMulti(dhs,delta_hs);

            //get derivatives, dWhh
            double[][] dWhh = new double[hs_prev.length][dhs.length];
            getderivatevethetaslayer(dWhh, hs_prev, dhs); //dWhh
            dWhh = matrixOps.stackBias(dhs, dWhh);
            dthetasAccL.get(1).add(dWhh); //add dWhy to W Accumulator; 0:Wxh; 1:Whh; 2:Why

            //get derivatives, dWxh
            double[][] dWxh = new double[xl_this.length][dhs.length];
            getderivatevethetaslayer(dWxh, xl_this, dhs); //dWxh
            dthetasAccL.get(0).add(dWxh); //add dWhy to W Accumulator; 0:Wxh; 1:Whh; 2:Why

            //delta_hs_next;
            delta_hs_next = matrixOps.MatrixVectormultiplication(Whh_, dhs);

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
