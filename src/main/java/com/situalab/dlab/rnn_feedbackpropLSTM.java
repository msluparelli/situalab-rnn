package com.situalab.dlab;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.util.DoubleAccumulator;

import java.io.BufferedWriter;
import java.io.Serializable;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.lang.Math.log;


public class rnn_feedbackpropLSTM implements Serializable {
    
    public static void LSTMfeedprop(String propagation,
                                    String inputx,
                                    List<double[][]> WiL,
                                    List<double[][]> WfL,
                                    List<double[][]> WoL,
                                    List<double[][]> WgL,
                                    List<DoubleAccumulator> AccumList,
                                    List<rnn_thetasAccumM> dWiAccL,
                                    List<rnn_thetasAccumM> dWfAccL,
                                    List<rnn_thetasAccumM> dWoAccL,
                                    List<rnn_thetasAccumM> dWgAccL,
                                    rnn_thetasAccumV Jouthslayer,
                                    String RNNArchitecture){
        
        
        
        //skicarv classes
        rnn_mops matrixOps = new rnn_mops();
        rnn_cellLSTM LSTMcell = new rnn_cellLSTM();
        rnn rnn = new rnn();
        rnn_optimization opt = new rnn_optimization();


        
        //hidden states
        int hsdim = WiL.get(0)[0].length; //hidden state dimension


        //devops
        //System.out.println(inputx);
        /*
        String input0 = "0.07897063983471861,0.5287651154268962,0.5300696736340301,0.4606938255887969,0.1553685895606111,0.47253504941392777,0.497242945183263,0.5997095296614904cat0";
        String input1 = "0.10287723690911448,0.482594356907292,0.577007700770077,0.4605346912794398,0.16342196759733524,0.41369800045966443,0.41907233214401557,0.6001564071053515cat0";
        String input2 = "0.11356012024639099,0.5287651154268962,0.4829482948294829,0.4594207511139401,0.15352379939106003,0.47092622385658467,0.4978916639636717,0.6010501619930735cat0";
        String input3 = "0.06496219238303604,0.5760351777207767,0.5293362669600293,0.45973901973265435,0.15915936527606717,0.4113996782348885,0.5001621796951021,0.6004915651882471cat0";
        //inputx = input0+";"+input1+";"+input2+";"+input3;
        //System.out.println(inputx);
        */

        
        //input pre process
        List<double[]> X = rnn.getinputcat(inputx);
        double[] catArray = X.get(0);
        X.remove(0); //remove label from List




        
        
        //lists of hidden states
        List<double[]> HS = new ArrayList<>();HS.add(new double[hsdim]); //initial hs; hs<t-1>
        
        //lists of cells
        List<double[]> C = new ArrayList<>();C.add(new double[hsdim]); //initial c; c<t-1>
        
        //List of hiddenstate_input
        List<double[]> I = new ArrayList<>(); //input gate
        
        //list of hiddenstate_forget
        List<double[]> F = new ArrayList<>(); //forget gate
        
        //list of hiddenstate_output
        List<double[]> O = new ArrayList<>(); //output gate
        
        //list of hiddenstate_cellgate
        List<double[]> G = new ArrayList<>(); //cel gate
        
        //List of softmax(output)
        List<double[]> Y = new ArrayList<>(); //softmax(output)


        //output layer; softmax
        double[] biasUnit = {1};
        double[][] Why = WgL.get(2);
        double[] bias_hs_output = new double[hsdim];
        int outputdim = Why[0].length;
        double[] outputlayer = new double[outputdim];
        double[] ypredlayer = new double[outputdim];
        List<Double> JoutputLayerList = new ArrayList<Double>();



        //LSTM hidden states feedpropagation
        for (int t=0; t<X.size(); t++) {

            //System.out.println(t);
            //System.out.println(Arrays.toString(X.get(t)));

            List<double[]> celloutput = LSTMcell.cellLSTMfeedprop(X.get(t), HS.get(t), C.get(t), WiL, WfL, WoL, WgL);
            C.add(celloutput.get(0));
            HS.add(celloutput.get(1));
            I.add(celloutput.get(2));
            F.add(celloutput.get(3));
            O.add(celloutput.get(4));
            G.add(celloutput.get(5));


            //output layer; [bias,last hidden state]
            bias_hs_output = ArrayUtils.addAll(biasUnit, HS.get(HS.size()-1)); //add bias unit = 1;

            //output = Why h(t); last hidden state; many to one architecture
            outputlayer = matrixOps.MatrixVectormultiplication(matrixOps.getTranspose(Why), bias_hs_output); //output unit

            //output; softmax
            ypredlayer = opt.getsoftmax(outputlayer);
            Y.add(ypredlayer);

            //System.out.println(Arrays.toString(ypredlayer));

            //loss function, default:multi classification
            int indxcat = (int) catArray[t]; //layer category
            double Joutlayer = -log(ypredlayer[indxcat]); //output; i.e. softmax(Why hs<t>); - y log(y_sm); this is OK:MIT


            //add values to M2M architecture
            JoutputLayerList.add(log(ypredlayer[indxcat])); //add log pred
            try {
                Jouthslayer.addV(Joutlayer, t);
            } catch (ArrayIndexOutOfBoundsException error){
                //System.out.println(t+"out");
            }
        }


        //delta output layers List
        int indxclass = (int) catArray[catArray.length-1]; //correct clasification or index of classifitation; (int)x.label()
        List<double[]> delta_outList = new ArrayList<double[]>();
        for (int ly = 0; ly<Y.size(); ly++) {
            indxclass = (int) catArray[catArray.length - 1]; //correct clasification or index of classifitation; (int)x.label()
            double[] delta_outLayer = Y.get(ly).clone();
            delta_outLayer[indxclass] -= 1;
            delta_outList.add(delta_outLayer);
            //System.out.println(Arrays.toString(delta_outLayer));
        }


        //get cost function for M2O | M2M
        double[] ypred = Y.get(Y.size()-1); //RNN last layer prediction
        double Jout = 0;
        if (RNNArchitecture.equals("M2M")){

            //loss function; default:multi classification; Last Layer M2M
            Jout = -1 * JoutputLayerList.stream().mapToDouble(Double::doubleValue).sum();

        } else {

            //loss function; default:multi classification; Last Layer M2O
            Jout = -log(ypred[indxclass]); //output; i.e. softmax(Why hs<t>); - y log(y_sm); this is OK:MIT

        }
        AccumList.get(0).add(Jout); //Use this


        //Model performance; prediction output, (prop=="test" | prop=="val")
        int indxpred = indxArray(ypred);
        if (propagation.equals("test") | propagation.equals("val")){
            if (indxpred==indxclass) AccumList.get(2).add(1.0);
        }

        //prediction output; sequence predition
        if (propagation.equals("test")){
            String sequencepred = getindxOutput(Y);
            //System.out.println(sequencepred);
            System.out.println(indxclass+","+indxpred);

        }

        //backpropagation
        if (propagation.equals("back")){
            LSTMbackprop(RNNArchitecture, X, HS, C, I, F, O, G, WiL, WfL, WoL, WgL, delta_outList, dWiAccL, dWfAccL, dWoAccL, dWgAccL);
        }

        AccumList.get(1).add(1.0);
        
        
    }
    
    
    public static void LSTMbackprop(String RNNArchitecture,
                                    List<double[]> X,
                                    List<double[]> HS,
                                    List<double[]> C,
                                    List<double[]> I,
                                    List<double[]> F,
                                    List<double[]> O,
                                    List<double[]> G,
                                    List<double[][]> WiL,
                                    List<double[][]> WfL,
                                    List<double[][]> WoL,
                                    List<double[][]> WgL,
                                    List<double[]> delta_outList,
                                    List<rnn_thetasAccumM> dWiAccL,
                                    List<rnn_thetasAccumM> dWfAccL,
                                    List<rnn_thetasAccumM> dWoAccL,
                                    List<rnn_thetasAccumM> dWgAccL) {
        
        // https://wiseodd.github.io/techblog/2016/08/12/lstm-backprop/
        // Stanford RNN course
        
        
        
        //skicarv class
        rnn_mops matrixOps = new rnn_mops();
        
        //weights; remove bias
        double[][] Wihh = matrixOps.removeBias(WiL.get(1));
        double[][] Wfhh = matrixOps.removeBias(WfL.get(1));
        double[][] Wohh = matrixOps.removeBias(WoL.get(1));
        double[][] Wghh = matrixOps.removeBias(WgL.get(1));
        double[][] Why = matrixOps.removeBias(WgL.get(2));
        
        double[][] Wi = (double[][])ArrayUtils.addAll(Wihh, WiL.get(0)); //stacked W,U; bias excluded
        double[][] Wf = (double[][])ArrayUtils.addAll(Wfhh, WfL.get(0)); //stacked W,U; bias excluded
        double[][] Wo = (double[][])ArrayUtils.addAll(Wohh, WoL.get(0)); //stacked W,U; bias excluded
        double[][] Wg = (double[][])ArrayUtils.addAll(Wghh, WgL.get(0)); //stacked W,U; bias excluded


        //back prop arrays
        int deltadim = delta_outList.get(delta_outList.size()-1).length;
        double[] dypred = new double[deltadim]; //dypred
        
        double[] hs_this = new double[Wihh.length]; //hs<t>
        double[] hs_next = new double[Wihh.length]; //hs<t+1>
        double[] hs_prev = new double[Wihh.length]; //hs<t-1>
        
        double[] c_this = new double[Wihh.length]; //c<t>
        double[] c_next = new double[Wihh.length]; //c<t+1>
        double[] c_prev = new double[Wihh.length]; //c<t-1>
        
        double[] hsi = new double[Wihh.length]; //hsi<t>
        double[] hsf = new double[Wihh.length]; //hsf<t>
        double[] hso = new double[Wihh.length]; //hso<t>
        double[] hsg = new double[Wihh.length]; //hsc<t>
        
        double[] xl_this = new double[X.get(0).length]; //x<t>
        
        
        //derivatives, deltas arrays
        double[] dhs = new double[hs_this.length]; //delta_hs<t>
        double[] dhs_next = new double[hs_next.length]; //delta_hs<t+1>
        double[] dc = new double[c_this.length]; //delta_c<t>
        double[] dc_next = new double[c_next.length]; //delta_c<t+1>
        
        
        
        
        
        //layers
        int RNNlayers = X.size();
        for (int RNNlayer=(RNNlayers-1); RNNlayer>=0; RNNlayer--){
            
            //x<t>, hs<t> and c<t> layer index; this is OK
            int xl = RNNlayer; //x input; hidden state activation: i, f, o, g
            int hsl = RNNlayer+1; //hidden state
            int cl = RNNlayer+1; //cell gate


            //RNNArchitecture; M2O | M2M
            if (RNNArchitecture.equals("M2M")){

                //Many to Many architecture
                dypred = delta_outList.get(xl);

            } else {
                if (xl==X.size()-1){

                    //Many to One architecture;dypred = delta_outList.get(delta_outList.size()-1);
                    dypred = delta_outList.get(xl);
                } else {
                    dypred = new double[deltadim];
                }
            }


            
            
            //cell params; this is OK
            hs_this = HS.get(hsl); //hs<t>
            hs_prev = HS.get(hsl-1); //hs<t-1>
            
            xl_this = X.get(xl); //x<t>
            
            c_this = C.get(cl); //c<t>
            c_prev = C.get(cl-1); //c<t-1>
            
            
            //LSTM gates; this is OK
            hsi = I.get(xl); //input gate
            hsf = F.get(xl); //forget gate
            hso = O.get(xl); //output gate
            hsg = G.get(xl); //cell gate

            /*
            System.out.println(Arrays.toString(dypred));

            System.out.println(Arrays.toString(hs_this));
            System.out.println(Arrays.toString(hs_prev));
            System.out.println(Arrays.toString(xl_this));
            System.out.println(Arrays.toString(c_this));
            System.out.println(Arrays.toString(c_prev));
            System.out.println(Arrays.toString(hsi));
            System.out.println(Arrays.toString(hsf));
            System.out.println(Arrays.toString(hso));
            System.out.println(Arrays.toString(hsg));
            */

            
            //dWhy; this is OK
            double[][] dWhy = getdW(hs_this, dypred);
            dWhy = matrixOps.stackBias(dypred, dWhy); //stack bias
            dWgAccL.get(2).add(dWhy); //add dWhy to W Accumulator; 0:Wxh; 1:Whh; 2:Why

            
            //delta_hs, last hidden state derivative; this is OK
            dhs = matrixOps.MatrixVectormultiplication(Why, dypred);
            dhs = matrixOps.VectorSum(dhs, dhs_next); //backprop hs


            
            //dhso, output gate derivative; this is OK
            double[] dhso = Arrays.stream(c_this).map(c -> rnn_activation.ftanh.apply(c)).toArray();
            dhso = matrixOps.elementWiseMulti(dhso, dhs);
            double[] dsigm_hso = Arrays.stream(hso).map(o -> rnn_activation.dsigmoid_.apply(o)).toArray(); //this is OK
            dhso = matrixOps.elementWiseMulti(dsigm_hso, dhso);

            
            
            //delta_c
            double[] dtanh_c = Arrays.stream(c_this).map(c -> rnn_activation.dtanh.apply(c)).toArray(); //dtanh(c); c is not previously tanh activated
            dc = matrixOps.elementWiseMulti(dhs, dtanh_c);
            dc = matrixOps.elementWiseMulti(dc, hso);
            dc = matrixOps.VectorSum(dc, dc_next);
            
            
            
            
            
            //dhsf
            double[] dhsf = matrixOps.elementWiseMulti(c_prev, dc); //this relation is OK; should be dcnext
            double[] dsigm_delta_hsf = Arrays.stream(hsf).map(f -> rnn_activation.dsigmoid_.apply(f)).toArray(); //this is OK
            dhsf = matrixOps.elementWiseMulti(dsigm_delta_hsf, dhsf);
            
            
            //dhsi
            double[] dhsi = matrixOps.elementWiseMulti(hsg, dc); //this relation is OK; should be dcnext
            double[] dsigm_delta_hsi = Arrays.stream(hsi).map(f -> rnn_activation.dsigmoid_.apply(f)).toArray(); //this is OK
            dhsi = matrixOps.elementWiseMulti(dsigm_delta_hsi, dhsi);
            
            
            //dhc
            double[] dhsg = matrixOps.elementWiseMulti(hsi, dc); //this relation is OK; should be dcnext
            double[] dsigm_delta_hsc = Arrays.stream(hsg).map(f -> rnn_activation.dtanh_.apply(f)).toArray(); //dtanh_; hsg was previously tanh activated
            dhsg = matrixOps.elementWiseMulti(dsigm_delta_hsc, dhsg);
            
            


            
            
            //Accumulate parameter derivatives; this is OK
            
            //get derivatives, dWf
            double[] dXf = matrixOps.MatrixVectormultiplication(Wf,dhsf); //this is OK
            
            //dWhh
            double[][] dWfhh = getdW(hs_prev, dhsf);
            dWfhh = matrixOps.stackBias(dhsf, dWfhh);
            dWfAccL.get(1).add(dWfhh); //add dWhy to W Accumulator; 0:Wxh; 1:Whh
            
            //dWxh
            double[][] dWfxh = getdW(xl_this, dhsf);
            dWfAccL.get(0).add(dWfxh); //add dWhy to W Accumulator; 0:Wxh; 1:Whh
            
            
            
            //get derivatives, dWi
            double[] dXi = matrixOps.MatrixVectormultiplication(Wi,dhsi); //this is OK
            
            
            //dWhh
            double[][] dWihh = getdW(hs_prev, dhsi);
            dWihh = matrixOps.stackBias(dhsi, dWihh);
            dWiAccL.get(1).add(dWihh); //add dWhy to W Accumulator; 0:Wxh; 1:Whh
            
            //dWxh
            double[][] dWixh = getdW(xl_this, dhsi);
            dWiAccL.get(0).add(dWixh); //add dWhy to W Accumulator; 0:Wxh; 1:Whh
            



            
            
            
            //get derivatives, dWo
            double[] dXo = matrixOps.MatrixVectormultiplication(Wo,dhso); //this is OK
            
            //dWhh
            double[][] dWohh = getdW(hs_prev, dhso);
            dWohh = matrixOps.stackBias(dhso, dWohh);
            dWoAccL.get(1).add(dWohh); //add dWhy to W Accumulator; 0:Wxh; 1:Whh
            
            //dWxh
            double[][] dWoxh = getdW(xl_this, dhso);
            dWoAccL.get(0).add(dWoxh); //add dWhy to W Accumulator; 0:Wxh; 1:Whh


            
            
            
            
            
            
            //get derivatives, dWc
            double[] dXg = matrixOps.MatrixVectormultiplication(Wg,dhsg); //this is OK
            
            
            //dWhh
            double[][] dWghh = getdW(hs_prev, dhsg);
            dWghh = matrixOps.stackBias(dhsg, dWghh);
            dWgAccL.get(1).add(dWghh); //add dWhy to W Accumulator; 0:Wxh; 1:Whh
            
            //dWxh
            double[][] dWgxh = getdW(xl_this, dhsg);
            dWgAccL.get(0).add(dWgxh); //add dWhy to W Accumulator; 0:Wxh; 1:Whh




            
            
            
            
            //dhs_next;
            double[] dX = new double[dXi.length];
            for (int d=0; d<dX.length; d++){
                dX[d] = dXf[d]+dXi[d]+dXo[d]+dXg[d];
            }
            dhs_next = Arrays.copyOf(dX, hs_this.length); //get gradients of hs<t+1>
            
            //dc_next;
            dc_next = matrixOps.elementWiseMulti(hsf, dc);
            
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
    
    public static double[][] getdW (double[] row,
                                    double[] col){
        
        double[][] dW = new double[row.length][col.length];
        for (int i = 0; i < row.length; i++) {
            for (int j = 0; j < col.length; j++) {
                dW[i][j] = row[i]*col[j];
            }
        }
        
        return dW;
    }

    //get indx array
    private static int indxArray (double[] values){
        double maxpred = Arrays.stream(values).max().getAsDouble();
        int indx = 0;
        for (int k=0; k<values.length; k++){
            if (values[k]==maxpred) {
                indx+=k;
                break;
            }
        }
        return indx;
    }

    //get sequence output indx
    private static String getindxOutput(List<double[]> Ypred){
        String indxseq = "";
        for (int ly = 0; ly<Ypred.size(); ly++){
            int indxp = indxArray(Ypred.get(ly));
            indxseq += indxp+",";
        }
        indxseq = indxseq.substring(0,indxseq.length()-1);

        return indxseq;
    }



}
