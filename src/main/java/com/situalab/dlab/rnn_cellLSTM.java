package com.situalab.dlab;

import java.io.Serializable;

import org.apache.commons.lang.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class rnn_cellLSTM implements Serializable {

    public static List<double[]> cellLSTMfeedprop (double[] x,
                                                   double[] hs,
                                                   double[] cell,
                                                   List<double[][]> WiL,
                                                   List<double[][]> WfL,
                                                   List<double[][]> WoL,
                                                   List<double[][]> WcL) {

        //skicarv classes
        rnn_mops matrixOps = new rnn_mops(); //matrix ops

        //output value
        List<double[]> celloutput = new ArrayList<>();


        //simplified architecture for feed propagation
        double[] biasUnit = {1};
        double[] s_x = ArrayUtils.addAll(hs, x); //stacked hidden state(s),input(x)
        double[] bias_s_x = ArrayUtils.addAll(biasUnit, s_x); //add bias unit



        //Winput
        double[][] Wi = (double[][])ArrayUtils.addAll(WiL.get(1), WiL.get(0)); //stacked W,U; bias included

        //Wforget
        double[][] Wf = (double[][])ArrayUtils.addAll(WfL.get(1), WfL.get(0)); //stacked W,U; bias included

        //Woutpuy
        double[][] Wo = (double[][])ArrayUtils.addAll(WoL.get(1), WoL.get(0)); //stacked W,U; bias included

        //Wgate
        double[][] Wc = (double[][])ArrayUtils.addAll(WcL.get(1), WcL.get(0)); //stacked W,U; bias include





        double[] inputgate = matrixOps.MatrixVectormultiplication(matrixOps.getTranspose(Wi), bias_s_x); //input gate = sigmoid (Wi[hs,x])
        double[] f_inputgate = Arrays.stream(inputgate).map(xoz -> rnn_activation.sigmoid.apply(xoz)).toArray(); //activation; sigmoid

        double[] forgetgate = matrixOps.MatrixVectormultiplication(matrixOps.getTranspose(Wf), bias_s_x); //forget gate = sigmoid (Wf[hs,x])
        double[] f_forgetgate = Arrays.stream(forgetgate).map(xoz -> rnn_activation.sigmoid.apply(xoz)).toArray(); //activation; sigmoid

        double[] outputgate = matrixOps.MatrixVectormultiplication(matrixOps.getTranspose(Wo), bias_s_x); //output gate = sigmoid (Wo[hs,x])
        double[] f_outputgate = Arrays.stream(outputgate).map(xoz -> rnn_activation.sigmoid.apply(xoz)).toArray(); //activation; sigmoid

        //candidate cell
        double[] cellgate = matrixOps.MatrixVectormultiplication(matrixOps.getTranspose(Wc), bias_s_x); //cell gate = tanh (Wg[hs,x])
        double[] f_cellgate = Arrays.stream(cellgate).map(xoz -> rnn_activation.ftanh.apply(xoz)).toArray(); //activation; tanh

        //cell value
        double[] cell_forget = matrixOps.elementWiseMulti(cell, f_forgetgate); //forget cell
        double[] cell_input = matrixOps.elementWiseMulti(f_cellgate, f_inputgate); //input cell
        cell = matrixOps.VectorSum(cell_forget, cell_input); //cell value = ct = ct-1 . forget + cellgate . input

        //hidden state value
        double[] f_cell = Arrays.stream(cell).map(xoz -> rnn_activation.ftanh.apply(xoz)).toArray(); //activation; tanh
        hs = matrixOps.elementWiseMulti(f_cell, f_outputgate); //hidden state value = tanh(ct) . output

        celloutput.add(cell);
        celloutput.add(hs);
        celloutput.add(f_inputgate);
        celloutput.add(f_forgetgate);
        celloutput.add(f_outputgate);
        celloutput.add(f_cellgate);

        return celloutput;

    }


}

