package com.situalab.dlab;

import java.io.Serializable;

import org.apache.commons.lang.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class rnn_cellVRNN implements Serializable {

    public static List<double[]> cellVRNNfeedprop (double[] x,
                                                   double[] hs,
                                                   double[][] Wxh,
                                                   double[][] Whh){

        List<double[]> cellactivationandhs = new ArrayList<>();

        double[] biasUnit = {1};
        rnn_mops matrixOps = new rnn_mops(); //matrix ops


        //simplified architecture for feed propagation
        double[] s_x = ArrayUtils.addAll(hs, x); //stacked hidden state(s),input(x)
        double[] bias_s_x = ArrayUtils.addAll(biasUnit, s_x); //add bias unit
        double[][] WU = (double[][])ArrayUtils.addAll(Whh, Wxh); //stacked W,U; bias included


        //feed propagation; initial hs:all zeros;
        double[] a = matrixOps.MatrixVectormultiplication(matrixOps.getTranspose(WU), bias_s_x); //new hidden state
        hs = Arrays.stream(a).map(xoz -> rnn_activation.ftanh.apply(xoz)).toArray(); //activation; tanh o ReLU

        cellactivationandhs.add(a);
        cellactivationandhs.add(hs);


        //update hidden state vector List
        return cellactivationandhs;

    }

}
