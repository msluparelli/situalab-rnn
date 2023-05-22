package com.situalab.dlab;

import org.apache.spark.util.AccumulatorV2;

import java.io.Serializable;


public class rnn_thetasAccumM extends AccumulatorV2<double[][], double[][]> implements Serializable {

    //attributes
    private double[][] dWA;
    private int iA;
    private int jA;

    //constructor
    public rnn_thetasAccumM(int i, int j){
        double[][] wA_ = new double[i][j];
        this.dWA = wA_;
        this.iA = i; //para utilizar en copy y reset
        this.jA = j; //para utilizar en copy y reset
    }


    @Override
    public boolean isZero() {
        return true;
    }

    @Override
    public AccumulatorV2<double[][], double[][]> copy() {
        return (new rnn_thetasAccumM(iA,jA));
    }

    @Override
    public void reset() {
        dWA = new double[iA][jA];

    }

    @Override
    public void add(double[][] dW) {
        for (int row=0; row<dW.length; row++) {
            for (int col = 0; col < dW[row].length; col++) {
                dWA[row][col] += dW[row][col];
            }
        }

    }

    @Override
    public void merge(AccumulatorV2<double[][], double[][]> other) {
        add(other.value());
    }

    @Override
    public double[][] value() {
        return dWA;
    }

}
