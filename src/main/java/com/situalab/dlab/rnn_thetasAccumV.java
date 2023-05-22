package com.situalab.dlab;

import org.apache.spark.util.AccumulatorV2;

import java.io.Serializable;


public class rnn_thetasAccumV extends AccumulatorV2<double[], double[]> implements Serializable {

    //attributes
    private double[] dbiasA;
    private int iA;

    //constructor
    public rnn_thetasAccumV(int i){
        double[] dbiasA_ = new double[i];
        this.dbiasA = dbiasA_;
        this.iA = i; //para utilizar en copy y reset
    }

    @Override
    public boolean isZero() {
        return true;
    }

    @Override
    public AccumulatorV2<double[], double[]> copy() {
        return (new rnn_thetasAccumV(iA));
    }

    @Override
    public void reset() {
        dbiasA = new double[iA];

    }

    @Override
    public void add(double[] dbias) {
        for (int row=0; row<dbias.length; row++) {
            dbiasA[row] += dbias[row];
        }

    }

    public void addV(double Jval, int i){
        dbiasA[i] += Jval;
    }

    @Override
    public void merge(AccumulatorV2<double[], double[]> other) {
        add(other.value());

    }

    @Override
    public double[] value() {
        return dbiasA;
    }


}
