package com.situalab.dlab;

import java.io.Serializable;
import java.util.Arrays;

import static java.lang.Math.*;


public class rnn_activation implements Serializable {

    @FunctionalInterface //declare my own FunctionalInterface
    interface activateV<Double> {
        Double apply(Double xoz);
    }


    //exp(value)
    static activateV<Double> expV = (activateV<Double> & Serializable) xoz -> (exp(xoz));
    //log(value)
    static activateV<Double> logV = (activateV<Double> & Serializable) xoz -> (log(xoz));

    //sigmoid
    static activateV<Double> sigmoid = (activateV<Double> & Serializable) xoz -> (1/(1+ exp(-xoz)));
    static activateV<Double> dsigmoid = (activateV<Double> & Serializable) xoz -> rnn_activation.sigmoid.apply(xoz) * (1 - rnn_activation.sigmoid.apply(xoz));
    static activateV<Double> dsigmoid_ = (activateV<Double> & Serializable) xoz -> xoz * (1 - xoz);


    //tanh
    static activateV<Double> ftanh = (activateV<Double> & Serializable) xoz -> ((exp(xoz)-exp(-xoz)) / (exp(xoz)+exp(-xoz)));
    static activateV<Double> dtanh = (activateV<Double> & Serializable) xoz -> (1-Math.pow(rnn_activation.ftanh.apply(xoz),2));
    static activateV<Double> dtanh_ = (activateV<Double> & Serializable) xoz -> (1-Math.pow(xoz,2));


    //ReLu
    static activateV<Double> ReLU = (activateV<Double> & Serializable) xoz -> (max(0.,xoz));

    public static double dReLUa (Double xoz, Double av){
        if (xoz >=0.) av = 1.0;
        return av;
    }
    static activateV<Double> dReLU = (activateV<Double> & Serializable) xoz -> (dReLUa(xoz, 0.0));


    //LReLu
    public static double LeakyReLU (Double xoz, Double alphaLReLU){
        double av = xoz*alphaLReLU;
        if (xoz >=0.) av = xoz;
        return av;
    }
    static activateV<Double> LReLU = (activateV<Double> & Serializable) xoz -> (LeakyReLU(xoz, 0.01));

    public static double dLeakyReLu (Double xoz, Double alphaLReLU){
        double a = alphaLReLU;
        if (xoz >=0.) a = 1.0;
        return a;
    }
    static activateV<Double> dLReLU = (activateV<Double> & Serializable) xoz -> (dLeakyReLu(xoz, 0.01)); // alphaLReLU = 0.01


    //ELU
    public static double ELUv (Double xoz, Double alphaELU){
        double av = alphaELU*(exp(xoz)-1);
        if (xoz>=0.) av = xoz;
        return av;
    }
    static activateV<Double> ELU = (activateV<Double> & Serializable) xoz -> (ELUv(xoz, 0.05)); // alphaELU = 0.05

    public static double dELUv (Double xoz, Double alphaELU){
        double av = (alphaELU*(exp(xoz)-1))+alphaELU;
        if (xoz>=0.) av = 1.;
        return av;
    }
    static activateV<Double> dELU = (activateV<Double> & Serializable) xoz -> (dELUv(xoz, 0.05)); // alphaELU = 0.05

    static activateV<Double> predres = xoz -> max(min(xoz, (1-10e-15)),10e-15);




}
