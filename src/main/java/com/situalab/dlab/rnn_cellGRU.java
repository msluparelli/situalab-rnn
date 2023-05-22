package com.situalab.dlab;

import org.apache.commons.lang.ArrayUtils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class rnn_cellGRU implements Serializable {

    public static List<double[]> cellGRUfeedprop (double[] x,
                                                  double[] hs,
                                                  List<double[][]> WrL,
                                                  List<double[][]> WuL,
                                                  List<double[][]> WcL){

        //skicarv classes
        rnn_mops matrixOps = new rnn_mops(); //matrix ops


        //output value
        List<double[]> celloutput = new ArrayList<>();


        //simplified architecture for feed propagation
        double[] biasUnit = {1};
        double[] hs_x = ArrayUtils.addAll(hs, x); //stacked hidden state(s),input(x)
        double[] bias_hs_x = ArrayUtils.addAll(biasUnit, hs_x); //add bias unit


        //Wreset [or relevant]
        double[][] Wr = (double[][])ArrayUtils.addAll(WrL.get(0), WrL.get(1)); //stacked W,U; bias included

        //Wupdate
        double[][] Wu = (double[][])ArrayUtils.addAll(WuL.get(0), WuL.get(1)); //stacked W,U; bias included

        //Wcell
        double[][] Wc = (double[][])ArrayUtils.addAll(WcL.get(0), WcL.get(1)); //stacked W,U; bias included


        //reset gate [or relevant]
        double[] resetgate = matrixOps.MatrixVectormultiplication(matrixOps.getTranspose(Wr), bias_hs_x); //reset gate = sigmoid (Wr[hs,x])
        double[] f_resetgate = Arrays.stream(resetgate).map(xoz -> rnn_activation.sigmoid.apply(xoz)).toArray(); //activation; sigmoid


        //update gate
        double[] updategate = matrixOps.MatrixVectormultiplication(matrixOps.getTranspose(Wu), bias_hs_x); //update gate = sigmoid (Wz[hs,x])
        double[] f_updategate = Arrays.stream(updategate).map(xoz -> rnn_activation.sigmoid.apply(xoz)).toArray(); //activation; sigmoid


        //candidate cell
        double[] reset_hiddenstate = matrixOps.elementWiseMulti(f_resetgate,hs);
        double[] rs_x = ArrayUtils.addAll(reset_hiddenstate, x); //stacked reset_hiddenstate(rs),input(x)
        double[] bias_rs_x = ArrayUtils.addAll(biasUnit, rs_x); //add bias unit [rt . hst-1, x]
        double[] memorycell = matrixOps.MatrixVectormultiplication(matrixOps.getTranspose(Wc), bias_rs_x); //candidate cell = tanh(Wc[rt.hst-1,x])
        double[] f_memorycell = Arrays.stream(memorycell).map(xoz -> rnn_activation.ftanh.apply(xoz)).toArray(); //activation; tanh


        //hidden state
        double[] update_memorycell = matrixOps.elementWiseMulti(f_updategate, f_memorycell); //update gate . candidate cell + (1 - update gate) . hst-1
        double[] update_diff = Arrays.stream(f_updategate).map(z -> 1-z).toArray();
        double[] update_hs = matrixOps.elementWiseMulti(update_diff,hs);
        hs = matrixOps.VectorSum(update_memorycell,update_hs);

        //update hidden state values
        celloutput.add(f_memorycell); //update memorycell
        celloutput.add(hs); //update hidden state values
        celloutput.add(f_resetgate); //reset gate
        celloutput.add(f_updategate); //update gate

        return celloutput;

    }

}
