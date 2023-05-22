package com.situalab.dlab;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.lang.Math.pow;
import static java.lang.Math.sqrt;


public class rnn_optimization implements Serializable {

    //derivative Gradient Descent, vanilla & momentum
    public List<double[][]> getderivativeBackProp (List<double[][]> thetasL,
                                                   List<rnn_thetasAccumM> dthetasAccL,
                                                   double lambda,
                                                   double mGD){

        List<double[][]> gthetasBackProp = new ArrayList<double[][]>();
        double epsilon = 1.e-5;
        double mgd = 1./mGD;
        for (int layer=0; layer <thetasL.size(); layer++){
            double[][] thetasLayer = thetasL.get(layer);
            double[][] thetasAccLayer = dthetasAccL.get(layer).value();
            int row = thetasLayer.length;
            int col = thetasLayer[0].length;
            double[][] thetaGrad = new double[row][col];
            //bias gradient, verificado con CS231
            for (int j=0; j<col; j++) {
                thetaGrad[0][j] = mgd*thetasAccLayer[0][j];
            }
            //weights gradient, verificado con CS231;
            int bias_indx = 0;
            if (layer == 1){
                bias_indx = 1;
            }
            for (int i=bias_indx; i<row; i++){
                for (int j=0; j<col; j++){
                    thetaGrad[i][j] = (mgd*thetasAccLayer[i][j])+(lambda*thetasLayer[i][j]);
                }
            }
            gthetasBackProp.add(thetaGrad);
        }
        return gthetasBackProp;
    }


    //momentum Update, velocity
    public List<double[][]> getmuVelocity(List<double[][]> thetasG,
                                          List<double[][]> velocityL,
                                          double mu,
                                          double learning,
                                          List<double[][]> gthetasAcc,
                                          String learning_mod){

        List<double[][]> velocityO = new ArrayList<double[][]>();
        for (int layer=0; layer<velocityL.size(); layer++){

            double[][] thetaG = thetasG.get(layer); //derivative
            double[][] thetaG_clip = clippingGradient(thetaG); //clipping gradients before parameters update
            double[][] veloc = velocityL.get(layer); //velocity
            double[][] gt = gthetasAcc.get(layer);

            //adaptive learning
            double learningRate = learning;

            if (learning_mod.equals("ADA")) learningRate = learning / getAlearningLayer(gt); //adaptive learning rate

            for(int i=0; i<veloc.length; i++) {
                for (int j = 0; j < veloc[0].length; j++) {
                    veloc[i][j] = (mu*veloc[i][j]) - (learningRate*thetaG_clip[i][j]); //ADAlearning vs learning; clipping gradients
                }
            }
            velocityO.add(veloc); //velocity update per layer
        }
        return velocityO;
    }


    //Vanilla Update
    public List<double[][]> getthetasVanillaUpdated(List<double[][]> thetasL,
                                                    List<double[][]> thetasG,
                                                    double learning,
                                                    List<double[][]> gthetasAcc,
                                                    double mGD, double lmbda, double n){



        List<double[][]> ThetasUpdated = new ArrayList<double[][]>();
        for (int layer=0; layer<thetasL.size(); layer++){

            //weight decay update (regularization)
            double weightDecay = 1;

            double[][] thetaW = thetasL.get(layer);
            double[][] thetaG = thetasG.get(layer);
            double[][] thetaG_clip = clippingGradient(thetaG); //clipping gradients before parameters update
            double[][] gt = gthetasAcc.get(layer);

            //adaptive learning
            double ADAlearning = learning / getAlearningLayer(gt); //adaptive learning rate

            //bias gradient, nesterov, revisar
            for (int j=0; j<thetaW[0].length; j++) {
                thetaW[0][j] -= (ADAlearning)*thetaG_clip[0][j];
            }
            //weight gradient update
            for(int i=1; i<thetaW.length; i++) {
                for (int j = 0; j <thetaW[0].length; j++) {

                    weightDecay = (1-(ADAlearning*lmbda/n)); //(1-(learning*lmbda/n)) OJO!! es n no m
                    thetaW[i][j] *= weightDecay;
                    thetaW[i][j] -= (ADAlearning)*thetaG_clip[i][j]; //learning/mGD -> StochasticGradientDescent (better without learning/n)
                }
            }
            ThetasUpdated.add(thetaW);
        }
        return ThetasUpdated;
    }



    //Momentum Update
    public List<double[][]> getthetasMuUpdated(List<double[][]> thetasL,
                                               List<double[][]> velocity){

        List<double[][]> ThetasUpdated = new ArrayList<double[][]>();
        for (int layer=0; layer<thetasL.size(); layer++){
            double[][] thetaW = thetasL.get(layer);
            double[][] v = velocity.get(layer);
            for(int i=0; i<thetaW.length; i++) {
                for (int j = 0; j <thetaW[0].length; j++) {
                    thetaW[i][j] += v[i][j];
                }
            }
            ThetasUpdated.add(thetaW);
        }
        return ThetasUpdated;
    }

    //accumulate gradients, ADAGRAD
    public List<double[][]> accumGradients (List<double[][]> gthetas,
                                            List<double[][]> gthetasAcc){

        List<double[][]> gaccumulated = new ArrayList<double[][]>();
        for (int layer=0; layer<gthetasAcc.size(); layer++){

            double[][] gt = gthetas.get(layer); //gradients
            double[][] gtA = gthetasAcc.get(layer); //gradients Acc
            for(int i=0; i<gtA.length; i++) {
                for (int j = 0; j < gtA[0].length; j++) {
                    gtA[i][j] += gt[i][j]; //accumulate gradients
                }
            }
            gaccumulated.add(gtA); //accumulated gradients per layer
        }
        return gaccumulated;
    }




    //gradient, l2 norm
    public double getAlearningLayer(double[][] gthetas){
        double ADAlearning = 0.0;
        for (int i = 0; i < gthetas.length; i++) {
            for (int j = 0; j < gthetas[0].length; j++) {
                ADAlearning += pow(gthetas[i][j],2);
            }
        }
        ADAlearning = sqrt(ADAlearning);
        return ADAlearning;
    }


    //softmax function
    public double[] getsoftmax(double[] output){

        double[] exp_output = Arrays.stream(output).map(xoz -> rnn_activation.expV.apply(xoz)).toArray(); //
        double Eexp_output = Arrays.stream(exp_output).sum();
        double[] ypred = Arrays.stream(exp_output).map(zj -> zj/Eexp_output).toArray(); //softmax output; this is OK

        return ypred;
    }


    //clipping gradients
    public double[][] clippingGradient(double[][] dW){

        double[][] dWc = new double[dW.length][dW[0].length];
        double v = 1; //could be between 1 and 10

        //get Matrix norm [max abs column sum]
        int col = dW[0].length;
        int row = dW.length;
        double[] sumcol = new double[col];
        for (int i=0; i<row; i++){
            for (int j=0; j<col; j++){
                sumcol[j] += Math.abs(dW[i][j]);
            }
        }

        double gnorm = Arrays.stream(sumcol).max().getAsDouble();

        for (int i=0; i<row; i++){
            for (int j=0; j<col; j++){
                if (gnorm > v){
                    dWc[i][j] += (dW[i][j]*v)/gnorm;
                } else {
                    dWc[i][j] += dW[i][j];
                }

            }
        }

        return dWc;

    }

}
