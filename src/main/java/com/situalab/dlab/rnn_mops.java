package com.situalab.dlab;

import java.io.Serializable;

public class rnn_mops implements Serializable {

    public void printMatrix(double[][] matrixPrint){

        System.out.print("\nprint matrix method\n");
        for (int row = 0; row < matrixPrint.length; row++) {
            for (int column = 0; column < matrixPrint[row].length; column++) {
                System.out.print(matrixPrint[row][column] + " ");
            }
            System.out.println();
        }

    }

    //print Accumulator
    public void printAccMatrix(rnn_thetasAccumM matrixPrint){

        System.out.print("\nprint matrix method\n");
        for (int row = 0; row < matrixPrint.value().length; row++) {
            for (int column = 0; column < matrixPrint.value()[row].length; column++) {
                System.out.print(matrixPrint.value()[row][column] + " ");
            }
            System.out.println();
        }

    }




    public double[] MatrixVectormultiplication(double[][] M, double[] v){
        double[] Mv = new double[M.length];
        for (int row=0; row<M.length; row++){
            double rowSum = 0;
            for (int col=0; col<M[row].length; col++){
                rowSum += (M[row][col]*v[col]);
            }
            Mv[row] = rowSum;
        }
        return Mv;
    }


    //vector sum
    public double[] VectorSum(double[] v1,
                              double[] v2){

        double[] v = new double[v1.length];
        for (int i=0; i<v1.length; i++){
            v[i] += v1[i]+v2[i];
        }
        return v;
    }


    //element wise multiplication
    public double[] elementWiseMulti(double[] v1,
                                     double[] v2){

        double[] v = new double[v1.length];
        for (int i=0; i<v1.length; i++){
            v[i] += v1[i]*v2[i];
        }
        return v;
    }


    //get outer product
    public double[][] VectorMultiplication(double[] v1,
                                           double[] v2){

        double[][] matrix = new double[v1.length][v2.length];

        for (int row = 0; row < v1.length; row++) {
            for (int col = 0; col < v2.length; col++) {
                matrix[row][col] = v1[row]*v2[col];
            }
        }

        return matrix;
    }


    //get outer product
    public void getouterproduct(double[][] dthetas,
                                double[] state,
                                double[] delta){
        for (int row = 0; row < state.length; row++) {
            for (int col = 0; col < delta.length; col++) {
                dthetas[row][col] = state[row]*delta[col];
            }
        }
    }

    //transpose Matrix
    public double[][] getTranspose(double[][] initMatrix){
        int row = initMatrix.length;
        int col = initMatrix[0].length;
        double[][] outputMatrix = new double[col][row];
        for (int j=0; j<col; j++) {
            for (int i = 0; i<row; i++) {
                outputMatrix[j][i] = initMatrix[i][j];
            }
        }
        return outputMatrix;
    }

    //stack bias into matrix
    public double[][] stackBias(double[] bias,
                                double[][] Weights){

        double[][] biasWeights = new double[Weights.length+1][Weights[0].length];
        int row = biasWeights.length;
        int col = biasWeights[0].length;

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (i==0){
                    biasWeights[i][j] = bias[j];
                } else {
                    biasWeights[i][j] = Weights[i-1][j];
                }

            }
        }

        return biasWeights;

    }

    //remove bias from matrix
    public double[][] removeBias(double[][] Weights){

        double[][] Weights_ = new double[Weights.length-1][Weights[0].length];
        int row = Weights_.length;
        int col = Weights_[0].length;

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {

                Weights_[i][j] = Weights[i+1][j];
            }
        }

        return Weights_;

    }


}
