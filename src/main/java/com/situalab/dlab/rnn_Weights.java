package com.situalab.dlab;

import org.apache.commons.lang.ArrayUtils;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import static java.lang.Math.sqrt;


public class rnn_Weights {

    public static List<double[][]> initWeights (int input,
                                                int hidden,
                                                int output,
                                                int seedrand) {

        //weights List
        List<double[][]> WeightsL = new ArrayList<>();

        //thetas architecture
        int cellthetas = (input * hidden) + ((hidden+1) * hidden);
        int outpthetas = ((hidden+1) * output);
        List<Integer> thetasArchitecture = new ArrayList<>();
        thetasArchitecture.add(input);
        thetasArchitecture.add(hidden);
        thetasArchitecture.add(hidden);
        thetasArchitecture.add(output);


        //random values
        double[] inputinterval = {-1/(sqrt(input)),1/(sqrt(input))};
        double[] hiddeninterval = {-1/(sqrt(hidden)),1/(sqrt(hidden))};

        Random cellrand = new Random();
        cellrand.setSeed(seedrand);
        cellrand.longs(cellthetas);

        Random outrand = new Random();
        int outseedrand = seedrand*2; //different seed for a different random distribution
        outrand.setSeed(outseedrand);
        outrand.longs(outpthetas);

        double[] cellthetasvalues = new double[cellthetas];
        for (int t=0; t<cellthetas; t++){
            double theta = inputinterval[0]+(inputinterval[1]-inputinterval[0]) * cellrand.nextDouble();
            cellthetasvalues[t] = theta;
        }
        double[] outthetasvalues = new double[outpthetas];
        for (int t=0; t<outpthetas; t++){
            double theta = hiddeninterval[0]+(hiddeninterval[1]-hiddeninterval[0]) * outrand.nextDouble();
            outthetasvalues[t] = theta;
        }
        double[] thetasvalues = ArrayUtils.addAll(cellthetasvalues, outthetasvalues);
        //System.out.print("cells:"+cellthetas+" out:"+outpthetas+" thetas:"+thetasvalues.length+" architecture:"+thetasArchitecture);



        //weigths matrix
        int thstart = 0;
        for (int w=0; w<thetasArchitecture.size()-1; w++){
            int row = thetasArchitecture.get(w); //rows
            if (w != 0) {
                row +=1;
            }
            int col = thetasArchitecture.get(w+1); //cols
            int thstop = thstart + (row*col); //stop
            double[] thetasArray = Arrays.copyOfRange(thetasvalues, thstart, thstop); //thetas
            thstart = thstop; //update start

            int ij = 0;
            double[][] thetaslayer = new double[row][col];
            for (int j=0; j<col; j++) {
                for (int i=0; i<row; i++){
                    thetaslayer[i][j] = thetasArray[ij];
                    if (i==0) {thetaslayer[0][j] = thetasArray[ij]*0;} //bias init at zero (ReLU, 0.1)
                    ij +=1;
                }
            }

            //System.out.println(row+":"+col+":"+thetasArray.length);
            WeightsL.add(thetaslayer);
        }

        return WeightsL;

    }


    //init Weights LSTM and GRU
    public static List<double[][]> initWeightsGates (int input,
                                                     int hidden,
                                                     int seedrand) {

        //weights List
        List<double[][]> WeightsL = new ArrayList<>();

        //thetas architecture
        int cellthetas = (input * hidden) + ((hidden+1) * hidden);
        List<Integer> thetasArchitecture = new ArrayList<>();
        thetasArchitecture.add(input);
        thetasArchitecture.add(hidden);
        thetasArchitecture.add(hidden);


        //random values
        double[] inputinterval = {-1/(sqrt(input)),1/(sqrt(input))};
        double[] hiddeninterval = {-1/(sqrt(hidden)),1/(sqrt(hidden))};

        Random cellrand = new Random();
        cellrand.setSeed(seedrand);
        cellrand.longs(cellthetas);


        double[] cellthetasvalues = new double[cellthetas];
        for (int t=0; t<cellthetas; t++){
            double theta = inputinterval[0]+(inputinterval[1]-inputinterval[0]) * cellrand.nextDouble();
            cellthetasvalues[t] = theta;
        }



        //weigths matrix
        int thstart = 0;
        for (int w=0; w<thetasArchitecture.size()-1; w++){
            int row = thetasArchitecture.get(w); //rows
            if (w != 0) {
                row +=1;
            }
            int col = thetasArchitecture.get(w+1); //cols
            int thstop = thstart + (row*col); //stop
            double[] thetasArray = Arrays.copyOfRange(cellthetasvalues, thstart, thstop); //thetas
            thstart = thstop; //update start

            int ij = 0;
            double[][] thetaslayer = new double[row][col];
            for (int j=0; j<col; j++) {
                for (int i=0; i<row; i++){
                    thetaslayer[i][j] = thetasArray[ij];
                    if (i==0) {thetaslayer[0][j] = thetasArray[ij]*0;} //bias init at zero (ReLU, 0.1)
                    ij +=1;
                }
            }

            //System.out.println(row+":"+col+":"+thetasArray.length);
            WeightsL.add(thetaslayer);
        }

        return WeightsL;

    }









    //transpose List of Thetas
    public List<double[][]> getThetasLTranspose(List<double[][]> weightsList){
        List<double[][]> weightsListTransposed = new ArrayList<double[][]>();
        for (int layer=0; layer<weightsList.size(); layer++){
            double [][] wT = getTranspose(weightsList.get(layer));
            weightsListTransposed.add(wT);
        }
        return weightsListTransposed;
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

    //get square roots of weights
    public double getthetassqw(List<double[][]> thetas){
        double sqw = 0;
        for (int layer=0; layer<thetas.size(); layer++){
            double[][] W = thetas.get(layer);
            //not including bias
            for(int i=1; i<W.length; i++){
                for (int j=0; j<W[0].length; j++){
                    sqw += Math.pow(W[i][j],2);
                }
            }
        }
        return sqw;
    }



    //Accumulate derivative[gradients]
    public List<rnn_thetasAccumM> getAccThetasL(List<double[][]> Thetas){

        List<rnn_thetasAccumM> dthetasAccL = new ArrayList<rnn_thetasAccumM>();

        for (int layer=0; layer<Thetas.size(); layer++){
            int i = Thetas.get(layer).length;
            int j = Thetas.get(layer)[0].length;
            rnn_thetasAccumM WAcc = new rnn_thetasAccumM(i,j);
            dthetasAccL.add(WAcc);
        }
        return dthetasAccL;
    }


    //zero thetas
    public List<double[][]> getThetasLZeros(List<double[][]> MList){
        List <double[][]> WeightsZeroList = new ArrayList<double[][]>();
        for (int layer=0; layer<MList.size(); layer++){
            int row = MList.get(layer).length;
            int col = MList.get(layer)[0].length;
            double[][] WZeros = new double[row][col];
            WeightsZeroList.add(WZeros);
        }
        return WeightsZeroList;
    }



    //get thetas+epsilon+gradient to compute Hd
    public List<double[][]> getthetasepsilon(List<double[][]> thetasL,
                                             List<double[][]> gthetas,
                                             List<double[][]> thetaseL){
        double epsilon = 1.e-4;
        for (int layer=0; layer<thetasL.size(); layer++){
            double[][] thetas = thetasL.get(layer);
            double[][] gradient = gthetas.get(layer);
            int row = thetas.length;
            int col = thetas[0].length;
            double[][] thetase = new double[row][col];

            for (int i=0; i<row; i++){
                for (int j=0; j<col ; j++){
                    thetase[i][j] = thetas[i][j]+(epsilon*-gradient[i][j]); //d = -gradient f(thetas)
                }
            }

            thetaseL.add(thetase);
        }


        return thetaseL;
    }


    //flatten Matrix
    public double[] getflatThetas(List<double[][]> thetasL){
        double[] thetasflat = new double[0];
        for (int layer=0; layer<thetasL.size(); layer++){
            double[] tflat = Arrays.stream(thetasL.get(layer)).flatMapToDouble(Arrays::stream).toArray();
            thetasflat = ArrayUtils.addAll(thetasflat, tflat);
        }
        return thetasflat;
    }

    //flatten Matrix
    public double[] getflatThetasAcc(List<rnn_thetasAccumM> dthetasL){
        double[] thetasflat = new double[0];
        for (int layer=0; layer<dthetasL.size(); layer++){
            double[] tflat = Arrays.stream(dthetasL.get(layer).value()).flatMapToDouble(Arrays::stream).toArray();
            thetasflat = ArrayUtils.addAll(thetasflat, tflat);
        }
        return thetasflat;
    }


    //get trained thetas
    public List<double[][]> getVRNNtrainedThetasL(double[] thetastrained,
                                                  String RNNhead){

        int thstart = 0; //start
        List<double[][]> thetasTrainedList = new ArrayList<double[][]>();
        String[] RNNheader = RNNhead.split("_");
        String[] thetasArch = Arrays.copyOfRange(RNNheader, 0,RNNheader.length-1);

        //Wxh
        int row = Integer.parseInt(thetasArch[0]); //rows
        int col = Integer.parseInt(thetasArch[1]); //columns
        int thstop = thstart + (row*col); //stop
        double[] WxhthetasArray = Arrays.copyOfRange(thetastrained, thstart, thstop); //trained thetas
        thstart = thstop; //update start

        int ij = 0;
        double[][] Wxh = new double[row][col];
        for (int i=0; i<row; i++){
            for (int j=0; j<col; j++){
                Wxh[i][j] = WxhthetasArray[ij];
                ij +=1;
            }
        }
        thetasTrainedList.add(Wxh);

        //Whh
        row = Integer.parseInt(thetasArch[1])+1; //rows
        col = Integer.parseInt(thetasArch[1]); //columns
        thstop = thstart + (row*col); //stop
        double[] WhhthetasArray = Arrays.copyOfRange(thetastrained, thstart, thstop); //trained thetas
        thstart = thstop; //update start

        ij = 0;
        double[][] Whh = new double[row][col];
        for (int i=0; i<row; i++){
            for (int j=0; j<col; j++){
                Whh[i][j] = WhhthetasArray[ij];
                ij +=1;
            }
        }
        thetasTrainedList.add(Whh);


        //Why
        row = Integer.parseInt(thetasArch[1])+1; //rows
        col = Integer.parseInt(thetasArch[2]); //columns
        thstop = thstart + (row*col); //stop
        double[] WhythetasArray = Arrays.copyOfRange(thetastrained, thstart, thstop); //trained thetas
        thstart = thstop; //update start

        ij = 0;
        double[][] Why = new double[row][col];
        for (int i=0; i<row; i++){
            for (int j=0; j<col; j++){
                Why[i][j] = WhythetasArray[ij];
                ij +=1;
            }
        }
        thetasTrainedList.add(Why);


        return thetasTrainedList;
    }






    //get trained thetas
    public List<double[][]> getGATEtrainedThetasL(double[] thetastrained,
                                                  String RNNhead){

        int thstart = 0; //start
        List<double[][]> thetasTrainedList = new ArrayList<double[][]>();
        String[] RNNheader = RNNhead.split("_");
        String[] thetasArch = Arrays.copyOfRange(RNNheader, 0,RNNheader.length-1);

        //Wxh
        int row = Integer.parseInt(thetasArch[0]); //rows
        int col = Integer.parseInt(thetasArch[1]); //columns
        int thstop = thstart + (row*col); //stop
        double[] WxhthetasArray = Arrays.copyOfRange(thetastrained, thstart, thstop); //trained thetas
        thstart = thstop; //update start

        int ij = 0;
        double[][] Wxh = new double[row][col];
        for (int i=0; i<row; i++){
            for (int j=0; j<col; j++){
                Wxh[i][j] = WxhthetasArray[ij];
                ij +=1;
            }
        }
        thetasTrainedList.add(Wxh);

        //Whh
        row = Integer.parseInt(thetasArch[1])+1; //rows
        col = Integer.parseInt(thetasArch[1]); //columns
        thstop = thstart + (row*col); //stop
        double[] WhhthetasArray = Arrays.copyOfRange(thetastrained, thstart, thstop); //trained thetas
        thstart = thstop; //update start

        ij = 0;
        double[][] Whh = new double[row][col];
        for (int i=0; i<row; i++){
            for (int j=0; j<col; j++){
                Whh[i][j] = WhhthetasArray[ij];
                ij +=1;
            }
        }
        thetasTrainedList.add(Whh);

        return thetasTrainedList;
    }



}
