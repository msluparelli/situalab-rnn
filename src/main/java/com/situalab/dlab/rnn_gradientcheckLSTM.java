package com.situalab.dlab;

import org.apache.spark.util.DoubleAccumulator;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.DoubleStream;


public class rnn_gradientcheckLSTM {

    public void checkGradient(){

        //skicarv Java Classes
        rnn_mops matrixOps = new rnn_mops(); //matrix operations
        rnn_Weights initW = new rnn_Weights(); //weights
        rnn_feedbackpropLSTM LSTM = new rnn_feedbackpropLSTM(); //feedpropagation

        String inputx = "2;0.882,0.791;0.834,0.715";
        //inputx = "2;0,0.882,0.791;0.006,0.834,0.715;0.076,0,0;0.138,0.499,0.303;0.152,0.662,0.391;0.159,0.693,0.424";
        //inputx = "2;0.882,0.791";
        //inputx = "2;0,0.882,0.791;0.006,0.834,0.715";
        //inputx = "1;0,0.674,0.629;0.013,0.829,0.694;0.146,0.251,0.195;0.166,0.376,0.281;0.27,0.657,0.466;0.305,0.934,0.628";

        inputx = "0,0.674,0.629cat1;0.013,0.829,0.694cat1;0.146,0.251,0.195cat1;0.166,0.376,0.281cat1;0.27,0.657,0.466cat1;0.305,0.934,0.628cat1";

        //inputx = "0.343191487538661,0.548590956542323,0.332420354996865,0.303635054188913,0.336553979283461,0.678438450879189,0.630121712270858,0.532991189386003,0.524282853958149,0.622083290029152-1;0.606759260053366,0.465597449779104,0.121382822338164,0.932096706197433,0.828640202778531,0.646440331827992,0.594848183185999,0.622702192865197,0.673734667041526,0.679400872221992-1;0.661134732590842,0.584724802158795,0.502983895187281,0.5028950676335,0.592141486597209,0.683958188092326,0.651965381965114,0.647401901109718,0.601225928775445,0.61529666086412-1;0.593566711654348,0.534946691270611,0.445779931803753,0.475402385048442,0.453509337603235,0.548643841355991,0.557051144411912,0.587128179645539,0.613203344770557,0.555638682024346-1;0.575870097674732,0.525918136284466,0.503807280570434,0.399676941137245,0.503300706183091,0.487768530271817,0.712770367327147,0.78040113306281,0.653821453410035,0.430638088507269-1;0.341982297591412,0.626369045152124,0.731334984836979,0.758036697349422,0.803985889677443,0.788917234487354,0.738301419042765,0.715893050208154,0.698074605599524,0.683420980778726-1;0.661009971761985,0.698885816015112,0.743714096421346,0.671726427710727,0.596246969000102,0.611562723491091,0.634527752779846,0.662272615446273,0.686607353461624,0.691652235750188-1;0.608543911137584,0.687949667442105,0.695372888121448,0.738859077994191,0.820570000581775,0.826625959640209,0.778746068720625,0.783016585049444,0.779153867197536,0.764344028115834-1;0.73086993047913,0.581434909073938,0.421019015408117,0.456560033444445,0.651445212212188,0.692670155336178,0.748677356576354,0.820114287054633,0.839645662719483,0.815995055781932-1;0.349597361762107,0.347050769906459,0.281837404301166,0.248566877515798,0.284513585018119,0.257829891101347,0.255355784969367,0.334308743668031,0.318406287497643,0.275689845536586-1;0.242935802807328,0.236875414725677,0.261771736499057,0.326394127007303,0.336556222261511,0.368723949881569,0.336095780069889,0.311374672169239,0.297521276879984,0.36464771872851-1;0.35276930088712,0.291004010543881,0.220000054621973,0.245497639824054,0.276201185451225,0.308078017297166,0.362820843620099,0.346114392441367,0.432966908677229,0.546984455255242-1;0.554051411437045,0.584205336306451,0.712725453841954,0.731156380837088,0.704353338923773,0.636074948847391,0.648245034190455,0.607253817543289,0.511848590094046,0.0979028930130746-1;0.742254245337958,0.799356030852908,0.826149473582698,0.84826364906996,0.824872545476307,0.825157179728744,0.839548138475131,0.807469589289573,0.790040767309826,0.796307421099804-1;0.77530598549886,0.682130652471605,0.64480571636917,0.531543267961411,0.675290203963085,0.682135698651729,0.74261235000488,0.806248447417314,0.942835465934033,0.967770073922267-1;0.941007130243597,0.829171257307386,0.759432784264217,0.712485420361403,0.657385585432306,0.550760654602998,0.471716381798324,0.732451708756875,0.745454343442241,0.796864312366133-1;0.804611835371245,0.748598598343172,0.738989189408602,0.771876046303071,0.71875316456758,0.675075210361683,0.587234465497315,0.497655860106103,0.419499579403381,0.413488090880166-1;0.487439199556289,0.592782655633989,0.569556914868655,0.489048670104173,0.527668912044786,0.609715087878973,0.447478405814877,0.723578505484093,0.906135190102993,0.85114355337755-1;0.780476482265716,0.627303540510331,0.667736706257971,0.811165967809675,0.880935393081713,0.718683984004161,0.733756808648763,0.7538133482741,0.791073295815166,0.816465634106289-1;0.746531716673217,0.700255995931275,0.833730979885495,0.425169774355493,0.69024378313652,0.787019684127709,0.706133993944081,0.564299431370775,0.776400860821342,0.774762520842434-1;0.782552378602084,0.780915479594872,0.706508615798973,0.640860724421326,0.60573052381658,0.589735436576949,0.63554056311641,0.60362381406006,0.555235812285594,0.60118958822222-1;0.602496226154702,0.608936430890065,0.536932480142843,0.540108753930724,0.430080808191361,0.46464547722373,0.513841153473879,0.533868263726914,0.633114076894432,0.676047642770153-1;0.658527607860755,0.79251103327551,0.78567960427641,0.65045802109237,0.718692533227814,0.754244212536058,0.745208235617387,0.695336947254632,0.740016542072886,0.691773436591693-1;0.758338849510414,0.730146652935926,0.690486705220653,0.591400744206423,0.58243641921154,0.514703457903094,0.415368315173299,0.354433278345254,0.355139286095456,0.372748444249081-1;0.340503512442849,0.281581251106248,0.304681553393169,0.330803528711329,0.393259588651522,0.460841638689578,0.53812786347722,0.60807027020328,0.591292479412189,0.598476217175573-1;0.60686756817914,0.699970011046952,0.815554483956284,0.817049681395438,0.714360315107615,0.772193766556486,0.715233682003274,0.694382023661511,0.563184129847776,0.59953802373138-1;0.263682021594756,0.155146496879027,0.288203731533912,0.451865123016644,0.457302721646287,0.409517397605116,0.507300756291837,0.573822445856693,0.649736978937921,0.73191289545096-1;0.659301734663478,0.708834755336408,0.85100705841789,0.855621993862987,0.75662170621003,0.735751995964201,0.790212558758054,0.774082238190433,0.716804952293842,0.746501214809935-1;0.791309503107083,0.693910203836824,0.696713743105783,0.714465272073857,0.720206718060341,0.703149194337144,0.710225026846023,0.695000863870717,0.744555229470945,0.776430628520855-1;0.765542414552738,0.724046856439999,0.684120390456196,0.682188072667175,0.707419569275658,0.703069778244275,0.699274304294643,0.663214437448792,0.682136184288458,0.733932188005094-1;0.671292713090107,0.663964647782335,0.715117853437519,0.689943783925794,0.663526078162099,0.742163135651026,0.630603777112961,0.591891018023703,0.600103064611583,0.630927303410289-1;0.413527836619436,0.286204500930536,0.288342737477961,0.413144551255951,0.465415324866363,0.448279410425814,0.426995253963372,0.283940805789684,0.633398613084496,0.736050278300152-1;0.665268070323858,0.529939122983081,0.792085054419958,0.746252534928666,0.601562039030768,0.582560606300765,0.684698547096142,0.623211402790818,0.578148513591124,0.619176780421603-1;0.572173374010836,0.493677200871373,0.560108411999419,0.670006341764928,0.701635098560515,0.692231274577562,0.671646062872717,0.622802530121386,0.600485282369482,0.57617587275438-1;0.565964295929537,0.543118245911547,0.621658822509052,0.693286317720133,0.703495800753542,0.69392296125925,0.745476787867698,0.690504730288134,0.640544104559735,0.553601585150746-1;0.535300426790142,0.521919755243488,0.484055416748296,0.38949084881758,0.355132659323291,0.395191863642491,0.41110980998647,0.403391198654513,0.426179812029256,0.414928540387009-1;0.388791708668631,0.41419928664426,0.509625824870397,0.610227525840405,0.637181853784736,0.589781293624017,0.617290783765515,0.514279362593265,0.441098599034079,0.414059585938439-1;0.286049208072055,0.490920446930479,0.29066772294517,0.443099995085896,0.123082355462354,0.366281274564683,0.370914835560249,0.239015092081091,0.193879613561269,0.331103145684694-1;0.338256709274104,1,0.745808568110592,0.571964174627008,0.759593323880318,0.870397939931275,0.845308055955019,0.443464264561325,0.429019008325786,0.691290184501653-1;0.647125578012486,0.35067630800369,0.413768789181181,0.662044780990238,0.498504002004791,0.137075480968182,0.160077519236932,0.62803573624925,0.502231034573129,0.279534196484206-1;0.291491339478096,0.357641574330662,0.226393003061961,0.145994173005916,0.289726264028792,0.40074872319884,0.371735642951292,0.119854878072956,0.235495142332913,0.514031562917879-1;0.535028463130007,0.534717507166184,0.543438423326184,0.533030035377966,0.598990178847865,0.609073434415822,0.566236957103701,0.631658080505085,0.706650869097678,0.742687837716582-1;0.709187404124309,0.660282276860856,0.501237655955778,0.588309497726246,0.536288685239419,0.666802136330988,0.808426518789268,0.852924753777177,0.901275798148321,0.727174674260723-1;0.500122300424262,0.567194595480471,0.581522417709247,0.578208669314593,0.607527730060781,0.656059051099748,0.690186395489396,0.463852118330653,0.367680336767363,0.71950377825-1;0.73060816721186,0.646872621199704,0.666495579407046,0.689429556405901,0.682452249929068,0.655932998862094,0.576448606955312,0.653266807335833,0.704595144632552,0.719083186667298-1;0.594247473488071,0.56286666753919,0.641979452326693,0.619423446537873,0.557852876001072,0.45880198164365,0.499359759226831,0.545926127360197,0.514170422445339,0.423640046552016-1";

        //RNN hyperparams
        double lmbda = 1.e-10;
        int inputxdim = inputx.split(";")[1].split(",").length; //input dimension
        int hiddencells = 4; //hidden states dimension
        int outputcat = 3; //output classification dimension
        double epsilon = 10.e-4; //this is OK

        //thetas
        List<double[][]> WiL = initW.initWeightsGates(inputxdim,hiddencells,234819);
        List<double[][]> WfL = initW.initWeightsGates(inputxdim,hiddencells,109274);
        List<double[][]> WoL = initW.initWeightsGates(inputxdim,hiddencells,554637);
        List<double[][]> WgL = initW.initWeights(inputxdim,hiddencells,outputcat,718274); //contains Why

        List<rnn_thetasAccumM> dWiAccL = initW.getAccThetasL(WiL); //derivative [gradients] thetas, Accumulator
        List<rnn_thetasAccumM> dWfAccL = initW.getAccThetasL(WfL); //derivative [gradients] thetas, Accumulator
        List<rnn_thetasAccumM> dWoAccL = initW.getAccThetasL(WoL); //derivative [gradients] thetas, Accumulator
        List<rnn_thetasAccumM> dWgAccL = initW.getAccThetasL(WgL); //derivative [gradients] thetas, Accumulator

        //params
        DoubleAccumulator costA = new DoubleAccumulator(); //cost
        DoubleAccumulator mproA = new DoubleAccumulator(); //events
        List<DoubleAccumulator> AccumList = new ArrayList<>(); //init accumulators, backpropagation
        AccumList.add(costA); //add accumulator
        AccumList.add(mproA); //add accumulator
        rnn_thetasAccumV Jouthslayer = new rnn_thetasAccumV(3);
        String RNNArchitecture = "M2M";
        LSTM.LSTMfeedprop("back", inputx, WiL, WfL, WoL, WgL, AccumList, dWiAccL, dWfAccL, dWoAccL, dWgAccL, Jouthslayer, RNNArchitecture);
        //System.out.println("GC Jout:"+AccumList.get(0).value());

        double[] dWiV = initW.getflatThetasAcc(dWiAccL);
        double[] dWfV = initW.getflatThetasAcc(dWfAccL);
        double[] dWoV = initW.getflatThetasAcc(dWoAccL);
        double[] dWgV = initW.getflatThetasAcc(dWgAccL);

        double gcA = 0.0;


        String check = "yes"; //yes;no

        if (check.equals("yes")){

            //dWi
            //gradient checking equations
            List<double[][]> giAprox = initW.getThetasLZeros(WiL);
            for (int layer=0; layer<WiL.size(); layer++){

                for (int i=0; i<WiL.get(layer).length; i++){
                    for (int j=0; j<WiL.get(layer)[0].length; j++){

                        //thetaPlus
                        List<double[][]> thetaplus = initW.initWeightsGates(inputxdim,hiddencells,234819);
                        thetaplus.get(layer)[i][j] += epsilon; //10.e-4
                        AccumList.get(0).reset();
                        AccumList.get(1).reset();
                        LSTM.LSTMfeedprop("feed", inputx, thetaplus, WfL, WoL, WgL, AccumList, dWiAccL, dWfAccL, dWoAccL, dWgAccL, Jouthslayer, RNNArchitecture); //LSTM RNN feedpropagation algorithm
                        double sqwtplu = initW.getthetassqw(thetaplus); //thetas squared
                        double costplus = (1./AccumList.get(1).value() * AccumList.get(0).value()) + (lmbda/2*sqwtplu); //cost reg


                        //thetasMinus
                        List<double[][]> thetasmin = initW.initWeightsGates(inputxdim,hiddencells,234819);
                        thetasmin.get(layer)[i][j] -= epsilon; //10.e-4
                        AccumList.get(0).reset();
                        AccumList.get(1).reset();
                        LSTM.LSTMfeedprop("feed", inputx, thetasmin, WfL, WoL, WgL, AccumList, dWiAccL, dWfAccL, dWoAccL, dWgAccL, Jouthslayer, RNNArchitecture); //LSTM RNN feedpropagation algorithm
                        double sqwtmin = initW.getthetassqw(thetasmin); //thetas squared
                        double costmin = (1./AccumList.get(1).value() * AccumList.get(0).value()) + (lmbda/2*sqwtmin); //cost reg

                        //gradient check
                        giAprox.get(layer)[i][j] = (costplus-costmin) / (2*epsilon);

                    }
                }
            }
            double[] giAproxV = initW.getflatThetas(giAprox);

            double aproxgradient = DoubleStream.of(giAproxV).sum();
            double feedbackgradient = DoubleStream.of(dWiV).sum();
            double gradientc = aproxgradient-feedbackgradient;
            gcA+= gradientc;


            NumberFormat formatter = new DecimalFormat("#0.0000");
            String feedbackgradient_ = formatter.format(feedbackgradient);
            String aproxgradient_ = formatter.format(aproxgradient);
            String gradientc_ = formatter.format(gradientc);


            System.out.println("aproximation:"+aproxgradient_);
            System.out.println("feedbackpropagation:"+feedbackgradient_);
            System.out.println("dWi Gradient Checking:"+gradientc_);





            //dWf
            //gradient checking equations
            List<double[][]> gfAprox = initW.getThetasLZeros(WfL);
            for (int layer=0; layer<WfL.size(); layer++){

                for (int i=0; i<WfL.get(layer).length; i++){
                    for (int j=0; j<WfL.get(layer)[0].length; j++){

                        //thetaPlus
                        List<double[][]> thetaplus = initW.initWeightsGates(inputxdim,hiddencells,109274);
                        thetaplus.get(layer)[i][j] += epsilon; //10.e-4
                        AccumList.get(0).reset();
                        AccumList.get(1).reset();
                        LSTM.LSTMfeedprop("feed", inputx, WiL, thetaplus, WoL, WgL, AccumList, dWiAccL, dWfAccL, dWoAccL, dWgAccL, Jouthslayer, RNNArchitecture); //LSTM RNN feedpropagation algorithm
                        double sqwtplu = initW.getthetassqw(thetaplus); //thetas squared
                        double costplus = (1./AccumList.get(1).value() * AccumList.get(0).value()) + (lmbda/2*sqwtplu); //cost reg


                        //thetasMinus
                        List<double[][]> thetasmin = initW.initWeightsGates(inputxdim,hiddencells,109274);
                        thetasmin.get(layer)[i][j] -= epsilon; //10.e-4
                        AccumList.get(0).reset();
                        AccumList.get(1).reset();
                        LSTM.LSTMfeedprop("feed", inputx, WiL, thetasmin, WoL, WgL, AccumList, dWiAccL, dWfAccL, dWoAccL, dWgAccL, Jouthslayer, RNNArchitecture); //LSTM RNN feedpropagation algorithm
                        double sqwtmin = initW.getthetassqw(thetasmin); //thetas squared
                        double costmin = (1./AccumList.get(1).value() * AccumList.get(0).value()) + (lmbda/2*sqwtmin); //cost reg

                        //gradient check
                        gfAprox.get(layer)[i][j] = (costplus-costmin) / (2*epsilon);

                    }
                }
            }


            double[] gfAproxV = initW.getflatThetas(gfAprox);

            aproxgradient = DoubleStream.of(gfAproxV).sum();
            feedbackgradient = DoubleStream.of(dWfV).sum();
            gradientc = aproxgradient-feedbackgradient;



            feedbackgradient_ = formatter.format(feedbackgradient);
            aproxgradient_ = formatter.format(aproxgradient);
            gradientc_ = formatter.format(gradientc);
            gcA+= gradientc;


            System.out.println("aproximation:"+aproxgradient_);
            System.out.println("feedbackpropagation:"+feedbackgradient_);
            System.out.println("dWf Gradient Checking:"+gradientc_);





            //dWo
            //gradient checking equations
            List<double[][]> goAprox = initW.getThetasLZeros(WoL);
            for (int layer=0; layer<WoL.size(); layer++){

                for (int i=0; i<WoL.get(layer).length; i++){
                    for (int j=0; j<WoL.get(layer)[0].length; j++){

                        //thetaPlus
                        List<double[][]> thetaplus = initW.initWeightsGates(inputxdim,hiddencells,554637);
                        thetaplus.get(layer)[i][j] += epsilon; //10.e-4
                        AccumList.get(0).reset();
                        AccumList.get(1).reset();
                        LSTM.LSTMfeedprop("feed", inputx, WiL, WfL, thetaplus, WgL, AccumList, dWiAccL, dWfAccL, dWoAccL, dWgAccL, Jouthslayer, RNNArchitecture); //LSTM RNN feedpropagation algorithm
                        double sqwtplu = initW.getthetassqw(thetaplus); //thetas squared
                        double costplus = (1./AccumList.get(1).value() * AccumList.get(0).value()) + (lmbda/2*sqwtplu); //cost reg


                        //thetasMinus
                        List<double[][]> thetasmin = initW.initWeightsGates(inputxdim,hiddencells,554637);
                        thetasmin.get(layer)[i][j] -= epsilon; //10.e-4
                        AccumList.get(0).reset();
                        AccumList.get(1).reset();
                        LSTM.LSTMfeedprop("feed", inputx, WiL, WfL, thetasmin, WgL, AccumList, dWiAccL, dWfAccL, dWoAccL, dWgAccL, Jouthslayer, RNNArchitecture); //LSTM RNN feedpropagation algorithm
                        double sqwtmin = initW.getthetassqw(thetasmin); //thetas squared
                        double costmin = (1./AccumList.get(1).value() * AccumList.get(0).value()) + (lmbda/2*sqwtmin); //cost reg

                        //gradient check
                        goAprox.get(layer)[i][j] = (costplus-costmin) / (2*epsilon);

                    }
                }
            }


            double[] goAproxV = initW.getflatThetas(goAprox);

            aproxgradient = DoubleStream.of(goAproxV).sum();
            feedbackgradient = DoubleStream.of(dWoV).sum();
            gradientc = aproxgradient-feedbackgradient;

            feedbackgradient_ = formatter.format(feedbackgradient);
            aproxgradient_ = formatter.format(aproxgradient);
            gradientc_ = formatter.format(gradientc);
            gcA+= gradientc;


            System.out.println("aproximation:"+aproxgradient_);
            System.out.println("feedbackpropagation:"+feedbackgradient_);
            System.out.println("dWo Gradient Checking:"+gradientc_);



            //dWc
            //gradient checking equations
            List<double[][]> ggAprox = initW.getThetasLZeros(WgL);
            for (int layer=0; layer<WgL.size(); layer++){


                for (int i=0; i<WgL.get(layer).length; i++){
                    for (int j=0; j<WgL.get(layer)[0].length; j++){

                        //thetaPlus
                        List<double[][]> thetaplus = initW.initWeights(inputxdim,hiddencells,outputcat,718274);
                        thetaplus.get(layer)[i][j] += epsilon; //10.e-4
                        AccumList.get(0).reset();
                        AccumList.get(1).reset();
                        LSTM.LSTMfeedprop("feed", inputx, WiL, WfL, WoL, thetaplus, AccumList, dWiAccL, dWfAccL, dWoAccL, dWgAccL, Jouthslayer, RNNArchitecture); //LSTM RNN feedpropagation algorithm
                        double sqwtplu = initW.getthetassqw(thetaplus); //thetas squared
                        double costplus = (1./AccumList.get(1).value() * AccumList.get(0).value()) + (lmbda/2*sqwtplu); //cost reg


                        //thetasMinus
                        List<double[][]> thetasmin = initW.initWeights(inputxdim,hiddencells,outputcat,718274);
                        thetasmin.get(layer)[i][j] -= epsilon; //10.e-4
                        AccumList.get(0).reset();
                        AccumList.get(1).reset();
                        LSTM.LSTMfeedprop("feed", inputx, WiL, WfL, WoL, thetasmin, AccumList, dWiAccL, dWfAccL, dWoAccL, dWgAccL, Jouthslayer, RNNArchitecture); //LSTM RNN feedpropagation algorithm
                        double sqwtmin = initW.getthetassqw(thetasmin); //thetas squared
                        double costmin = (1./AccumList.get(1).value() * AccumList.get(0).value()) + (lmbda/2*sqwtmin); //cost reg

                        //gradient check
                        ggAprox.get(layer)[i][j] = (costplus-costmin) / (2*epsilon);

                    }
                }
            }


            double[] ggAproxV = initW.getflatThetas(ggAprox);

            aproxgradient = DoubleStream.of(ggAproxV).sum();
            feedbackgradient = DoubleStream.of(dWgV).sum();
            gradientc = aproxgradient-feedbackgradient;
            gcA+= gradientc;

            feedbackgradient_ = formatter.format(feedbackgradient);
            aproxgradient_ = formatter.format(aproxgradient);
            gradientc_ = formatter.format(gradientc);


            System.out.println("aproximation:"+aproxgradient_);
            System.out.println("feedbackpropagation:"+feedbackgradient_);
            System.out.println("dWc Gradient Checking:"+gradientc_);

            String gcA_ = formatter.format(gcA);
            System.out.println("\nLSTM Gradient Checking:"+gcA_+"\n");



        }



    }

}
