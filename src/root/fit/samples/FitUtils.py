#################################################                                 
# Tools for creating fitting functions as well  #     
# multidimensional fitting models, plotting     #
# the pdf and pull distributions.               #             
#################################################

##### Run from within Dstar_Fit

import ROOT
from ROOT import gROOT
from ROOT import gSystem
from ROOT import gStyle
from ROOT import gPad
import time 

gSystem.Load('libRooFit')
#gROOT.ProcessLine(".x ./PDFs/RooJohnsonSU.cc+")
gROOT.ProcessLine(".x ./PDFs/RooCruijff.cc+")
gROOT.ProcessLine(".x ./PDFs/RooDoubleCB.cc+")
gROOT.ProcessLine(".L ./PDFs/RooCruijffExtended.C")
gROOT.ProcessLine(".L ./PDFs/RooBkg.cpp")
gSystem.Load('libRooFit')

from lhcbStyle import setLHCbStyle
import HistoUtils
import subprocess



def Dstar_4comp(entries, DM, deltaM, a, p_row, splot):
    #### Default fit

    mean = ROOT.RooRealVar ("mean", "mean", 1866, 1860, 1872)
    sigma1 = ROOT.RooRealVar ("sigma1", "sigma1", 15, 8, 30)
    gaus1 = ROOT.RooGaussian("gaus1", "gaus1", DM, mean, sigma1)
    #    mean = RooRealVar ("mean", "mean", 1866, 1700, 2000)
    sigma2 = ROOT.RooRealVar ("sigma2", "sigma2", 5, 4, 10)
    gaus2 = ROOT.RooGaussian("gaus2", "gaus2", DM, mean, sigma2)

    cf = ROOT.RooRealVar ("cf", "cf", .25, 0, 1.)

    signalD = ROOT.RooAddPdf ( "signalD", "signalD", ROOT.RooArgList( gaus1, gaus2),ROOT.RooArgList(cf) ) 

    gaus1B = ROOT.RooGaussian("gaus1B", "gausB", DM, mean, sigma1)
    gaus2B = ROOT.RooGaussian("gaus2B", "gausB", DM, mean, sigma2)
    signalB = ROOT.RooAddPdf ( "signalB", "signalB",ROOT.RooArgList( gaus1B, gaus2B), ROOT.RooArgList(cf) ) 

    meanDelta   = ROOT.RooRealVar ("meanDelta",    "meanDelta", 145.5, 140, 150)
    meanDelta2  = ROOT.RooRealVar ("meanDelta2",   "meanDelta2",145.5, 140, 150)
    sigmaDelta  = ROOT.RooRealVar ("sigmaDelta",   "sigmaDelta", 0.5, 0.1, 10)
    gausDelta   = ROOT.RooGaussian("gausDelta",    "gausDelta", deltaM, meanDelta, sigmaDelta)
    sigmaDelta2 = ROOT.RooRealVar ("sigmaDelta2",  "sigmaDelta2", .8, 0.1, 10)
    sigmaDelta3 = ROOT.RooRealVar ("sigmaDelta3",  "sigmaDelta3", .8, 0.1, 10)
    gausDelta2  = ROOT.RooGaussian("gausDelta2",   "gausDelta2", deltaM, meanDelta, sigmaDelta3)
    sigmaDelta4 = ROOT.RooRealVar ("sigmaDelta4",  "sigmaDelta3", .8, 0.1, 10)
    gausDelta4  = ROOT.RooGaussian("gausDelta4",   "gausDelta2", deltaM, meanDelta, sigmaDelta4)
    nDelta      = ROOT.RooRealVar ("nDelta",       "nDelta",    -1.3, -10, -0.1)
    alphaDelta  = ROOT.RooRealVar ("alphaDelta",   "alphaDelta", 40, 30, 80)    # 40, 30, 80
    cbDelta2    = ROOT.RooCBShape ("cbDelta2",     "cbDelta2", deltaM, meanDelta2, sigmaDelta2, nDelta, alphaDelta)    
    cfDelta     = ROOT.RooRealVar ("cfDelta",      "cfDelta", .46, 0, 1)
    cfDelta2    = ROOT.RooRealVar ("cfDelta2",      "cfDelta2", .02, 0,0.1)    # .02, 0,0.1
    cfDelta3    = ROOT.RooRealVar ("cf3Delta",      "cf3Delta", .02, 0,0.1)
    signalDelta = ROOT.RooAddPdf  ("signalDelta",  "signalDelta", ROOT.RooArgList( gausDelta, gausDelta2, cbDelta2), ROOT.RooArgList ( cfDelta,cfDelta2 ))
    signalDelta2 = ROOT.RooAddPdf  ("signalDelta2",  "signalDelta2", ROOT.RooArgList( gausDelta, gausDelta2, cbDelta2), ROOT.RooArgList ( cfDelta, cfDelta2 ))
    
    bg0 = ROOT.RooRealVar("bg0", "bg0", -0.27, -1, 1)
    bg1 = ROOT.RooRealVar("bg1", "bg1", -0.27, -1, 1)
    bg2 = ROOT.RooRealVar("bg2", "bg2", 1e-7, 1e-10,1e-5)
    bg3 = ROOT.RooRealVar("bg3", "bg3", 1e-7, 1e-10,1e-5)

    bkg = ROOT.RooChebychev ( "bkg", "bkg", DM, ROOT.RooArgList(bg0) )
    bkgrDs = ROOT.RooChebychev ( "bkgrDs", "bkgrDs", DM, ROOT.RooArgList(bg1) )
    #bkg = RooExponential( "bkg", "bkg", DM, bg2 )
    #bkgrDs = ROOT.RooExponential( "bkgrDs","bkgrDs",DM,bg3)


    bg0B = ROOT.RooRealVar("bg0B", "bg0", -4, -10, 10)
    bg1B = ROOT.RooRealVar("bg1B", "bg1", -0.7, -10, 100)
    bg2B = ROOT.RooRealVar("bg2B", "bg2", 0, -1, 1)

    bkgB = ROOT.RooGenericPdf ( "bkgB", "bkg", "sqrt((@0)/139.57 -1)*exp(@1*(@0)/139.57)", ROOT.RooArgList( deltaM, bg0B) )
    #    bkgB = RooChebychev ( "bkgB", "bkgB", deltaM, RooArgList(bg0B, bg1B) )

    bg0Delta = ROOT.RooRealVar("bg0Delta", "bg0", 0, -10, 10)
    bg1Delta = ROOT.RooRealVar("bg1Delta", "bg1", -0.6, -10, 10)
    bg2Delta = ROOT.RooRealVar("bg2Delta", "bg2", 0, -1, 1)

    #    bkgDelta = RooChebychev ( "bkgDelta", "bkg", deltaM, RooArgList(bg0Delta, bg1Delta) )
    #RooGenericPdf delmBkgModel(m_delmBkgModelName, " ", "sqrt((@0)/139.57 -1)*exp(@1*(@0)/139.57)",RooArgList(*delm,c));

    threshold = ROOT.RooRealVar("threshold", "threshold", 139.57, 139, 142)    
    bkgDelta = ROOT.RooGenericPdf ( "bkgDelta", "bkg", "sqrt((@0)/@2 -1)*exp(@1*(@0)/@2)", ROOT.RooArgList( deltaM, bg0B, threshold))

    signal = ROOT.RooProdPdf ( "signal", "",        ROOT.RooArgList( signalD, signalDelta ) )
    nrD    = ROOT.RooProdPdf ( "nrD", "",           ROOT.RooArgList( signalB, bkgB     ) )
    combo  = ROOT.RooProdPdf ( "combinatorial", "", ROOT.RooArgList(bkg, bkgDelta   ) )
    mbDs   = ROOT.RooProdPdf ( "MultibodyrealDstar", "",ROOT.RooArgList(bkgrDs,signalDelta2) )
    
    nTot = entries
    nSig = ROOT.RooRealVar ( "nSig", "nSig", 0.6 * nTot, 0, nTot)
    nNR  = ROOT.RooRealVar ( "nNR" , "nNR" , 0.15 * nTot, 0, nTot)
    nCmb = ROOT.RooRealVar ( "nCmb", "nCmb", 0.2 * nTot, 0, nTot)
    nRDs = ROOT.RooRealVar ( "nRDs", "nRDs", 0.05 * nTot, 0,nTot)


    if splot:
    ### Runs if the efficiency calculation is required by Dstar_Fit, must be different from usual fit as nBkg cannot be a RooFormularVar for the sPlot function to run
        bkg_total = ROOT.RooAddPdf( "bkg_total", "bkg_total", ROOT.RooArgList(nrD, combo, mbDs), ROOT.RooArgList(nNR, nCmb, nRDs) )
        #nBkg = ROOT.RooFormulaVar( "nBkg", "nNR + nCmb + nRDs", ROOT.RooArgList(nNR, nCmb, nRDs) )
        nBkg = ROOT.RooRealVar( "nBkg", "nBkg", 0.3 * nTot, 0, nTot)
            
        model = ROOT.RooAddPdf ( "model", "model", ROOT.RooArgList(signal, bkg_total), ROOT.RooArgList(nSig, nBkg) )

    else:
        model = ROOT.RooAddPdf ( "model", "model", ROOT.RooArgList(signal, nrD,  combo, mbDs) , ROOT.RooArgList(nSig, nNR, nCmb, nRDs) )
   

    work_space = ROOT.RooWorkspace("w")
    getattr(work_space,'import')(model)

    return work_space




def Dstar_run3(entries, DM, deltaM, bkg_option, bin_no, decay_mode, splot, nSig_value, nNR_value, nCmb_value):
#### Alternate fits

    signalD = ''
    signalDelta = ''
    signalB = ''
    bkgB = ''
    bkg = ''
    bkgDelta = ''
    bkgrDs = ''
    signalDelta2 = ''

    mean_low = 145.42 - 2
    mean_high = 145.42 + 2

    
    ##########################################################################################
    ##########################################################################################
    ### mD Model
    ##########################################################################################
    ##########################################################################################

    #alpha_L_list = [0.17455, 0.20148, 0.18476, 0.18730, 0.18517, 0.17481, 0.17647, 0.17363]
    #alpha_R_list = [0.14390, 0.14161, 0.14130, 0.14502, 0.14336, 0.13970, 0.14037, 0.14237]
    #beta_C_list = [0.0000078059, 0.000029506, 0.000023003, 0.000020691, 0.000013000, 0.0000045044, 0.0000052003, 0.0000058715]
    #frac_list = [0.89355, 0.73717, 0.89806, 0.82704, 0.77225, 0.85302, 0.83410, 0.88657]
    
    alpha_L_list = [0.17455, 0.20138, 0.18497, 0.18731, 0.18511, 0.17487, 0.17647, 0.17365]
    alpha_R_list = [0.14389, 0.14153, 0.14145, 0.14502, 0.14332, 0.13975, 0.14037, 0.14238]
    beta_C_list = [0.0000078024, 0.000029394, 0.000023224, 0.000020722, 0.000012944, 0.0000045679, 0.0000051972, 0.0000058998]
    frac_list = [0.89356, 0.73767, 0.89636, 0.82715, 0.77264, 0.85249, 0.83406, 0.88649]


    #if bin_no == 1 or bin_no == 2:
    alpha_L = ROOT.RooRealVar("alpha_L","alpha_L", alpha_L_list[bin_no], -10.0, 10.0)
    alpha_R = ROOT.RooRealVar("alpha_R","alpha_R", alpha_R_list[bin_no], -10.0, 10.0)
    
    #else:
    #alpha_L = ROOT.RooRealVar("alpha_L","alpha_L", alpha_L_list[bin_no])
    #alpha_R = ROOT.RooRealVar("alpha_R","alpha_R", alpha_R_list[bin_no])

    #if bin_no == 1:
    #beta_C = ROOT.RooRealVar("beta_C","beta_C", beta_C_list[bin_no], -1.0, 1.0)
    #else:    
    beta_C = ROOT.RooRealVar("beta_C","beta_C", beta_C_list[bin_no])



    #frac = ROOT.RooRealVar("frac","frac", frac_list[bin_no], 0.01, 1.0)
    frac = ROOT.RooRealVar("frac","frac", frac_list[bin_no])


    mean = ROOT.RooRealVar("mean", "mean", 1866., 1800., 1930.)
    sigma = ROOT.RooRealVar("sigma", "sigma", 10., 0.00001, 200.)

    cruijffE = ROOT.RooCruijffExtended("cruijffE", "cruijffE", DM, mean, sigma, alpha_L, alpha_R, beta_C)

    gaus = ROOT.RooGaussian("gaus", "gaus", DM, mean, sigma)

    signalD = ROOT.RooAddPdf( "signalD", "signalD", ROOT.RooArgList(cruijffE,gaus), ROOT.RooArgList(frac))
    signalB = ROOT.RooAddPdf( "signalB", "signalB", ROOT.RooArgList(cruijffE,gaus), ROOT.RooArgList(frac))


    ##########################################################################################
    ##########################################################################################
    ### deltaM Model
    ##########################################################################################
    ##########################################################################################

    #alpha_LDelta_list = [0.14700, 0.15280, 0.15705, 0.15386, 0.15070, 0.14974, 0.14767, 0.14672]
    #alpha_RDelta_list = [0.18689, 0.16492, 0.18048, 0.17724, 0.17762, 0.18016, 0.17962, 0.18552]
    #beta_CDelta_list = [0.0013024, 0.00048662, 0.00081796, 0.00093552, 0.0012026, 0.00081637, 0.0010859, 0.0011915]
    #cfDelta_list = [0.86070, 0.95864, 0.85694, 0.89523, 0.90641, 0.85413, 0.88552, 0.86332]
    #scale_sigmaDelta2_list = [0.53538, 0.50380, 0.57015, 0.54809, 0.54420, 0.56599, 0.53477, 0.53607]

    alpha_LDelta_list = [0.16720, 0.15676, 0.17042, 0.16463, 0.16097, 0.16548, 0.16236, 0.16633]
    alpha_RDelta_list = [0.20355, 0.16882, 0.19245, 0.18715, 0.18676, 0.19340, 0.19243, 0.20180]
    beta_CDelta_list = [0.0044547, 0.0013480, 0.0034584, 0.0030529, 0.0031348, 0.0035943, 0.0036283, 0.0042713]


    #if bin_no == 1 or bin_no == 2:
    alpha_LDelta = ROOT.RooRealVar("alpha_LDelta","alpha_LDelta", alpha_LDelta_list[bin_no], -1.0, 1.0)
    alpha_RDelta = ROOT.RooRealVar("alpha_RDelta","alpha_RDelta", alpha_RDelta_list[bin_no], -1.0, 1.0)
    #beta_CDelta = ROOT.RooRealVar("beta_CDelta","beta_CDelta", beta_CDelta_list[bin_no], -1.0, 1.0)

    #else:
    #alpha_LDelta = ROOT.RooRealVar("alpha_LDelta","alpha_LDelta", alpha_LDelta_list[bin_no])
    #alpha_RDelta = ROOT.RooRealVar("alpha_RDelta","alpha_RDelta", alpha_RDelta_list[bin_no])
    beta_CDelta = ROOT.RooRealVar("beta_CDelta","beta_CDelta", beta_CDelta_list[bin_no])


    #cfDelta = ROOT.RooRealVar("cfDelta","cfDelta", cfDelta_list[bin_no]) 
    #scale_sigmaDelta2 = ROOT.RooRealVar("scale_sigmaDelta2","scale_sigmaDelta2", scale_sigmaDelta2_list[bin_no])
    ###scale_sigmaDelta2 = ROOT.RooRealVar("scale_sigmaDelta2","scale_sigmaDelta2", 0.5, 0.1, 10.0)
         
      

    meanDelta = ROOT.RooRealVar("meanDelta", "meanDelta", 144., mean_low, mean_high)
    sigmaDelta = ROOT.RooRealVar("sigmaDelta", "sigmaDelta", 0.6, 0.01, 6.)  
           
    cruijffEDelta = ROOT.RooCruijffExtended("cruijffEDelta", "cruijffEDelta", deltaM, meanDelta, sigmaDelta, alpha_LDelta, alpha_RDelta, beta_CDelta)
    cruijffEDelta2 = ROOT.RooCruijffExtended("cruijffEDelta2", "cruijffEDelta2", deltaM, meanDelta, sigmaDelta, alpha_LDelta, alpha_RDelta, beta_CDelta)

    #sigmaDelta2 = ROOT.RooFormulaVar("sigmaDelta2", "@0*@1",ROOT.RooArgList(sigmaDelta,scale_sigmaDelta2))
    #gausDelta = ROOT.RooGaussian("gausDelta", "gausDelta", deltaM, meanDelta, sigmaDelta2)

    #signalDelta = ROOT.RooAddPdf( "signalDelta", "signalDelta", ROOT.RooArgList(cruijffEDelta,gausDelta),ROOT.RooArgList(cfDelta))
    #signalDelta2 = ROOT.RooAddPdf( "signalDelta2", "signalDelta2",ROOT.RooArgList(cruijffEDelta,gausDelta), ROOT.RooArgList(cfDelta))
    signalDelta = cruijffEDelta
    signalDelta2 = cruijffEDelta2

    ##########################################################################################
    ##########################################################################################
    ### Multibody background
    ##########################################################################################
    ##########################################################################################

    b0_list_PiP = [-0.421022, -1.15223, -0.893271, -0.333530, -0.354454, -0.463006, -0.188743, -0.294997]
    b1_list_PiP = [0.11304, 0.12759, 0.20003, 0.055392, 0.084300, 0.13474, -0.00627153, 0.075110]

    b0_list_KM = [-0.421022, -0.280649, -0.228693, -1.10390, -0.831957, -0.484473, -0.818137, -0.814227]
    b1_list_KM = [0.11304, 0.074796, 0.041038, 0.20031, 0.29951, 0.085794, 0.20702, 0.22887 ]

    mean_list_PiP = [0, 1900.3, 1897.8, 0, 0, 0, 0, 0]
    sigma_list_PiP = [0, 12.782, 6.2445, 0, 0, 0, 0, 0]
    cf_list_PiP = [0, 0.86260, 0.95999, 0, 0, 0, 0, 0,]

    mean_list_KM = [0, 0, 0, 1902.9, 1902.6, 0, 0, 1896.3]
    sigma_list_KM = [0, 0, 0, 5.3473, 2.2904, 0, 0, 5.6240]
    cf_list_KM = [0, 0, 0, 0.94215, 0.99113, 0, 0, 0.97514]

    ### Boolean lists denotes whether a gaussian is required in this bin to describe the multibody background (See RapidSim chapters of logbook)
    factor_list_PiP = [0.0154, 0.0085, 0.0079, 0.0227, 0.0223, 0.0147, 0.0204, 0.0218 ]
    bkgauss_list_PiP = [False, True, True, False, False, False, False, False]

    factor_list_KM = [0.0154, 0.0272, 0.0274, 0.0043, 0.0063, 0.0164, 0.0051, 0.0081]
    bkgauss_list_KM = [False, False, False, True, True, False, False, True]

    if decay_mode == "DSt_PiP" or decay_mode == "DSt_PiM":
        b0 = ROOT.RooRealVar("b0", "b0", b0_list_PiP[bin_no])  #
        b1 = ROOT.RooRealVar("b1", "b1", b1_list_PiP[bin_no])  #
        factor = factor_list_PiP[bin_no]
    if decay_mode == "DSt_KM" or decay_mode == "DSt_KP":
        b0 = ROOT.RooRealVar("b0", "b0", b0_list_KM[bin_no])  #
        b1 = ROOT.RooRealVar("b1", "b1", b1_list_KM[bin_no])  #
        factor = factor_list_KM[bin_no]

   
    if (decay_mode == "DSt_PiP" or decay_mode == "DSt_PiM") and not bkgauss_list_PiP[bin_no]: 
        bkgrDs = ROOT.RooChebychev("bkgrDs", "bkgrDs", DM, ROOT.RooArgList(b0, b1))

    elif bkgauss_list_PiP[bin_no] and (decay_mode == "DSt_PiP" or decay_mode == "DSt_PiM"):
        bkgrDsfirst = ROOT.RooChebychev("bkgrDsfirst", "bkgrDsfirst", DM, ROOT.RooArgList(b0, b1)) 
        mean_bkg = ROOT.RooRealVar("mean_bkg", "mean_bkg", mean_list_PiP[bin_no])
        sigma_bkg = ROOT.RooRealVar("sigma_bkg", "sigma_bkg", sigma_list_PiP[bin_no])
        gauss_bkg = ROOT.RooGaussian("gauss_bkg", "gauss_bkg", DM, mean_bkg, sigma_bkg)

        cf_bkg = ROOT.RooRealVar("cf_bkg", "cf_bkg", cf_list_PiP[bin_no])
        
        bkgrDs = ROOT.RooAddPdf( "bkgrDs", "bkgrDs", ROOT.RooArgList(bkgrDsfirst, gauss_bkg), ROOT.RooArgList(cf_bkg))    

    elif (decay_mode == "DSt_KM" or decay_mode == "DSt_KP") and not bkgauss_list_KM[bin_no]:
        bkgrDs = ROOT.RooChebychev("bkgrDs", "bkgrDs", DM, ROOT.RooArgList(b0, b1)) 

    elif bkgauss_list_KM[bin_no] and (decay_mode == "DSt_KM" or decay_mode == "DSt_KP"):
        bkgrDsfirst = ROOT.RooChebychev("bkgrDsfirst", "bkgrDsfirst", DM, ROOT.RooArgList(b0, b1)) 
        mean_bkg = ROOT.RooRealVar("mean_bkg", "mean_bkg", mean_list_KM[bin_no])
        sigma_bkg = ROOT.RooRealVar("sigma_bkg", "sigma_bkg", sigma_list_KM[bin_no])
        gauss_bkg = ROOT.RooGaussian("gauss_bkg", "gauss_bkg", DM, mean_bkg, sigma_bkg)

        cf_bkg = ROOT.RooRealVar("cf_bkg", "cf_bkg", cf_list_KM[bin_no])
        
        bkgrDs = ROOT.RooAddPdf( "bkgrDs", "bkgrDs", ROOT.RooArgList(bkgrDsfirst, gauss_bkg), ROOT.RooArgList(cf_bkg))

 
    ##########################################################################################
    ##########################################################################################
    ### Random Pi + Combinatorial background 
    ##########################################################################################
    ##########################################################################################

    bg0 = ROOT.RooRealVar("bg0", "bg0", -0.1, -10.0, 0.0)
    bkg = ROOT.RooChebychev("bkg", "bkg", DM, ROOT.RooArgList(bg0))


    threshold = ROOT.RooRealVar("threshold", "threshold", 139.57)
    dm0 = ROOT.RooRealVar("dm0", "dm0", 139.57, 139., 142.)
    
    

    if bkg_option == 'dstar_bkg':
        bg0B = ROOT.RooRealVar("bg0B", "bg0B", 0.0, -10.0, 10.0) 
        bg1B = ROOT.RooRealVar("bg1B", "bg1B", 0.0, -10.0, 10.0)

        bkgB = ROOT.RooBkg("bkgB", "bkgB", deltaM, threshold, bg0B, bg1B)
        bkgDelta = ROOT.RooBkg("bkgDelta", "bkgDelta", deltaM, dm0, bg0B, bg1B)


    elif bkg_option == 'dstar_bkg1':
        bg0B = ROOT.RooRealVar("bg0B", "bg0B", 0.0, -10.0, 10.0)

        bkgB = ROOT.RooGenericPdf ( "bkgB", "bkgB", "sqrt((@0)/139.57 -1)*exp(@1*(@0)/139.57)", ROOT.RooArgList( deltaM, bg0B) )
        bkgDelta = ROOT.RooGenericPdf ( "bkgDelta", "bkgDelta", "sqrt((@0)/@2 -1)*exp(@1*(@0)/@2)", ROOT.RooArgList( deltaM, bg0B, dm0))
    
    elif bkg_option == 'dstar_bkg2':
        bg0B = ROOT.RooRealVar("bg0B", "bg0B", 0.0, -10.0, 10.0) 
        bg1B = ROOT.RooRealVar("bg1B", "bg1B", 0.0, -10.0, 10.0)

        bkgB = ROOT.RooGenericPdf ( "bkgB", "bkgB", "TMath::Power(@0-139.57,@1)*exp(@2*(@0-139.57))",ROOT.RooArgList(deltaM, bg0B,bg1B))
        bkgDelta = ROOT.RooGenericPdf ( "bkgDelta", "bkgDelta", "TMath::Power(@0-@1,@2)*exp(@3*(@0-@1))", ROOT.RooArgList( deltaM,dm0,bg0B,bg1B))

    elif bkg_option == 'dstar_bkg3':
        bg0B = ROOT.RooRealVar("bg0B", "bg0B", 0.0, -10.0, 10.0) 
        bg1B = ROOT.RooRealVar("bg1B", "bg1B", 0.0, -10.0, 10.0)
        bg2B = ROOT.RooRealVar("bg2B", "bg2B", 0.0, -10.0, 10.0)

        bkgB = ROOT.RooDstD0BG("bkgB","bkgB", deltaM, threshold, bg0B, bg1B, bg2B)
        bkgDelta = ROOT.RooDstD0BG("bkgDelta","bkgDelta", deltaM, dm0, bg0B, bg1B, bg2B)


    ##########################################################################################
    ##########################################################################################
    ### TOTAL 2D MODEL 
    ##########################################################################################
    ##########################################################################################

    signal = ROOT.RooProdPdf ( "signal", "",        ROOT.RooArgList( signalD, signalDelta ) )
    nrD    = ROOT.RooProdPdf ( "nrD", "",           ROOT.RooArgList( signalB, bkgB     ) )
    combo  = ROOT.RooProdPdf ( "combinatorial", "", ROOT.RooArgList( bkg, bkgDelta   ) )
    mbDs = ROOT.RooProdPdf ( "MultibodyrealDstar", "",ROOT.RooArgList( bkgrDs, signalDelta2) )

    nTot = entries

    nSig = ROOT.RooRealVar ( "nSig", "nSig", 0.7 * nTot, 0, nTot)

    nNR  = ROOT.RooRealVar ( "nNR" , "nNR" , 0.18 * nTot, 0, nTot) #0.2
    nCmb = ROOT.RooRealVar ( "nCmb", "nCmb", 0.1 * nTot, 0, nTot)
    #nRDs = ROOT.RooRealVar ( "nRDs", "nRDs", 0.05 * nTot, 0,nTot)
    nRDs = ROOT.RooFormulaVar("nRDs", "{}*nSig".format(factor), ROOT.RooArgList(nSig))


    if splot:
    ### Runs if the efficiency calculation is required by Dstar_Fit, must be different from usual fit as nBkg cannot be a RooFormularVar for the sPlot function to run
        if nSig_value != '':
            nRDs_fixed = ROOT.RooRealVar("nRDs_fixed", "nRDs_fixed", nSig_value*factor)
            nNR_fixed = ROOT.RooRealVar("nNR_fixed", "nNR_fixed", nNR_value)
            nCmb_fixed = ROOT.RooRealVar("nCmb_fixed", "nCmb_fixed", nCmb_value)
            nBkg_value = nNR_value + nCmb_value + (nSig_value*factor)

            bkg_total = ROOT.RooAddPdf( "bkg_total", "bkg_total", ROOT.RooArgList(nrD, combo, mbDs), ROOT.RooArgList(nNR_fixed, nCmb_fixed, nRDs_fixed) )

            nBkg = ROOT.RooRealVar( "nBkg", "nBkg", 0.3 * nTot, 0, nTot)

        else:
            bkg_total = ROOT.RooAddPdf( "bkg_total", "bkg_total", ROOT.RooArgList(nrD, combo, mbDs), ROOT.RooArgList(nNR, nCmb, nRDs) )
            nBkg = ROOT.RooFormulaVar( "nBkg", "nNR + nCmb + nRDs", ROOT.RooArgList(nNR, nCmb, nRDs) )
            
        model = ROOT.RooAddPdf ( "model", "model", ROOT.RooArgList(signal, bkg_total), ROOT.RooArgList(nSig, nBkg) )

    else:
        model = ROOT.RooAddPdf ( "model", "model", ROOT.RooArgList(signal, nrD,  combo, mbDs) , ROOT.RooArgList(nSig, nNR, nCmb, nRDs) )
        

    work_space = ROOT.RooWorkspace("w")
    getattr(work_space,'import')(model)

    return work_space





def Dstar_run3_3comp(entries, DM, deltaM, bkg_option, bin_no, decay_mode, splot):
#### Alternate fits

    signalD = ''
    signalDelta = ''
    signalB = ''
    bkgB = ''
    bkg = ''
    bkgDelta = ''

    mean_low = 145.42 - 2
    mean_high = 145.42 + 2

    
    ##########################################################################################
    ##########################################################################################
    ### mD Model
    ##########################################################################################
    ##########################################################################################

    #alpha_L_list = [0.17455, 0.20148, 0.18476, 0.18730, 0.18517, 0.17481, 0.17647, 0.17363]
    #alpha_R_list = [0.14390, 0.14161, 0.14130, 0.14502, 0.14336, 0.13970, 0.14037, 0.14237]
    #beta_C_list = [0.0000078059, 0.000029506, 0.000023003, 0.000020691, 0.000013000, 0.0000045044, 0.0000052003, 0.0000058715]
    #frac_list = [0.89355, 0.73717, 0.89806, 0.82704, 0.77225, 0.85302, 0.83410, 0.88657]
    
    alpha_L_list = [0.17455, 0.20138, 0.18497, 0.18731, 0.18511, 0.17487, 0.17647, 0.17365]
    alpha_R_list = [0.14389, 0.14153, 0.14145, 0.14502, 0.14332, 0.13975, 0.14037, 0.14238]
    beta_C_list = [0.0000078024, 0.000029394, 0.000023224, 0.000020722, 0.000012944, 0.0000045679, 0.0000051972, 0.0000058998]
    frac_list = [0.89356, 0.73767, 0.89636, 0.82715, 0.77264, 0.85249, 0.83406, 0.88649]


    #if bin_no == 1 or bin_no == 2:
    #alpha_L = ROOT.RooRealVar("alpha_L","alpha_L", alpha_L_list[bin_no], -1.0, 1.0)
    alpha_R = ROOT.RooRealVar("alpha_R","alpha_R", alpha_R_list[bin_no], -1.0, 1.0)
    
    #else:
    alpha_L = ROOT.RooRealVar("alpha_L","alpha_L", alpha_L_list[bin_no])
    #alpha_R = ROOT.RooRealVar("alpha_R","alpha_R", alpha_R_list[bin_no])

    #if bin_no == 1:
    #beta_C = ROOT.RooRealVar("beta_C","beta_C", beta_C_list[bin_no], -1.0, 1.0)
    #else:    
    beta_C = ROOT.RooRealVar("beta_C","beta_C", beta_C_list[bin_no])



    #frac = ROOT.RooRealVar("frac","frac", frac_list[bin_no], 0.01, 1.0)
    frac = ROOT.RooRealVar("frac","frac", frac_list[bin_no])


    mean = ROOT.RooRealVar("mean", "mean", 1866., 1860., 1872.)
    sigma = ROOT.RooRealVar("sigma", "sigma", 3., 0.00001, 30.)

    cruijffE = ROOT.RooCruijffExtended("cruijffE", "cruijffE", DM, mean, sigma, alpha_L, alpha_R, beta_C)

    gaus = ROOT.RooGaussian("gaus", "gaus", DM, mean, sigma)

    signalD = ROOT.RooAddPdf( "signalD", "signalD", ROOT.RooArgList(cruijffE,gaus), ROOT.RooArgList(frac))
    signalB = ROOT.RooAddPdf( "signalB", "signalB", ROOT.RooArgList(cruijffE,gaus), ROOT.RooArgList(frac))


    ##########################################################################################
    ##########################################################################################
    ### deltaM Model
    ##########################################################################################
    ##########################################################################################

    #alpha_LDelta_list = [0.14700, 0.15280, 0.15705, 0.15386, 0.15070, 0.14974, 0.14767, 0.14672]
    #alpha_RDelta_list = [0.18689, 0.16492, 0.18048, 0.17724, 0.17762, 0.18016, 0.17962, 0.18552]
    #beta_CDelta_list = [0.0013024, 0.00048662, 0.00081796, 0.00093552, 0.0012026, 0.00081637, 0.0010859, 0.0011915]
    #cfDelta_list = [0.86070, 0.95864, 0.85694, 0.89523, 0.90641, 0.85413, 0.88552, 0.86332]
    #scale_sigmaDelta2_list = [0.53538, 0.50380, 0.57015, 0.54809, 0.54420, 0.56599, 0.53477, 0.53607]

    alpha_LDelta_list = [0.16720, 0.15676, 0.17042, 0.16463, 0.16097, 0.16548, 0.16236, 0.16633]
    alpha_RDelta_list = [0.20355, 0.16882, 0.19245, 0.18715, 0.18676, 0.19340, 0.19243, 0.20180]
    beta_CDelta_list = [0.0044547, 0.0013480, 0.0034584, 0.0030529, 0.0031348, 0.0035943, 0.0036283, 0.0042713]


    #if bin_no == 1 or bin_no == 2:
    #alpha_LDelta = ROOT.RooRealVar("alpha_LDelta","alpha_LDelta", alpha_LDelta_list[bin_no], -1.0, 1.0)
    #alpha_RDelta = ROOT.RooRealVar("alpha_RDelta","alpha_RDelta", alpha_RDelta_list[bin_no], -1.0, 1.0)
    #beta_CDelta = ROOT.RooRealVar("beta_CDelta","beta_CDelta", beta_CDelta_list[bin_no], -1.0, 1.0)

    #else:
    alpha_LDelta = ROOT.RooRealVar("alpha_LDelta","alpha_LDelta", alpha_LDelta_list[bin_no])
    alpha_RDelta = ROOT.RooRealVar("alpha_RDelta","alpha_RDelta", alpha_RDelta_list[bin_no])
    beta_CDelta = ROOT.RooRealVar("beta_CDelta","beta_CDelta", beta_CDelta_list[bin_no])


    #cfDelta = ROOT.RooRealVar("cfDelta","cfDelta", cfDelta_list[bin_no]) 
    #scale_sigmaDelta2 = ROOT.RooRealVar("scale_sigmaDelta2","scale_sigmaDelta2", scale_sigmaDelta2_list[bin_no])
    ###scale_sigmaDelta2 = ROOT.RooRealVar("scale_sigmaDelta2","scale_sigmaDelta2", 0.5, 0.1, 10.0)
         
      

    meanDelta = ROOT.RooRealVar("meanDelta", "meanDelta", 144., mean_low, mean_high)
    sigmaDelta = ROOT.RooRealVar("sigmaDelta", "sigmaDelta", 0.6, 0.01, 6.)  
           
    cruijffEDelta = ROOT.RooCruijffExtended("cruijffEDelta", "cruijffEDelta", deltaM, meanDelta, sigmaDelta, alpha_LDelta, alpha_RDelta, beta_CDelta)

    #sigmaDelta2 = ROOT.RooFormulaVar("sigmaDelta2", "@0*@1",ROOT.RooArgList(sigmaDelta,scale_sigmaDelta2))
    #gausDelta = ROOT.RooGaussian("gausDelta", "gausDelta", deltaM, meanDelta, sigmaDelta2)

    #signalDelta = ROOT.RooAddPdf( "signalDelta", "signalDelta", ROOT.RooArgList(cruijffEDelta,gausDelta),ROOT.RooArgList(cfDelta))
    signalDelta = cruijffEDelta


    ##########################################################################################
    ##########################################################################################
    ### Multibody background
    ##########################################################################################
    ##########################################################################################

    ##########################################################################################
    ##########################################################################################
    ### Random Pi + Combinatorial background 
    ##########################################################################################
    ##########################################################################################

    bg0 = ROOT.RooRealVar("bg0", "bg0", -0.1, -10.0, 0.0)
    bkg = ROOT.RooChebychev("bkg", "bkg", DM, ROOT.RooArgList(bg0))


    threshold = ROOT.RooRealVar("threshold", "threshold", 139.57)
    dm0 = ROOT.RooRealVar("dm0", "dm0", 139.57, 139., 142.)
    
    

    if bkg_option == 'dstar_bkg':
        bg0B = ROOT.RooRealVar("bg0B", "bg0B", 0.0, -10.0, 10.0) 
        bg1B = ROOT.RooRealVar("bg1B", "bg1B", 0.0, -10.0, 10.0)

        bkgB = ROOT.RooBkg("bkgB", "bkgB", deltaM, threshold, bg0B, bg1B)
        bkgDelta = ROOT.RooBkg("bkgDelta", "bkgDelta", deltaM, dm0, bg0B, bg1B)


    elif bkg_option == 'dstar_bkg1':
        bg0B = ROOT.RooRealVar("bg0B", "bg0B", 0.0, -10.0, 10.0)

        bkgB = ROOT.RooGenericPdf ( "bkgB", "bkgB", "sqrt((@0)/139.57 -1)*exp(@1*(@0)/139.57)", ROOT.RooArgList( deltaM, bg0B) )
        bkgDelta = ROOT.RooGenericPdf ( "bkgDelta", "bkgDelta", "sqrt((@0)/@2 -1)*exp(@1*(@0)/@2)", ROOT.RooArgList( deltaM, bg0B, dm0))
    
    elif bkg_option == 'dstar_bkg2':
        bg0B = ROOT.RooRealVar("bg0B", "bg0B", 0.0, -10.0, 10.0) 
        bg1B = ROOT.RooRealVar("bg1B", "bg1B", 0.0, -10.0, 10.0)

        bkgB = ROOT.RooGenericPdf ( "bkgB", "bkgB", "TMath::Power(@0-139.57,@1)*exp(@2*(@0-139.57))",ROOT.RooArgList(deltaM, bg0B,bg1B))
        bkgDelta = ROOT.RooGenericPdf ( "bkgDelta", "bkgDelta", "TMath::Power(@0-@1,@2)*exp(@3*(@0-@1))", ROOT.RooArgList( deltaM,dm0,bg0B,bg1B))

    elif bkg_option == 'dstar_bkg3':
        bg0B = ROOT.RooRealVar("bg0B", "bg0B", 0.0, -10.0, 10.0) 
        bg1B = ROOT.RooRealVar("bg1B", "bg1B", 0.0, -10.0, 10.0)
        bg2B = ROOT.RooRealVar("bg2B", "bg2B", 0.0, -10.0, 10.0)

        bkgB = ROOT.RooDstD0BG("bkgB","bkgB", deltaM, threshold, bg0B, bg1B, bg2B)
        bkgDelta = ROOT.RooDstD0BG("bkgDelta","bkgDelta", deltaM, dm0, bg0B, bg1B, bg2B)


    ##########################################################################################
    ##########################################################################################
    ### TOTAL 2D MODEL 
    ##########################################################################################
    ##########################################################################################

    signal = ROOT.RooProdPdf ( "signal", "",        ROOT.RooArgList( signalD, signalDelta ) )
    nrD    = ROOT.RooProdPdf ( "nrD", "",           ROOT.RooArgList( signalB, bkgB     ) )
    combo  = ROOT.RooProdPdf ( "combinatorial", "", ROOT.RooArgList( bkg, bkgDelta   ) )

    nTot = entries

    nSig = ROOT.RooRealVar ( "nSig", "nSig", 0.7 * nTot, 0, nTot)

    nNR  = ROOT.RooRealVar ( "nNR" , "nNR" , 0.18 * nTot, 0, nTot) #0.2
    nCmb = ROOT.RooRealVar ( "nCmb", "nCmb", 0.1 * nTot, 0, nTot)


    bkg_add_D = ROOT.RooAddPdf( "bkg_add_D", "bkg_add_D", ROOT.RooArgList(signalB, bkg))
    bkg_add_deltaM = ROOT.RooAddPdf( "bkg_add_deltaM", "bkg_add_deltaM", ROOT.RooArgList(bkgB, bkgDelta))

    bkg_total = ROOT.RooProdPdf( "bkg_total", "bkg_total", ROOT.RooArgList(bkg_add_D, bkg_add_deltaM))

 

    #if splot:
    ### Runs if the efficiency calculation is required by Dstar_Fit, must be different from usual fit as nBkg cannot be a RooFormularVar for the sPlot function to run

        #bkg_total = ROOT.RooAddPdf( "bkg_total", "bkg_total", ROOT.RooArgList(nrD, combo), ROOT.RooArgList(nNR, nCmb) )
    #bkg_total = ROOT.RooAddPdf( "bkg_total", "bkg_total", ROOT.RooArgList(nrD, combo))
    nBkg = ROOT.RooRealVar( "nBkg", "nBkg", 0.3 * nTot, 0, nTot)    
            
    model = ROOT.RooAddPdf ( "model", "model", ROOT.RooArgList(signal, bkg_total), ROOT.RooArgList(nSig, nBkg) )

    #else:
        #model = ROOT.RooAddPdf ( "model", "model", ROOT.RooArgList(signal, nrD,  combo) , ROOT.RooArgList(nSig, nNR, nCmb) )
        

    work_space = ROOT.RooWorkspace("w")
    getattr(work_space,'import')(model)

    return work_space



            






def print_fit_result(fit_result):
    print("EDM: {0}, minNLL: {1}, status: {2}, covQual: {3}".format(fit_result.edm(),fit_result.minNll(),fit_result.status(),fit_result.covQual()))
    return fit_result

def chi2Fit(model, arglist, data):
    
    r = model.chi2FitTo(data,arglist)
    
    count = 0
    while(not (r.covQual() == 3 and r.status() == 0)):
        if count >= 20:
            break
        print("\n")
        print("edm: {0}".format(r.edm()))
        print("Attempting new fit: ")
        r = model.chi2FitTo(data,arglist)
        print("\n")
        print("\n")
        count += 1

    return r,model

def fitNLL(model, arglist, data):
    r = model.fitTo(data,arglist)
    
    while(not (r.covQual() == 3 and r.status() == 0)):# and r.edm() < 0.001)):
        print("\n")
        print("edm: {0}".format(r.edm()))
        print("Attempting new fit: ")
        r = model.fitTo(data,arglist)
        print("\n")
        print("\n")

    return r,model


def plotFit(model, var, data, NBins, minimum, maximum, name, save_file, leg_coords, pt_coords=[], varlist=[], load_plot=False, log=True):
    
    cdata = ROOT.TCanvas("cdata","cdata", 0, 0, 800, 800)    # no pull: 800, 600  
    
    frame = var.frame()    
    frame_pull = var.frame()
    dh = ROOT.TH1D()
        
    data.plotOn(frame,ROOT.RooFit.LineColor(ROOT.kBlack),ROOT.RooFit.Name("data"))
    model.plotOn(frame,ROOT.RooFit.Name("model"), ROOT.RooFit.LineColor(ROOT.kAzure))            
    hresid = frame.pullHist()
    
    comp = "signalD"
    if var.GetName() == "deltaM":
        #comp = 'signalDelta'
        comp = "cruijffEDelta"

        #model.plotOn(frame, ROOT.RooFit.Name("gaussian_comp"), ROOT.RooFit.Components("gausDelta"), ROOT.RooFit.LineColor(ROOT.kGreen))
        
    comp2 = "signalB"
    if var.GetName() == "deltaM":
        comp2 = "bkgB"

    comp3 = "bkg"
    if var.GetName() == "deltaM":
        comp3 = "bkgDelta"

    comp4 = "bkgrDs"
    if var.GetName() == "deltaM":
        ##comp4 = "signalDelta2"
        comp4 = "cruijffEDelta2"

    model.plotOn(frame,ROOT.RooFit.Name("Signal"),ROOT.RooFit.Components(comp),ROOT.RooFit.LineStyle(ROOT.kDashed),ROOT.RooFit.LineColor(ROOT.kRed))
    model.plotOn(frame,ROOT.RooFit.Name("Random"),ROOT.RooFit.Components(comp2),ROOT.RooFit.LineColor(ROOT.kOrange+2))
    model.plotOn(frame,ROOT.RooFit.Name("Comb"),ROOT.RooFit.Components(comp3),ROOT.RooFit.LineColor(ROOT.kSpring-1))
    model.plotOn(frame,ROOT.RooFit.Name("Multi"),ROOT.RooFit.Components(comp4),ROOT.RooFit.LineColor(ROOT.kViolet+8))
    #model.plotOn(frame,ROOT.RooFit.Name("other"),ROOT.RooFit.Components('gaus_bkg'),ROOT.RooFit.LineColor(ROOT.kViolet))

    data.plotOn(frame,ROOT.RooFit.LineColor(ROOT.kBlack),ROOT.RooFit.Name("data"))


    leg_entries = {}
    leg_entries["1"] = [frame.findObject("data"),"Data", "pl"]
    leg_entries["2"] = [frame.findObject("model"),"Model","l"]
    leg_entries["3"] = [frame.findObject("Signal"),"Signal","l"]
    leg_entries["4"] = [frame.findObject("Random"),"Random #pi^{}","l"]
    leg_entries["5"] = [frame.findObject("Comb"),"Comb","l"]
    leg_entries["6"] = [frame.findObject("Multi"),"Multibody","l"] #Multibody D^{*}

    hresid.SetFillColor(ROOT.kAzure)
    hresid.SetLineColor(ROOT.kBlack)
    hresid.SetLineWidth(3)
    hresid.SetMarkerColor(ROOT.kBlue)

    frame_pull.addPlotable(hresid,"B X0")
 
    pad2 = ROOT.TPad("pad2","",0.005,0.01,0.995,0.25)
    pad1 = ROOT.TPad("pad1","",0.005,0.25,0.995,0.995)    #for 1 pad only: 0.005,0.01,0.995,0.995
    pad1.Draw()
    pad2.Draw("sames")
    pad1.cd()
    if var.GetName() == "DM":
        frame.SetMinimum(0.1)
        leg_coords = [0.70, 0.24, 0.85, 0.47]

    if var.GetName() == "deltaM":
        frame.SetMinimum(0.1)

    if log:
        pad1.SetLogy()
    


    pt_coords = [[0.22, 0.80, 0.35, 0.9]]

    frame.Draw("a2")
    if pt_coords:
        if len(pt_coords) > 1:
            pts = HistoUtils.createPaveText(pt_coords,varlist,True)
            for pt in pts:
                pt.SetTextSize(0.06)
                pt.Draw("sames")
        else:
            pt = HistoUtils.createPaveText(pt_coords[0],varlist)
            pt.SetTextSize(0.038)
            pt.Draw("sames")
        
    if leg_coords and var.GetName() == "deltaM":
        leg = HistoUtils.createLegend(leg_coords,leg_entries)
        leg.SetTextSize(0.045)
        leg.Draw("sames")
    
    frame.GetXaxis().SetTitle(name)
    binning_range = (maximum - minimum) / NBins

    #title = "Candidates per {0} MeV/c^{{2}}"
    title = "Candidates"

    frame.GetYaxis().SetTitle(title.format(round(binning_range,2)))
    frame.GetYaxis().SetTitleOffset(1.1) #1.5
    frame.GetXaxis().SetTitleOffset(1.3)
    frame.GetXaxis().SetLabelSize(0.06)
    frame.GetXaxis().SetTitleSize(0.06)
    frame.GetYaxis().SetTitleSize(0.06)
    frame.GetYaxis().SetLabelSize(0.045)
    
    pad2.cd()
    pad2.SetGridy()
   
    #frame_pull.GetYaxis().SetTitle("#Delta/#sigma")
    frame_pull.GetYaxis().SetTitle("Pull")
    frame_pull.Draw("a2")
       
    frame_pull.SetMaximum(7.)
    frame_pull.SetMinimum(-7.)

    frame_pull.GetYaxis().SetLabelFont(132)
    frame_pull.GetYaxis().SetTitleFont(132)
    frame_pull.GetYaxis().SetTitleSize(0.15)
    frame_pull.GetYaxis().SetTitleOffset(0.51)
    frame_pull.GetXaxis().SetLabelFont(132)
    frame_pull.GetYaxis().SetLabelSize(0.15)
    frame_pull.GetXaxis().SetLabelSize(0)
    frame_pull.GetXaxis().SetTitleSize(0)

    line2 = ROOT.TLine(minimum,3,maximum,3)
    line2.SetLineColorAlpha(ROOT.kRed,0.6)
    line2.SetLineStyle(1)
    line2.Draw()
    line3 = ROOT.TLine(minimum,-3,maximum,-3)
    line3.SetLineColorAlpha(ROOT.kRed,0.6)
    line3.SetLineStyle(1)
    line3.Draw("sames")

    cdata.Update()

    if save_file != '':
        cdata.SaveAs(save_file)
    if load_plot:
        subprocess.call("eog " + save_file + " &",shell=True)



def plotFit_noPulls(model, var, data, NBins, minimum, maximum, name, save_file, leg_coords, pt_coords=[], varlist=[], load_plot=False, log=True):
    
    #cdata = ROOT.TCanvas("cdata","cdata", 0, 0, 800, 800)    # no pull: 800, 600
    #cdata = ROOT.TCanvas("cdata","cdata")
    cdata = ROOT.TCanvas("cdata","cdata", 0, 0, 800, 610)

    
    frame = var.frame()    
    dh = ROOT.TH1D()
        
    data.plotOn(frame,ROOT.RooFit.LineColor(ROOT.kBlack),ROOT.RooFit.Name("data"))
    model.plotOn(frame,ROOT.RooFit.Name("model"), ROOT.RooFit.LineColor(ROOT.kAzure))            

    
    comp = "signalD"
    if var.GetName() == "deltaM":
        #comp = 'signalDelta'
        comp = "cruijffEDelta"

        #model.plotOn(frame, ROOT.RooFit.Name("gaussian_comp"), ROOT.RooFit.Components("gausDelta"), ROOT.RooFit.LineColor(ROOT.kGreen))
        
    comp2 = "signalB"
    if var.GetName() == "deltaM":
        comp2 = "bkgB"

    comp3 = "bkg"
    if var.GetName() == "deltaM":
        comp3 = "bkgDelta"

    #comp4 = "bkgrDs"
    #if var.GetName() == "deltaM":
        ##comp4 = "signalDelta2"
        #comp4 = "cruijffEDelta2"

    model.plotOn(frame,ROOT.RooFit.Name("Signal"),ROOT.RooFit.Components(comp),ROOT.RooFit.LineStyle(ROOT.kDashed),ROOT.RooFit.LineColor(ROOT.kRed))
    #model.plotOn(frame,ROOT.RooFit.Name("Random"),ROOT.RooFit.Components(comp2),ROOT.RooFit.LineColor(ROOT.kOrange+2))
    model.plotOn(frame,ROOT.RooFit.Name("Background"),ROOT.RooFit.Components(comp2),ROOT.RooFit.LineStyle(ROOT.kDashed),ROOT.RooFit.LineColor(ROOT.kSpring-1))
    #model.plotOn(frame,ROOT.RooFit.Name("Multi"),ROOT.RooFit.Components(comp4),ROOT.RooFit.LineColor(ROOT.kViolet+8))
    #model.plotOn(frame,ROOT.RooFit.Name("other"),ROOT.RooFit.Components('gaus_bkg'),ROOT.RooFit.LineColor(ROOT.kViolet))

    model.plotOn(frame,ROOT.RooFit.Name("model"), ROOT.RooFit.LineColor(ROOT.kAzure)) 
    #data.plotOn(frame,ROOT.RooFit.LineColor(ROOT.kBlack),ROOT.RooFit.Name("data"))


    leg_entries = {}
    leg_entries["1"] = [frame.findObject("data"),"Data", "pl"]
    leg_entries["2"] = [frame.findObject("model"),"Model","l"]
    leg_entries["3"] = [frame.findObject("Signal"),"Signal","l"]
    leg_entries["4"] = [frame.findObject("Background"),"Background","l"]
    #leg_entries["6"] = [frame.findObject("Multi"),"Multibody","l"] #Multibody D^{*}

    #pad1 = ROOT.TPad("pad1","",0.005,0.25,0.995,0.995)    #for 1 pad only: 0.005,0.01,0.995,0.995
    #pad1 = ROOT.TPad("pad1","",0.005,0.01,0.995,0.995)

    pad1 = ROOT.TPad("pad1","",0.0001,0.01,0.995,0.995)

    pad1.Draw()
    pad1.cd()
    if var.GetName() == "DM":
        frame.SetMinimum(0.1)
        leg_coords = [0.70, 0.24, 0.85, 0.47]

    if var.GetName() == "deltaM":
        frame.SetMinimum(0.1)

    if log:
        pad1.SetLogy()
    


    pt_coords = [[0.22, 0.80, 0.35, 0.9]]

    frame.Draw("a2")
    if pt_coords:
        if len(pt_coords) > 1:
            pts = HistoUtils.createPaveText(pt_coords,varlist,True)
            for pt in pts:
                pt.SetTextSize(0.06)
                pt.Draw("sames")
        else:
            pt = HistoUtils.createPaveText(pt_coords[0],varlist)
            pt.SetTextSize(0.038)
            pt.Draw("sames")
        
    if leg_coords and var.GetName() == "deltaM":
        leg = HistoUtils.createLegend(leg_coords,leg_entries)
        leg.SetTextSize(0.045)
        leg.Draw("sames")
    
    frame.GetXaxis().SetTitle(name)
    binning_range = (maximum - minimum) / NBins

    #title = "Candidates per {0} MeV/c^{{2}}"
    title = "Candidates"

    frame.GetYaxis().SetTitle(title.format(round(binning_range,2)))
    frame.GetYaxis().SetTitleOffset(1.1) #1.5
    frame.GetXaxis().SetTitleOffset(1.3)
    frame.GetXaxis().SetLabelSize(0.06)
    frame.GetXaxis().SetTitleSize(0.06)
    frame.GetYaxis().SetTitleSize(0.06)
    frame.GetYaxis().SetLabelSize(0.045)
    
    cdata.Update()
    
    if save_file != '':
        cdata.SaveAs(save_file)
    if load_plot:
        subprocess.call("eog " + save_file + " &",shell=True)