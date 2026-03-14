import ROOT
from ROOT import gSystem
import myHistoUtils as HistoUtils

gSystem.Load('libRooFit')

def Dstar_run3(entries, DM, deltaM, splot, nSig_value, nNR_value, nCmb_value):

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

    # parameters for D_M
    alpha_L = ROOT.RooRealVar("alpha_L","alpha_L", 0.1, -10.0, 10.0)
    alpha_R = ROOT.RooRealVar("alpha_R","alpha_R", 0.1, -10.0, 10.0)
    beta_C = ROOT.RooRealVar("beta_C","beta_C", 0.00003)
    frac = ROOT.RooRealVar("frac","frac", 0.8)
    mean = ROOT.RooRealVar("mean", "mean", 1866., 1800., 1930.)
    sigma = ROOT.RooRealVar("sigma", "sigma", 10., 0.00001, 200.)

    # cruijffE = ROOT.RooCruijffExtended("cruijffE", "cruijffE", DM, mean, sigma, alpha_L, alpha_R, beta_C)
    gaus = ROOT.RooGaussian("gaus", "gaus", DM, mean, sigma)

    # signalD = ROOT.RooAddPdf( "signalD", "signalD", ROOT.RooArgList(cruijffE,gaus), ROOT.RooArgList(frac))
    # signalB = ROOT.RooAddPdf( "signalB", "signalB", ROOT.RooArgList(cruijffE,gaus), ROOT.RooArgList(frac))
    signalD = ROOT.RooAddPdf( "signalD", "signalD", ROOT.RooArgList(gaus), ROOT.RooArgList(frac))
    signalB = ROOT.RooAddPdf( "signalB", "signalB", ROOT.RooArgList(gaus), ROOT.RooArgList(frac))

    # parameters for delta_M
    alpha_LDelta = ROOT.RooRealVar("alpha_LDelta","alpha_LDelta", 0.15, -1.0, 1.0)
    alpha_RDelta = ROOT.RooRealVar("alpha_RDelta","alpha_RDelta", 0.15, -1.0, 1.0)
    beta_CDelta = ROOT.RooRealVar("beta_CDelta","beta_CDelta", 0.003)
    meanDelta = ROOT.RooRealVar("meanDelta", "meanDelta", 144., mean_low, mean_high)
    sigmaDelta = ROOT.RooRealVar("sigmaDelta", "sigmaDelta", 0.6, 0.01, 6.)  
           
    # cruijffEDelta = ROOT.RooCruijffExtended("cruijffEDelta", "cruijffEDelta", deltaM, meanDelta, sigmaDelta, alpha_LDelta, alpha_RDelta, beta_CDelta)
    # cruijffEDelta2 = ROOT.RooCruijffExtended("cruijffEDelta2", "cruijffEDelta2", deltaM, meanDelta, sigmaDelta, alpha_LDelta, alpha_RDelta, beta_CDelta)

    # signalDelta = cruijffEDelta
    # signalDelta2 = cruijffEDelta2
    signalDelta = ROOT.RooGaussian("signalDelta", "signalDelta", deltaM, meanDelta, sigmaDelta)
    signalDelta2 = ROOT.RooGaussian("signalDelta2", "signalDelta2", deltaM, meanDelta, sigmaDelta)

    # parameters for multibody background
    b0 = ROOT.RooRealVar("b0", "b0", -0.4)
    b1 = ROOT.RooRealVar("b1", "b1", 0.1)
    factor = 0.02
    bkgrDsfirst = ROOT.RooChebychev("bkgrDsfirst", "bkgrDsfirst", DM, ROOT.RooArgList(b0, b1)) 
    mean_bkg = ROOT.RooRealVar("mean_bkg", "mean_bkg", 1900)
    sigma_bkg = ROOT.RooRealVar("sigma_bkg", "sigma_bkg", 10)
    gauss_bkg = ROOT.RooGaussian("gauss_bkg", "gauss_bkg", DM, mean_bkg, sigma_bkg)

    cf_bkg = ROOT.RooRealVar("cf_bkg", "cf_bkg", 0.9)
    
    bkgrDs = ROOT.RooAddPdf( "bkgrDs", "bkgrDs", ROOT.RooArgList(bkgrDsfirst, gauss_bkg), ROOT.RooArgList(cf_bkg))

    # randompi + combinatorial background
    bg0 = ROOT.RooRealVar("bg0", "bg0", -0.1, -10.0, 0.0)
    bkg = ROOT.RooChebychev("bkg", "bkg", DM, ROOT.RooArgList(bg0))

    threshold = ROOT.RooRealVar("threshold", "threshold", 139.57)
    dm0 = ROOT.RooRealVar("dm0", "dm0", 139.57, 139., 142.)
    bg0B = ROOT.RooRealVar("bg0B", "bg0B", 0.0, -10.0, 10.0) 
    bg1B = ROOT.RooRealVar("bg1B", "bg1B", 0.0, -10.0, 10.0)
    bg2B = ROOT.RooRealVar("bg2B", "bg2B", 0.0, -10.0, 10.0)

    bkgB = ROOT.RooDstD0BG("bkgB","bkgB", deltaM, threshold, bg0B, bg1B, bg2B)
    bkgDelta = ROOT.RooDstD0BG("bkgDelta","bkgDelta", deltaM, dm0, bg0B, bg1B, bg2B)

    # sum
    signal = ROOT.RooProdPdf ( "signal", "",        ROOT.RooArgList( signalD, signalDelta ) )
    nrD    = ROOT.RooProdPdf ( "nrD", "",           ROOT.RooArgList( signalB, bkgB     ) )
    combo  = ROOT.RooProdPdf ( "combinatorial", "", ROOT.RooArgList( bkg, bkgDelta   ) )
    mbDs = ROOT.RooProdPdf ( "MultibodyrealDstar", "",ROOT.RooArgList( bkgrDs, signalDelta2) )
    nTot = entries
    nSig = ROOT.RooRealVar ( "nSig", "nSig", 0.7 * nTot, 0, nTot)
    nNR  = ROOT.RooRealVar ( "nNR" , "nNR" , 0.18 * nTot, 0, nTot) #0.2
    nCmb = ROOT.RooRealVar ( "nCmb", "nCmb", 0.1 * nTot, 0, nTot)
    nRDs = ROOT.RooFormulaVar("nRDs", "{}*nSig".format(factor), ROOT.RooArgList(nSig))

    if splot:
    ### Runs if the efficiency calculation is required by Dstar_Fit, must be different from usual fit as nBkg cannot be a RooFormularVar for the sPlot function to run
        # if nSig_value != '':
        nRDs_fixed = ROOT.RooRealVar("nRDs_fixed", "nRDs_fixed", nSig_value*factor)
        nNR_fixed = ROOT.RooRealVar("nNR_fixed", "nNR_fixed", nNR_value)
        nCmb_fixed = ROOT.RooRealVar("nCmb_fixed", "nCmb_fixed", nCmb_value)
        # nBkg_value = nNR_value + nCmb_value + (nSig_value*factor)

        bkg_total = ROOT.RooAddPdf( "bkg_total", "bkg_total", ROOT.RooArgList(nrD, combo, mbDs), ROOT.RooArgList(nNR_fixed, nCmb_fixed, nRDs_fixed) )

        nBkg = ROOT.RooRealVar( "nBkg", "nBkg", 0.3 * nTot, 0, nTot)

        # else:
        #     bkg_total = ROOT.RooAddPdf( "bkg_total", "bkg_total", ROOT.RooArgList(nrD, combo, mbDs), ROOT.RooArgList(nNR, nCmb, nRDs) )
        #     nBkg = ROOT.RooFormulaVar( "nBkg", "nNR + nCmb + nRDs", ROOT.RooArgList(nNR, nCmb, nRDs) )
            
        model = ROOT.RooAddPdf ( "model", "model", ROOT.RooArgList(signal, bkg_total), ROOT.RooArgList(nSig, nBkg) )

    else:
        model = ROOT.RooAddPdf ( "model", "model", ROOT.RooArgList(signal, nrD,  combo, mbDs) , ROOT.RooArgList(nSig, nNR, nCmb, nRDs) )

    work_space = ROOT.RooWorkspace("w")
    getattr(work_space,'import')(model)

    return work_space


def plotFit(model, var, data, NBins, minimum, maximum, name, save_file, leg_coords, pt_coords=[], varlist=[], load_plot=False, log=True):
    
    cdata = ROOT.TCanvas("cdata","cdata", 0, 0, 800, 800)    # no pull: 800, 600  
    
    frame = var.frame()    
    frame_pull = var.frame()
    dh = ROOT.TH1D()
        
    data.plotOn(frame,ROOT.RooFit.LineColor(ROOT.kBlack),ROOT.RooFit.Name("data"))
    model.plotOn(frame,ROOT.RooFit.Name("model"), ROOT.RooFit.LineColor(ROOT.kAzure))            
    hresid = frame.pullHist()
    
    is_deltam = var.GetName() == "deltaM"
    comp = "signalD" if is_deltam else "signal"
    comp2 = "signalB" if is_deltam else "bkgB"
    comp3 = "bkg" if is_deltam else "bkgDelta"
    comp4 = "bkgrDs" if is_deltam else "cruijffEDelta2"

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
    # if load_plot:
    #     subprocess.call("eog " + save_file + " &",shell=True)