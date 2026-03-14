###################################################                         
# Main script for running fits to data and        # 
# producing plots of data, background, and fits.  #
# Also allows for the finidng of efficiencies via #
# an sWeight method or via  a fit and count       #
# method.                                         #
# Can be run in unbinned data, binned data or     #
# both.                                           #
# Returns plots and text files of the parameters  #
# found by the fit.                               #
###################################################

##### Script convention:
##### Mass of D^0 denoted by DM
##### Mass of D* - D^0 denoted by deltaM

##### Arguments required:
##### Decay Mode [DSt_PiP, DSt_PiM, DSt_KM, DSt_KP]
##### Year [2015, 2016, 2017, 2018, All]
##### Polarity [MagUp, MagDown, All]
##### Type of calculation [Projections, Efficiency]
##### Fit model [Run2, Run3]
##### Should the fit be run in kinemtatic bins? [y, n, both]
##### Cut (float)
##### number of files (integer, Note: 0 files will correspond to all files)

##### Example use:
##### Dstar_Fit.py DSt_PiP 2016 All Efficiency CruijffE JohnsonSUGauss n 0.0 10

import ROOT

import time

from ROOT import gROOT
from ROOT import gSystem
from ROOT import gStyle
from ROOT import gPad
gSystem.Load('libRooFit')

from lhcbStyle import setLHCbStyle
from array import array

import FitUtils
import HistoUtils
import subprocess
#ßimport ProcessingUtils
import pandas as pd
import root_pandas

import numpy as np
from tqdm import tqdm

import itertools

import argparse 
parser = argparse.ArgumentParser(description='fit')
parser.add_argument("decay_mode",type=str,choices=['DSt_PiP', 'DSt_PiM', 'DSt_KM', 'DSt_KP'])
parser.add_argument("year",type=str,choices=['2015', '2016', '2017', '2018', 'All', '2022'])
parser.add_argument("polarity",type=str,choices=['MagUp', 'MagDown', 'All'])
parser.add_argument("calc_type",type=str,choices=['Projections', 'Efficiency'])
parser.add_argument("fit_model",type=str,choices=['old', 'new', 'new_3comp'])
parser.add_argument("bkg_option",type=str,choices=['dstar_bkg', 'dstar_bkg1', 'dstar_bkg2', 'dstar_bkg3'])

parser.add_argument("kin_bins",type=str,choices=['y', 'n', 'both'])
parser.add_argument("cut",type=str, choices=['0', '1', '2', '5', '12'])
parser.add_argument("file_no", type = int) 

args = parser.parse_args()
setLHCbStyle()

### python Dstar_Fit.py DSt_PiP 2022 MagUp Projections new dstar_bkg n 5 0
### (python Dstar_Fit.py DSt_PiP 2022 MagUp Projections new_3comp dstar_bkg n 1 0)


cut_value = args.cut
'''
if args.decay_mode=='DSt_PiP' or args.decay_mode=='DSt_PiM':
    cut_type = '<'
else:
    cut_type = '>'
'''
cut_type = '>'



if args.decay_mode=='DSt_PiP' or args.decay_mode=='DSt_PiM':
    probe = 'pi'
elif args.decay_mode=='DSt_KP' or args.decay_mode=='DSt_KM':
    probe = 'K'

if args.decay_mode=='DSt_PiP' or args.decay_mode=='DSt_KP':
    cut_charge = 'trackcharge>0'
    qp = 'plus'
elif args.decay_mode=='DSt_PiM' or args.decay_mode=='DSt_KM':
    cut_charge = 'trackcharge<0'
    qp = 'minus'


#cut = "probe_PIDK{0}{1}".format(cut_type, cut_value)
#cut = "K_PID_K{0}{1}".format(cut_type, cut_value)
#cut = "pi_PID_K{0}{1}".format(cut_type, cut_value)

#cut = "pidk{0}{1}".format(cut_type, cut_value)
cut = "probe_PIDK{0}{1}".format(cut_type, cut_value)

NBins = 65

#### Designed for use in the 7 bin binning scheme (see README.txt)
if args.kin_bins == 'y' or args.kin_bins == 'both':
    p_bins_all = [2, 19, 39, 200]
    eta_bins_1 = [1.6, 2.9, 5.0]
    eta_bins_2 = [1.6, 2.9, 3.4, 5.0]
    eta_bins_3 = [1.6, 3.4, 5.0]

#### Allows all files to be examined if the argument for file_no is 0 
if args.file_no != 0:
    file_no = args.file_no
else:
    file_no = "all"


gROOT.SetBatch(ROOT.kTRUE)

eff_sweight = [] 
eff_fitncount = []


class DataPoint:

    def __init__(self, DM, deltaM, probe_P, probe_ETA, probe_PIDK, probe_PT, ntr):

        self.DM = DM
        self.deltaM = deltaM
        self.probe_P = probe_P
        self.probe_ETA = probe_ETA
        self.probe_PIDK = probe_PIDK
        self.probe_PT = probe_PT
        self.ntr = ntr

        
    def __str__(self):
        return "DM: {0} deltaM: {1} probe_P: {2} probe_ETA: {3} probe_PIDK: {4} probe_PT: {5} ntr: {6}".format(self.DM, self.deltaM, self.probe_P, self.probe_ETA, self.probe_PIDK, self.probe_PT, self.ntr)



def load_file() -> tuple:
    """
    Load the file and return the tuple of list of DataPoints based on eta and p bins
    """
    path_files = f'/data/lhcb/users/suljik/rta_pid_calib/data_2016/merged_20files'
    
    files = [f'{path_files}/data_2016_MagUp_20files.root']

    tree = "DSt_PiPTuple/DecayTree"

    ### Currently creates different datasets for each of the different bins regardless of whether a binned fit is needed, this could be improved
    data0 = data1 = data2 = data3 = data4 = data5 = data6 = data7 = []    

    columns = ['Dst_M','Dz_M','probe_P','probe_ETA', 'probe_PIDK', 'probe_PT', 'nTracks']

    if file_no != "all":
        files = files[0:args.file_no]


    for temp_file in files:
        print("Reading file {0}/{1}".format(files.index(temp_file)+1, len(files)))
        for df in root_pandas.read_root(temp_file, columns=columns, key=tree, chunksize=100000):
            df['p'] = df['probe_P']
            df = df[df['p'] < 200*1000]  # less than 200 GeV
            df = df[df['p'] > 2*1000]  # greater than 2 GeV

            df['deltaM'] = df['Dst_M'] - df['Dz_M']
            df['dm'] = df['deltaM']
            df['md'] = df['Dz_M']
            df['DM'] = df['Dz_M']
            df = df[(df['deltaM'] > 139.57) & (df['deltaM'] < 155)]
            df = df[(df['Dz_M'] > 1825) & (df['Dz_M'] < 1910)]
            
            df['pt'] = df['probe_PT']
            df['eta'] = df['probe_ETA']
            df['pidk'] = df['probe_PIDK']
            df['ntr'] = df['nTracks']

            df0 = df
            
            df1 = df.query('eta>1.6 and eta<2.9 and (p/1000.)>2.0 and (p/1000.)<19.0')
            df2 = df.query('eta>2.9 and eta<5.0 and (p/1000.)>2.0 and (p/1000.)<19.0')
            df3 = df.query('eta>1.6 and eta<2.9 and (p/1000.)>19.0 and (p/1000.)<39.0')
            df4 = df.query('eta>2.9 and eta<3.4 and (p/1000.)>19.0 and (p/1000.)<39.0')
            df5 = df.query('eta>3.4 and eta<5.0 and (p/1000.)>19.0 and (p/1000.)<39.0')
            df6 = df.query('eta>1.6 and eta<3.4 and (p/1000.)>39.0 and (p/1000.)<200.0')
            df7 = df.query('eta>3.4 and eta<5.0 and (p/1000.)>39.0 and (p/1000.)<200.0')
            
                
            for temp_dm,temp_deltam,temp_probe_P,temp_probe_ETA,temp_probe_PIDK,temp_probe_PT,temp_ntr in zip(list(df0['md'].values.flatten()),list(df0['deltaM'].values.flatten()),list(df0['p'].values.flatten()),list(df0['eta'].values.flatten()),list(df0['pidk'].values.flatten()),list(df0['pt'].values.flatten()),list(df0['ntr'].values.flatten())):
                temp_point = DataPoint(temp_dm,temp_deltam,temp_probe_P/1000.,temp_probe_ETA,temp_probe_PIDK,temp_probe_PT/1000.,temp_ntr)
                data0.append(temp_point)
            for temp_dm,temp_deltam,temp_probe_P,temp_probe_ETA,temp_probe_PIDK,temp_probe_PT,temp_ntr in zip(list(df1['md'].values.flatten()),list(df1['deltaM'].values.flatten()),list(df1['p'].values.flatten()),list(df1['eta'].values.flatten()),list(df1['pidk'].values.flatten()),list(df0['pt'].values.flatten()),list(df0['ntr'].values.flatten())):
                temp_point = DataPoint(temp_dm,temp_deltam,temp_probe_P/1000.,temp_probe_ETA,temp_probe_PIDK,temp_probe_PT/1000.,temp_ntr)
                data1.append(temp_point)
            for temp_dm,temp_deltam,temp_probe_P,temp_probe_ETA,temp_probe_PIDK,temp_probe_PT,temp_ntr in zip(list(df2['md'].values.flatten()),list(df2['deltaM'].values.flatten()),list(df2['p'].values.flatten()),list(df2['eta'].values.flatten()),list(df2['pidk'].values.flatten()),list(df0['pt'].values.flatten()),list(df0['ntr'].values.flatten())):
                temp_point = DataPoint(temp_dm,temp_deltam,temp_probe_P/1000.,temp_probe_ETA,temp_probe_PIDK,temp_probe_PT/1000.,temp_ntr)
                data2.append(temp_point)
            for temp_dm,temp_deltam,temp_probe_P,temp_probe_ETA,temp_probe_PIDK,temp_probe_PT,temp_ntr in zip(list(df3['md'].values.flatten()),list(df3['deltaM'].values.flatten()),list(df3['p'].values.flatten()),list(df3['eta'].values.flatten()),list(df3['pidk'].values.flatten()),list(df0['pt'].values.flatten()),list(df0['ntr'].values.flatten())):
                temp_point = DataPoint(temp_dm,temp_deltam,temp_probe_P/1000.,temp_probe_ETA,temp_probe_PIDK,temp_probe_PT/1000.,temp_ntr)
                data3.append(temp_point)
            for temp_dm,temp_deltam,temp_probe_P,temp_probe_ETA,temp_probe_PIDK,temp_probe_PT,temp_ntr in zip(list(df4['md'].values.flatten()),list(df4['deltaM'].values.flatten()),list(df4['p'].values.flatten()),list(df4['eta'].values.flatten()),list(df4['pidk'].values.flatten()),list(df0['pt'].values.flatten()),list(df0['ntr'].values.flatten())):
                temp_point = DataPoint(temp_dm,temp_deltam,temp_probe_P/1000.,temp_probe_ETA,temp_probe_PIDK,temp_probe_PT/1000.,temp_ntr)
                data4.append(temp_point)
            for temp_dm,temp_deltam,temp_probe_P,temp_probe_ETA,temp_probe_PIDK,temp_probe_PT,temp_ntr in zip(list(df5['md'].values.flatten()),list(df5['deltaM'].values.flatten()),list(df5['p'].values.flatten()),list(df5['eta'].values.flatten()),list(df5['pidk'].values.flatten()),list(df0['pt'].values.flatten()),list(df0['ntr'].values.flatten())):
                temp_point = DataPoint(temp_dm,temp_deltam,temp_probe_P/1000.,temp_probe_ETA,temp_probe_PIDK,temp_probe_PT/1000.,temp_ntr)
                data5.append(temp_point)
            for temp_dm,temp_deltam,temp_probe_P,temp_probe_ETA,temp_probe_PIDK,temp_probe_PT,temp_ntr in zip(list(df6['md'].values.flatten()),list(df6['deltaM'].values.flatten()),list(df6['p'].values.flatten()),list(df6['eta'].values.flatten()),list(df6['pidk'].values.flatten()),list(df0['pt'].values.flatten()),list(df0['ntr'].values.flatten())):
                temp_point = DataPoint(temp_dm,temp_deltam,temp_probe_P/1000.,temp_probe_ETA,temp_probe_PIDK,temp_probe_PT/1000.,temp_ntr)
                data6.append(temp_point)
            for temp_dm,temp_deltam,temp_probe_P,temp_probe_ETA,temp_probe_PIDK,temp_probe_PT,temp_ntr in zip(list(df7['md'].values.flatten()),list(df7['deltaM'].values.flatten()),list(df7['p'].values.flatten()),list(df7['eta'].values.flatten()),list(df7['pidk'].values.flatten()),list(df0['pt'].values.flatten()),list(df0['ntr'].values.flatten())):
                temp_point = DataPoint(temp_dm,temp_deltam,temp_probe_P/1000.,temp_probe_ETA,temp_probe_PIDK,temp_probe_PT/1000.,temp_ntr)
                data7.append(temp_point)
               
    
    return data0, data1, data2, data3, data4, data5, data6, data7


def create_2d_dataset(data,var1,var2,var1_name,var2_name) -> ROOT.RooDataSet:
    """
    Create a 2D dataset(`var1`, `var2`) from the `data` and return as `ROOT.RooDataSet`
    """

    dataset = ROOT.RooDataSet("data2D","data2D",ROOT.RooArgSet(var1,var2))
    print("Creating 2D dataset..........")
    
    for entry in data:
        var1.setVal(getattr(entry,var1_name))
        var2.setVal(getattr(entry,var2_name))
        dataset.add(ROOT.RooArgSet(var1,var2))
    
    return dataset

def create_splot_dataset(data,var1,var2,var3,var4,var5,var6,var7,var1_name,var2_name,var3_name,var4_name,var5_name,var6_name,var7_name) -> ROOT.RooDataSet:
    """
    Create a 7D dataset(`var1`, `var2`, `var3`, `var4`, `var5`, `var6`, `var7`) from the `data` and return as `ROOT.RooDataSet`
    """
    dataset_splot = ROOT.RooDataSet("data_splot","data_splot",ROOT.RooArgSet(var1,var2,var3,var4,var5,var6,var7))

    for entry in data:
        var1.setVal(getattr(entry,var1_name))
        var2.setVal(getattr(entry,var2_name))
        var3.setVal(getattr(entry,var3_name))
        var4.setVal(getattr(entry,var4_name))
        var5.setVal(getattr(entry,var5_name))
        var6.setVal(getattr(entry,var6_name))
        var7.setVal(getattr(entry,var7_name))
        dataset_splot.add(ROOT.RooArgSet(var1,var2,var3,var4,var5,var6,var7))
    
    return dataset_splot    
    
def create_1d_dataset(data,var,var_name) -> ROOT.RooDataSet:
    """
    Create a 1D dataset(`var`) from the `data` and return as `ROOT.RooDataSet`
    """
    dataset = ROOT.RooDataSet("data{0}".format(var_name),"data{0}".format(var_name),ROOT.RooArgSet(var))
    print("Creating 1D dataset..........")

    for var_temp in data:
        var.setVal(getattr(var_temp,var_name))
        dataset.add(ROOT.RooArgSet(var))

    return dataset


def find_sweights(workspace, data, low_mass_deltam, high_mass_deltam, low_mass_dm, high_mass_dm, nSig_value, nNR_value, nCmb_value, tag='',i='',j=''):
#### Fits to the model when the efficiencies are required, must be done in a different way as nSig and nBkg must be RooRealVars for the sPlot function (see Chapter 7 of logbook)

    DM = ROOT.RooRealVar("DM","DM",low_mass_dm,high_mass_dm)
    deltaM = ROOT.RooRealVar("deltaM","deltaM",low_mass_deltam,high_mass_deltam)

    probe_P = ROOT.RooRealVar("probe_P","probe_P", 0, 400000)
    probe_PT = ROOT.RooRealVar("probe_PT","probe_PT", 0, 40000)
    probe_ETA = ROOT.RooRealVar("probe_ETA","probe_ETA", 0, 6)
    probe_PIDK = ROOT.RooRealVar("probe_PIDK","probe_PIDK", -200, 200)
    ntr = ROOT.RooRealVar("ntr","ntr", 0, 500)

    entries = len(data)
    print("Fitting candidates: ------> {0}".format(entries))

    DM.setBins(NBins)
    deltaM.setBins(NBins)

    DM_dataset = create_1d_dataset(data,DM,'DM')
    deltaM_dataset = create_1d_dataset(data,deltaM,'deltaM')

    twoD_dataset = create_2d_dataset(data,DM,deltaM,'DM','deltaM')

    splot = True

    #w = FitUtils.Dstar_run3(entries, DM, deltaM, args.bkg_option, bin_no, args.decay_mode, splot, nSig_value, nNR_value, nCmb_value)
    if args.fit_model == 'old':
        w = FitUtils.Dstar_4comp(entries, DM, deltaM, a, p_row, splot)
    elif args.fit_model == 'new':
        w = FitUtils.Dstar_run3(entries, DM, deltaM, args.bkg_option, bin_no, args.decay_mode, splot, nSig_value, nNR_value, nCmb_value)
    else:
        w = FitUtils.Dstar_run3_3comp(entries, DM, deltaM, args.bkg_option, bin_no, args.decay_mode, splot)   

    model = w.pdf('model')

    binned = ROOT.RooDataHist("binned","",ROOT.RooArgSet(DM,deltaM),twoD_dataset)

    ll = ROOT.RooLinkedList()
    cmd1 = ROOT.RooFit.Extended(ROOT.kTRUE)
    ll.Add(cmd1)
    cmd2 = ROOT.RooFit.NumCPU(2)
    ll.Add(cmd2)
    cmd3 = ROOT.RooFit.Save()
    ll.Add(cmd3)
    cmd4 = ROOT.RooFit.Minimizer("Minuit","Migrad") 
    ll.Add(cmd4)                                                                
    cmd5 = ROOT.RooFit.PrintLevel(1)
    ll.Add(cmd5)
    
    print("Attempting Fit......")

    params = model.getVariables()
    params.writeToFile("./plots_data_fit/parameters{0}_{1}files_{2}_fit{3}.txt".format(args.year + args.polarity + args.decay_mode+tag+i+j+str(p_row)+str(a),str(file_no), args.bkg_option, args.fit_model))



    temp_eff_sweight = 0.0

    ### Sets parameters constant before sPlot is carried out (will need adjusting if fits other than Extended Cruijff and gaussian or JohnsonSU and gaussian are to be used for deltam. 
   
    if args.fit_model == 'old':
        w.var("mean").setConstant(True)
        w.var("sigma1").setConstant(True)
        w.var("sigma2").setConstant(True)
        w.var("cf").setConstant(True)

        w.var("meanDelta").setConstant(True)
        w.var("sigmaDelta").setConstant(True)
        w.var("sigmaDelta3").setConstant(True)
        w.var("meanDelta2").setConstant(True)
        w.var("sigmaDelta2").setConstant(True)
        w.var("cfDelta").setConstant(True)
        w.var("cfDelta2").setConstant(True)

        w.var("nDelta").setConstant(True)
        w.var("alphaDelta").setConstant(True)

        w.var("bg0").setConstant(True)
        w.var("bg1").setConstant(True)

        w.var("bg0B").setConstant(True)
        w.var("threshold").setConstant(True)


    elif args.fit_model == 'new':
        w.var("mean").setConstant(True)
        w.var("sigma").setConstant(True)

        w.var("alpha_L").setConstant(True)
        w.var("alpha_R").setConstant(True)
        w.var("beta_C").setConstant(True)
        w.var("frac").setConstant(True)

        w.var("meanDelta").setConstant(True)
        w.var("sigmaDelta").setConstant(True)

        w.var("alpha_LDelta").setConstant(True)
        w.var("alpha_RDelta").setConstant(True)
        w.var("beta_CDelta").setConstant(True)

        w.var("b0").setConstant(True)
        w.var("b1").setConstant(True)
        #w.var("mean_bkg").setConstant(True) #if gaussian
        #w.var("sigma_bkg").setConstant(True) #if gaussian
        #w.var("cf_bkg").setConstant(True) #if gaussian 

        w.var("bg0").setConstant(True)
        w.var("threshold").setConstant(True)
        w.var("dm0").setConstant(True)

        w.var("bg0B").setConstant(True)
        w.var("bg1B").setConstant(True) #if not dstar_bkg1
        #w.var("bg2B").setConstant(True) #if bkg_dstar3 
    
    else:
        w.var("mean").setConstant(True)
        w.var("sigma").setConstant(True)

        w.var("alpha_L").setConstant(True)
        w.var("alpha_R").setConstant(True)
        w.var("beta_C").setConstant(True)
        w.var("frac").setConstant(True)

        w.var("meanDelta").setConstant(True)
        w.var("sigmaDelta").setConstant(True)
        #w.var("sigmaDelta2").setConstant(True) #RooFormulaVar
        #w.var("scale_sigmaDelta2").setConstant(True)

        w.var("alpha_LDelta").setConstant(True)
        w.var("alpha_RDelta").setConstant(True)
        w.var("beta_CDelta").setConstant(True)
        #w.var("cfDelta").setConstant(True)

        w.var("bg0").setConstant(True)
        w.var("threshold").setConstant(True)
        w.var("dm0").setConstant(True)

        w.var("bg0B").setConstant(True)
        w.var("bg1B").setConstant(True) #if not dstar_bkg1
        #w.var("bg2B").setConstant(True) #if bkg_dstar3    

    fit_result = model.fitTo(binned, ll)

    while (not(fit_result.status() == 0 and fit_result.covQual() == 3)):
        fit_result = model.fitTo(binned, ll)

    nSig = w.var('nSig')
    nBkg = w.var('nBkg')

    paramscheck = model.getVariables()
    paramscheck.writeToFile("./plots_data_fit/parameters_check_{0}_{1}files_{2}_fit{3}.txt".format(args.year + args.polarity + args.decay_mode+tag+i+j+str(p_row)+str(a),str(file_no), args.bkg_option, args.fit_model))

    ### Uses sPlot method to find the sWeights so that the efficiences can be calculated
    splot_dataset = create_splot_dataset(data, DM, deltaM, probe_P, probe_ETA, probe_PIDK, probe_PT, ntr, 'DM', 'deltaM', 'probe_P', 'probe_ETA', 'probe_PIDK', 'probe_PT', 'ntr')
    sData = ROOT.RooStats.SPlot("sData","sPlot",splot_dataset,model,ROOT.RooArgList(nSig, nBkg))                                 
    print("Check sWeights: ")
    print("Signal Yield: {0}, From sWeights: {1}".format(nSig.getVal(),sData.GetYieldFromSWeight("nSig")))              
    print("Background Yield: {0}, From sWeights: {1}".format(nBkg.getVal(),sData.GetYieldFromSWeight("nBkg")))   
   
    data_splot = ROOT.RooDataSet("sPlot", "sPlot", splot_dataset, splot_dataset.get(), "", 'nSig_sw')
    data_splot_unweighted = ROOT.RooDataSet("sPlot_unweighted", "sPlot_unweighted", splot_dataset, splot_dataset.get(), "", '')

    pre_cut_sweight = data_splot.sumEntries()

    data_splot_cut = ROOT.RooDataSet("sPlot_cut", "sPlot_cut", splot_dataset, splot_dataset.get(), "{}".format(cut), 'nSig_sw')
    post_cut_sweight = data_splot_cut.sumEntries()

    ### Saves the sWeights as a root file so the efficiencies can also be calculated externally
    fout2 = ROOT.TFile.Open('sweights_data_{0}_{1}_{2}.root'.format(probe, qp, str(p_row)+str(a)), 'recreate')
    splot_tree =  ROOT.RooStats.GetAsTTree("splot_data","splot_data",data_splot_unweighted)
    fout2.cd()
    splot_tree.Write()
    fout2.Close()

    if pre_cut_sweight != 0:
        temp_eff_sweight = post_cut_sweight / pre_cut_sweight

    leg_coords = [0.70,0.55,0.85,0.85]
    pt_coords = [[0.23,0.77,0.42,0.9]]

    leg_coords_log = [0.70, 0.25, 0.85, 0.50]
    pt_coords_log = [[0.10, 0.77, 0.30, 0.9]]

    cases = {'DM': ["m(D^{0}) [MeV/c^{2}]",DM,DM_dataset,low_mass_dm,high_mass_dm],'deltaM':["#Deltam [MeV/c^{2}]",deltaM,deltaM_dataset,low_mass_deltam,high_mass_deltam]}

    #model_DM = model.createProjection(ROOT.RooArgSet(deltaM))
    #model_deltaM = model.createProjection(ROOT.RooArgSet(DM))

    for case in cases:
        element = ''
        if args.calc_type == 'Efficiency':
            element = 'Eff'
            if args.kin_bins == 'n':
                save_file = "./plots_data_eff/{0}EfficiencyData_{1}files_fit{2}.pdf".format(args.year+args.polarity+args.decay_mode+case+args.bkg_option+element+tag,str(file_no), args.fit_model)  
                save_file_log = "./plots_data_eff/{0}EfficiencyData_{1}files_fit{2}_log.pdf".format(args.year+args.polarity+args.decay_mode+case+args.bkg_option+element+tag,str(file_no), args.fit_model) 
            else:
                save_file = "./plots_data_eff/{0}EfficiencyData_{1}files_fit{2}.pdf".format(args.year+args.polarity+args.decay_mode+case+args.bkg_option+element+tag+str(p_row)+str(a),str(file_no), args.fit_model)
                save_file_log = "./plots_data_eff/{0}EfficiencyData_{1}files_fit{2}_log.pdf".format(args.year+args.polarity+args.decay_mode+case+args.bkg_option+element+tag+str(p_row)+str(a),str(file_no), args.fit_model)
        else:
            if args.kin_bins == 'n':
                save_file = "./plots_data_fit/{0}ProjectionData_{1}files_fit{2}.pdf".format(args.year+args.polarity+args.decay_mode+case+args.bkg_option+element+tag,str(file_no), args.fit_model)
                save_file_log = "./plots_data_fit/{0}ProjectionData_{1}files_fit{2}_log.pdf".format(args.year+args.polarity+args.decay_mode+case+args.bkg_option+element+tag,str(file_no), args.fit_model)   
            else:
                save_file = "./plots_data_fit/{0}ProjectionData_{1}files_fit{2}.pdf".format(args.year+args.polarity+args.decay_mode+case+args.bkg_option+element+tag+str(p_row)+str(a),str(file_no), args.fit_model)
                save_file_log = "./plots_data_fit/{0}ProjectionData_{1}files_fit{2}_log.pdf".format(args.year+args.polarity+args.decay_mode+case+args.bkg_option+element+tag+str(p_row)+str(a),str(file_no), args.fit_model)   
        #temp_model = ''
        #if case == "DM":
        #    temp_model = model_DM
        #else:
        #    temp_model = model_deltaM
        #Make sure to add ks_type to the title                                                                                                                               
        if args.polarity == "All":
            FitUtils.plotFit(model, cases[case][1], cases[case][2], NBins, cases[case][3], cases[case][4], cases[case][0], save_file, leg_coords, pt_coords, [args.year,args.decay_mode], load_plot=False, log=False)
            FitUtils.plotFit(model, cases[case][1], cases[case][2], NBins, cases[case][3], cases[case][4], cases[case][0], save_file_log, leg_coords, pt_coords, [args.year,args.decay_mode], load_plot=False, log=True)
        else:
            FitUtils.plotFit(model, cases[case][1], cases[case][2], NBins,cases[case][3], cases[case][4], cases[case][0], save_file, leg_coords, pt_coords, ['LHCb Run 3','Preliminary'], load_plot=False, log=False)
            FitUtils.plotFit(model, cases[case][1], cases[case][2], NBins,cases[case][3], cases[case][4], cases[case][0], save_file_log, leg_coords, pt_coords, ['LHCb Run 3','Preliminary'], load_plot=False, log=True)

    return temp_eff_sweight

def perform_fit(data,low_mass_deltam,high_mass_deltam,low_mass_dm,high_mass_dm,tag='',i='',j=''):
#### Fits to the model when the efficieinces are not required, in this case nBkg can be a RooFormulaVar

    DM = ROOT.RooRealVar("DM","DM",low_mass_dm,high_mass_dm)
    deltaM = ROOT.RooRealVar("deltaM","deltaM",low_mass_deltam,high_mass_deltam)
    #probe_PIDK = ROOT.RooRealVar("probe_PIDK","probe_PIDK", -100, 100)

    entries = len(data)
    print("Fitting candidates: ------> {0}".format(entries))

    splot = False

    if args.fit_model == 'old':
        w = FitUtils.Dstar_4comp(entries, DM, deltaM, a, p_row, splot)
    elif args.fit_model == 'new':
        w = FitUtils.Dstar_run3(entries, DM, deltaM, args.bkg_option, bin_no, args.decay_mode, splot, '', '', '')
    else:
        w = FitUtils.Dstar_run3_3comp(entries, DM, deltaM, args.bkg_option, bin_no, args.decay_mode, splot)    

    model = w.pdf('model')

    DM_dataset = create_1d_dataset(data,DM,'DM')
    deltaM_dataset = create_1d_dataset(data,deltaM,'deltaM')

    twoD_dataset = create_2d_dataset(data,DM,deltaM,'DM','deltaM')
    
    DM.setBins(NBins)
    deltaM.setBins(NBins)        
    binned = ROOT.RooDataHist("binned","",ROOT.RooArgSet(DM,deltaM),twoD_dataset)

    ll = ROOT.RooLinkedList()
    cmd1 = ROOT.RooFit.Extended(ROOT.kTRUE)
    ll.Add(cmd1)
    cmd2 = ROOT.RooFit.NumCPU(2)
    ll.Add(cmd2)
    cmd3 = ROOT.RooFit.Save()
    ll.Add(cmd3)
    cmd4 = ROOT.RooFit.Minimizer("Minuit","Migrad") 
    ll.Add(cmd4)                                                                
    cmd5 = ROOT.RooFit.PrintLevel(1)
    ll.Add(cmd5)
    

    #ROOT.RooMsgService.instance().setSilentMode(ROOT.kTRUE)

    #fixAtFirstIteration = ["nDelta","alphaDelta", "cfDelta"]
    #params = model.getVariables()
    #iter = params.createIterator()
    #var = iter.Next()
    #while var:
    #    if var.GetName() in fixAtFirstIteration:
    #        print(var.GetName())
    #        w.var(var.GetName()).setConstant()
    #    var = iter.Next()
    
    print("Attempting Fit......")
    '''ROOT.RooMsgService.instance().setSilentMode(ROOT.kTRUE)'''


    fit_result = model.fitTo(binned, ll)

    #iter = params.createIterator()
    #var = iter.Next()
    #while var:
    #    if var.GetName() in fixAtFirstIteration:
    #        print(var.GetName())
    #        w.var(var.GetName()).setConstant(False)
    #    var = iter.Next()
    #fit_result = model.fitTo(binned,ll)

    #continues until model is fitted
    n_maxfits = 1
    while (not(fit_result.status() == 0 and fit_result.covQual() == 3) and n_maxfits<6):
        fit_result = model.fitTo(binned, ll)
        n_maxfits += 1

    params = model.getVariables()
    params.writeToFile("./plots_data_fit/parameters{0}_{1}files_{2}_fit{3}.txt".format(args.year + args.polarity + args.decay_mode+tag+i+j+str(p_row)+str(a), str(file_no), args.bkg_option, args.fit_model))
    print("Parameters saved to parameters{0}_{1}files_{2}_fit{3}.txt".format(args.year + args.polarity + args.decay_mode+tag+i+j+str(p_row)+str(a), str(file_no), args.bkg_option, args.fit_model))
    '''if args.year == 'All':                                                                                                                                                                                               
efficiency    '''

    res_status = str(fit_result.status())
    res_covQual = str(fit_result.covQual())
    res_nfits = str(n_maxfits)

    with open("./plots_data_fit/parameters{0}_{1}files_{2}_fit{3}.txt".format(args.year + args.polarity + args.decay_mode+tag+i+j+str(p_row)+str(a), str(file_no), args.bkg_option, args.fit_model), 'a') as out_f:
        out_f.write("\n")
        out_f.write(f'fit status      = {res_status}')
        out_f.write("\n")
        out_f.write(f'fit covQual     = {res_covQual}')
        out_f.write("\n")
        out_f.write(f'number of fits  = {res_nfits}')

    # legend coords
    leg_coords = [0.70,0.55,0.85,0.85]
    pt_coords = [[0.23,0.77,0.42,0.9]]


    leg_coords_log = [0.70, 0.25, 0.85, 0.50]
    pt_coords_log = [[0.10, 0.77, 0.30, 0.9]]


    cases = {'DM': ["m(D^{0}) [MeV/c^{2}]",DM,DM_dataset,low_mass_dm,high_mass_dm],'deltaM':["#Deltam [MeV/c^{2}]",deltaM,deltaM_dataset,low_mass_deltam,high_mass_deltam]}

    #model_DM = model.createProjection(ROOT.RooArgSet(deltaM))
    #model_deltaM = model.createProjection(ROOT.RooArgSet(DM))

    for case in cases:
        element = ''
        if args.calc_type == 'Efficiency':
            element = 'Eff'
            if args.kin_bins == 'n':
                save_file = "./plots_data_eff/{0}EfficiencyData_{1}files_fit{2}.pdf".format(args.year+args.polarity+args.decay_mode+case+args.bkg_option+element+tag+i+j,str(file_no), args.fit_model)
                save_file_log = "./plots_data_eff/{0}EfficiencyData_{1}files_fit{2}_log.pdf".format(args.year+args.polarity+args.decay_mode+case+args.bkg_option+element+tag+i+j,str(file_no), args.fit_model)                      
            else:
                save_file = "./plots_data_eff/{0}EfficiencyData_{1}files_fit{2}.pdf".format(args.year+args.polarity+args.decay_mode+case+args.bkg_option+element+tag+i+j+str(p_row)+str(a),str(file_no), args.fit_model)
                save_file_log = "./plots_data_eff/{0}EfficiencyData_{1}files_fit{2}_log.pdf".format(args.year+args.polarity+args.decay_mode+case+args.bkg_option+element+tag+i+j+str(p_row)+str(a),str(file_no), args.fit_model)   
        else:
            if args.kin_bins == 'n':
                save_file = "./plots_data_fit/{0}ProjectionData_{1}files_fit{2}.pdf".format(args.year+args.polarity+args.decay_mode+case+args.bkg_option+element+tag+i+j,str(file_no), args.fit_model)
                save_file_log = "./plots_data_fit/{0}ProjectionData_{1}files_fit{2}_log.pdf".format(args.year+args.polarity+args.decay_mode+case+args.bkg_option+element+tag+i+j,str(file_no), args.fit_model)   
            else:
                save_file = "./plots_data_fit/{0}ProjectionData_{1}files_fit{2}.pdf".format(args.year+args.polarity+args.decay_mode+case+args.bkg_option+element+tag+i+j+str(p_row)+str(a),str(file_no), args.fit_model)
                save_file_log = "./plots_data_fit/{0}ProjectionData_{1}files_fit{2}_log.pdf".format(args.year+args.polarity+args.decay_mode+case+args.bkg_option+element+tag+i+j+str(p_row)+str(a),str(file_no), args.fit_model)   
        #temp_model = ''
        #if case == "DM":
        #    temp_model = model_DM
        #else:
        #    temp_model = model_deltaM
        #Make sure to add ks_type to the title                                                                                                                               
        if args.polarity == "All":
            FitUtils.plotFit(model, cases[case][1], cases[case][2], NBins, cases[case][3], cases[case][4], cases[case][0], save_file, leg_coords, pt_coords, [args.year, args.decay_mode], load_plot=False, log=False)
            FitUtils.plotFit(model, cases[case][1], cases[case][2], NBins, cases[case][3], cases[case][4], cases[case][0], save_file_log, leg_coords, pt_coords, [args.year, args.decay_mode], load_plot=False, log=True)
        else:
            FitUtils.plotFit(model, cases[case][1], cases[case][2], NBins, cases[case][3], cases[case][4], cases[case][0], save_file, leg_coords, pt_coords, ['LHCb Run 3','Preliminary'], load_plot=False, log=False)
            FitUtils.plotFit(model, cases[case][1], cases[case][2], NBins, cases[case][3], cases[case][4], cases[case][0], save_file_log, leg_coords, pt_coords, ['LHCb Run 3', 'Preliminary'], load_plot=False, log=True)

    return w

def split_x_data(data,bins,var):
    
    vals = []
    for entry in data:
        vals.append(getattr(entry,var))
    
    vals = np.array(vals)
    bins = np.array(bins)
    
    bin_locations = np.digitize(vals,bins,right=True)
    split_data = [[] for i in range(len(bins)-1)]
    
    print('Splitting data into bins......... {0}'.format(len(split_data)))
    for cand,cand_bin in zip(data,bin_locations):
        if cand_bin < 0 or cand_bin > len(bins):
            print(cand,cand_bin)
        split_data[cand_bin-1].append(cand)
    
    return split_data

def split_2d_data(data,x_bins,y_bins,x_var,y_var):
    
    x_vals = []
    x_binned_data = [[] for i in range(len(x_bins)-1)]

    for cand in data:
        x_vals.append(getattr(cand,x_var))

    x_vals = np.asarray(x_vals)
    x_bins = np.asarray(x_bins)
    bin_locations = np.digitize(x_vals,x_bins)

    bin_locations = bin_locations.tolist()

    print('Splitting data into bins...........{0}'.format(len(x_bins)-1))
    
    binned_data = []
    for cand,cand_bin in zip(data,bin_locations):
        x_binned_data[cand_bin-1].append(cand)

    for sub_list in x_binned_data:
        binned_data.append(split_x_data(sub_list,y_bins,y_var))

    return binned_data

def main():

    low_mass_deltam = min([d.deltaM for d in data])
    #high_mass_deltam = max([d.deltaM for d in data])

    #low_mass_dm = min([d.DM for d in data])
    #high_mass_dm = max([d.DM for d in data])

    #low_mass_deltam = 141.0
    high_mass_deltam = 153.0
    low_mass_dm = 1825.0 #DEFAULT
    #low_mass_dm = 1800.0
    high_mass_dm = 1910.0 #DEFAULT
    #high_mass_dm = 1930.0


    leg_coords = [0.75,0.55,0.9,0.85]
    pt_coords = [[0.23,0.6,0.42,0.9]]

    DM = ROOT.RooRealVar("DM","DM",low_mass_dm,high_mass_dm)
    deltaM = ROOT.RooRealVar("deltaM","deltaM",low_mass_deltam,high_mass_deltam)

    if args.calc_type == 'Projections':
        #splot = False
        w = perform_fit(data,low_mass_deltam,high_mass_deltam,low_mass_dm,high_mass_dm)
        model = w.pdf('model')
        
    else:
        temp_eff = 0.0

        if len(data) != 0:
            #splot = False
            w = perform_fit(data,low_mass_deltam,high_mass_deltam,low_mass_dm,high_mass_dm,'Pre')
            pre_cut = w.var('nSig').getValV()

            nSig_value = w.var("nSig").getVal()
            #nNR_value = w.var("nNR").getVal()
            #nCmb_value = w.var("nCmb").getVal()
            nNR_value = 0.0
            nCmb_value = 0.0

            temp_eff_sweight= find_sweights(w, data,low_mass_deltam,high_mass_deltam,low_mass_dm,high_mass_dm, nSig_value, nNR_value, nCmb_value)
            eff_sweight.append(temp_eff_sweight)

            ## Uncomment if the fit and count method is required
            #post_data = [d for d in data if d.probe_PIDK > cut_value]
            #w_post = perform_fit(post_data,low_mass_deltam,high_mass_deltam,low_mass_dm,high_mass_dm,'Post')
            #post_cut = w_post.var('nSig').getValV()
            #temp_eff = post_cut / pre_cut

        #eff_fitncount.append(temp_eff)  
    return

if __name__ == "__main__":
    data0, data1, data2, data3, data4, data5, data6, data7 = load_file()
    
    if args.kin_bins == 'n' or args.kin_bins == 'both':
        p_row = 0
        a = 'unbinned'	
        bin_no = 0
        data = data0
        main()
        
    if args.kin_bins == 'y' or args.kin_bins == 'both':
        for a in range(len(eta_bins_1)-1):
            p_row = 1
            print('Running for {0} <= p < {1}, {2} <= eta < {3}'.format(p_bins_all[0], p_bins_all[1], eta_bins_1[a], eta_bins_1[a+1]))
            if a == 0: 
                data = data1
                bin_no = 1
            if a == 1:
                data = data2
                bin_no = 2
            main()
           
        for a in range(len(eta_bins_2)-1):
            if a == 0: 
                data = data3
                bin_no = 3
            if a == 1:
                data = data4
                bin_no = 4
            if a == 2: 
                data = data5
                bin_no = 5
            p_row = 2
            print('Running for {0} <= p < {1}, {2} <= eta < {3}'.format(p_bins_all[1], p_bins_all[2], eta_bins_2[a], eta_bins_2[a+1]))
            main()

        for a in range(len(eta_bins_3)-1):
            if a == 0: 
                data = data6
                bin_no = 6
            if a == 1:
                data = data7
                bin_no = 7
            p_row = 3
            print('Running for {0} <= p < {1}, {2} <= eta < {3}'.format(p_bins_all[2], p_bins_all[3], eta_bins_3[a], eta_bins_3[a+1]))
            main()

if args.calc_type == 'Efficiency':
    print(eff_fitncount)
    print(eff_sweight)





