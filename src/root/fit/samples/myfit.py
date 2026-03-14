import os
import pandas as pd
import ROOT
import uproot
import myFitUtils as FitUtils

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

def load_file() -> pd.DataFrame:
    cwd = os.path.dirname(os.path.abspath(__file__))  # <proj_dir>/src/root/fit/samples
    proj_dir = os.path.join(cwd, os.pardir, os.pardir, os.pardir)  # <proj_dir>
    filename = f'{proj_dir}/data/full.root'
    treename = 'DecayTree'
    
    df = uproot.open(filename)[treename]
    df = df.arrays(filter_name=["*"], library="pd")[['D_M', 'delta_M']]
    data = [
        DataPoint(
            row['D_M'],
            row['delta_M'],
            0, 0, 0, 0, 0
        ) for _, row in df.iterrows()
    ]
    return data

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

def perform_fit(data, DM, deltaM, masscuts, nbins=100):
    entries = len(data)
    splot = False
    w = FitUtils.Dstar_run3(entries, DM, deltaM, splot, '', '', '')
    model = w.pdf("model")

    print("Creating datasets..........")
    DM_dataset = create_1d_dataset(data,DM,'DM')
    deltaM_dataset = create_1d_dataset(data,deltaM,'deltaM')
    twoD_dataset = create_2d_dataset(data,DM,deltaM,'DM','deltaM')
    print("Datasets created successfully")
    
    print("Binning the 2D dataset..........")
    DM.setBins(nbins)
    deltaM.setBins(nbins)        
    binned = ROOT.RooDataHist("binned","",ROOT.RooArgSet(DM,deltaM),twoD_dataset)
    print("2D dataset binned successfully")

    print("Setting up the fit..........")
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
    print("Fit setup completed")

    print("Fitting the model..........")
    fit_result = model.fitTo(binned, ll)

    n_maxfits = 1
    while (not(fit_result.status() == 0 and fit_result.covQual() == 3) and n_maxfits<6):
        fit_result = model.fitTo(binned, ll)
        n_maxfits += 1

    params = model.getVariables()
    filename = "./plots_data_fit/fitparams.txt"
    params.writeToFile(filename)
    print("Parameters saved to fitparams.txt")

    res_status = str(fit_result.status())
    res_covQual = str(fit_result.covQual())
    res_nfits = str(n_maxfits)

    with open(filename, 'a') as out_f:
        out_f.write("\n")
        out_f.write(f'fit status      = {res_status}')
        out_f.write("\n")
        out_f.write(f'fit covQual     = {res_covQual}')
        out_f.write("\n")
        out_f.write(f'number of fits  = {res_nfits}')
    
    leg_coords = [0.70,0.55,0.85,0.85]
    pt_coords = [[0.23,0.77,0.42,0.9]]

    cases = {
        'DM': ["m(D^{0}) [MeV/c^{2}]",DM,DM_dataset,masscuts['dmmin'],masscuts['dmmax']],
        'deltaM':["#Deltam [MeV/c^{2}]",deltaM,deltaM_dataset,masscuts['deltammin'],masscuts['deltammax']]
    }

    for case in cases:
        # suffix = "eff" if args.calc_type == 'Efficiency' else "fit"
        suffix = "fit"
        save_file = f'./plots_data_{suffix}/data.pdf'
        save_file_log = f'./plots_data_{suffix}/data_log.pdf'
        FitUtils.plotFit(model, cases[case][1], cases[case][2], nbins,cases[case][3], cases[case][4], cases[case][0], save_file, leg_coords, pt_coords, ['LHCb Run 3','Preliminary'], load_plot=False, log=False)
        FitUtils.plotFit(model, cases[case][1], cases[case][2], nbins,cases[case][3], cases[case][4], cases[case][0], save_file_log, leg_coords, pt_coords, ['LHCb Run 3','Preliminary'], load_plot=False, log=True)
        
    return w


def main():
    print("Loading data from file..........")
    data = load_file()
    print("Data loaded successfully")

    # TODO: import from src/utils/consts.py
    # sneha_masscuts = {'dmmin': 1820, 'dmmax': 1900, 'deltammin': min([d.deltaM for d in data]), 'deltammax': 152}
    sneha_masscuts = {'dmmin': 1820, 'dmmax': 1900, 'deltammin': 139, 'deltammax': 152}

    # Define the mass ranges
    DM = ROOT.RooRealVar("DM", "DM", sneha_masscuts['dmmin'], sneha_masscuts['dmmax'])
    deltaM = ROOT.RooRealVar("deltaM", "deltaM", sneha_masscuts['deltammin'], sneha_masscuts['deltammax'])

    # projections
    w = perform_fit(data, DM, deltaM, sneha_masscuts)
    # model = w.pdf("model")

    # TODO: non-projections

if __name__ == '__main__':
    main()