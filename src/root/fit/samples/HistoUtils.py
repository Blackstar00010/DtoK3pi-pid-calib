#################################################
# Tools for performing operation on histograms  #
# in particular plotting, saving, and adding    #
# legends as well as pavetext                   #
#################################################
import ROOT
import os
import subprocess
from lhcbStyle import setLHCbStyle

setLHCbStyle()

    
def SetAxisAttributes(hist):
   
    for axis in [hist.GetXaxis(),hist.GetYaxis()]:
        axis.SetTitleSize(0.06)
        axis.SetTitleSize(0.06)
        axis.SetTitleOffset(1.1)
    
    return hist

def createLegend(coords,entries):

    leg = ROOT.TLegend(coords[0],coords[1],coords[2],coords[3])
    leg.SetFillColor(ROOT.kWhite)
    
    for key,value in entries.items():
        leg.AddEntry(value[0],value[1],value[2])
    
    return leg

def createPaveText(coords,varlist,multiple=False):
    
    pt_list = []
    if multiple:
        for pos,entries in zip(coords,varlist):
            pt = ROOT.TPaveText(pos[0],pos[1],pos[2],pos[3],'BNDC')
            for var in entries:
                pt.AddText(var)
            
            pt.SetFillColor(0)
            pt.SetBorderSize(0)
            
            pt_list.append(pt)

        return pt_list

    else:
        
        pt = ROOT.TPaveText(coords[0],coords[1],coords[2],coords[3],'BNDC')
        for var in varlist:
                pt.AddText(var)
        
        pt.SetFillColor(0)
        pt.SetBorderSize(0)

        return pt

def TF2ToHist(func,coords=[],bins=500):
    
    x_min = ROOT.Double(0)
    x_max = ROOT.Double(0)
    y_min = ROOT.Double(0)
    y_max = ROOT.Double(0)
    if not coords:
        func.GetRange(x_min,y_min,x_max,y_max)
    
    hist = ROOT.TH2D("{0}".format(func.GetName()),"",bins,x_min,x_max,bins,y_min,y_max)
    for i in range(1,bins+2):
        for j in range(1,bins+2):
            hist.SetBinContent(i,j,func.Eval(hist.GetXaxis().GetBinCenter(i),hist.GetYaxis().GetBinCenter(j)))
    return hist,func

def Create2Dpulls(func,hist,save_file):
    
    bins = hist.GetNbinsX()
    
    min_val = hist.GetXais().GetXmin()
    max_val = hist.GetXaxis().GetXmax()

    pull_hist = ROOT.TH2D("pull_hist","",bins,min_val,max_val,bins,min_val,max_val)
    
    for i in range(1,bins+2):
        for j in range(1,bins+2):
            hist_val = hist.GetBinContent(i,j)
            if his_val == 0:
                continue
            temp_x = hist.GetBinCenter(i)
            temp_y = hist.GetBinCenter(j)
            func_val = func.Eval(temp_x,temp_y)
            if func_val == 0:
                continue
            pull = (func_val - hist_val) / hist.GetBinError(i,j)
            pull_hist.SetBinContent(i,j,pull)

    return pull_hist

def writeObjectToFile(obj,save_file,name):

    if os.path.isfile(save_file):
        temp_file = ROOT.TFile.Open(save_file,"UPDATE")
        temp_file.cd()
        if temp_file.GetListOfKeys().Contains(name):
            ROOT.gDirectory.Delete(name + ";1")
        obj.Write(name)
        temp_file.Close()

    else:
        temp_file = ROOT.TFile.Open(save_file,"RECREATE")
        temp_file.cd()
        obj.Write(name)
        #temp_file.Write()
        temp_file.Close()

    print("Written: {0}, to: {1}".format(name,save_file))
    

    


