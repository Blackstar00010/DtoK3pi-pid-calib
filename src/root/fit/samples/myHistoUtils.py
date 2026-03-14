import ROOT

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

def createLegend(coords,entries):

    leg = ROOT.TLegend(coords[0],coords[1],coords[2],coords[3])
    leg.SetFillColor(ROOT.kWhite)
    
    for _, value in entries.items():
        leg.AddEntry(value[0],value[1],value[2])
    
    return leg