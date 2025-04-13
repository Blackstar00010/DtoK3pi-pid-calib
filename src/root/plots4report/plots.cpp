#include <iostream>
#include <vector>
#include <string>
#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TGaxis.h>
// #include <TBox.h>
#include <TEllipse.h>
#include <TGraph.h>
#include <TLegend.h>
#include "../utils/config.cpp"
#include "../utils/treeops.cpp"
#include "../utils/check.cpp"
#include "../utils/strops.cpp"
#include "../utils/fileio.cpp"
#include "../utils/consts.cpp"
#include "consts.cpp"  // rename?

std::vector<double> getRange(TTree* tree, const std::vector<std::string>& targetBranches) {
    double min = tree->GetMinimum(targetBranches[0].c_str());
    double max = tree->GetMaximum(targetBranches[0].c_str());
    for (const std::string& branch : targetBranches) {
        double tempMin = tree->GetMinimum(branch.c_str());
        double tempMax = tree->GetMaximum(branch.c_str());
        if (tempMin < min) {
            min = tempMin;
        }
        if (tempMax > max) {
            max = tempMax;
        }
    }
    return {min, max};
}
int getColour(int i) {
    std::vector<int> colours = {
        kRed,
        kGreen + 2, // not visible with kGreen
        kBlue,
        kCyan + 1, // not visible with kCyan
        kMagenta,
        kYellow + 1, // not visible with kYellow
        kOrange,
        kViolet,
        kTeal,
        kAzure,
        kSpring,
        kPink,
        kBlack
    };
    return colours[i % colours.size()];
}
std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& data) {
    int nRows = data.size();
    int nCols = data[0].size();
    std::vector<std::vector<double>> transposed(nCols, std::vector<double>(nRows, 0.0));
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            transposed[j][i] = data[i][j];
        }
    }
    return transposed;
}

/**
 * Plot the 2D mass distribution of branches "D_M" and "delta_M"
 * @param tree: TTree pointer
 * @param filename: output file name
 */
void plot2DmassWithBox(TTree* tree, const std::string filename) {
    if (!tree) {
        std::cerr << "Error: TTree pointer is null!" << std::endl;
        return;
    }

    double dmmin = tree->GetMinimum("D_M");
    double dmmax = tree->GetMaximum("D_M");
    double deltammin = tree->GetMinimum("delta_M");
    double deltammax = tree->GetMaximum("delta_M");

    // Create a canvas
    TCanvas* c = new TCanvas("c", "c", 800, 600);
    // Create a 2D histogram
    TH2F* hist2D = new TH2F("hist2D", "2D mass distribution", 100, dmmin, dmmax, 100, deltammin, deltammax);
    tree->Draw("delta_M:D_M>>hist2D", "", "colz");
    // double dmmean = 1865.4285;
    // double deltamean = 145.4904;
    // double dmstd = 7.5276;
    // double deltastd = 0.7585;
    // TEllipse* box = new TEllipse(dmmean, deltamean, 5*dmstd, 5*deltastd, 0, 360, 0);
    // TEllipse* box = new TEllipse(consts::dmmean, consts::deltamean, consts::cutStd*consts::dmstd, consts::cutStd*consts::deltastd, 0, 360, 0);
    TBox* box = new TBox(consts::dmmean - consts::cutStd*consts::dmstd, consts::deltamean - consts::cutStd*consts::deltastd, consts::dmmean + consts::cutStd*consts::dmstd, consts::deltamean + consts::cutStd*consts::deltastd);
    box->SetFillStyle(0);
    box->SetLineColor(kRed);
    box->SetLineWidth(2);
    // TBox* box = new TBox(dmmean - dmstd, deltamean - deltastd, dmmean + dmstd, deltamean + deltastd);
    // box->SetFillStyle(0);
    // box->SetLineColor(kRed);
    // box->SetLineWidth(2);
    
    hist2D->Draw("colz");
    box->Draw();
    hist2D->SetStats(0);
    
    // ticks: 3 ticks at 1800, 1850, 1900 for the x axis, 3 ticks at 140, 146, 152 for the y axis
    hist2D->GetXaxis()->SetNdivisions(0);
    hist2D->GetYaxis()->SetNdivisions(0);
    // 510 means 5 primary ticks and 10 secondary ticks
    TGaxis* xaxis = new TGaxis(1800, deltammin, 1900, deltammin, 1800, 1900, 503, "");
    xaxis->SetLabelSize(consts::xLabelSize);
    xaxis->SetLabelFont(consts::labelFont);
    xaxis->SetTitle("#it{m}_{D} [MeV/#it{c}^{2}]");
    xaxis->SetTitleSize(consts::xTitleSize);
    // xaxis->SetTitleOffset(0.8);
    xaxis->SetTitleFont(consts::titleFont);
    xaxis->CenterTitle(consts::centerTitle);
    xaxis->Draw();
    TGaxis* yaxis = new TGaxis(dmmin, 135, dmmin, 155, 135, 155, 3, "");
    yaxis->SetLabelSize(consts::yLabelSize);
    yaxis->SetLabelFont(consts::labelFont);
    yaxis->SetTitle("#Delta #it{m} [MeV/#it{c}^{2}]");
    yaxis->SetTitleSize(consts::yTitleSize);
    // yaxis->SetTitleOffset(0.9);
    yaxis->SetTitleFont(consts::titleFont);
    yaxis->CenterTitle(consts::centerTitle);
    yaxis->Draw();
    hist2D->GetZaxis()->SetLabelSize(consts::zLabelSize);
    hist2D->SetTitle("");
    c->SetMargin(0.14, 0.15, 0.15, 0.05);
    c->SetFrameLineWidth(consts::frameWidth);
    // Save the plot
    c->SaveAs(filename.c_str());
    // Clean up
    delete box;
    delete hist2D;
    delete c;
}

void plotScoreDistribution(TTree* scoreTree, const std::string filename) {
    // if (!tree) {
    if (!scoreTree) {
        std::cerr << "Error: Tree is null!" << std::endl;
        return;
    }

    // Define the range of the histogram
    double min = -0.5;
    double max = 1.5;
    int nBins = 100;

    // Define histograms for each (is_test, ans) combination
    TH1F* h00 = new TH1F("h00", "Score Distributions", nBins, min, max);
    TH1F* h01 = new TH1F("h01", "Score Distributions", nBins, min, max);
    TH1F* h10 = new TH1F("h10", "Score Distributions", nBins, min, max);
    TH1F* h11 = new TH1F("h11", "Score Distributions", nBins, min, max);

    h00->SetLineColor(getColour(0));
    h01->SetLineColor(getColour(1));
    h10->SetLineColor(getColour(2));
    h11->SetLineColor(getColour(3+1));
    h10->SetFillColor(getColour(2));
    h11->SetFillColor(getColour(3+1));
    h10->SetFillStyle(3004);  // y=x cross-hatched
    h11->SetFillStyle(3005);  // y=-x cross-hatched

    h00->SetLineWidth(consts::lineWidth);
    h01->SetLineWidth(consts::lineWidth);
    h10->SetLineWidth(consts::lineWidth-1);
    h11->SetLineWidth(consts::lineWidth-1);

    // Set branch addresses
    Long64_t is_test, ans;
    double score;
    scoreTree->SetBranchAddress("_is_test_all", &is_test);
    scoreTree->SetBranchAddress("_ans_all", &ans);
    scoreTree->SetBranchAddress("bdt_all", &score);

    // Loop over tree entries
    Long64_t nEntries = scoreTree->GetEntries();
    for (Long64_t i = 0; i < nEntries; i++) {
        scoreTree->GetEntry(i);
        if (is_test == 0 && ans == 0) h00->Fill(score);
        if (is_test == 0 && ans == 1) h01->Fill(score);
        if (is_test == 1 && ans == 0) h10->Fill(score);
        if (is_test == 1 && ans == 1) h11->Fill(score);
    }

    h00->Scale(1.0 / h00->Integral());
    h01->Scale(1.0 / h01->Integral());
    h10->Scale(1.0 / h10->Integral());
    h11->Scale(1.0 / h11->Integral());

    // Create Canvas
    TCanvas* c = new TCanvas("c", "Score Distributions", 800, 600);

    // Draw histograms
    h00->Draw("HIST");
    h01->Draw("HIST SAME");
    h10->Draw("HIST SAME");
    h11->Draw("HIST SAME");

    // cut x range
    h00->GetXaxis()->SetRangeUser(-0.4, 1.4);
    h00->GetYaxis()->SetRangeUser(0, 0.1);

    // Create legend
    TLegend* legend = new TLegend(0.44, 0.65, 0.64, 0.94);
    legend->AddEntry(h00, "train-bkg", "l");
    legend->AddEntry(h01, "train-sig", "l");
    legend->AddEntry(h10, "test-bkg", "f");
    legend->AddEntry(h11, "test-sig", "f");
    legend->SetTextFont(consts::legendFont);
    legend->SetBorderSize(consts::legendWidth);
    legend->SetTextSize(consts::legendSize);
    legend->Draw();

    // Axis titles
    h00->SetStats(0);
    h00->SetTitle("");
    h00->GetXaxis()->SetTitle("BDT Scores");
    h00->GetYaxis()->SetTitle("Normalised Entries");
    h00->GetXaxis()->SetTitleSize(consts::xTitleSize);
    h00->GetYaxis()->SetTitleSize(consts::yTitleSize);
    h00->GetYaxis()->SetTitleOffset(0.9);
    h00->GetXaxis()->SetTitleFont(consts::titleFont);
    h00->GetYaxis()->SetTitleFont(consts::titleFont);
    h00->GetXaxis()->CenterTitle(consts::centerTitle);
    h00->GetYaxis()->CenterTitle(consts::centerTitle);

    // Axis labels
    h00->GetXaxis()->SetNdivisions(4);
    h00->GetYaxis()->SetNdivisions(2);
    h00->GetXaxis()->SetLabelSize(consts::xLabelSize);
    h00->GetYaxis()->SetLabelSize(consts::yLabelSize);
    h00->GetXaxis()->SetLabelFont(consts::labelFont);
    h00->GetYaxis()->SetLabelFont(consts::labelFont);

    // Save the plot
    c->SetMargin(0.14, 0.05, 0.15, 0.05);
    c->SetFrameLineWidth(consts::frameWidth);
    c->SaveAs(filename.c_str());

    // Clean up
    delete h00;
    delete h01;
    delete h10;
    delete h11;
    delete legend;
    delete c;
}

/**
 * Plot the comparison of the signal distribution of the full and the proba trees
 * @param fandptree: TTree pointer of the full and proba tree
 * @param filename: output file name
 */
void plotFullSigComp(TTree* fandptree, const std::string filename) {
    if (!fandptree) {
        std::cerr << "Error: TTree pointer is null!" << std::endl;
        return;
    }

    double dmmin = fandptree->GetMinimum("D_M");
    double dmmax = fandptree->GetMaximum("D_M");

    TCanvas* c = new TCanvas("canvas", "canvas", 800, 600);
    TH1F* histfull = new TH1F("histfull", "Full tree", 100, dmmin, dmmax);
    TH1F* histbdt50 = new TH1F("histfandp", "Proba tree", 100, dmmin, dmmax);
    fandptree->Draw("D_M>>histfull", "", "goff");
    fandptree->Draw("D_M>>histfandp", "bdt_100 > 0.5", "goff");
    histfull->Scale(1.0 / histfull->Integral());
    histbdt50->Scale(1.0 / histbdt50->Integral());
    histbdt50->Draw("HIST");
    histfull->Draw("HIST SAME");
    histfull->SetStats(0);
    histbdt50->SetStats(0);
    histbdt50->SetTitle("");
    histbdt50->GetXaxis()->SetTitle("#it{m}_{D} [MeV/#it{c}^{2}]");
    histbdt50->GetYaxis()->SetTitle("Normalised Entries");
    // set title offset
    // histbdt50->GetXaxis()->SetTitleOffset(0.8);
    histbdt50->GetYaxis()->SetTitleOffset(0.9);
    histbdt50->GetXaxis()->SetNdivisions(4);
    histbdt50->GetYaxis()->SetNdivisions(3);
    // set title center
    histbdt50->GetXaxis()->CenterTitle(consts::centerTitle);
    histbdt50->GetYaxis()->CenterTitle(consts::centerTitle);
    // set font size
    histbdt50->GetXaxis()->SetTitleSize(consts::xTitleSize);
    histbdt50->GetYaxis()->SetTitleSize(consts::yTitleSize);
    histbdt50->GetXaxis()->SetLabelSize(consts::xLabelSize);
    histbdt50->GetYaxis()->SetLabelSize(consts::yLabelSize);
    histbdt50->GetXaxis()->SetTitleFont(consts::titleFont);
    histbdt50->GetYaxis()->SetTitleFont(consts::titleFont);
    histbdt50->GetXaxis()->SetLabelFont(consts::labelFont);
    histbdt50->GetYaxis()->SetLabelFont(consts::labelFont);
    // line thickness
    histfull->SetLineWidth(consts::lineWidth);
    histbdt50->SetLineWidth(consts::lineWidth);
    histfull->SetLineColor(kBlue);
    histbdt50->SetLineColor(kRed);
    histfull->SetMinimum(0);
    histbdt50->SetMinimum(0);
    TLegend* leg = new TLegend(0.65, 0.7, 0.94, 0.94);
    leg->AddEntry(histfull, "Full Data", "l");
    leg->AddEntry(histbdt50, "Selected Data", "l");
    // font size
    leg->SetTextSize(consts::legendSize);
    leg->SetTextFont(consts::legendFont);
    leg->SetBorderSize(consts::legendWidth);
    leg->Draw();
    // set margin
    c->SetMargin(0.14, 0.05, 0.15, 0.05);
    c->SetFrameLineWidth(consts::frameWidth);
    c->SaveAs(filename.c_str());
    delete c;
}

/**
 * @brief Plot efficiency 1D histograms for given branches. (N+M) histograms are plotted where targetBranches.size() = N and targetBranchSets.size() = M.
 * 
 * @param tree The TTree containing the data.
 * @param targetBranchSets The vectors of vectors of branch names to plot. Each vector of branch names will be summed to calculate the efficiency.
 * @param weightBranch The name of the branch containing the weights.
 * @param criteriaBranch The name of the branch containing the criteria.
 * @param criteriaValue The value of the criteria.
 * @param outputDir The directory to save the plots.
 * @param givenRepresentativeNames The names that can represent each vector in the targetBranchSets vector. Default is the vector of the first branch name.
 * @param givenRepresentativeName The name that can represent the all the branches that are plotted. Default is the first branch name of the targetBranches.
 * @param nBins The number of bins for the histogram. Default is 100.
 * @param range The range of the histogram. Default is the range of the tree.
 * @param higherIsBetter Whether values higher than the criteria are considered better. Default is true.
 * @param saturate Whether to saturate the efficiency values, i.e. limit them to [0, 1]. Default is true.
 * @param filename The name of the file to save the plot. Default is the representative name. File saved as "<outputDir>eff1d_<filename>.png".
 * @return void
 * 
 * @example PlotEff1ds(tree, {{"K+_P"}, {"Pi0_P"}, {"K-1_P", "K-2_P"}, {"Pi+1_P", "Pi+2_P", "Pi+3_P"}}, "weight", "criteria", {5, 0, -5}, "output/", {"", "", "K-_P", "Pi+_P"}, "P", 100, {0, 1000}, true, true);
 * @example PlotEff1ds(tree, {{"K+_P"}}, "weight", "criteria", {5, 0, -5}, "output/"});  // plots one branch with auto-bin, auto-range
 * 
 */
void PlotEff1ds(
    TTree* tree,
    const std::vector<std::vector<std::string>>& targetBranchSets,
    const std::string& weightBranch,
    const std::string& criteriaBranch,
    const std::vector<double>& criteriaValues,
    const std::string& outputDir,
    const std::vector<std::string>& givenRepresentativeNames = {},
    const std::string& givenRepresentativeName = "",
    int nBins = 100,
    std::vector<double> range = {1, 0},
    bool higherIsBetter = true,
    bool saturate = true,
    const std::string& givenFilename = ""
) {
    // if (targetBranches.empty() || targetBranchSets.empty()) {
    if (targetBranchSets.empty()) {
        std::cerr << "Error: No target branches or sets provided." << std::endl;
        return;
    }
    for (const std::vector<std::string>& branchSet : targetBranchSets) {
        if (branchSet.empty()) {
            std::cerr << "Error: Empty branch set provided." << std::endl;
            return;
        }
        if (!check::CheckTreeBranches(tree, branchSet)) {
            return;
        }
    }
    if (nBins <= 0) {
        std::cerr << "Error: Invalid number of bins: " << nBins << std::endl;
        return;
    }
    if (criteriaValues.empty()) {
        std::cerr << "Error: No criteria values provided." << std::endl;
        return;
    }
    std::vector<std::string> representativeNames = givenRepresentativeNames;
    if (representativeNames.empty()) {
        std::cout << "Empty representative names provided. Using the first branch name as the representative." << std::endl;
        for (const std::vector<std::string>& branchSet : targetBranchSets) {
            representativeNames.push_back(branchSet[0]);
        }
    }
    for (int i = 0; i < representativeNames.size(); ++i) {
        if (representativeNames[i].empty()) {
            std::cout << "Empty representative name provided. Using the first branch name of the set as the representative." << std::endl;
            representativeNames[i] = targetBranchSets[i][0];
        }
    }
    std::string representativeName = givenRepresentativeName;
    if (representativeName.empty()) {
        std::cout << "Empty representative name provided. Using the first branch name as the representative." << std::endl;
        representativeName = representativeNames[0];
    }
    std::string filename = givenFilename;
    if (filename.empty()) {
        std::cout << "Empty filename provided. Using the representative name as the filename." << std::endl;
        filename = representativeName;
    }
    std::cout << "Plotting 1D efficiency histograms for branch sets: {";
    for (int i = 0; i < targetBranchSets.size(); ++i) {
        // std::cout << representativeNames[i] << " = {";
        for (const std::string& branch : targetBranchSets[i]) {
            // std::cout << branch << ", ";
        }
        // std::cout << "}, ";
    }
    std::cout << "} using weights from branch " << weightBranch << " and criteria from branch " << criteriaBranch << std::endl;
    std::string histid = strops::randstr(4);
    std::cout << "Plot ID generated: " << histid << std::endl;

    std::cout << "Checking for variation in branches..." << std::endl;
    std::vector<double> minmax = getRange(tree, targetBranchSets[0]);
    double min = minmax[0];
    double max = minmax[1];
    for (const std::vector<std::string>& branchSet : targetBranchSets) {
        std::vector<double> tempMinMax = getRange(tree, branchSet);
        if (tempMinMax[0] < min) {
            min = tempMinMax[0];
        }
        if (tempMinMax[1] > max) {
            max = tempMinMax[1];
        }
    }
    if (min == max) {
        std::cerr << "Error: Branches have no variation." << std::endl;
        return;
    }
    if (range.size() != 2) {
        std::cout << "Invalid range provided; size 2 expected but got " << range.size() << ". The range will be determined by the tree." << std::endl;
    } else if (range[0] >= range[1]) {
        std::cout << "Invalid range provided (" << range[0] << ", " << range[1] << "). The range will be determined by the tree." << std::endl;
    } else {
        std::cout << "Using provided range of (" << range[0] << ", " << range[1] << ")" << std::endl;
        min = range[0];
        max = range[1];
    }

    // Define histograms
    std::cout << "Creating histograms..." << std::endl;
    std::vector<TH1F*> totalHists;  // 1D vector because totalHists[criteria]
    std::vector<std::vector<TH1F*>> passedHistsSets;  // 2D vector because passedHistsSets[branchSet][criteria]
    std::vector<std::vector<TH1F*>> effHistsSets;  // 2D vector because effHistsSets[branchSet][criteria]
    for (int i = 0; i < targetBranchSets.size(); ++i) {
        TH1F* totalHist = new TH1F(("totalHist_" + representativeNames[i] + histid).c_str(), ("total_" + representativeNames[i] + histid).c_str(), nBins, min, max);
        std::vector<TH1F*> passedHists;  // 1D vector because passedHists[criteria]
        std::vector<TH1F*> effHists;  // 1D vector because effHists[criteria]
        for (int j = 0; j < criteriaValues.size(); ++j) {
            std::string commonName = representativeNames[i] + std::to_string(criteriaValues[j]) + histid;
            TH1F* passedHist = new TH1F(("passedHist_" + commonName).c_str(), ("passed_" + representativeNames[i] + commonName).c_str(), nBins, min, max);
            TH1F* effHist = new TH1F(("effHist_" + commonName).c_str(), ("eff_" + representativeNames[i] + commonName).c_str(), nBins, min, max);
            passedHists.push_back(passedHist);
            effHists.push_back(effHist);
        }
        totalHists.push_back(totalHist);
        passedHistsSets.push_back(passedHists);
        effHistsSets.push_back(effHists);
    }
    std::cout << "Histograms created." << std::endl;

    // Set up branches
    std::cout << "Setting up branches..." << std::endl;
    std::vector<std::vector<double>> values(targetBranchSets.size());  // 2D vector because values[branchSet][branch]
    double weight, criteria;
    tree->ResetBranchAddresses();
    for (int i = 0; i < targetBranchSets.size(); ++i) {
        values[i].resize(targetBranchSets[i].size());
        for (int j = 0; j < targetBranchSets[i].size(); ++j) {
            tree->SetBranchAddress(targetBranchSets[i][j].c_str(), &values[i][j]);
        }
    }
    tree->SetBranchAddress(weightBranch.c_str(), &weight);
    tree->SetBranchAddress(criteriaBranch.c_str(), &criteria);
    std::cout << "Branches set up." << std::endl;

    // Loop through tree entries
    std::cout << "Filling histograms..." << std::endl;
    // damn what a mess but let me explain
//  for each row
//      for each branch set
//          for each branch
//              fill total histogram
//              for each criteria
//                  if criteria is met, fill passed histogram
    for (Long64_t i = 0; i < tree->GetEntries(); ++i) {
        tree->GetEntry(i);
        for (int j = 0; j < targetBranchSets.size(); ++j) {
            for (int k = 0; k < targetBranchSets[j].size(); ++k) {
                totalHists[j]->Fill(values[j][k], weight);
                for (int l = 0; l < criteriaValues.size(); ++l) {
                    if (((criteria > criteriaValues[l]) && higherIsBetter) || ((criteria < criteriaValues[l]) && !higherIsBetter)) {
                        passedHistsSets[j][l]->Fill(values[j][k], weight);
                    }
                }
            }
        }
    }
    std::cout << "Histograms filled." << std::endl;

    // Calculate efficiency
    std::cout << "Calculating efficiencies..." << std::endl;
//  for each branch set
//      for each criteria
//          divide passed by total to get efficiency
    for (int i = 0; i < targetBranchSets.size(); ++i) {
        for (int j = 0; j < criteriaValues.size(); ++j) {
            effHistsSets[i][j]->Divide(passedHistsSets[i][j], totalHists[i], 1, 1, "B");
        }
    }
    std::cout << "Efficiencies calculated." << std::endl;

    // Plot histograms
    std::cout << "Plotting histograms..." << std::endl;
    TCanvas* canvas = new TCanvas(("canvas_" + representativeName + histid).c_str(), ("Eff1D_" + representativeName + histid).c_str(), 1000, 700);
    canvas->SetMargin(0.12, 0.05, 0.14, 0.05);
    canvas->SetFrameLineWidth(consts::frameWidth);
    // effHistsSets[0][0]->Draw("HIST");
    effHistsSets[0][0]->Draw("E1");
    effHistsSets[0][0]->SetLineColor(getColour(0));
    effHistsSets[0][0]->SetLineWidth(consts::lineWidth);
    std::vector<double> coords;
    if (strops::startswith(criteriaBranch, "K_PID")) {
        coords = {0.13, 0.15, 0.45, 0.35};
    } else {
        coords = {0.13, 0.75, 0.45, 0.94};
    }
    TLegend* legend = new TLegend(coords[0], coords[1], coords[2], coords[3]);
    legend->SetTextSize(consts::legendSize);
    legend->SetTextFont(consts::legendFont);
    legend->SetBorderSize(consts::legendWidth);
    // std::vector<std::vector<std::string>> legends(targetBranchSets.size(), std::vector<std::string>(criteriaValues.size()));
    std::string sign = higherIsBetter ? ">" : "<";
    int histCount = 0;
    for (int i = 0; i < targetBranchSets.size(); ++i) {
        for (int j = 0; j < criteriaValues.size(); ++j) {
            // effHistsSets[i][j]->Draw("HIST SAME");
            effHistsSets[i][j]->Draw("E1 SAME");
            effHistsSets[i][j]->SetLineColor(getColour(i * criteriaValues.size() + j));
            effHistsSets[i][j]->SetLineWidth(consts::lineWidth);
            // legends[i][j] = representativeNames[i] + " " + sign + " " + std::to_string(criteriaValues[j]);
            legend->AddEntry(effHistsSets[i][j], ("PID_K " + sign + " " + std::to_string(int(criteriaValues[j]))).c_str(), "l");
            ++histCount;
        }
    }
    // title
    effHistsSets[0][0]->SetTitle("");
    // axes labels
    effHistsSets[0][0]->GetXaxis()->SetTitle(representativeName.c_str());
    effHistsSets[0][0]->GetYaxis()->SetTitle("Efficiency");
    effHistsSets[0][0]->GetXaxis()->SetTitleSize(consts::xTitleSize);
    effHistsSets[0][0]->GetYaxis()->SetTitleSize(consts::yTitleSize);
    effHistsSets[0][0]->GetXaxis()->SetTitleFont(consts::titleFont);
    effHistsSets[0][0]->GetYaxis()->SetTitleFont(consts::titleFont);
    // effHistsSets[0][0]->GetXaxis()->SetTitleOffset(0.9);
    effHistsSets[0][0]->GetYaxis()->SetTitleOffset(0.8);
    effHistsSets[0][0]->GetXaxis()->CenterTitle(consts::centerTitle);
    effHistsSets[0][0]->GetYaxis()->CenterTitle(consts::centerTitle);
    // change x range from MeV/c to GeV/c if we are plotting momentum
    if (strops::strin(representativeName, "GeV")) {
        for (int i = 0; i < targetBranchSets.size(); ++i) {
            for (int j = 0; j < criteriaValues.size(); ++j) {
                effHistsSets[i][j]->GetXaxis()->SetLimits(min / 1000, max / 1000);
            }
        }
    }
    // fewer ticks: 5
    effHistsSets[0][0]->GetXaxis()->SetNdivisions(5);
    effHistsSets[0][0]->GetYaxis()->SetNdivisions(5);
    // font size
    effHistsSets[0][0]->GetXaxis()->SetLabelSize(consts::xLabelSize);
    effHistsSets[0][0]->GetYaxis()->SetLabelSize(consts::yLabelSize);
    effHistsSets[0][0]->GetXaxis()->SetLabelFont(consts::labelFont);
    effHistsSets[0][0]->GetYaxis()->SetLabelFont(consts::labelFont);
    // Style settings
    if (gStyle->GetOptStat() != 0) {
        gStyle->SetOptStat(0);
    }
    if (histCount > 1) {
        legend->Draw();
    }
    // y range from 0 to 1
    if (saturate) {
        effHistsSets[0][0]->SetMinimum(0);
        effHistsSets[0][0]->SetMaximum(1);
    }
    std::cout << "Histograms plotted." << std::endl;

    // Save the plot
    filename = outputDir + "eff1d_" + filename + ".png";
    std::cout << "Saving efficiency plot to " << filename << std::endl;
    // canvas->Update();
    // canvas->Modified();
    canvas->SaveAs(filename.c_str());
    // gSystem->ProcessEvents();
    std::cout << "Efficiency plot saved." << std::endl;

    // Cleanup
    delete legend;
    delete canvas;
    for (int i = 0; i < targetBranchSets.size(); ++i) {
        for (int j = 0; j < criteriaValues.size(); ++j) {
            delete effHistsSets[i][j];
            delete passedHistsSets[i][j];
        }
        delete totalHists[i];
    }
    tree->ResetBranchAddresses();
    std::cout << std::endl;
}

std::vector<double> sortVectorByAnotherVector(std::vector<double> target, std::vector<double> standard) {
    std::vector<std::pair<double, double>> pairs;
    for (size_t i = 0; i < target.size(); ++i) {
        pairs.push_back({target[i], standard[i]});
    }
    std::sort(pairs.begin(), pairs.end(), [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
        return a.second < b.second;
    });
    std::vector<double> result;
    for (const auto& pair : pairs) {
        result.push_back(pair.first);
    }
    return result;
}

std::vector<std::vector<double>> calc2dROC(std::vector<double> valuesX, std::vector<double> valuesY, std::vector<double> weightsX, std::vector<double> weightsY) {
    weightsX = sortVectorByAnotherVector(weightsX, valuesX);
    valuesX = sortVectorByAnotherVector(valuesX, valuesX);
    weightsY = sortVectorByAnotherVector(weightsY, valuesY);
    valuesY = sortVectorByAnotherVector(valuesY, valuesY);
    double sumWeightsX = 0;
    for (double weight : weightsX) {
        sumWeightsX += weight;
    }
    double sumWeightsY = 0;
    for (double weight : weightsY) {
        sumWeightsY += weight;
    }
    std::vector<std::vector<double>> result;
    double effX = 0;
    double effY = 0;
    int previousIndexY = 0;
    for (long i = 0; i < valuesX.size(); i++) {
        double cutX = valuesX[i];
        effX += weightsX[i] / sumWeightsX;
        for (long j = previousIndexY; j < valuesY.size(); j++) {
            if (valuesY[j] > cutX) {
                previousIndexY = j;
                break;
            }
            effY += weightsY[j] / sumWeightsY;
        }
        result.push_back({1 - effX, 1 - effY});
        // result.push_back({effY, effX});
    }
    return result;
}

void PlotROC(
    TTree* tree,
    const std::string& criteriaBranchX,
    const std::string& criteriaBranchY,
    const std::string& weightBranch,
    std::vector<std::vector<double>> comparisonROC,
    const std::string& outputDir,
    std::vector<double> criteriaRange = {1, 0},
    int resolution = 100
) {
    if (resolution <= 0) {
        std::cerr << "Error: Invalid resolution: " << resolution << std::endl;
        return;
    }
    if (!check::CheckTreeBranches(tree, {criteriaBranchX, criteriaBranchY}) || !check::CheckTreeBranches(tree, {weightBranch})) {
        return;
    }
    std::string histid = strops::randstr(4);
    std::cout << "Plot ID generated: " << histid << std::endl;

    // for criteria branches <particle>_PID_K, new tree with 3000 < <particle>_P < 100000 and 1.5 < <particle>_ETA < 5.0
    TTree* cutTree = nullptr;
    gROOT->cd();
    std::cout << "Cutting tree for the kinetic region of interest... (nEntries: " << tree->GetEntries() << ")" << std::endl;
    for (const std::string& branch : {criteriaBranchX, criteriaBranchY}) {
        std::string particle = strops::split(branch, "_")[0];
        if (strops::endswith(branch, "_PID_K")) {
            if (!check::CheckTreeBranches(tree, {particle + "_P", particle + "_ETA"})) {
                continue;
            }
            std::string pBranch = particle + "_P";
            std::string etaBranch = particle + "_ETA";
            std::string cut = pBranch + " > 3000 && " + pBranch + " < 100000 && " + etaBranch + " > 1.5 && " + etaBranch + " < 5.0";
            cutTree = tree->CopyTree(cut.c_str());
            if (cutTree->GetEntries() == 0) {
                std::cerr << "Error: No entries after applying the cut!" << std::endl;
                continue;
            }
            tree = cutTree;
        }
    }
    std::cout << "Tree cut. (nEntries: " << tree->GetEntries() << ")" << std::endl;

    // Find the range of the criteria
    std::cout << "Checking for variation in branches..." << std::endl;
    double min = tree->GetMinimum(criteriaBranchX.c_str());
    double max = tree->GetMaximum(criteriaBranchX.c_str());
    double tempMin = tree->GetMinimum(criteriaBranchY.c_str());
    double tempMax = tree->GetMaximum(criteriaBranchY.c_str());
    if (tempMin < min) {
        min = tempMin;
    }
    if (tempMax > max) {
        max = tempMax;
    }
    if (min == max) {
        std::cerr << "Error: Branches " << criteriaBranchX << " and " << criteriaBranchY << " have no variation." << std::endl;
        return;
    }

    // Prepare storage for branch values
    std::cout << "Setting up branch values..." << std::endl;
    double xValue;
    double yValue;
    double weight;

    tree->ResetBranchAddresses();
    tree->SetBranchAddress(criteriaBranchX.c_str(), &xValue);
    tree->SetBranchAddress(criteriaBranchY.c_str(), &yValue);
    tree->SetBranchAddress(weightBranch.c_str(), &weight);
    std::cout << "Branch values set up." << std::endl;

    // Loop over tree entries and fill the histogram
    std::cout << "Calculating efficiencies..." << std::endl;
    Long64_t nEntries = tree->GetEntries();
    std::vector<double> xVals(nEntries);
    std::vector<double> yVals(nEntries);
    std::vector<double> xWeight(nEntries);
    std::vector<double> yWeight(nEntries);
    double total = 0;
    for (long i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        xVals[i] = xValue;
        yVals[i] = yValue;
        xWeight[i] = weight;
        yWeight[i] = weight;
        total += weight;
    }
    std::cout << "Efficiencies calculated." << std::endl;
    std::cout << "Calculating ROC curve..." << std::endl;
    std::vector<std::vector<double>> roc = calc2dROC(xVals, yVals, xWeight, yWeight);
    std::vector<std::vector<double>> cutRoc;
    for (size_t i = 0; i < roc.size(); ++i) {
        if (roc[i][1] < 0.01) {
            continue;
        }
        cutRoc.push_back({roc[i][0], roc[i][1]});
    }
    std::cout << "ROC curve calculated." << std::endl;
    std::cout << "Extracting points from the ROC curve..." << std::endl;
    // get `resolution` points from the ROC curve
    std::vector<double> xEffs(resolution);
    std::vector<double> yEffs(resolution);
    for (int i = 0; i < resolution; i++) {
        xEffs[i] = cutRoc[i * cutRoc.size() / resolution][0];
        yEffs[i] = cutRoc[i * cutRoc.size() / resolution][1];
    }
    std::cout << "Points extracted." << std::endl;
    std::cout << "Extracting points from the comparison ROC curve..." << std::endl;
    std::vector<double> xEffsComp = transpose(comparisonROC)[0];
    std::vector<double> yEffsComp = transpose(comparisonROC)[1];
    std::cout << "Comparison ROC curve extracted." << std::endl;

    // Plot the ROC curve (line plot)
    std::cout << "Plotting ROC curve..." << std::endl;
    TCanvas* canvas = new TCanvas(("canvas_" + criteriaBranchX + "_" + criteriaBranchY + histid).c_str(), ("ROC Curve (" + criteriaBranchX + " vs " + criteriaBranchY + ")" + histid).c_str(), 1000, 700);
    TGraph* graph = new TGraph(xEffs.size(), xEffs.data(), yEffs.data());
    TGraph* graphComp = new TGraph(xEffsComp.size(), xEffsComp.data(), yEffsComp.data());
    // graph->SetTitle(("ROC Curve; " + criteriaBranchX + " efficiency; " + criteriaBranchY + " efficiency").c_str());
    graph->SetTitle("");
    graph->GetXaxis()->SetTitle("#varepsilon(#it{K} #rightarrow #it{K})");
    graph->GetYaxis()->SetTitle("#varepsilon(#pi #rightarrow #it{K})");
    graph->GetXaxis()->SetTitleSize(consts::xTitleSize);
    graph->GetYaxis()->SetTitleSize(consts::yTitleSize);
    graph->GetXaxis()->SetTitleFont(consts::titleFont);
    graph->GetYaxis()->SetTitleFont(consts::titleFont);
    // graph->GetXaxis()->SetTitleOffset(1);
    graph->GetYaxis()->SetTitleOffset(0.9);
    graph->GetXaxis()->CenterTitle(consts::centerTitle);
    graph->GetYaxis()->CenterTitle(consts::centerTitle);
    graph->GetXaxis()->SetNdivisions(3);
    graph->GetYaxis()->SetNdivisions(3);
    graph->GetXaxis()->SetLabelSize(consts::xLabelSize);
    graph->GetYaxis()->SetLabelSize(consts::yLabelSize);
    graph->GetXaxis()->SetLabelFont(consts::labelFont);
    graph->GetYaxis()->SetLabelFont(consts::labelFont);
    graph->SetMarkerStyle(0);  // remove marker and leave line only
    graphComp->SetMarkerStyle(0);
    graph->SetMarkerSize(0.5);
    graphComp->SetMarkerSize(0.5);
    graph->SetMarkerColor(kBlue);
    graphComp->SetMarkerColor(kRed);
    graph->SetLineColor(kBlue);
    graphComp->SetLineColor(kRed);
    canvas->SetLogy();
    // graph->GetXaxis()->SetRangeUser(-0.1, 1.1);
    // graph->GetYaxis()->SetRangeUser(-0.1, 1.1);
    graph->SetLineWidth(consts::lineWidth);
    graphComp->SetLineWidth(consts::lineWidth);
    graph->Draw("AL");  // axis, line
    graphComp->Draw("L SAME");

    TLegend* legend = new TLegend(0.15, 0.7, 0.45, 0.9);
    legend->SetTextSize(consts::legendSize);
    legend->SetTextFont(consts::legendFont);
    legend->SetBorderSize(consts::legendWidth);
    legend->AddEntry(graph, "#it{K}#pi", "l");
    legend->AddEntry(graphComp, "#it{K}3#pi", "l");
    legend->Draw();

    canvas->SetFrameLineWidth(consts::frameWidth);
    canvas->SetMargin(0.13, 0.05, 0.14, 0.05);
    canvas->SaveAs((outputDir + "/roc_" + criteriaBranchX + "_" + criteriaBranchY + ".png").c_str());
    std::cout << "ROC curve plotted." << std::endl;
    
    // Clean up
    delete canvas;
    delete graph;
    tree->ResetBranchAddresses();
    delete cutTree;
    std::cout << std::endl;
}

void plots() {
    std::string plot_dir = config::plot_dir6;

    // TODO: TT score distribution not ready
    // auto [ttFile, ttTree] = treeops::loadFileAndTree(config::tt_score_file);
    // if (!ttFile || !ttTree) {
    //     std::cerr << "Error: Failed to load file and tree" << std::endl;
    //     return;
    // }
    // plotScoreDistribution(ttTree, plot_dir + "tt_score.png");
    // ttFile->Close();

    auto [longFile, longTree] = treeops::loadFileAndTree(config::long_root_file);
    if (!longFile || !longTree) {
        std::cerr << "Error: Failed to load file and tree" << std::endl;
        return;
    }
    plot2DmassWithBox(longTree, plot_dir + "2dmass.png");
    longFile->Close();

    auto [fnpFile, fnpTree] = treeops::loadFileAndTree(config::long_wscore_file);
    if (!fnpFile || !fnpTree) {
        std::cerr << "Error: Failed to load file and tree" << std::endl;
        return;
    }
    plotFullSigComp(fnpTree, plot_dir + "dmbdtcomp.png");
    fnpFile->Close();

    auto [swfile, swtree] = treeops::loadFileAndTree(config::long_sweight_file);
    if (!swfile || !swtree) {
        std::cerr << "Error: Failed to load file and tree" << std::endl;
        return;
    }
    std::string weightCol = "bdt_100_50_ss";
    double pidCut = 5;
    std::vector<double> pRange = {3000, 100000};
    std::vector<double> etaRange = {1.5, 5.0};
    int pBins = 18;
    int etaBins = 4;
    PlotEff1ds(swtree, {{"K_P"}}, weightCol, "K_PID_K", {pidCut, 0, -pidCut}, plot_dir, {""}, "Momentum [GeV/#it{c}]", pBins, pRange, true, true, "K_P_by_KPIDK");
    PlotEff1ds(swtree, {{"K_ETA"}}, weightCol, "K_PID_K", {pidCut, 0, -pidCut}, plot_dir, {""}, "Pseudorapidity", etaBins, etaRange, true, true, "K_ETA_by_KPIDK");
    PlotEff1ds(swtree, {{"pi1_P"}}, weightCol, "pi1_PID_K", {pidCut, 0, -pidCut}, plot_dir, {""}, "Momentum [GeV/#it{c}]", pBins, pRange, true, true, "pi1_P_by_pi1PIDK");
    PlotEff1ds(swtree, {{"pi3_P"}}, weightCol, "pi3_PID_K", {pidCut, 0, -pidCut}, plot_dir, {""}, "Momentum [GeV/#it{c}]", pBins, pRange, true, true, "pi3_P_by_pi3PIDK");

    std::vector<std::vector<double>> prevROC = fileio::readCSV("roc.csv");
    PlotROC(swtree, "K_PID_K", "pi1_PID_K", weightCol, prevROC, plot_dir);

    swfile->Close();
    return;
}