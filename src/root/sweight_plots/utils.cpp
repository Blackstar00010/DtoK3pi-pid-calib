#include <iostream>
#include <string>
#include <fstream>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TROOT.h>
#include <TSystem.h>
#include "../utils/config.cpp"
#include "../utils/treeops.cpp"
#include "../utils/plotter.cpp"
#include "../utils/strops.cpp"
#include "../utils/check.cpp"

// This file plots some 2D histograms and efficiency plots using sweights
// This file does not plot all the files under plots/4_massfit_cpp/2d/
// because some require models itself rather than just the sweights
// Those plots are done in ../fit/fit2d.cpp

// targetBranch : for one branch
// targetBranches : for multiple branches when not joined. i.e. plot N graphs for vector of N branches
// targetBranchSet : for multiple branches when joined. i.e. plot 1 graph for vector of N branches
// targetBranchSets : for multiple branches when joined at first level. i.e. plot N graphs for vector of N vector of branches

// TODO: add verbosity options

namespace commonVars {
    const std::string swFilename = "sWeights_sorted.root";
    const double cut = 0.5;
    const std::string modelName = "bdt_100";
    const std::string weightCol = modelName + "_" + std::to_string(int(cut * 100)) + "_ss";
}

int getColour(int i) {
    std::vector<int> colours = {
        kRed,
        kGreen + 2, // not visible with kGreen
        kBlue,
        kCyan + 2, // not visible with kCyan
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

/**
 * @brief Load the TTree from the sWeights file.
 * 
 * @return std::tuple<TFile*, TTree*> The TFile and TTree containing the sWeights data.
 *        If the file does not exist, returns {nullptr, nullptr}.
 */
std::tuple<TFile*, TTree*> loadSweight() {
    std::cout << "Loading TTree from file..." << std::endl;
    TFile* file;
    TTree* tree;
    std::string targetFileName = config::output_dir + "sWeights_sorted.root";
    // std::string targetFileName = sweight_full_file;
    if (std::filesystem::exists(targetFileName)) {
        std::cout << "Reading " << targetFileName << "..." << std::endl;
        // TODO: replace load file and tree to load tree only
        auto [tempfile, temptree] = treeops::loadFileAndTree(targetFileName);
        file = tempfile;
        tree = temptree;
    } else {
        std::cerr << "Error: File " << targetFileName << " does not exist!" << std::endl;
        return {nullptr, nullptr};
    }
    std::cout << "Loaded TTree from file" << std::endl;
    return {file, tree};
}

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

/**
 * @brief Plot the weighted vs unweighted histograms for a given branch.
 * 
 * @param tree The TTree containing the data.
 * @param targetBranch The name of the branch to plot.
 * @param weightBranch The name of the branch containing the weights.
 * @param outputDir The directory to save the plots.
 * @param nBins The number of bins for the histogram. Default is 100.
 * @param normalize Whether to normalize the histograms. Default is true.
 * @return void
 */
void PlotWeightedVSUnweighted(
    TTree* tree, 
    const std::string& targetBranch, 
    const std::string& weightBranch, 
    const std::string& outputDir,
    std::vector<double> range = {1, 0},
    int nBins = 100,
    bool normalize = true
) {
    std::cout << "Plotting weighted vs unweighted histograms for branch " << targetBranch << " using weights from branch " << weightBranch << std::endl;
    if (!check::CheckTreeBranches(tree, {targetBranch, weightBranch})) {
        return;
    }
    if (nBins <= 0) {
        std::cerr << "Error: Invalid number of bins: " << nBins << std::endl;
        return;
    }
    std::string histid = strops::randstr(4);
    std::cout << "Plot ID generated: " << histid << std::endl;

    std::cout << "Checking for variation in branch..." << std::endl;
    double min = tree->GetMinimum(targetBranch.c_str());
    double max = tree->GetMaximum(targetBranch.c_str());
    if (min == max) {
        std::cerr << "Error: Branch " << targetBranch << " has no variation." << std::endl;
        return;
    }
    std::cout << "Branch has variation in the range [" << min << ", " << max << "]" << std::endl;
    if (range[0] > range[1]) {
        std::cerr << "Error: Branch " << targetBranch << " has invalid range [" << range[0] << ", " << range[1] << "]" << std::endl;
        std::cout << "Using min and max values from the branch." << std::endl;
        range = {min, max};
    } else {
        min = range[0];
        max = range[1];
    }

    // Define histograms
    std::cout << "Creating histograms..." << std::endl;
    TH1F* histWeighted = new TH1F(("histWeighted" + histid).c_str(), ("Weighted Dist;" + targetBranch + histid).c_str(), nBins, min, max);
    TH1F* histUnweighted = new TH1F(("histUnweighted" + histid).c_str(), ("Unweighted Dist;" + targetBranch + histid).c_str(), nBins, min, max);
    std::cout << "Histograms created." << std::endl;

    // Fill the histograms using TTree::Draw
    std::cout << "Filling histograms..." << std::endl;
    tree->Draw((targetBranch + ">>histWeighted" + histid).c_str(), weightBranch.c_str(), "goff");
    tree->Draw((targetBranch + ">>histUnweighted" + histid).c_str(), "", "goff");
    std::cout << "Histograms filled." << std::endl;

    // Normalize histograms to compare shapes
    if (normalize) {
        std::cout << "Normalizing histograms..." << std::endl;
        histWeighted->Scale(1.0 / histWeighted->Integral());
        histUnweighted->Scale(1.0 / histUnweighted->Integral());
        std::cout << "Histograms normalized." << std::endl;
    }
    
    // Plot histograms
    std::cout << "Plotting histograms..." << std::endl;
    TCanvas* canvas = new TCanvas(("WvUcanvas_" + targetBranch + histid).c_str(), ("Weighted vs Unweighted (" + targetBranch + ")" + histid).c_str(), 1000, 700);
    histWeighted->SetLineColor(getColour(0));
    histUnweighted->SetLineColor(getColour(1));
    histWeighted->SetLineWidth(2);
    histUnweighted->SetLineWidth(2);
    histWeighted->Draw("HIST");
    histUnweighted->Draw("HIST SAME");
    // title
    histWeighted->SetTitle(("Weighted vs Unweighted " + targetBranch + " Distribution").c_str());
    // axes labels
    histWeighted->GetXaxis()->SetTitle(targetBranch.c_str());
    histWeighted->GetYaxis()->SetTitle("Events");
    // legend
    if (gStyle->GetOptStat() != 0) {
        gStyle->SetOptStat(0);
    }
    TLegend* legend = new TLegend(0.7, 0.7, 0.9, 0.9);
    legend->AddEntry(histWeighted, "Weighted Data", "l");
    legend->AddEntry(histUnweighted, "Unweighted Data", "l");
    legend->Draw();
    std::cout << "Histograms plotted." << std::endl;

    // Save plot
    std::string filename = outputDir + "sWeighted_" + targetBranch + ".png";
    std::cout << "Saving plot to " << filename << std::endl;
    canvas->Update();
    canvas->Modified();
    canvas->SaveAs(filename.c_str());
    gSystem->ProcessEvents();
    std::cout << "Plot saved." << std::endl;

    // Cleanup
    delete legend;
    delete canvas;
    delete histUnweighted;
    delete histWeighted;
    tree->ResetBranchAddresses();
    std::cout << std::endl;
};

/**
 * @brief Plot the weighted vs unweighted histograms for given branches. Because the branches are plotted together, branches must have the same quality and quantity of data.
 */
void PlotWeightedVSUnweighted(
    TTree* tree, 
    const std::vector<std::string>& targetBranches,
    const std::string& weightBranch,
    const std::string& outputDir,
    const std::string& givenRepresentativeName = "",
    std::vector<double> range = {1, 0},
    int nBins = 100,
    bool normalize = true
) {
    if (targetBranches.empty()) {
        std::cerr << "Error: No target branches provided." << std::endl;
        return;
    }
    if (!check::CheckTreeBranches(tree, targetBranches) || !check::CheckTreeBranches(tree, {weightBranch})) {
        return;
    }
    if (nBins <= 0) {
        std::cerr << "Error: Invalid number of bins: " << nBins << std::endl;
        return;
    }
    std::string representativeName = givenRepresentativeName;
    if (representativeName.empty()) {
        std::cout << "Empty representative name provided. Using the first branch name as the representative." << std::endl;
        representativeName = targetBranches[0];
    }
    
    std::cout << "Plotting weighted vs unweighted histograms for branches: " << givenRepresentativeName << " = {";
    for (const std::string& branch : targetBranches) {
        std::cout << branch << ", ";
    }
    std::cout << "} using weights from branch " << weightBranch << std::endl;
    std::string histid = strops::randstr(4);
    std::cout << "Plot ID generated: " << histid << std::endl;

    // TODO: some optimisation
    std::cout << "Checking for variation in branches..." << std::endl;
    auto tempRange = getRange(tree, targetBranches);
    double min = tempRange[0];
    double max = tempRange[1];
    if (min == max) {
        std::cerr << "Error: Branches " << representativeName << " have no variation." << std::endl;
        return;
    }
    std::cout << "Branches have variation in the range [" << min << ", " << max << "]" << std::endl;
    if (range[0] > range[1]) {
        std::cerr << "Error: Branches " << representativeName << " have invalid range [" << range[0] << ", " << range[1] << "]" << std::endl;
        std::cout << "Using min and max values from the branches." << std::endl;
        range = tempRange;
    } else {
        min = range[0];
        max = range[1];
    }

    // Define histograms
    std::cout << "Creating histograms..." << std::endl;
    TH1F* histWeighted = new TH1F(("histWeighted_" + representativeName + histid).c_str(), ("Weighted Dist;" + representativeName + histid).c_str(), nBins, min, max);
    TH1F* histUnweighted = new TH1F(("histUnweighted_" + representativeName + histid).c_str(), ("Unweighted Dist;" + representativeName + histid).c_str(), nBins, min, max);
    std::cout << "Histograms created." << std::endl;

    // Prepare storage for branch values
    std::cout << "Setting up branch values..." << std::endl;
    std::vector<Double_t> values(targetBranches.size());
    double weight;
    tree->ResetBranchAddresses();
    for (size_t i = 0; i < targetBranches.size(); ++i) {
        tree->SetBranchAddress(targetBranches[i].c_str(), &values[i]);
    }
    tree->SetBranchAddress(weightBranch.c_str(), &weight);
    std::cout << "Branch values set up." << std::endl;

    // Loop over tree entries and fill the histogram
    std::cout << "Filling histograms..." << std::endl;
    Long64_t nEntries = tree->GetEntries();
    for (Long64_t i = 0; i < nEntries; i++) {
        tree->GetEntry(i);
        for (double value : values) {
            histUnweighted->Fill(value);
            histWeighted->Fill(value, weight);
        }
    }
    std::cout << "Histograms filled." << std::endl;

    // Normalize histograms to compare shapes
    if (normalize) {
        std::cout << "Normalizing histograms..." << std::endl;
        histWeighted->Scale(1.0 / histWeighted->Integral());
        histUnweighted->Scale(1.0 / histUnweighted->Integral());
        std::cout << "Histograms normalized." << std::endl;
    }

    // Plot histograms
    std::cout << "Plotting histograms..." << std::endl;
    TCanvas* canvas = new TCanvas(("canvas_" + representativeName + histid).c_str(), ("Weighted vs Unweighted Histogram" + histid).c_str(), 1000, 700);
    histWeighted->SetLineColor(getColour(0));
    histUnweighted->SetLineColor(getColour(1));
    histWeighted->SetLineWidth(2);
    histUnweighted->SetLineWidth(2);
    histWeighted->Draw("HIST");
    histUnweighted->Draw("HIST SAME");
    // title
    histWeighted->SetTitle(("Weighted vs Unweighted " + representativeName + " Distribution").c_str());
    // axes labels
    histWeighted->GetXaxis()->SetTitle(representativeName.c_str());
    histWeighted->GetYaxis()->SetTitle("Events");
    // legend
    if (gStyle->GetOptStat() != 0) {
        gStyle->SetOptStat(0);
    }
    TLegend* legend = new TLegend(0.7, 0.7, 0.9, 0.9);
    legend->AddEntry(histWeighted, "Weighted Data", "l");
    legend->AddEntry(histUnweighted, "Unweighted Data", "l");
    legend->Draw();
    std::cout << "Histograms plotted." << std::endl;

    // Save plot
    std::string filename = outputDir + "sWeighted_" + representativeName + ".png";
    std::cout << "Saving plot to " << filename << std::endl;
    canvas->Update();
    canvas->Modified();
    canvas->SaveAs(filename.c_str());
    gSystem->ProcessEvents();
    std::cout << "Plot saved." << std::endl;

    // Cleanup
    delete legend;
    delete canvas;
    delete histUnweighted;
    delete histWeighted;
    tree->ResetBranchAddresses();
    std::cout << std::endl;
}

/**
 * @brief Plot a 2D histogram of two branches with weights.
 * 
 * @param tree The TTree containing the data.
 * @param targetBranchX The name of the X-axis branch.
 * @param targetBranchY The name of the Y-axis branch.
 * @param weightBranch The name of the branch containing the weights.
 * @param outputDir The directory to save the plots.
 * @param nBinsX The number of bins for the X-axis. Default is 100.
 * @param nBinsY The number of bins for the Y-axis. Default is 100.
 */
void PlotWeightd2dHist(
    TTree* tree,
    const std::string& targetBranchX,
    const std::string& targetBranchY,
    const std::string& weightBranch,
    const std::string& outputDir,
    int nBinsX = 100,
    int nBinsY = 100,
    std::vector<double> rangeX = {1, 0},
    std::vector<double> rangeY = {1, 0}
) {
    std::cout << "Plotting 2D weighted histogram for branches " << targetBranchX << " and " << targetBranchY << " using weights from branch " << weightBranch << std::endl;
    if (!check::CheckTreeBranches(tree, {targetBranchX, targetBranchY, weightBranch})) {
        return;
    }
    if (nBinsX <= 0 || nBinsY <= 0) {
        std::cerr << "Error: Invalid number of bins: " << nBinsX << " x " << nBinsY << std::endl;
        return;
    }
    std::string histid = strops::randstr(4);
    std::cout << "Plot ID generated: " << histid << std::endl;

    std::cout << "Checking for variation in branches..." << std::endl;
    double minX = tree->GetMinimum(targetBranchX.c_str());
    double maxX = tree->GetMaximum(targetBranchX.c_str());
    double minY = tree->GetMinimum(targetBranchY.c_str());
    double maxY = tree->GetMaximum(targetBranchY.c_str());
    if (minX == maxX) {
        std::cerr << "Error: Branch " << targetBranchX << " has no variation." << std::endl;
        return;
    }
    if (minY == maxY) {
        std::cerr << "Error: Branch " << targetBranchY << " has no variation." << std::endl;
        return;
    }
    std::cout << "Branches have variation in the range [" << minX << ", " << maxX << "] x [" << minY << ", " << maxY << "]" << std::endl;
    if (rangeX.size() != 2 || rangeY.size() != 2) {
        std::cout << "Invalid range provided; size 2 expected but got " << rangeX.size() << " x " << rangeY.size() << ". The range will be determined by the tree." << std::endl;
    } else if (rangeX[0] >= rangeX[1] || rangeY[0] >= rangeY[1]) {
        std::cout << "Invalid range provided (" << rangeX[0] << ", " << rangeX[1] << ") x (" << rangeY[0] << ", " << rangeY[1] << "). The range will be determined by the tree." << std::endl;
    } else {
        minX = rangeX[0];
        maxX = rangeX[1];
        minY = rangeY[0];
        maxY = rangeY[1];
        std::cout << "Using provided range (" << minX << ", " << maxX << ") x (" << minY << ", " << maxY << ")" << std::endl;
    }

    // Create 2D histogram
    std::cout << "Creating histograms..." << std::endl;
    TH2F* histWeighted = new TH2F(
        ("histWeighted" + histid).c_str(),
        ("Weighted " + targetBranchX + " vs " + targetBranchY + histid).c_str(),
        nBinsX, minX, maxX,
        nBinsY, minY, maxY
    );
    std::cout << "Histograms created." << std::endl;

    // Fill the histogram using TTree::Draw
    std::cout << "Filling histograms..." << std::endl;
    tree->ResetBranchAddresses();
    tree->Draw((targetBranchY + ":" + targetBranchX + ">>histWeighted" + histid).c_str(), weightBranch.c_str(), "goff");
    std::cout << "Histograms filled." << std::endl;

    // Create canvas
    std::cout << "Plotting histograms..." << std::endl;
    TCanvas* canvas = new TCanvas(("W2Dcanvas_" + targetBranchX + "_vs_" + targetBranchY + histid).c_str(), ("Weighted2D_" + targetBranchX + "_vs_" + targetBranchY + histid).c_str(), 1000, 700);
    canvas->SetRightMargin(0.15);  // Adjust margin for color bar
    histWeighted->Draw("COLZ");
    histWeighted->SetLineWidth(2);
    // title
    histWeighted->SetTitle(("sWeighted " + targetBranchX + " vs " + targetBranchY + " Distribution").c_str());
    // axes labels
    histWeighted->GetXaxis()->SetTitle(targetBranchX.c_str());
    histWeighted->GetYaxis()->SetTitle(targetBranchY.c_str());
    // Style settings
    if (gStyle->GetOptStat() != 0) {
        gStyle->SetOptStat(0);
    }
    gStyle->SetPalette(kBird);  // Adjust color palette
    std::cout << "Histograms plotted." << std::endl;

    // Save the plot
    std::string filename = outputDir + "sWeighted_" + targetBranchX + "_vs_" + targetBranchY + ".png";
    std::cout << "Saving weighted 2d histogram to " << filename << std::endl;
    canvas->Update();
    canvas->Modified();
    canvas->SaveAs(filename.c_str());
    gSystem->ProcessEvents();
    std::cout << "Weighted 2d histogram saved." << std::endl;

    // Cleanup
    delete canvas;
    delete histWeighted;
    tree->ResetBranchAddresses();
    std::cout << std::endl;
}

/**
 * @brief Plot a efficiency 1D histogram for a given branch. 1 histogram is plotted.
 * 
 * @param tree The TTree containing the data.
 * @param targetBranch The name of the branch to plot.
 * @param weightBranch The name of the branch containing the weights.
 * @param criteriaBranch The name of the branch containing the criteria.
 * @param criteriaValue The value of the criteria.
 * @param outputDir The directory to save the plots.
 * @param nBins The number of bins for the histogram. Default is 100.
 * @param range The range of the histogram. Default is the range of the tree.
 * @param higherIsBetter Whether values higher than the criteria are considered better. Default is true.
 * @param saturate Whether to saturate the efficiency values, i.e. limit them to [0, 1]. Default is true.
 */
void PlotEff1d(
    TTree* tree,
    const std::string& targetBranch,
    const std::string& weightBranch,
    const std::string& criteriaBranch,
    const double criteriaValue,
    const std::string& outputDir,
    int nBins = 100,
    std::vector<double> range = {1, 0},
    bool higherIsBetter = true,
    bool saturate = true
) {
    std::cout << "Plotting 1D efficiency histogram for branch " << targetBranch << " using weights from branch " << weightBranch << " and criteria from branch " << criteriaBranch << std::endl;
    if (!check::CheckTreeBranches(tree, {targetBranch, weightBranch, criteriaBranch})) {
        return;
    }
    if (nBins <= 0) {
        std::cerr << "Error: Invalid number of bins: " << nBins << std::endl;
        return;
    }
    std::string histid = strops::randstr(4);
    std::cout << "Plot ID generated: " << histid << std::endl;

    std::cout << "Checking for variation in branch..." << std::endl;
    double min = tree->GetMinimum(targetBranch.c_str());
    double max = tree->GetMaximum(targetBranch.c_str());
    if (min == max) {
        std::cerr << "Error: Branch " << targetBranch << " has no variation." << std::endl;
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
    TH1F* totalHist = new TH1F(("totalHist" + histid).c_str(), ("total_" + targetBranch + histid).c_str(), nBins, min, max);
    TH1F* passedHist = new TH1F(("passedHist" + histid).c_str(), ("passed_" + targetBranch + histid).c_str(), nBins, min, max);
    TH1F* effHist = new TH1F(("effHist" + histid).c_str(), ("eff_" + targetBranch + histid).c_str(), nBins, min, max);
    std::cout << "Histograms created." << std::endl;

    // Set up branches
    std::cout << "Setting up branches..." << std::endl;
    double value, weight, criteria;
    tree->ResetBranchAddresses();
    tree->SetBranchAddress(targetBranch.c_str(), &value);
    tree->SetBranchAddress(weightBranch.c_str(), &weight);
    tree->SetBranchAddress(criteriaBranch.c_str(), &criteria);
    std::cout << "Branches set up." << std::endl;
 
    // Loop through tree entries
    std::cout << "Filling histograms..." << std::endl;
    for (Long64_t i = 0; i < tree->GetEntries(); ++i) {
        tree->GetEntry(i);
        totalHist->Fill(value, weight);
        if (((criteria > criteriaValue) && higherIsBetter) || ((criteria < criteriaValue) && !higherIsBetter)) {
            passedHist->Fill(value, weight);
        }
    }
    std::cout << "Histograms filled." << std::endl;

    // Calculate efficiency
    effHist->Divide(passedHist, totalHist);
    if (saturate) {
        std::cout << "Clipping efficiency values..." << std::endl;
        effHist->SetMinimum(0);
        effHist->SetMaximum(1);
        std::cout << "Efficiency values saturated." << std::endl;
    }

    // Plot histograms
    std::cout << "Plotting histograms..." << std::endl;
    TCanvas* canvas = new TCanvas(("canvas_" + targetBranch + histid).c_str(), ("Eff1D_" + targetBranch + histid).c_str(), 1000, 700);
    effHist->Draw("HIST");
    effHist->SetLineWidth(2);
    // title
    effHist->SetTitle(("Efficiency " + targetBranch + " Distribution").c_str());
    // axes labels
    effHist->GetXaxis()->SetTitle(targetBranch.c_str());
    effHist->GetYaxis()->SetTitle("Efficiency");
    // Style settings
    if (gStyle->GetOptStat() != 0) {
        gStyle->SetOptStat(0);
    }
    std::cout << "Histograms plotted." << std::endl;

    // Save the plot
    std::string filename = outputDir + "eff1d_" + targetBranch + ".png";
    std::cout << "Saving efficiency plot to " << filename << std::endl;
    canvas->Update();
    canvas->Modified();
    canvas->SaveAs(filename.c_str());
    gSystem->ProcessEvents();
    std::cout << "Efficiency plot saved." << std::endl;

    // Cleanup
    delete canvas;
    delete effHist;
    delete passedHist;
    delete totalHist;
    tree->ResetBranchAddresses();
    std::cout << std::endl;
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
 * @example PlotEff1dsUltimate(tree, {{"K+_P"}, {"Pi0_P"}, {"K-1_P", "K-2_P"}, {"Pi+1_P", "Pi+2_P", "Pi+3_P"}}, "weight", "criteria", {5, 0, -5}, "output/", {"", "", "K-_P", "Pi+_P"}, "P", 100, {0, 1000}, true, true);
 * @example PlotEff1dsUltimate(tree, {{"K+_P"}}, "weight", "criteria", {5, 0, -5}, "output/"});  // plots one branch with auto-bin, auto-range
 * 
 */
void PlotEff1dsUltimate(
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
        std::cout << representativeNames[i] << " = {";
        for (const std::string& branch : targetBranchSets[i]) {
            std::cout << branch << ", ";
        }
        std::cout << "}, ";
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
            // for (int k = 1; k <= nBins; ++k) {
            //     double passed = passedHistsSets[i][j]->GetBinContent(k);
            //     double total = totalHists[i]->GetBinContent(k);
            //     double eff = total == 0 ? 0 : passed / total;
            //     effHistsSets[i][j]->SetBinContent(k, eff);
            //     double err = total == 0 ? 0 : std::sqrt(eff * (1 - eff) / total);
            //     effHistsSets[i][j]->SetBinError(k, err);
            // }
            // if (saturate) {
            //     effHistsSets[i][j]->SetMinimum(0);
            //     effHistsSets[i][j]->SetMaximum(1);
            // }
        }
    }
    std::cout << "Efficiencies calculated." << std::endl;

    // Plot histograms
    std::cout << "Plotting histograms..." << std::endl;
    TCanvas* canvas = new TCanvas(("canvas_" + representativeName + histid).c_str(), ("Eff1D_" + representativeName + histid).c_str(), 1000, 700);
    // effHistsSets[0][0]->Draw("HIST");
    effHistsSets[0][0]->Draw("E1");
    effHistsSets[0][0]->SetLineColor(getColour(0));
    effHistsSets[0][0]->SetLineWidth(2);
    TLegend* legend = new TLegend(0.1, 0.4, 0.3, 0.6);
    // std::vector<std::vector<std::string>> legends(targetBranchSets.size(), std::vector<std::string>(criteriaValues.size()));
    std::string sign = higherIsBetter ? ">" : "<";
    int histCount = 0;
    for (int i = 0; i < targetBranchSets.size(); ++i) {
        for (int j = 0; j < criteriaValues.size(); ++j) {
            // effHistsSets[i][j]->Draw("HIST SAME");
            effHistsSets[i][j]->Draw("E1 SAME");
            effHistsSets[i][j]->SetLineColor(getColour(i * criteriaValues.size() + j));
            effHistsSets[i][j]->SetLineWidth(2);
            // legends[i][j] = representativeNames[i] + " " + sign + " " + std::to_string(criteriaValues[j]);
            legend->AddEntry(effHistsSets[i][j], (representativeNames[i] + " : " + criteriaBranch + sign + " " + std::to_string(int(criteriaValues[j]))).c_str(), "l");
            ++histCount;
        }
    }
    // set y axis range from 0 to 1
    effHistsSets[0][0]->SetMinimum(0);
    effHistsSets[0][0]->SetMaximum(1);
    // title
    effHistsSets[0][0]->SetTitle("Efficiency Distribution");
    // axes labels
    effHistsSets[0][0]->GetXaxis()->SetTitle(representativeName.c_str());
    effHistsSets[0][0]->GetYaxis()->SetTitle("Efficiency");
    // Style settings
    if (gStyle->GetOptStat() != 0) {
        gStyle->SetOptStat(0);
    }
    if (histCount > 1) {
        legend->Draw();
    }
    std::cout << "Histograms plotted." << std::endl;

    // Save the plot
    filename = outputDir + "eff1d_" + filename + ".png";
    std::cout << "Saving efficiency plot to " << filename << std::endl;
    canvas->Update();
    canvas->Modified();
    canvas->SaveAs(filename.c_str());
    gSystem->ProcessEvents();
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

// recommended for cases where the targetBranchSets is a vector of vectors of size 1
void PlotEff1dsUltimate(
    TTree* tree,
    const std::vector<std::vector<std::string>>& targetBranchSets,
    const std::string& weightBranch,
    const std::string& criteriaBranch,
    const std::vector<double>& criteriaValues,
    const std::string& outputDir,
    int nBins = 100,
    std::vector<double> range = {1, 0},
    bool higherIsBetter = true,
    bool saturate = true,
    const std::string& givenFilename = ""
) {
    PlotEff1dsUltimate(tree, targetBranchSets, weightBranch, criteriaBranch, criteriaValues, outputDir, {}, "", nBins, range, higherIsBetter, saturate, givenFilename);
}

/**
 * @brief Plot the efficiency 2D histograms for a given branch.
 * 
 * @param tree The TTree containing the data.
 * @param targetBranchX The name of the X-axis branch.
 * @param targetBranchY The name of the Y-axis branch.
 * @param weightBranch The name of the branch containing the weights.
 * @param criteriaBranch The name of the branch containing the criteria.
 * @param criteriaValue The value of the criteria.
 * @param outputDir The directory to save the plots. The saved file will be named "eff2d_<targetBranchX>_vs_<targetBranchY>_<criteriaBranch>_<criteriaValue>.png" under this directory.
 * @param nBinsX The number of bins for the X-axis. Default is 100.
 * @param nBinsY The number of bins for the Y-axis. Default is 100.
 * @param xRange The range of the X-axis. Default is the range of the tree.
 * @param yRange The range of the Y-axis. Default is the range of the tree.
 * @param higherIsBetter Whether values higher than the criteria are considered better. Default is true.
 * @param saturate Whether to saturate the efficiency values, i.e. limiit them to [0, 1]. Default is false.
 * @return void
 */
void PlotEff2d(
    TTree* tree,
    const std::string& targetBranchX,
    const std::string& targetBranchY,
    const std::string& weightBranch,
    const std::string& criteriaBranch,
    const double criteriaValue,
    const std::string& outputDir,
    int nBinsX = 100,
    int nBinsY = 100,
    std::vector<double> xRange = {1, 0},
    std::vector<double> yRange = {1, 0},
    bool higherIsBetter = true,
    bool saturate = true,
    bool isEfficiency = false,
    const std::string& givenFilename = ""
) {
    std::cout << "Plotting 2D " << (isEfficiency? "": "in") << "efficiency histogram for branches " << targetBranchX << " and " << targetBranchY << " using weights from branch " << weightBranch << " and criteria from branch " << criteriaBranch << std::endl;
    if (!check::CheckTreeBranches(tree, {targetBranchX, targetBranchY, weightBranch, criteriaBranch})) {
        return;
    }
    if (nBinsX <= 0 || nBinsY <= 0) {
        std::cerr << "Error: Invalid number of bins: " << nBinsX << " x " << nBinsY << std::endl;
        return;
    }
    std::string filename = givenFilename;
    std::string histid = strops::randstr(4);
    std::cout << "Plot ID generated: " << histid << std::endl;

    // double minX, maxX, minY, maxY;
    std::cout << "Checking for variation in branches..." << std::endl;
    double minX = tree->GetMinimum(targetBranchX.c_str());
    double maxX = tree->GetMaximum(targetBranchX.c_str());
    double minY = tree->GetMinimum(targetBranchY.c_str());
    double maxY = tree->GetMaximum(targetBranchY.c_str());
    if (xRange.size() != 2) {
        std::cout << "xRange is invalid; size 2 expected but got " << xRange.size() << ". The range will be determined by the tree." << std::endl;
    } else if (xRange[0] >= xRange[1]) {
        std::cout << "Because the xRange is invalid (" << xRange[0] << ", " << xRange[1] <<"), the range will be determined by the tree." << std::endl;
    } else {
        std::cout << "Using provided xRange of (" << xRange[0] << ", " << xRange[1] << ")" << std::endl;
        minX = xRange[0];
        maxX = xRange[1];
    }
    if (minX == maxX) {
        std::cerr << "Error: Branch " << targetBranchX << " has no variation." << std::endl;
        return;
    }
    if (yRange.size() != 2) {
        std::cout << "yRange is invalid; size 2 expected but got " << yRange.size() << ". The range will be determined by the tree." << std::endl;
    } else if (yRange[0] >= yRange[1]) {
        std::cout << "Because the yRange is invalid (" << yRange[0] << ", " << yRange[1] <<"), the range will be determined by the tree." << std::endl;
    } else {
        std::cout << "Using provided yRange of (" << yRange[0] << ", " << yRange[1] << ")" << std::endl;
        minY = yRange[0];
        maxY = yRange[1];
    }
    if (minY == maxY) {
        std::cerr << "Error: Branch " << targetBranchY << " has no variation." << std::endl;
        return;
    }

    std::cout << "Creating histograms..." << std::endl;
    std::string inequal = higherIsBetter ? "_>_" : "_<_";
    std::string basicInfo = targetBranchX + "_vs_" + targetBranchY;
    std::string critInfo = criteriaBranch + inequal + std::to_string(int(criteriaValue));
    std::string plotTitle = (isEfficiency ? "Eff_" : "Ineff_") + targetBranchX + "_vs_" + targetBranchY + "_" + criteriaBranch + inequal + std::to_string(int(criteriaValue));
    std::cout << "Creating histograms with name: " << plotTitle << std::endl;
    TH2F* totalHist = new TH2F("totalHist", ("total_" + plotTitle).c_str(), nBinsX, minX, maxX, nBinsY, minY, maxY);
    TH2F* passedHist = new TH2F("passedHist", ("passed_" + plotTitle).c_str(), nBinsX, minX, maxX, nBinsY, minY, maxY);
    TH2F* effHist = new TH2F("effHist", ("eff_" + plotTitle).c_str(), nBinsX, minX, maxX, nBinsY, minY, maxY);
    std::cout << "Histograms created." << std::endl;

    // Set up branches
    std::cout << "Setting up branches..." << std::endl;
    double xVal, yVal, weight, criteria;
    tree->ResetBranchAddresses();
    tree->SetBranchAddress(targetBranchX.c_str(), &xVal);
    tree->SetBranchAddress(targetBranchY.c_str(), &yVal);
    tree->SetBranchAddress(weightBranch.c_str(), &weight);
    tree->SetBranchAddress(criteriaBranch.c_str(), &criteria);
    std::cout << "Branches set up." << std::endl;

    // Loop through tree entries
    std::cout << "Filling histograms..." << std::endl;
    for (Long64_t i = 0; i < tree->GetEntries(); ++i) {
        tree->GetEntry(i);
        totalHist->Fill(xVal, yVal, weight);
        if (((criteria > criteriaValue) && higherIsBetter) || ((criteria < criteriaValue) && !higherIsBetter)) {
            passedHist->Fill(xVal, yVal, weight);
        }
    }
    std::cout << "Histograms filled." << std::endl;

    // Calculate efficiency
    std::cout << "Calculating efficiency..." << std::endl;
    effHist->Divide(passedHist, totalHist);
    if (saturate) {
        std::cout << "Clipping (in)efficiency values..." << std::endl;
        effHist->SetMinimum(0);
        effHist->SetMaximum(1);
        std::cout << "(in)Efficiency values saturated." << std::endl;
    }
    // // lines below are for verbose / debug
    // for (Long64_t iX = 1; iX <= nBinsX; ++iX) {
    //     for (Long64_t iY = 1; iY <= nBinsY; ++iY) {
    //         double total = totalHist->GetBinContent(iX, iY);
    //         double passed = passedHist->GetBinContent(iX, iY);
    //         double var = (total == 0) ? 0 : (passed / total);
    //         var = (var > 1) ? 1 : ((var < 0) ? 0 : var);
    //         effHist->SetBinContent(iX, iY, var);
    //         std::cout << passed << "/" << total << "  ";
    //     }
    //     std::cout << std::endl;
    // }
    std::cout << "(in)Efficiency calculated." << std::endl;
    
    std::cout << "Creating canvas and plotting..." << std::endl;
    TCanvas* canvas = new TCanvas(("canvas_" + basicInfo + "_" + critInfo).c_str(), ("Eff2D_" + basicInfo + "_" + critInfo).c_str(), 1000, 700);
    canvas->SetRightMargin(0.15);  // Adjust margin for color bar
    effHist->Draw("COLZ");
    effHist->SetLineWidth(2);
    // title
    effHist->SetTitle(
        (
            (isEfficiency ? "E" : "Ine") + std::string("fficiency Distribution (") + 
            criteriaBranch + strops::replace(inequal, "_", " ") + std::to_string(int(criteriaValue)) + ")").c_str());
    // axes labels
    effHist->GetXaxis()->SetTitle(targetBranchX.c_str());
    effHist->GetYaxis()->SetTitle(targetBranchY.c_str());
    // Style settings
    if (gStyle->GetOptStat() != 0) {
        gStyle->SetOptStat(0);
    }
    gStyle->SetPalette(kViridis);
    // TColor::InvertPalette();
    std::cout << "Finished plotting on canvas." << std::endl;

    // Save the plot
    // std::string filename = outputDir + (isEfficiency ? "eff2d_" : "ineff2d_") + basicInfo + "_" + critInfo + ".png";
    if (filename.empty()) {
        filename = basicInfo + "_" + critInfo;
    }
    filename = outputDir + (isEfficiency ? "eff2d_" : "ineff2d_") + filename + ".png";
    std::cout << "Saving (in)efficiency plot to file: " << filename << std::endl;
    canvas->Update();
    canvas->Modified();
    canvas->SaveAs(filename.c_str());
    gSystem->ProcessEvents();
    std::cout << "(in)Efficiency plot saved to file." << std::endl;

    // Cleanup
    delete canvas;
    delete effHist;
    delete passedHist;
    delete totalHist;
    tree->ResetBranchAddresses();
    std::cout << std::endl;
}

/**
 * @brief Plot the efficiency 2D histograms for given branches. Because the branches are plotted together, branches must have the same quality and quantity of data.
 * 
 * @param tree The TTree containing the data.
 * @param targetBranchesX The names of the X-axis branches.
 * @param targetBranchesY The names of the Y-axis branches.
 * @param weightBranch The name of the branch containing the weights.
 * @param criteriaBranches The names of the branches containing the criteria.
 * @param criteriaValue The value of the criteria.
 * @param outputDir The directory to save the plots.
 * @param givenRepresentativeNameX The name of the representative branch for the X-axis. Default is the first branch name.
 * @param givenRepresentativeNameY The name of the representative branch for the Y-axis. Default is the first branch name.
 * @param givenRepresentativeNameCriteria The name of the representative branch for the criteria. Default is the first branch name.
 * @param nBinsX The number of bins for the X-axis. Default is 100.
 * @param nBinsY The number of bins for the Y-axis. Default is 100.
 * @param xRange The range of the X-axis. Default is the range of the tree.
 * @param yRange The range of the Y-axis. Default is the range of the tree.
 * @param higherIsBetter Whether values higher than the criteria are considered better. Default is true.
 * @param saturate Whether to saturate the efficiency values, i.e. limiit them to [0, 1]. Default is false.
 * @return void
 */
void PlotEff2d(
    TTree* tree,
    const std::vector<std::string>& targetBranchesX,
    const std::vector<std::string>& targetBranchesY,
    const std::string& weightBranch,
    const std::vector<std::string>& criteriaBranches,
    const double criteriaValue,
    const std::string& outputDir,
    const std::string& givenRepresentativeNameX = "",
    const std::string& givenRepresentativeNameY = "",
    const std::string& givenRepresentativeNameCriteria = "",
    int nBinsX = 100,
    int nBinsY = 100,
    std::vector<double> xRange = {1, 0},
    std::vector<double> yRange = {1, 0},
    bool higherIsBetter = true,
    bool saturate = true,
    bool isEfficiency = false
) {
    if (targetBranchesX.empty() || targetBranchesY.empty() || criteriaBranches.empty()) {
        std::cerr << "Error: No target branches provided." << std::endl;
        return;
    }
    std::string representativeNameX = givenRepresentativeNameX;
    if (representativeNameX.empty()) {
        std::cout << "Empty representative name provided for X. Using the first branch name as the representative." << std::endl;
        representativeNameX = targetBranchesX[0];
    }
    std::string representativeNameY = givenRepresentativeNameY;
    if (representativeNameY.empty()) {
        std::cout << "Empty representative name provided for Y. Using the first branch name as the representative." << std::endl;
        representativeNameY = targetBranchesY[0];
    }
    std::string representativeNameCriteria = givenRepresentativeNameCriteria;
    if (representativeNameCriteria.empty()) {
        std::cout << "Empty representative name provided for criteria. Using the first branch name as the representative." << std::endl;
        representativeNameCriteria = criteriaBranches[0];
    }
    if (targetBranchesX.size() != targetBranchesY.size() || targetBranchesX.size() != criteriaBranches.size()) {
        std::cerr << "Error: Branch vectors must have the same size." << std::endl
                    << representativeNameX << ": " << targetBranchesX.size() << std::endl
                    << representativeNameY << ": " << targetBranchesY.size() << std::endl
                    << representativeNameCriteria << ": " << criteriaBranches.size() << std::endl;
        return;
    }
    std::cout << "Plotting 2D (in)efficiency histogram for branches: " << givenRepresentativeNameX << " and " << givenRepresentativeNameY << " using weights from branch " << weightBranch << " and criteria from branch " << givenRepresentativeNameCriteria << std::endl;
    if (!check::CheckTreeBranches(tree, targetBranchesX) || !check::CheckTreeBranches(tree, targetBranchesY) || !check::CheckTreeBranches(tree, criteriaBranches) || !tree->GetBranch(weightBranch.c_str())) {
        return;
    }
    if (nBinsX <= 0 || nBinsY <= 0) {
        std::cerr << "Error: Invalid number of bins: " << nBinsX << " x " << nBinsY << std::endl;
        return;
    }
    std::string histid = strops::randstr(4);
    std::cout << "Plot ID generated: " << histid << std::endl;

    std::cout << "Checking for variation in branches..." << std::endl;
    double minX = tree->GetMinimum(targetBranchesX[0].c_str());
    double maxX = tree->GetMaximum(targetBranchesX[0].c_str());
    double minY = tree->GetMinimum(targetBranchesY[0].c_str());
    double maxY = tree->GetMaximum(targetBranchesY[0].c_str());
    if (xRange.size() != 2) {
        std::cout << "xRange is invalid; size 2 expected but got " << xRange.size() << ". The range will be determined by the tree." << std::endl;
    } else if (xRange[0] >= xRange[1]) {
        std::cout << "Because the xRange is invalid (" << xRange[0] << ", " << xRange[1] <<"), the range will be determined by the tree." << std::endl;
    } else {
        std::cout << "Using provided xRange of (" << xRange[0] << ", " << xRange[1] << ")" << std::endl;
        minX = xRange[0];
        maxX = xRange[1];
    }
    if (minX == maxX) {
        std::cerr << "Error: Branch " << targetBranchesX[0] << " has no variation." << std::endl;
        return;
    }
    if (yRange.size() != 2) {
        std::cout << "yRange is invalid; size 2 expected but got " << yRange.size() << ". The range will be determined by the tree." << std::endl;
    } else if (yRange[0] >= yRange[1]) {
        std::cout << "Because the yRange is invalid (" << yRange[0] << ", " << yRange[1] <<"), the range will be determined by the tree." << std::endl;
    } else {
        std::cout << "Using provided yRange of (" << yRange[0] << ", " << yRange[1] << ")" << std::endl;
        minY = yRange[0];
        maxY = yRange[1];
    }
    if (minY == maxY) {
        std::cerr << "Error: Branch " << targetBranchesY[0] << " has no variation." << std::endl;
        return;
    }

    std::cout << "Generating histograms..." << std::endl;
    // std::string plotTitle = representativeNameX + " vs " + representativeNameY + "; " + representativeNameX + "; " + representativeNameY;
    std::string plotTitle = representativeNameX + "_vs_" + representativeNameY + "_" + representativeNameCriteria + "_" + std::to_string(int(criteriaValue));
    TH2F* totalHist = new TH2F("totalHist", plotTitle.c_str(), nBinsX, minX, maxX, nBinsY, minY, maxY);
    TH2F* passedHist = new TH2F("passedHist", plotTitle.c_str(), nBinsX, minX, maxX, nBinsY, minY, maxY);
    TH2F* effHist = new TH2F("effHist", plotTitle.c_str(), nBinsX, minX, maxX, nBinsY, minY, maxY);
    std::cout << "Histograms created." << std::endl;

    // Prepare storage for branch values
    std::cout << "Setting up branch values..." << std::endl;
    std::vector<double> xValues(targetBranchesX.size());
    std::vector<double> yValues(targetBranchesY.size());
    double weight;
    std::vector<double> criteriaValues(criteriaBranches.size());

    tree->ResetBranchAddresses();
    for (size_t i = 0; i < targetBranchesX.size(); ++i) {
        tree->SetBranchAddress(targetBranchesX[i].c_str(), &xValues[i]);
    }
    for (size_t i = 0; i < targetBranchesY.size(); ++i) {
        tree->SetBranchAddress(targetBranchesY[i].c_str(), &yValues[i]);
    }
    tree->SetBranchAddress(weightBranch.c_str(), &weight);
    for (size_t i = 0; i < criteriaBranches.size(); ++i) {
        tree->SetBranchAddress(criteriaBranches[i].c_str(), &criteriaValues[i]);
    }
    std::cout << "Branch values set up." << std::endl;

    // Loop over tree entries and fill the histogram
    std::cout << "Filling histograms..." << std::endl;
    Long64_t nEntries = tree->GetEntries();
    // for ith entry,
    //    for jth (x_ij, y_ij, criteria_ij) set,
    //        fill the total histogram with weight_i
    //        if criteria_ij is satisfied, fill the passed histogram with weight_i
    for (Long64_t i = 0; i < nEntries; i++) {
        tree->GetEntry(i);
        for (size_t j = 0; j < targetBranchesX.size(); j++) {
            // std::cout << "x:" << xValues[j] << " y:" << yValues[j] << " criteria:" << criteriaValues[j] << " weight:" << weight << std::endl;
            totalHist->Fill(xValues[j], yValues[j], weight);
            if (((criteriaValues[j] > criteriaValue) && higherIsBetter) || ((criteriaValues[j] < criteriaValue) && !higherIsBetter)) {
                passedHist->Fill(xValues[j], yValues[j], weight);
            }
        }
    }
    std::cout << "Histograms filled." << std::endl;

    // Calculate efficiency
    std::cout << "Calculating (in)efficiency..." << std::endl;
    effHist->Divide(passedHist, totalHist);
    // // lines below are for verbose / debug
    // for (Long64_t iX = 1; iX <= nBinsX; ++iX) {
    //     for (Long64_t iY = 1; iY <= nBinsY; ++iY) {
    //         double total = totalHist->GetBinContent(iX, iY);
    //         double passed = passedHist->GetBinContent(iX, iY);
    //         double var = (total == 0) ? 0 : (passed / total);
    //         effHist->SetBinContent(iX, iY, var);
    //         std::cout << passed << "/" << total << "  ";
    //     }
    //     std::cout << std::endl;
    // }
    std::cout << "(in)Efficiency calculated." << std::endl;

    if (saturate) {
        std::cout << "Clipping (in)efficiency values..." << std::endl;
        effHist->SetMinimum(0);
        effHist->SetMaximum(1);
        std::cout << "(in)Efficiency values saturated." << std::endl;
    }

    // Plot histograms
    std::cout << "Creating canvas and plotting..." << std::endl;
    TCanvas* canvas = new TCanvas(("canvas_" + representativeNameX + "_vs_" + representativeNameY).c_str(), "2D (in)Efficiency", 1000, 700);
    canvas->SetRightMargin(0.15);  // Adjust margin for color bar
    effHist->Draw("COLZ");
    effHist->SetLineWidth(2);
    // title
    std::string inequality = higherIsBetter ? " > " : " < ";
    // effHist->SetTitle(("Efficiency " + representativeNameX + " vs " + representativeNameY + "(" + representativeNameCriteria + " " + inequality + " " + std::to_string(int(criteriaValue)) + ")").c_str());
    effHist->SetTitle((
        (isEfficiency ? "Efficiency " : "Inefficiency ") + std::string("Distribution (") + 
        representativeNameCriteria + inequality + std::to_string(int(criteriaValue)) + ")").c_str());
    // axes labels
    effHist->GetXaxis()->SetTitle(representativeNameX.c_str());
    effHist->GetYaxis()->SetTitle(representativeNameY.c_str());
    // Style settings
    if (gStyle->GetOptStat() != 0) {
        gStyle->SetOptStat(0);
    }
    gStyle->SetPalette(kViridis);
    // TColor::InvertPalette();
    std::cout << "Finished plotting on canvas." << std::endl;

    // Save the plot
    std::string filename = outputDir + (isEfficiency ? "eff2d_": "ineff2d_") + representativeNameX + "_vs_" + representativeNameY + "_" + representativeNameCriteria + inequality + std::to_string(int(criteriaValue)) + ".png";
    std::cout << "Saving (in)efficiency plot to file: " << filename << std::endl;
    canvas->Update();
    canvas->Modified();
    canvas->SaveAs(filename.c_str());
    gSystem->ProcessEvents();
    std::cout << "(in)Efficiency plot saved to file." << std::endl;

    // Cleanup
    delete canvas;
    delete effHist;
    delete passedHist;
    delete totalHist;
    tree->ResetBranchAddresses();
    std::cout << std::endl;
}

// signal-to-background ratio
void PlotSBR2d(
    TTree* tree,
    const std::string& targetBranchX,
    const std::string& targetBranchY,
    const std::string& weightBranch,
    // const std::string& criteriaBranch,
    // const double criteriaValue,
    const std::string& outputDir,
    int nBinsX = 100,
    int nBinsY = 100,
    std::vector<double> xRange = {1, 0},
    std::vector<double> yRange = {1, 0},
    // bool higherIsBetter = true,
    bool saturate = true,
    // bool isEfficiency = false,
    const std::string& givenFilename = ""
) {
    // std::cout << "Plotting 2D " << (isEfficiency? "": "in") << "efficiency histogram for branches " << targetBranchX << " and " << targetBranchY << " using weights from branch " << weightBranch << " and criteria from branch " << criteriaBranch << std::endl;
    std::cout << "Plotting 2D SBR histogram for branches " << targetBranchX << " and " << targetBranchY << " using weights from branch " << weightBranch << std::endl;
    // if (!check::CheckTreeBranches(tree, {targetBranchX, targetBranchY, weightBranch, criteriaBranch})) {
    if (!check::CheckTreeBranches(tree, {targetBranchX, targetBranchY, weightBranch})) {
        return;
    }
    if (nBinsX <= 0 || nBinsY <= 0) {
        std::cerr << "Error: Invalid number of bins: " << nBinsX << " x " << nBinsY << std::endl;
        return;
    }
    std::string filename = givenFilename;
    std::string histid = strops::randstr(4);
    std::cout << "Plot ID generated: " << histid << std::endl;

    // double minX, maxX, minY, maxY;
    std::cout << "Checking for variation in branches..." << std::endl;
    double minX = tree->GetMinimum(targetBranchX.c_str());
    double maxX = tree->GetMaximum(targetBranchX.c_str());
    double minY = tree->GetMinimum(targetBranchY.c_str());
    double maxY = tree->GetMaximum(targetBranchY.c_str());
    if (xRange.size() != 2) {
        std::cout << "xRange is invalid; size 2 expected but got " << xRange.size() << ". The range will be determined by the tree." << std::endl;
    } else if (xRange[0] >= xRange[1]) {
        std::cout << "Because the xRange is invalid (" << xRange[0] << ", " << xRange[1] <<"), the range will be determined by the tree." << std::endl;
    } else {
        std::cout << "Using provided xRange of (" << xRange[0] << ", " << xRange[1] << ")" << std::endl;
        minX = xRange[0];
        maxX = xRange[1];
    }
    if (minX == maxX) {
        std::cerr << "Error: Branch " << targetBranchX << " has no variation." << std::endl;
        return;
    }
    if (yRange.size() != 2) {
        std::cout << "yRange is invalid; size 2 expected but got " << yRange.size() << ". The range will be determined by the tree." << std::endl;
    } else if (yRange[0] >= yRange[1]) {
        std::cout << "Because the yRange is invalid (" << yRange[0] << ", " << yRange[1] <<"), the range will be determined by the tree." << std::endl;
    } else {
        std::cout << "Using provided yRange of (" << yRange[0] << ", " << yRange[1] << ")" << std::endl;
        minY = yRange[0];
        maxY = yRange[1];
    }
    if (minY == maxY) {
        std::cerr << "Error: Branch " << targetBranchY << " has no variation." << std::endl;
        return;
    }

    std::cout << "Creating histograms..." << std::endl;
    // std::string inequal = higherIsBetter ? "_>_" : "_<_";
    std::string basicInfo = targetBranchX + "_vs_" + targetBranchY;
    // std::string critInfo = criteriaBranch + inequal + std::to_string(int(criteriaValue));
    // std::string plotTitle = (isEfficiency ? "Eff_" : "Ineff_") + targetBranchX + "_vs_" + targetBranchY + "_" + criteriaBranch + inequal + std::to_string(int(criteriaValue));
    std::string plotTitle = "SBR_" + targetBranchX + "_vs_" + targetBranchY;
    std::cout << "Creating histograms with name: " << plotTitle << std::endl;
    TH2F* bkgHist = new TH2F("bkgHist", ("bkg_" + plotTitle).c_str(), nBinsX, minX, maxX, nBinsY, minY, maxY);
    TH2F* sigHist = new TH2F("sigHist", ("sig_" + plotTitle).c_str(), nBinsX, minX, maxX, nBinsY, minY, maxY);
    TH2F* sbrHist = new TH2F("sbrHist", ("sbr_" + plotTitle).c_str(), nBinsX, minX, maxX, nBinsY, minY, maxY);
    std::cout << "Histograms created." << std::endl;

    // Set up branches
    std::cout << "Setting up branches..." << std::endl;
    // double xVal, yVal, weight, criteria;
    double xVal, yVal, weight;
    tree->ResetBranchAddresses();
    tree->SetBranchAddress(targetBranchX.c_str(), &xVal);
    tree->SetBranchAddress(targetBranchY.c_str(), &yVal);
    tree->SetBranchAddress(weightBranch.c_str(), &weight);
    // tree->SetBranchAddress(criteriaBranch.c_str(), &criteria);
    std::cout << "Branches set up." << std::endl;

    // Loop through tree entries
    std::cout << "Filling histograms..." << std::endl;
    for (Long64_t i = 0; i < tree->GetEntries(); ++i) {
        tree->GetEntry(i);
        bkgHist->Fill(xVal, yVal, 1-weight);
        sigHist->Fill(xVal, yVal, weight);
        // totalHist->
        // signalHist->Fill(xVal, yVal, weight);
        // if (((criteria > criteriaValue) && higherIsBetter) || ((criteria < criteriaValue) && !higherIsBetter)) {
        //     passedHist->Fill(xVal, yVal, weight);
        // }
    }
    std::cout << "Histograms filled." << std::endl;

    // Calculate efficiency
    std::cout << "Calculating efficiency..." << std::endl;
    // effHist->Divide(passedHist, totalHist);
    sbrHist->Divide(sigHist, bkgHist);
    if (saturate) {
        std::cout << "Clipping SBR values..." << std::endl;
        sbrHist->SetMinimum(-25);
        // sbrHist->SetMaximum(25);
        std::cout << "SBR values saturated." << std::endl;
    }
    // // lines below are for verbose / debug
    // for (Long64_t iX = 1; iX <= nBinsX; ++iX) {
    //     for (Long64_t iY = 1; iY <= nBinsY; ++iY) {
    //         double total = totalHist->GetBinContent(iX, iY);
    //         double passed = passedHist->GetBinContent(iX, iY);
    //         double var = (total == 0) ? 0 : (passed / total);
    //         var = (var > 1) ? 1 : ((var < 0) ? 0 : var);
    //         effHist->SetBinContent(iX, iY, var);
    //         std::cout << passed << "/" << total << "  ";
    //     }
    //     std::cout << std::endl;
    // }
    std::cout << "(in)Efficiency calculated." << std::endl;
    
    std::cout << "Creating canvas and plotting..." << std::endl;
    // TCanvas* canvas = new TCanvas(("canvas_" + basicInfo + "_" + critInfo).c_str(), ("Eff2D_" + basicInfo + "_" + critInfo).c_str(), 1000, 700);
    TCanvas* canvas = new TCanvas(("canvas_" + basicInfo).c_str(), ("SBR2D_" + basicInfo).c_str(), 1000, 700);
    canvas->SetRightMargin(0.15);  // Adjust margin for color bar
    // effHist->Draw("COLZ");
    sbrHist->Draw("COLZ");
    sbrHist->SetLineWidth(2);
    // title
    sbrHist->SetTitle("Signal-to-Background Ratio Distribution");
    // effHist->SetTitle(
    //     (
    //         (isEfficiency ? "E" : "Ine") + std::string("fficiency Distribution (") + 
    //         criteriaBranch + strops::replace(inequal, "_", " ") + std::to_string(int(criteriaValue)) + ")").c_str());
    // axes labels
    sbrHist->GetXaxis()->SetTitle(targetBranchX.c_str());
    sbrHist->GetYaxis()->SetTitle(targetBranchY.c_str());
    // Style settings
    if (gStyle->GetOptStat() != 0) {
        gStyle->SetOptStat(0);
    }
    gStyle->SetPalette(kViridis);
    // TColor::InvertPalette();
    std::cout << "Finished plotting on canvas." << std::endl;

    // Save the plot
    // std::string filename = outputDir + (isEfficiency ? "eff2d_" : "ineff2d_") + basicInfo + "_" + critInfo + ".png";
    // std::string filename = givenFilename;
    if (filename.empty()) {
        filename = "sbr2d_" + basicInfo;
    }
    // if (filename.empty()) {
    //     filename = basicInfo + "_" + critInfo;
    // }
    // filename = outputDir + (isEfficiency ? "eff2d_" : "ineff2d_") + filename + ".png";
    filename = outputDir + filename + ".png";
    // std::cout << "Saving (in)efficiency plot to file: " << filename << std::endl;
    std::cout << "Saving SBR plot to file: " << filename << std::endl;
    canvas->Update();
    canvas->Modified();
    canvas->SaveAs(filename.c_str());
    gSystem->ProcessEvents();
    // std::cout << "(in)Efficiency plot saved to file." << std::endl;
    std::cout << "SBR plot saved to file." << std::endl;

    // Cleanup
    delete canvas;
    delete sbrHist;
    delete sigHist;
    delete bkgHist;
    tree->ResetBranchAddresses();
    std::cout << std::endl;
}


void utils() {
    std::cout << "main.cpp is not meant to be run as a standalone program." << std::endl;
}
