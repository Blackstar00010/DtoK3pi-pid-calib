#include "utils.cpp"
#include "TGraph.h"
#include "TLine.h"

// TODO: move this to common.cpp
void printProgress(int progress, int total) {
    std::cout << "[";
    for (int i = 0; i < progress; i++) {
        std::cout << "=";
    }
    for (int i = progress; i < total; i++) {
        std::cout << " ";
    }
    std::cout << "] " << progress << "/" << total << std::endl;
}

// TODO: move this to utils.cpp
void PlotROCOld(
    TTree* tree,
    const std::vector<std::string>& criteriaBranchesX,
    const std::vector<std::string>& criteriaBranchesY,
    const std::string& weightBranch,
    const std::string& outputDir,
    const std::string& givenRepresentativeNameX = "",
    const std::string& givenRepresentativeNameY = "",
    std::vector<double> criteriaRange = {1, 0},
    int resolution = 100
) {
    if (criteriaBranchesX.empty() || criteriaBranchesY.empty()) {
        std::cerr << "Error: No target branches provided." << std::endl;
        return;
    }
    if (resolution <= 0) {
        std::cerr << "Error: Invalid resolution: " << resolution << std::endl;
        return;
    }
    if (!check::CheckTreeBranches(tree, criteriaBranchesX) || !check::CheckTreeBranches(tree, criteriaBranchesY) || !check::CheckTreeBranches(tree, {weightBranch})) {
        return;
    }
    std::string representativeNameX = givenRepresentativeNameX;
    if (representativeNameX.empty()) {
        std::cout << "Empty representative name provided for X. Using the first branch name as the representative." << std::endl;
        representativeNameX = criteriaBranchesX[0];
    }
    std::string representativeNameY = givenRepresentativeNameY;
    if (representativeNameY.empty()) {
        std::cout << "Empty representative name provided for Y. Using the first branch name as the representative." << std::endl;
        representativeNameY = criteriaBranchesY[0];
    }
    std::string histid = strops::randstr(4);
    std::cout << "Plot ID generated: " << histid << std::endl;

    // Find the range of the criteria
    std::cout << "Checking for variation in branches..." << std::endl;
    double min = tree->GetMinimum(criteriaBranchesX[0].c_str());
    double max = tree->GetMaximum(criteriaBranchesX[0].c_str());
    for (const std::string& branch : criteriaBranchesX) {
        double tempMin = tree->GetMinimum(branch.c_str());
        double tempMax = tree->GetMaximum(branch.c_str());
        if (tempMin < min) {
            min = tempMin;
        }
        if (tempMax > max) {
            max = tempMax;
        }
    }
    for (const std::string& branch : criteriaBranchesY) {
        double tempMin = tree->GetMinimum(branch.c_str());
        double tempMax = tree->GetMaximum(branch.c_str());
        if (tempMin < min) {
            min = tempMin;
        }
        if (tempMax > max) {
            max = tempMax;
        }
    }
    if (min == max) {
        std::cerr << "Error: Branches " << representativeNameX << " and " << representativeNameY << " have no variation." << std::endl;
        return;
    }

    // Prepare storage for branch values
    std::cout << "Setting up branch values..." << std::endl;
    std::vector<double> xValues(criteriaBranchesX.size());
    std::vector<double> yValues(criteriaBranchesY.size());
    double weight;
    double criteriaValue;

    tree->ResetBranchAddresses();
    for (size_t i = 0; i < criteriaBranchesX.size(); ++i) {
        tree->SetBranchAddress(criteriaBranchesX[i].c_str(), &xValues[i]);
    }
    for (size_t i = 0; i < criteriaBranchesY.size(); ++i) {
        tree->SetBranchAddress(criteriaBranchesY[i].c_str(), &yValues[i]);
    }
    tree->SetBranchAddress(weightBranch.c_str(), &weight);
    std::cout << "Branch values set up." << std::endl;

    // Loop over tree entries and fill the histogram
    std::cout << "Calculating efficiencies..." << std::endl;
    Long64_t nEntries = tree->GetEntries();
    std::vector<double> xEffs(resolution);
    std::vector<double> yEffs(resolution);
    for (int i = 0; i < resolution; i++) {
        double criteria = min + (max - min) * i / resolution;
        double xEff = 0;
        double yEff = 0;
        double xTotal = 0;
        double yTotal = 0;
        for (Long64_t j = 0; j < nEntries; j++) {
            tree->GetEntry(j);
            for (size_t k = 0; k < criteriaBranchesX.size(); ++k) {
                if (xValues[k] > criteria) {
                    xEff += weight;
                }
                xTotal += weight;
            }
            for (size_t k = 0; k < criteriaBranchesY.size(); ++k) {
                if (yValues[k] > criteria) {
                    yEff += weight;
                }
                yTotal += weight;
            }
        }
        // xEff /= (nEntries * criteriaBranchesX.size());
        // yEff /= (nEntries * criteriaBranchesY.size());
        xEffs[i] = xEff / xTotal;
        yEffs[i] = yEff / yTotal;
        printProgress(i + 1, resolution);
    }

    // Plot the ROC curve (line plot)
    std::cout << "Plotting ROC curve..." << std::endl;
    TCanvas* canvas = new TCanvas(("canvas_" + representativeNameX + "_" + representativeNameY + histid).c_str(), ("ROC Curve (" + representativeNameX + " vs " + representativeNameY + ")" + histid).c_str(), 1000, 700);
    TGraph* graph = new TGraph(resolution, xEffs.data(), yEffs.data());
    graph->SetTitle(("ROC Curve; " + representativeNameX + " efficiency; " + representativeNameY + " efficiency").c_str());
    graph->SetMarkerStyle(20);
    graph->SetMarkerSize(0.5);
    graph->SetMarkerColor(kBlue);
    graph->SetLineColor(kBlue);
    graph->SetLineWidth(2);
    // graph->GetXaxis()->SetRangeUser(0.9, 1e0);
    // graph->GetYaxis()->SetRangeUser(0.9, 1e0);
    graph->Draw("AL");  // axis, points, line

    // draw y=x line
    TLine* line = new TLine(0, 0, 1, 1);
    line->SetLineColor(kRed);
    line->SetLineWidth(2);
    line->Draw("same");

    canvas->SaveAs((outputDir + "/roc_old_" + representativeNameX + "_" + representativeNameY + ".png").c_str());
    std::cout << "ROC curve plotted." << std::endl;

    // Clean up
    delete canvas;
    delete graph;
    tree->ResetBranchAddresses();
    std::cout << std::endl;
}

/**
 * @brief Sort a vector by another vector. For example, if target={1, 2, 3} and standard={3, 1, 2}, the result will be {2, 3, 1}.
 */
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

// return {cut, ROC val at the cut}
std::vector<std::vector<double>> calc1dROC(std::vector<double> values, std::vector<double> weights) {
    weights = sortVectorByAnotherVector(weights, values);
    values = sortVectorByAnotherVector(values, values);
    double sumWeights = 0;
    for (double weight : weights) {
        sumWeights += weight;
    }
    std::vector<std::vector<double>> result;
    double eff = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        double cut = values[i];
        eff += weights[i] / sumWeights;
        result.push_back({cut, eff});
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
    }
    return result;
}

std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix) {
    std::vector<std::vector<double>> result(matrix[0].size(), std::vector<double>(matrix.size()));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

void PlotROC(
    TTree* tree,
    const std::string& criteriaBranchX,
    const std::string& criteriaBranchY,
    const std::string& weightBranch,
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
        // if (roc[i][0] < 0.9) {
        //     continue;
        // }
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

    // Plot the ROC curve (line plot)
    std::cout << "Plotting ROC curve..." << std::endl;
    TCanvas* canvas = new TCanvas(("canvas_" + criteriaBranchX + "_" + criteriaBranchY + histid).c_str(), ("ROC Curve (" + criteriaBranchX + " vs " + criteriaBranchY + ")" + histid).c_str(), 1000, 700);
    TGraph* graph = new TGraph(xEffs.size(), xEffs.data(), yEffs.data());
    graph->SetTitle(("ROC Curve; " + criteriaBranchX + " efficiency; " + criteriaBranchY + " efficiency").c_str());
    // graph->SetMarkerStyle(20);
    // remove market and leave line only
    graph->SetMarkerStyle(0);
    graph->SetMarkerSize(0.5);
    graph->SetMarkerColor(kBlue);
    graph->SetLineColor(kBlue);
    canvas->SetLogy();
    // graph->GetXaxis()->SetRangeUser(0.9, 1e0);
    // graph->GetYaxis()->SetRangeUser(0.9, 1e0);
    graph->SetLineWidth(2);
    graph->Draw("AL");  // axis, points, line
    canvas->SaveAs((outputDir + "/roc_" + criteriaBranchX + "_" + criteriaBranchY + ".png").c_str());
    std::cout << "ROC curve plotted." << std::endl;
    
    // Clean up
    delete canvas;
    delete graph;
    tree->ResetBranchAddresses();
    delete cutTree;
    std::cout << std::endl;
}


int main() {
    auto [file, tree] = loadSweight();
    if (!file || !tree) {
        return 1;
    }
    std::string compHistOutDir = config::plot_dir5(commonVars::modelName, commonVars::cut);
    std::string weightCol = commonVars::weightCol;

    // PlotROC(tree, {"K_PID_K"}, {"pi1_PID_K", "pi2_PID_K", "pi3_PID_K"}, weightCol, compHistOutDir, "K_PID_K", "pi_PID_K", {1, 0}, 20);
    PlotROC(tree, "K_PID_K", "pi1_PID_K", weightCol, compHistOutDir, {1, 0}, 100);
    PlotROC(tree, "K_PID_K", "pi2_PID_K", weightCol, compHistOutDir, {1, 0}, 100);
    PlotROC(tree, "K_PID_K", "pi3_PID_K", weightCol, compHistOutDir, {1, 0}, 100);
    PlotROCOld(tree, {"K_PID_K"}, {"pi1_PID_K"}, weightCol, compHistOutDir);  // old version
    PlotROCOld(tree, {"K_PID_K"}, {"pi2_PID_K"}, weightCol, compHistOutDir);  // old version
    PlotROCOld(tree, {"K_PID_K"}, {"pi3_PID_K"}, weightCol, compHistOutDir);  // old version

    // Cleanup
    file->Close();
    return 0;
}
void plotROC() {
    main();
}