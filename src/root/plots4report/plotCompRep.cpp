#include <iostream>
#include <vector>
#include <string>
#include <TTree.h>
#include "../utils/fileio.cpp"
#include "../sweight_plots/utils.cpp"
#include "consts.cpp"
// #include "../utils/check.cpp"  # included in utils.cpp
// #include "../utils/treeops.cpp"
// #include "../utils/strops.cpp"

// comparisons with marcelo's data

void print2Dvector(const std::vector<std::vector<double>>& data) {
    for (const auto& row : data) {
        for (const auto& cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Sum all elements in a 2D vector
 * 
 * @param data The 2D vector to sum
 * @return double The sum of all elements
 */
double sum2Dvector(const std::vector<std::vector<double>>& data) {
    double sum = 0.0;
    for (const auto& row : data) {
        for (const auto& cell : row) {
            sum += cell;
        }
    }
    return sum;
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
 * @brief Generate a 2D histogram from a TTree. The data's entry can be accessed by `data[x_index][y_index]`
 * 
 * @param tree The TTree to generate the data from
 * @param branches The branches to use as the x and y axes
 * @param weightBranch The branch to use as the weight
 * @param nBins The number of bins in the x and y directions
 * @param ranges The ranges of the x and y directions
 * @return std::vector<std::vector<double>> The 2D histogram data
 */
std::vector<std::vector<double>> generateSigYieldData(
    TTree* tree,
    const std::vector<std::string>& branches,
    const std::string& weightBranch,
    std::vector<int> nBins,
    std::vector<std::vector<double>> ranges 
) {
    std::string branchX = branches[0];
    std::string branchY = branches[1];
    if (!check::CheckTreeBranches(tree, {branchX, branchY, weightBranch})) {
        std::cerr << "Error: Branches not found!" << std::endl;
        return {{0.0}};
    }
    std::vector<std::vector<double>> data(nBins[0], std::vector<double>(nBins[1], 0.0));

    double x, y, w;
    tree->ResetBranchAddresses();
    tree->SetBranchAddress(branchX.c_str(), &x);
    tree->SetBranchAddress(branchY.c_str(), &y);
    tree->SetBranchAddress(weightBranch.c_str(), &w);
    
    double xBinWidth = (ranges[0][1] - ranges[0][0]) / nBins[0];
    double yBinWidth = (ranges[1][1] - ranges[1][0]) / nBins[1];
    for (int i = 0; i < tree->GetEntries(); i++) {
        tree->GetEntry(i);
        // if x \notin [xmin, xmax] or y \notin [ymin, ymax], skip
        if (x < ranges[0][0] || x > ranges[0][1] || y < ranges[1][0] || y > ranges[1][1]) {
            continue;
        }
        // if (x-xMin) = xWidth * (n + 0.xx), then fill in the (n+1)-th bin, or the bin with index n
        int xBin = (x - ranges[0][0]) / xBinWidth;
        int yBin = (y - ranges[1][0]) / yBinWidth;
        if (xBin > data.size() || yBin > data[0].size()) {
            // just in case
            std::cout << "There was an error with the binning" << std::endl;
            std::cout << "x: " << x << " y: " << y << std::endl;
            std::cout << "nBins[0]: " << nBins[0] << " nBins[1]: " << nBins[1] << std::endl;
            std::cout << "ranges[0][0]: " << ranges[0][0] << " ranges[0][1]: " << ranges[0][1] << std::endl;
            std::cout << "ranges[1][0]: " << ranges[1][0] << " ranges[1][1]: " << ranges[1][1] << std::endl;
            throw std::runtime_error("Error: Bin index out of range!");
        }
        data[xBin][yBin] += w;
    }
    // cleanup
    tree->ResetBranchAddresses();
    return data;
}

void Plot2dHist(
    std::vector<std::vector<double>> data, 
    const std::string& plotTitle,
    const std::string& filename,
    const std::string& xLabel,
    const std::string& yLabel,
    std::vector<std::vector<double>> plotRanges = {{0, 1}, {0, 1}},
    std::vector<double> zRanges = {1, 0}
) {
    int nRows = data.size();
    int nCols = data[0].size();
    // TH2F* hist = new TH2F("hist", plotTitle.c_str(), nCols, 0, nCols, nRows, 0, nRows);
    TH2F* hist = new TH2F("hist", plotTitle.c_str(), nCols, plotRanges[0][0], plotRanges[0][1], nRows, plotRanges[1][0], plotRanges[1][1]);
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            hist->SetBinContent(j+1, i+1, data[i][j]);
        }
    }
    if (zRanges[0] < zRanges[1]) {
        hist->SetMinimum(zRanges[0]);
        hist->SetMaximum(zRanges[1]);
    }
    TCanvas* canvas = new TCanvas("canvas", "canvas", 1000, 700);
    hist->SetTitle("");
    // convert x axis from Mev to GeV
    if (strops::strin(xLabel, "MeV")) {
        hist->GetXaxis()->SetLimits(plotRanges[0][0] / 1000, plotRanges[0][1] / 1000);
        hist->GetXaxis()->SetTitle(strops::replace(xLabel, "MeV", "GeV").c_str());
    } else {
        hist->GetXaxis()->SetTitle(xLabel.c_str());
    }
    hist->GetYaxis()->SetTitle(yLabel.c_str());
    hist->GetXaxis()->SetTitleSize(consts::xTitleSize);
    hist->GetYaxis()->SetTitleSize(consts::yTitleSize);
    hist->GetXaxis()->SetTitleFont(consts::titleFont);
    hist->GetYaxis()->SetTitleFont(consts::titleFont);
    // hist->GetXaxis()->SetTitleOffset(0.9);
    hist->GetYaxis()->SetTitleOffset(0.7);
    hist->GetXaxis()->CenterTitle(consts::centerTitle);
    hist->GetYaxis()->CenterTitle(consts::centerTitle);
    hist->GetXaxis()->SetNdivisions(5);
    hist->GetYaxis()->SetNdivisions(5);
    hist->GetXaxis()->SetLabelSize(consts::xLabelSize);
    hist->GetYaxis()->SetLabelSize(consts::yLabelSize);
    hist->GetZaxis()->SetLabelSize(consts::zLabelSize);
    hist->Draw("colz");
    hist->SetStats(0);
    canvas->SetMargin(0.11, 0.1, 0.14, 0.05);
    // colourbar's font larger
    canvas->SetFrameLineWidth(consts::frameWidth);
    canvas->SaveAs(filename.c_str());
    delete canvas;
    delete hist;
}

int main() {
    // std::string figDir = config::plot_dir5("bdt_all", 0.5, "comp");
    std::string figDir = config::plot_dir6;
    auto [file, tree] = loadSweight();

    std::vector<std::string> particles = {"K", "pi1", "pi2", "pi3"};
    std::vector<std::string> modes = {"fine"};
    for (const auto& particle : particles) {
        for (const auto& mode : modes) {
            std::cout << "Loading sample data of " << particle << "..." << std::endl;
            std::string sample_particle = particle;
            if (particle == "pi1" || particle == "pi2" || particle == "pi3") {
                sample_particle = "Pi";
            }
            std::string filename = config::plot_dir5("_samples") + sample_particle + "_p_vs_eta_" + mode + "_binning_raw_yields.csv";
            auto data = fileio::readCSV(filename);
            // print2Dvector(data);
            auto columns = fileio::readCSVcolumns(filename);
            auto index = fileio::readCSVindex(filename);
            double colMin = std::stod(strops::split(columns[1], "-")[0]);
            double colMax = std::stod(strops::split(columns[columns.size() - 1], "-")[1]);
            double rowMin = std::stod(strops::split(index[1], "-")[0]);
            double rowMax = std::stod(strops::split(index[index.size() - 1], "-")[1]);
            int colBins = columns.size() - 1;
            int rowBins = index.size() - 1;
            std::cout << "Sample data loaded." << std::endl;
            std::cout << "Size: " << data.size() << "x" << data[0].size() << std::endl;
            // Plot2dHist(
            //     transpose(data), "Sample data - " + particle + " (adj yield: " + std::to_string(int(sum2Dvector(data) * 2)) + ")",
            //     figDir + "sample_" + particle + "_" + mode + ".png",
            //     "p [MeV/c]", "\\eta", {{rowMin, rowMax}, {colMin, colMax}});

            std::cout << "Loading sWeighted data..." << std::endl;
            std::vector<std::vector<double>> myData = generateSigYieldData(tree, {particle + "_P", particle + "_ETA"}, "bdt_100_50_ss", {rowBins, colBins}, {{rowMin, rowMax}, {colMin, colMax}});
            // print2Dvector(myData);
            std::cout << "sWeighted data loaded." << std::endl;
            std::cout << "Size: " << myData.size() << "x" << myData[0].size() << std::endl;
            // Plot2dHist(
            //     transpose(myData), "sWeighted data - " + particle + " (adj yield: " + std::to_string(int(sum2Dvector(myData) * 50)) + ")",
            //     figDir + "sweighted_" + particle + "_" + mode + ".png",
            //     "p [MeV/c]", "\\eta", {{rowMin, rowMax}, {colMin, colMax}});

            std::cout << "Comparing data..." << std::endl;
            std::vector<std::vector<double>> diff(rowBins, std::vector<double>(colBins, 0.0));
            double sumData = sum2Dvector(data);
            double sumMyData = sum2Dvector(myData);
            double min = 0.0; double max = 2.0;  // range to clip
            for (int i = 0; i < rowBins; i++) {
                for (int j = 0; j < colBins; j++) {
                    double denom = data[i][j] / sumData;
                    if (data[i][j] == 0) {
                        denom = 1.0;
                        // this way, if myData is also 0, the ratio is 0 and if not, the ratio is something that might be clipped i.e. max
                    }
                    // clipping here instead of Plot2dHist's zRanges as the latter clips to range [min, max] and plots (min, max]
                    diff[i][j] = std::max((myData[i][j]/sumMyData) / denom, min+0.00001);
                }
            }
            // print2Dvector(diff);
            Plot2dHist(
                transpose(diff), "Signal Distribution Difference (K3pi / Kpi)", figDir + "comparison_" + particle + "_" + mode + ".png",
                "Momentum [MeV/#it{c}]", "Pseudorapidity", {{rowMin, rowMax}, {colMin, colMax}}, {min, max});
            std::cout << particle << " data compared." << std::endl;

            // cleanup
            data.clear();
            myData.clear();
            diff.clear();
        }
    }

    // cleanup
    file->Close();
    return 0;
}
void plotCompRep() {
    main();
}
