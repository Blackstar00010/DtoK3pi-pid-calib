#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <vector>
#include <string>
#include "../utils/common.cpp"
#include "../utils/config.cpp"
#include "../utils/plotter.cpp"


namespace main2 {
    void main();
}

std::vector<TTree*> split_data(TTree* tree) {
    if (!tree) {
        std::cerr << "Error: Null TTree pointer provided!" << std::endl;
        return {};
    }
    
    double dmmean = 1862.0;
    double deltamean = 145.35;
    double dmwidtho = 60.0;
    double deltawidtho = 5.0;
    double dmwidthi = 10.0;
    double deltawidthi = 1.0;
    // Create two new trees to store the signal and background data
    std::string dm_min_i = std::to_string(dmmean - dmwidthi/2);
    std::string dm_max_i = std::to_string(dmmean + dmwidthi/2);
    std::string delta_min_i = std::to_string(deltamean - deltawidthi/2);
    std::string delta_max_i = std::to_string(deltamean + deltawidthi/2);
    std::string dm_min_o = std::to_string(dmmean - dmwidtho/2);
    std::string dm_max_o = std::to_string(dmmean + dmwidtho/2);
    std::string delta_min_o = std::to_string(deltamean - deltawidtho/2);
    std::string delta_max_o = std::to_string(deltamean + deltawidtho/2);
    std::string cond_in = "(D_M > " + dm_min_i + ") && (D_M < " + dm_max_i + ") && (delta_M > " + delta_min_i + ") && (delta_M < " + delta_max_i + ")";
    std::string cond_out = "(D_M < " + dm_min_o + ") || (D_M > " + dm_max_o + ") || (delta_M < " + delta_min_o + ") || (delta_M > " + delta_max_o + ")";

    if (gDirectory != gROOT) {
        gROOT->cd();
    }
    TTree* tree1 = tree->CopyTree(cond_in.c_str());
    TTree* tree2 = tree->CopyTree(cond_out.c_str());
    return {tree1, tree2};
}

// plots 1. 1d histogram of D_M, 2. 1d hist of delta_M, 3. 2d hist of D_M and delta_M
void plotMasses(TTree* tree, std::string suffix) {
    std::string dmBranch = "D_M";
    std::string deltaBranch = "delta_M";

    // if 'delta_M' not in branch but 'Dst_M' is, calculate delta_M = Dst_M - D_M
    if (tman::getBranches(tree).end() == std::find(tman::getBranches(tree).begin(), tman::getBranches(tree).end(), deltaBranch)) {
        tree->SetAlias(deltaBranch.c_str(), "Dst_M - D_M");
    }

    // filenames = ./plots/2_sigbkg_cpp/_general/hist1d_dm_<suffix>.png
    std::string generalDir = plots_dir + "2_sigbkg_cpp/_general/";
    plotter.hist(tree, dmBranch, 300, {true, true}, {0.0, 0.0}, "D_M", "D_M Distribution", generalDir + "hist1d_dm_" + suffix + ".png");
    plotter.hist(tree, deltaBranch, 300, {true, true}, {0.0, 0.0}, "delta_M", "delta_M Distribution", generalDir + "hist1d_deltam_" + suffix + ".png");
    plotter.hist2d(tree, dmBranch, deltaBranch, 300, {true, true}, {0.0, 0.0}, {0.0, 0.0}, false, "D_M", "delta_M", "D_M vs delta_M", generalDir + "2dmass.png");
    
}

void main2::main() {
    // auto start = std::chrono::high_resolution_clock::now();

    TTree* tree = tman::load_tree("./data/full.root");
    std::vector<TTree*> trees = split_data(tree);
    TTree* tree1 = trees[0];
    TTree* tree2 = trees[1];

    std::vector<std::string> cols = tman::getBranches(tree);
    std::cout << "Plotting histograms for " << cols.size() << " columns" << std::endl;
    for (const auto& col : cols) {
        TCanvas* canvas = plotter::sigbkg(tree1, tree2, col, 300, true, {true, true}, {0.0, 0.0}, {"sig", "bkg"}, "", "", col);
        delete canvas;
    };
    
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Time taken: " << elapsed.count() << " s" << std::endl;
}