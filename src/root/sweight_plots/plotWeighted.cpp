#include "utils.cpp"

int main() {
    auto [file, tree] = loadSweight();
    if (!file || !tree) {
        return 1;
    }

    std::string compHistOutDir = config::plot_dir5(commonVars::modelName, commonVars::cut);
    std::string weightCol = commonVars::weightCol;
    std::vector<double> pRange = {3000, 100000};
    std::vector<double> etaRange = {1.5, 5.0};
    
    // Plot the weighted vs unweighted histograms
    PlotWeightedVSUnweighted(tree, "D_M", weightCol, compHistOutDir);
    PlotWeightedVSUnweighted(tree, "delta_M", weightCol, compHistOutDir);
    PlotWeightedVSUnweighted(tree, "K_P", weightCol, compHistOutDir, pRange);
    PlotWeightedVSUnweighted(tree, "K_ETA", weightCol, compHistOutDir, etaRange);
    PlotWeightedVSUnweighted(tree, "K_PID_K", weightCol, compHistOutDir);
    PlotWeightedVSUnweighted(tree, {"pi1_P", "pi2_P", "pi3_P"}, weightCol, compHistOutDir, "pi_P", pRange);
    PlotWeightedVSUnweighted(tree, {"pi1_ETA", "pi2_ETA", "pi3_ETA"}, weightCol, compHistOutDir, "pi_ETA", etaRange);
    PlotWeightedVSUnweighted(tree, {"pi1_PID_K", "pi2_PID_K", "pi3_PID_K"}, weightCol, compHistOutDir, "pi_PID_K");
    PlotWeightedVSUnweighted(tree, "pi1_PID_K", weightCol, compHistOutDir);
    PlotWeightedVSUnweighted(tree, "pi2_PID_K", weightCol, compHistOutDir);
    PlotWeightedVSUnweighted(tree, "pi3_PID_K", weightCol, compHistOutDir);

    // Plot the 2D weighted histograms
    PlotWeightd2dHist(tree, "D_M", "delta_M", weightCol, compHistOutDir);
    PlotWeightd2dHist(tree, "K_P", "K_ETA", weightCol, compHistOutDir, 100, 100, pRange, etaRange);
    // merging pions is not needed
    // PlotWeightd2dHist(tree, {"pi1_P", "pi2_P", "pi3_P"}, {"pi1_ETA", "pi2_ETA", "pi3_ETA"}, weightCol, compHistOutDir, "pi_P", "pi_ETA");
    PlotWeightd2dHist(tree, "pi1_P", "pi1_ETA", weightCol, compHistOutDir, 100, 100, pRange, etaRange);
    PlotWeightd2dHist(tree, "pi2_P", "pi2_ETA", weightCol, compHistOutDir, 100, 100, pRange, etaRange);
    PlotWeightd2dHist(tree, "pi3_P", "pi3_ETA", weightCol, compHistOutDir, 100, 100, pRange, etaRange);

    // Cleanup
    std::cout << "Closing file..." << std::endl;
    file->Close();
    std::cout << "File closed." << std::endl;

    return 0;
}

void plotWeighted() {
    main();
}