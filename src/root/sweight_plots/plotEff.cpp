#include "utils.cpp"

int main() {
    auto [file, tree] = loadSweight();
    if (!file || !tree) {
        return 1;
    }

    // Plot the weighted vs unweighted histograms
    // std::string compHistOutDir = config::plot_dir5("bdt_100", 0.5);
    std::string compHistOutDir = config::plot_dir5(commonVars::modelName, commonVars::cut);
    std::string weightCol = commonVars::weightCol;
    std::vector<double> pRange = {3000, 100000};
    std::vector<double> etaRange = {1.5, 5.0};
    int pBins = 18;
    int etaBins = 4;
    double pidCut = 5;

    // Plot the efficiency 1D histograms
    // p; K by K, pi1 by pi1, pi2 by pi2, pi3 by pi3
    PlotEff1dsUltimate(tree, {{"K_P"}}, weightCol, "K_PID_K", {pidCut, 0, -pidCut}, compHistOutDir, pBins, pRange, true, true, "K_P_by_KPIDK");
    PlotEff1dsUltimate(tree, {{"pi1_P"}}, weightCol, "pi1_PID_K", {pidCut, 0, -pidCut}, compHistOutDir, pBins, pRange, true, true, "pi1_P_by_pi1PIDK");
    PlotEff1dsUltimate(tree, {{"pi2_P"}}, weightCol, "pi2_PID_K", {pidCut, 0, -pidCut}, compHistOutDir, pBins, pRange, true, true, "pi2_P_by_pi2PIDK");
    PlotEff1dsUltimate(tree, {{"pi3_P"}}, weightCol, "pi3_PID_K", {pidCut, 0, -pidCut}, compHistOutDir, pBins, pRange, true, true, "pi3_P_by_pi3PIDK");
    // PlotEff1dsUltimate(tree, {{"K_P"}, {"pi1_P", "pi2_P", "pi3_P"}}, weightCol, "K_PID_K", {pidCut, 0, -pidCut}, compHistOutDir, {"", "pi_P"}, "P", pBins, pRange);

    // eta; K by K, pi1 by pi1, pi2 by pi2, pi3 by pi3
    PlotEff1dsUltimate(tree, {{"K_ETA"}}, weightCol, "K_PID_K", {pidCut, 0, -pidCut}, compHistOutDir, etaBins, etaRange, true, true, "K_ETA_by_KPIDK");
    PlotEff1dsUltimate(tree, {{"pi1_ETA"}}, weightCol, "pi1_PID_K", {pidCut, 0, -pidCut}, compHistOutDir, etaBins, etaRange, true, true, "pi1_ETA_by_pi1PIDK");
    PlotEff1dsUltimate(tree, {{"pi2_ETA"}}, weightCol, "pi2_PID_K", {pidCut, 0, -pidCut}, compHistOutDir, etaBins, etaRange, true, true, "pi2_ETA_by_pi2PIDK");
    PlotEff1dsUltimate(tree, {{"pi3_ETA"}}, weightCol, "pi3_PID_K", {pidCut, 0, -pidCut}, compHistOutDir, etaBins, etaRange, true, true, "pi3_ETA_by_pi3PIDK");
    // PlotEff1dsUltimate(tree, {{"K_ETA"}, {"pi1_ETA", "pi2_ETA", "pi3_ETA"}}, weightCol, {{"K_PID_K"}, {"pi1_PID_K", "pi2_PID_K", "pi3_PID_K"}}, {pidCut, 0, -pidCut}, compHistOutDir, etaBins, etaRange);  // TODO

    // efficiency 2d
    PlotEff2d(tree, "K_P", "K_ETA", weightCol, "K_PID_K", pidCut, compHistOutDir, pBins, etaBins, pRange, etaRange, true, true, true);
    // full range is not needed
    // PlotEff2d(tree, "K_P", "K_ETA", weightCol, "K_PID_K", pidCut, compHistOutDir, pBins*2, etaBins*2, {0, 200000}, {1, 6.5}, true, true, true, "K_full");
    PlotEff2d(tree, "pi1_P", "pi1_ETA", weightCol, "pi1_PID_K", -pidCut, compHistOutDir, pBins, etaBins, pRange, etaRange, false, true, true);
    PlotEff2d(tree, "pi2_P", "pi2_ETA", weightCol, "pi2_PID_K", -pidCut, compHistOutDir, pBins, etaBins, pRange, etaRange, false, true, true);
    PlotEff2d(tree, "pi3_P", "pi3_ETA", weightCol, "pi3_PID_K", -pidCut, compHistOutDir, pBins, etaBins, pRange, etaRange, false, true, true);
    // full range is not needed
    // PlotEff2d(tree, "pi3_P", "pi3_ETA", weightCol, "pi3_PID_K", -pidCut, compHistOutDir, pBins*2, etaBins*2, {0, 200000}, {1, 6.5}, false, true, true, "pi3_full");
    // merging pions is not needed
    // PlotEff2d(
    //     tree, {"pi1_P", "pi2_P", "pi3_P"}, {"pi1_ETA", "pi2_ETA", "pi3_ETA"}, 
    //     weightCol, {"pi1_PID_K", "pi2_PID_K", "pi3_PID_K"}, -pidCut, compHistOutDir, 
    //     "pi_P", "pi_ETA", "pi_PID_K",
    //     pBins, etaBins, pRange, etaRange, false, true, true
    // );

    // inefficiency 2d
    PlotEff2d(tree, "K_P", "K_ETA", weightCol, "K_PID_K", -pidCut, compHistOutDir, pBins, etaBins, pRange, etaRange, false, false, false);
    PlotEff2d(tree, "pi1_P", "pi1_ETA", weightCol, "pi1_PID_K", pidCut, compHistOutDir, pBins, etaBins, pRange, etaRange, true, true, false);
    PlotEff2d(tree, "pi2_P", "pi2_ETA", weightCol, "pi2_PID_K", pidCut, compHistOutDir, pBins, etaBins, pRange, etaRange, true, true, false);
    PlotEff2d(tree, "pi3_P", "pi3_ETA", weightCol, "pi3_PID_K", pidCut, compHistOutDir, pBins, etaBins, pRange, etaRange, true, true, false);
    // merging pions is not needed
    // PlotEff2d(
    //     tree, {"pi1_P", "pi2_P", "pi3_P"}, {"pi1_ETA", "pi2_ETA", "pi3_ETA"}, 
    //     weightCol, {"pi1_PID_K", "pi2_PID_K", "pi3_PID_K"}, pidCut, compHistOutDir, 
    //     "pi_P", "pi_ETA", "pi_PID_K",
    //     pBins, etaBins, pRange, etaRange,
    //     true, false, false
    // );

    // SBR
    PlotSBR2d(tree, "K_P", "K_ETA", weightCol, compHistOutDir, pBins, etaBins, pRange, etaRange);
    PlotSBR2d(tree, "pi1_P", "pi1_ETA", weightCol, compHistOutDir, pBins, etaBins, pRange, etaRange);
    PlotSBR2d(tree, "pi2_P", "pi2_ETA", weightCol, compHistOutDir, pBins, etaBins, pRange, etaRange);
    PlotSBR2d(tree, "pi3_P", "pi3_ETA", weightCol, compHistOutDir, pBins, etaBins, pRange, etaRange);

    // Cleanup
    std::cout << "Closing file..." << std::endl;
    file->Close();
    std::cout << "File closed." << std::endl;

    return 0;
}

void plotEff() {
    main();
}