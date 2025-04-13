#include <iostream>
#include <cstdio>  // for remove()
#include <TTree.h>
#include <TH1F.h>
#include <RooRealVar.h>
#include <RooDataHist.h>
#include <RooAddPdf.h>
#include <RooPlot.h>
#include <TCanvas.h>
#include "../utils/config.cpp"
#include "../utils/common.cpp"
#include "../utils/plotter.cpp"
#include "../utils/check.cpp"
#include "models.cpp"


void fit(TTree* tree, const std::string& branch, const std::string& model1, const std::string& model2, const std::string& outputDir) {
    if (!check::checkBranch(tree, branch)) {
        return;
    }
    // convert str to char*
    const char* branch_c = branch.c_str();
    std::cout << "Fitting the branch '" << branch << "' with models '" << model1 << "' and '" << model2 << "'..." << std::endl;

    // Define the range for the variable
    std::cout << "Defining the range for the branch..." << std::endl;
    Double_t min = tree->GetMinimum(branch_c);
    Double_t max = tree->GetMaximum(branch_c);

    // Create a histogram (binned data)
    std::cout << "Creating the histogram..." << std::endl;
    const int nBins = 200;
    TH1F* hist = new TH1F("hist", branch_c, nBins, min, max);
    tree->Draw((branch + ">>hist").c_str(), "", "goff");  // fill the histogram

    // Define independent variable x
    std::cout << "Defining the RooFit independent variable..." << std::endl;
    RooRealVar x("x", branch_c, min, max);

    // Import the histogram into RooFit
    std::cout << "Importing the histogram into RooFit..." << std::endl;
    // dataHist instead of dataSets for binned data
    RooDataHist dataHist("dataHist", "Binned Data", x, hist);

    // Generate the signal and background models
    std::cout << "Generating the signal and background models..." << std::endl;
    RooAbsPdf* signal = GenerateModel(model1, hist, x, "signal");
    if (!signal) {
        delete hist;
        return;
    }
    RooAbsPdf* background = GenerateModel(model2, hist, x, "background");
    if (!background) {
        delete hist;
        return;
    }

    // Combine the signal and background models
    std::cout << "Combining the signal and background models..." << std::endl;
    RooRealVar frac("frac", "Fraction of Gaussian", 0.5, 0.0, 1.0);
    RooAddPdf model("model", "Signal + Background", RooArgList(*signal, *background), RooArgList(frac));
    model.fitTo(dataHist);

    // Plot the data and fit
    TCanvas* canvas = new TCanvas("canvas", "Fit Result", 800, 600);
    RooPlot* frame = x.frame();
    dataHist.plotOn(frame);  // data histogram
    model.plotOn(frame, RooFit::Name("model"));  // combined model
    model.plotOn(frame, RooFit::Components("signal"), RooFit::LineStyle(kDashed), RooFit::LineColor(kRed));  // signal model
    model.plotOn(frame, RooFit::Components("background"), RooFit::LineStyle(kDashed), RooFit::LineColor(kGreen));  // background model
    frame->SetXTitle(branch.c_str());
    frame->SetYTitle("Events");
    frame->SetTitle((branch + " Fit: " + model1 + " + " + model2).c_str());
    frame->Draw();
    std::string suffix = strops::replace(branch, "_", "") + "_" + model1[0] + model2[0];
    canvas->SaveAs((outputDir + "overlay_" + suffix + ".png").c_str());

    // Plot residuals; TODO: NOT WORKING PIECE OF SHIT
    // TCanvas* canvasResidual = new TCanvas("canvasResidual", "Residual Plot", 800, 600);
    // RooPlot* residualFrame = x.frame();
    // residualFrame->SetTitle((branch + " Residuals: " + model1 + " + " + model2).c_str());
    // RooHist* residuals = frame->residHist();
    // residualFrame->addPlotable(residuals, "P");
    // residualFrame->SetXTitle(branch.c_str());
    // residualFrame->SetYTitle("Residuals");
    // residualFrame->SetTitle((branch + " Residuals: " + model1 + " + " + model2).c_str());
    // residualFrame->Draw();
    // canvasResidual->SaveAs((outputDir + "residuals_" + suffix + ".png").c_str());

    // Cleanup
    delete signal;
    delete background;
    delete canvas;
    delete hist; 
}

void fit1d() {
    std::cout << "Loading TTree from file..." << std::endl;
    auto [file, tree] = tman::loadFileAndTree(config::masscut_root_file); // TODO: masscut removed
    std::cout << "Loaded TTree from file" << std::endl;

    // TODO: cut based on probabilities
    float cut = 0.0;

    const std::string output_dir = config::plot_dir4(1, "bdt_all", cut);
    std::cout << "Fitting the models..." << std::endl;
    fit(tree, "D_M", "gaussian", "exp", output_dir);
    fit(tree, "D_M", "gaussian", "lin", output_dir);
    fit(tree, "D_M", "cb", "exp", output_dir);
    fit(tree, "D_M", "cb", "lin", output_dir);
    fit(tree, "delta_M", "gaussian", "dst0bg", output_dir);
    fit(tree, "delta_M", "cb", "dst0bg", output_dir);
    // fitting already prints too many messages, so no intermediate messages here
    std::cout << "Fitting for cut " << std::to_string(cut) << " complete!" << std::endl;

    std::cout << "Cleaning up..." << std::endl;
    delete tree;
    delete file;
    std::cout << "All done!" << std::endl;
}