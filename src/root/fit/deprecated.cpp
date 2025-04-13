#include <iostream>
#include <string>
#include <fstream>
#include <TFile.h>
#include <TTree.h>
#include <TH2F.h>
#include <TCanvas.h>
#include <RooRealVar.h>
#include <RooFormulaVar.h>
#include <RooGaussian.h>
#include <RooDataSet.h>
#include <RooArgList.h>
#include <RooProdPdf.h>
#include <RooPlot.h>
#include <RooDataHist.h>
#include <RooStats/SPlot.h>
#include "../utils/config.cpp"
#include "../utils/common.cpp"
#include "../utils/plotter.cpp"
#include "../utils/check.cpp"
#include "models.cpp"

// this file 1. fits, 2. sPlots, 3. plots the sWeighted histograms, in a single function, so deprecated


// Perform a 2D binned fit of the given tree and branches
void Fit(TTree* tree, const std::string& branchX, const std::string& branchY, 
         const std::string& model1s = "gauss", const std::string& model1b = "linear", 
         const std::string& model2s = "gauss", const std::string& model2b = "linear",
         const std::string& outputDir = config::temp_fig_dir) {
    if (!check::checkBranch(tree, branchX) || !check::checkBranch(tree, branchY)) {
        return;
    }
    std::cout << "WARNING: this function is optimised for fitting D_M and delta_M only." << std::endl;
    const char* branchX_c = branchX.c_str();
    const char* branchY_c = branchY.c_str();
    std::cout << "Fitting the branches '" << branchX << "' and '" << branchY << "' with models '" << model1s << "', '" << model1b << "', '" << model2s << "', and '" << model2b << "'..." << std::endl;

    // Define the range for the variables
    std::cout << "Defining the range for the branches..." << std::endl;
    Double_t xMin = tree->GetMinimum(branchX_c);
    Double_t xMax = tree->GetMaximum(branchX_c);
    Double_t yMin = tree->GetMinimum(branchY_c);
    Double_t yMax = tree->GetMaximum(branchY_c);

    // Create a 2D histogram (binned data)
    std::cout << "Creating the 2D histogram..." << std::endl;
    const int nBinsX = 100;
    const int nBinsY = 100;
    TH2F* hist2D = new TH2F("hist2D", "2D Histogram;X-axis;Y-axis", nBinsX, xMin, xMax, nBinsY, yMin, yMax);
    tree->Draw((branchY + ":" + branchX + ">>hist2D").c_str(), "", "goff");

    // Define the independent variables
    std::cout << "Defining the RooFit independent variables..." << std::endl;
    RooRealVar x(branchX.c_str(), branchX.c_str(), xMin, xMax);
    RooRealVar y(branchY.c_str(), branchY.c_str(), yMin, yMax);

    // Import the histogram into RooFit
    std::cout << "Importing the histogram into RooFit..." << std::endl;
    // dataHist instead of dataSets for binned data
    RooDataHist dataHist("dataHist", "Binned Data", RooArgSet(x, y), hist2D);

    // Generate the signal and background models for X
    std::cout << "Generating the signal and background models for X..." << std::endl;
    // PDF variables are named as: (type of model)(X/Y)_(type of X model)(type of Y model)
    TH1D* projX = hist2D->ProjectionX("projX");
    RooRealVar* xPtr = &x;
    // RooAbsPdf* signalX_ss = GenerateModel(model1s, projX, x, "signalXss");
    RooAbsPdf* signalX_ss = GenerateModel(model1s, projX, xPtr, "signalXss");
    if (!signalX_ss) {
        delete hist2D;
        return;
    }
    RooAbsPdf* signalX_sb = GenerateModel(model1s, projX, x, "signalXsb");
    if (!signalX_sb) {
        delete hist2D;
        return;
    }
    // RooAbsPdf* backgroundX_bs = GenerateModel(model1b, projX, x, "backgroundXbs");
    // if (!backgroundX_bs) {
    //     delete hist2D;
    //     return;
    // }
    RooAbsPdf* backgroundX_bb = GenerateModel(model1b, projX, x, "backgroundXbb");
    if (!backgroundX_bb) {
        delete hist2D;
        return;
    }
    
    // Generate the signal and background models for Y
    std::cout << "Generating the signal and background models for Y..." << std::endl;
    TH1D* projY = hist2D->ProjectionY("projY");
    RooAbsPdf* signalY_ss = GenerateModel(model2s, projY, y, "signalYss");
    if (!signalY_ss) {
        delete hist2D;
        return;
    }
    RooAbsPdf* backgroundY_sb = GenerateModel(model2b, projY, y, "backgroundYsb");
    if (!backgroundY_sb) {
        delete hist2D;
        return;
    }
    // RooAbsPdf* signalY_bs = GenerateModel(model2s, projY, y, "signalYbs");
    // if (!signalY_bs) {
    //     delete hist2D;
    //     return;
    // }
    RooAbsPdf* backgroundY_bb = GenerateModel(model2b, projY, y, "backgroundYbb");
    if (!backgroundY_bb) {
        delete hist2D;
        return;
    }

    // Combine the signal and background models for X
    std::cout << "Combining the signal and background models..." << std::endl;
    int nEntries = hist2D->GetEntries();
    RooRealVar n_ss("n_ss", "Number of signal-signal events", 0.6 * nEntries, 0, nEntries);
    RooRealVar n_sb("n_sb", "Number of signal-background events", 0.15 * nEntries, 0, nEntries);
    // RooRealVar n_bs("n_bs", "Number of background-signal events", 0.2 * nEntries, 0, nEntries);
    RooRealVar n_bb("n_bb", "Number of background-background events", 0.2 * nEntries, 0, nEntries);
    // RooFormulaVar n_bb("n_bb", "Number of background-background events",
    //                 //    TString::Format("%f - (@0 + @1 + @2)", float(nEntries)),
    //                 //    RooArgList(n_ss, n_sb, n_bs));
    //                    TString::Format("%f - (@0 + @1)", float(nEntries)),
    //                    RooArgList(n_ss, n_bs));

    RooProdPdf model_ss("model_ss", "Signal X + Signal Y", RooArgList(*signalX_ss, *signalY_ss));
    RooProdPdf model_sb("model_sb", "Signal X + Background Y", RooArgList(*signalX_ss, *backgroundY_bb));
    // RooProdPdf model_bs("model_bs", "Background X + Signal Y", RooArgList(*backgroundX_bs, *signalY_bs));
    RooProdPdf model_bb("model_bb", "Background X + Background Y", RooArgList(*backgroundX_bb, *backgroundY_bb));

    // RooAddPdf model("model", "Signal + Background", RooArgList(model_ss, model_sb, model_bs, model_bb), RooArgList(n_ss, n_sb, n_bs, n_bb));
    RooAddPdf model("model", "Signal + Background", RooArgList(model_ss, model_sb, model_bb), RooArgList(n_ss, n_sb, n_bb));

    std::cout << "Fitting the model to the data..." << std::endl;
    // model.fitTo(dataHist, RooFit::Extended(kTRUE), RooFit::Verbose(), RooFit::PrintLevel(10));  // extended maximum likelihood fit (the same result)
    model.fitTo(dataHist, RooFit::Verbose(kTRUE), RooFit::PrintLevel(10));

    // Create a residual heatmap
    std::cout << "Creating the residual heatmap..." << std::endl;
    TH2F* residualHist = new TH2F("residualHist", "Residual Heatmap;X-axis;Y-axis", nBinsX, xMin, xMax, nBinsY, yMin, yMax);
    RooArgSet vars(x, y);
    // Loop through each bin and calculate residuals
    std::cout << "Calculating residuals..." << std::endl;
    double xBinWidth = hist2D->GetXaxis()->GetBinWidth(1); // Get the bin width of the first X bin
    double yBinWidth = hist2D->GetYaxis()->GetBinWidth(1); // Get the bin width of the first Y bin
    for (int iX = 1; iX <= nBinsX; ++iX) {
        // Get the bin center
        double centerX = hist2D->GetXaxis()->GetBinCenter(iX);
        x.setVal(centerX);
        for (int iY = 1; iY <= nBinsY; ++iY) {
            double centerY = hist2D->GetYaxis()->GetBinCenter(iY);

            // Set the variable values for the model
            y.setVal(centerY);

            // Get observed and predicted densities
            double observed = hist2D->GetBinContent(iX, iY); // Observed count
            double predicted = model.getVal(&vars); // Predicted density
            predicted *= nEntries * xBinWidth * yBinWidth; // Scale the density to the number of entries

            // Calculate the residual
            double residual = observed - predicted;
            residual /= sqrt(observed);

            // Fill the residual histogram
            residualHist->SetBinContent(iX, iY, residual);
        }
    }

    // Create a canvas for the residual heatmap
    std::cout << "Creating the canvas for the residual heatmap..." << std::endl;
    TCanvas* heatmapCanvas = new TCanvas("heatmapCanvas", "Residual Heatmap", 1000, 800);
    int chi2 = int(model.createChi2(dataHist)->getVal());
    residualHist->Draw("COLZ"); // Draw the residual heatmap
    residualHist->GetXaxis()->SetTitle(branchX.c_str());
    residualHist->GetYaxis()->SetTitle(branchY.c_str());
    residualHist->SetTitle(("Heatmap of Residual/sqrt(Data) (Chi2: " + std::to_string(chi2) + ")").c_str());
    heatmapCanvas->SetRightMargin(0.15);  // color bar issue
    gStyle->SetOptStat(0);
    std::string suffix = std::string(1, model1s[0]) + model1b[0] + model2s[0] + model2b[0];
    heatmapCanvas->SaveAs((outputDir + "2dfit_hm_" + suffix + ".png").c_str());
    std::cout << "Residual heatmap saved to file." << std::endl;

    // Create a canvas for the 2D Gaussian fit projections
    std::cout << "Creating the canvas for the 2D Gaussian fit projections..." << std::endl;
    TCanvas* projCanvas = new TCanvas("projCanvas", "2D Gaussian Fit (Binned)", 1000, 800);
    projCanvas->Divide(1, 2);

    // X projection
    std::cout << "Creating the X projection(" << branchX << ")..." << std::endl;
    projCanvas->cd(1);
    RooPlot* frameX = x.frame();
    dataHist.plotOn(frameX, RooFit::Name("dataHist"));  // data
    model.plotOn(frameX, RooFit::Name("model"));  // model
    model.plotOn(frameX, RooFit::Components("model_ss"), RooFit::LineStyle(kDashed), RooFit::LineColor(kRed), RooFit::Name("model_ss"));  // signal-signal
    model.plotOn(frameX, RooFit::Components("model_sb"), RooFit::LineStyle(kDashed), RooFit::LineColor(kYellow), RooFit::Name("model_sb"));  // signal-background
    // model.plotOn(frameX, RooFit::Components("model_bs"), RooFit::LineStyle(kDashed), RooFit::LineColor(kBlue), RooFit::Name("model_bs"));  // background-signal
    model.plotOn(frameX, RooFit::Components("model_bb"), RooFit::LineStyle(kDashed), RooFit::LineColor(kGreen), RooFit::Name("model_bb"));  // background-background
    frameX->SetTitle((branchX + " projection").c_str());
    // legends
    TLegend* legendX = new TLegend(0.6, 0.5, 0.9, 0.9);
    legendX->AddEntry(frameX->findObject("dataHist"), "Data", "p");
    legendX->AddEntry(frameX->findObject("model"), "Model", "l");
    legendX->AddEntry(frameX->findObject("model_ss"), ("Sig(" + model1s + ")-Sig(" + model2s + ") : " + std::to_string(int(n_ss.getVal() / nEntries * 100)) + "%").c_str(), "l");
    legendX->AddEntry(frameX->findObject("model_sb"), ("Sig(" + model1s + ")-Bkg(" + model2b + ") : " + std::to_string(int(n_sb.getVal() / nEntries * 100)) + "%").c_str(), "l");
    // legendX->AddEntry(frameX->findObject("model_bs"), ("Bkg(" + model1b + ")-Sig(" + model2s + ") : " + std::to_string(int(n_bs.getVal() / nEntries * 100)) + "%").c_str(), "l");
    legendX->AddEntry(frameX->findObject("model_bb"), ("Bkg(" + model1b + ")-Bkg(" + model2b + ") : " + std::to_string(int(n_bb.getVal() / nEntries * 100)) + "%").c_str(), "l");
    frameX->Draw();
    legendX->Draw();

    // Y projection
    std::cout << "Creating the Y projection(" << branchY << ")..." << std::endl;
    projCanvas->cd(2);
    RooPlot* frameY = y.frame();
    dataHist.plotOn(frameY, RooFit::Name("dataHist"));  // data
    model.plotOn(frameY, RooFit::Name("model"));  // model
    model.plotOn(frameY, RooFit::Components("model_ss"), RooFit::LineStyle(kDashed), RooFit::LineColor(kRed), RooFit::Name("model_ss"));  // signal-signal
    model.plotOn(frameY, RooFit::Components("model_sb"), RooFit::LineStyle(kDashed), RooFit::LineColor(kYellow), RooFit::Name("model_sb"));
    // model.plotOn(frameY, RooFit::Components("model_bs"), RooFit::LineStyle(kDashed), RooFit::LineColor(kBlue), RooFit::Name("model_bs"));
    model.plotOn(frameY, RooFit::Components("model_bb"), RooFit::LineStyle(kDashed), RooFit::LineColor(kGreen), RooFit::Name("model_bb"));
    frameY->SetTitle((branchY + " projection").c_str());
    TLegend* legendY = new TLegend(0.6, 0.5, 0.9, 0.9);
    legendY->AddEntry(frameY->findObject("dataHist"), "Data", "p");
    legendY->AddEntry(frameY->findObject("model"), "Model", "l");
    legendY->AddEntry(frameY->findObject("model_ss"), ("Sig(" + model1s + ")-Sig(" + model2s + ") : " + std::to_string(int(n_ss.getVal() / nEntries * 100)) + "%").c_str(), "l");
    legendY->AddEntry(frameY->findObject("model_sb"), ("Sig(" + model1s + ")-Bkg(" + model2b + ") : " + std::to_string(int(n_sb.getVal() / nEntries * 100)) + "%").c_str(), "l");
    // legendY->AddEntry(frameY->findObject("model_bs"), ("Bkg(" + model1b + ")-Sig(" + model2s + ") : " + std::to_string(int(n_bs.getVal() / nEntries * 100)) + "%").c_str(), "l");
    legendY->AddEntry(frameY->findObject("model_bb"), ("Bkg(" + model1b + ")-Bkg(" + model2b + ") : " + std::to_string(int(n_bb.getVal() / nEntries * 100)) + "%").c_str(), "l");
    frameY->Draw();
    legendY->Draw();
    // Save the canvas
    projCanvas->SaveAs((outputDir + "2dfit_proj_" + suffix + ".png").c_str());

    // Compute sWeights
    RooDataSet data("data", "Unbinned Data", RooArgSet(x, y));
    RooStats::SPlot sData("sData", "sWeights", data, &model, RooArgList(n_ss, n_sb, n_bb));

    // Print first few sWeights
    // std::cout << "Event  x     sWeight(signal)  sWeight(background)" << std::endl;
    // for (int i = 0; i < 10; ++i) {
    //     const RooArgSet* row = data->get(i);
    //     double xVal = row->getRealValue("x");
    //     double swSignal = sData.GetSWeight(i, "nSignal");
    //     double swBackground = sData.GetSWeight(i, "nBackground");
    //     std::cout << i << "  " << xVal << "   " << swSignal << "   " << swBackground << std::endl;
    // }

    // Create new dataset weighted by sWeights (pure signal)
    RooDataSet* signalData = new RooDataSet("signalData", "Pure Signal Dataset", &data, *data.get(), 0, "n_ss_sw");

    // Plot the extracted signal-only dataset
    TCanvas* sWeightCanvas = new TCanvas("sWeightCanvas", "sWeighted Signal", 1000, 800);
    RooPlot* sWeightFrame = x.frame();
    signalData->plotOn(sWeightFrame);
    sWeightFrame->Draw();
    sWeightCanvas->SaveAs((outputDir + "2dfit_sweight_" + suffix + ".png").c_str());

    // Clean up memory
    delete hist2D;
    delete residualHist;
    delete heatmapCanvas;
    delete projCanvas;
    delete legendX;
    delete legendY;
    delete frameX;
    delete frameY;
    delete sWeightCanvas;
    delete sWeightFrame;
    delete signalData;
}

void PerformSPlot2D(TTree* tree, const std::string& branchX, const std::string& branchY, 
         const std::string& model1s = "gauss", const std::string& model1b = "linear", 
         const std::string& model2s = "gauss", const std::string& model2b = "linear",
         const std::string& outputDir = config::temp_fig_dir) {
    if (!check::CheckTreeBranches(tree, {branchX, branchY})) {
        return;
    }
    std::cout << "Fitting & sPlotting the branches '" << branchX << "' and '" << branchY << "' with models '" << model1s << "', '" << model1b << "', '" << model2s << "', and '" << model2b << "'..." << std::endl;

    // Define the range for the variables
    std::cout << "Defining the range for the branches..." << std::endl;
    Double_t xMin = tree->GetMinimum(branchX.c_str());
    Double_t xMax = tree->GetMaximum(branchX.c_str());
    Double_t yMin = tree->GetMinimum(branchY.c_str());
    Double_t yMax = tree->GetMaximum(branchY.c_str());
    std::cout << "Finished defining the range for the branches." << std::endl;

    // Define observables
    std::cout << "Defining the RooFit observables..." << std::endl;
    RooRealVar x(branchX.c_str(), branchX.c_str(), xMin, xMax);
    RooRealVar y(branchY.c_str(), branchY.c_str(), yMin, yMax);
    std::cout << "Finished defining the RooFit observables." << std::endl;

    // === Define Signal and Background Models ===
    std::cout << "Generating the signal and background models..." << std::endl;
    TH2F* hist2D = new TH2F("hist2D", "2D Histogram;X-axis;Y-axis", 100, xMin, xMax, 100, yMin, yMax);
    TH1D* projX = hist2D->ProjectionX("projX");
    RooAbsPdf* signalX_ss = GenerateModel(model1s, projX, x, "signalXss");
    // RooAbsPdf* signalX_sb = GenerateModel(model1s, projX, x, "signalXsb");
    // RooAbsPdf* backgroundX_bs = GenerateModel(model1b, projX, x, "backgroundXbs");
    RooAbsPdf* backgroundX_bb = GenerateModel(model1b, projX, x, "backgroundXbb");
    TH1D* projY = hist2D->ProjectionY("projY");
    RooAbsPdf* signalY_ss = GenerateModel(model2s, projY, y, "signalYss");
    // RooAbsPdf* backgroundY_sb = GenerateModel(model2b, projY, y, "backgroundYsb");
    // RooAbsPdf* signalY_bs = GenerateModel(model2s, projY, y, "signalYbs");
    RooAbsPdf* backgroundY_bb = GenerateModel(model2b, projY, y, "backgroundYbb");
    // if (!signalX_ss || !signalX_sb || !backgroundX_bs || !backgroundX_bb || !signalY_ss || !backgroundY_sb || !backgroundY_bb) {
    // if (!signalX_ss || !signalX_sb || !backgroundX_bb || !signalY_ss || !backgroundY_sb || !backgroundY_bb) {
    if (!signalX_ss || !backgroundX_bb || !signalY_ss || !backgroundY_bb) {
        return;
    }
    std::cout << "Finished generating the signal and background models." << std::endl;

    // === Define Combined Components ===
    std::cout << "Combining the signal and background models..." << std::endl;
    RooProdPdf model_ss("model_ss", "Signal X - Signal Y", RooArgList(*signalX_ss, *signalY_ss));
    RooProdPdf model_sb("model_sb", "Signal X - Background Y", RooArgList(*signalX_ss, *backgroundY_bb));
    // RooProdPdf model_bs("model_bs", "Background X - Signal Y", RooArgList(*backgroundX_bs, *signalY_bs));
    RooProdPdf model_bb("model_bb", "Background X - Background Y", RooArgList(*backgroundX_bb, *backgroundY_bb));
    std::cout << "Finished combining the signal and background models." << std::endl;

    // === Define Yields (Extended Fit) ===
    std::cout << "Defining the yields..." << std::endl;
    RooRealVar n_ss_1("n_ss_1", "Events (sigX-sigY)", 0.6 * tree->GetEntries(), 0.0001, tree->GetEntries());
    RooRealVar n_sb_1("n_sb_1", "Events (sigX-bkgY)", 0.15 * tree->GetEntries(), 0.0001, tree->GetEntries());
    // RooRealVar n_bs("n_bs", "Events (bkgX-sigY)", 0.2 * tree->GetEntries(), 0, tree->GetEntries());
    // RooRealVar n_bb("n_bb", "Events (bkgX-bkgY)", 0.25 * tree->GetEntries(), 0, tree->GetEntries());
    RooFormulaVar n_bb_1("n_bb_1", "Number of background-background events",
        // TString::Format("%f - (@0 + @1 + @2)", float(tree->GetEntries())),
        // RooArgList(n_ss, n_sb, n_bs));
                         TString::Format("%f - (@0 + @1)", float(tree->GetEntries())),
                         RooArgList(n_ss_1, n_sb_1));
    std::cout << "Finished defining the yields." << std::endl;

    // === Full Extended Model ===
    std::cout << "Defining the full extended model..." << std::endl;
    RooAddPdf model_1("model", "Total Model",
                    RooArgList(model_ss, model_sb, model_bb),
                    RooArgList(n_ss_1, n_sb_1, n_bb_1));
    std::cout << "Finished defining the full extended model." << std::endl;

    // === Create Unbinned Dataset ===
    std::cout << "Creating the unbinned dataset..." << std::endl;
    RooDataSet data("data", "Unbinned Data", RooArgSet(x, y), RooFit::Import(*tree));    
    std::cout << "Finished creating the unbinned dataset." << std::endl;

    // === Perform Fit ===
    std::cout << "Fitting the model to the data..." << std::endl;
    model_1.fitTo(data, RooFit::Extended(), RooFit::Verbose(kTRUE), RooFit::PrintLevel(10));
    // model.fitTo(data, RooFit::Extended(), RooFit::Verbose(kFALSE), RooFit::PrintLevel(-1));
    std::cout << "Sum of fitted yields: " << n_ss_1.getVal() + n_sb_1.getVal() + n_bb_1.getVal() << std::endl;
    std::cout << "True sum of yields: " << tree->GetEntries() << std::endl;

    // === Set values to constant
    std::cout << "Setting parameters to constant for sWeights..." << std::endl;
    RooArgSet* params = model_1.getParameters(RooArgSet());  // All parameters in the model
    for (auto param : *params) {
        // if (param->GetName() != n_ss.GetName() && param->GetName() != n_sb.GetName() && param->GetName() != n_bs.GetName() && param->GetName() != n_bb.GetName()) {
        if (param->GetName() != n_ss_1.GetName() && param->GetName() != n_sb_1.GetName() && param->GetName() != n_bb_1.GetName()) {
            param->setAttribute("Constant");
        }
    }
    std::cout << "Finished setting parameters to constant for sWeights." << std::endl;

    // === Second Fit ===
    std::cout << "Performing the second fit..." << std::endl;
    RooRealVar n_ss("n_ss", "Events (sigX-sigY)", n_ss_1.getVal(), 0, tree->GetEntries());
    RooRealVar n_sb("n_sb", "Events (sigX-bkgY)", n_sb_1.getVal(), 0, tree->GetEntries());
    // RooRealVar n_bs("n_bs", "Events (bkgX-sigY)", n_bs.getVal(), 0, tree->GetEntries());
    RooRealVar n_bb("n_bb", "Events (bkgX-bkgY)", n_bb_1.getVal(), 0, tree->GetEntries());
    RooAddPdf model("model", "Total Model",
                    RooArgList(model_ss, model_sb, model_bb),
                    RooArgList(n_ss, n_sb, n_bb));
    model.fitTo(data, RooFit::Extended(), RooFit::Verbose(kTRUE), RooFit::PrintLevel(10));
    std::cout << "Finished performing the second fit." << std::endl;


    // === Compute sWeights ===
    std::cout << "Computing sWeights..." << std::endl;
    // RooStats::SPlot sData("sData", "sWeights", data, &model, RooArgList(n_ss, n_sb, n_bs, n_bb));
    RooStats::SPlot sData("sData", "sWeights", data, &model, RooArgList(n_ss, n_sb, n_bb));

    // fit again
    // std::cout << "Skipping the second fit..." << std::endl;
    // std::cout << "Performing the second fit..." << std::endl;
    // model.fitTo(data, RooFit::Extended(), RooFit::Verbose(kFALSE), RooFit::PrintLevel(-1));
    std::cout << "Finished computing sWeights." << std::endl;

    std::cout << "Yield of Sig-Sig is " 
        << n_ss.getVal() << ".  From sWeights it is "
        << sData.GetYieldFromSWeight("n_ss") << std::endl
        << std::endl;
    std::cout << "Yield of Bkg-Sig is "
        << n_sb.getVal() << ".  From sWeights it is "
        << sData.GetYieldFromSWeight("n_sb") << std::endl
        << std::endl;
    std::cout << "Yield of Bkg-Bkg is "
        << n_bb.getVal() << ".  From sWeights it is "
        << sData.GetYieldFromSWeight("n_bb") << std::endl
        << std::endl;

    // === Extract Signal Dataset (sigX-sigY) ===
    // SPlot automatically adds variables with "_sw" suffix
    // RooDataSet* pureSignalData = new RooDataSet("pureSignalData", "Pure Signal Dataset", &data, *data.get(), 0, "n_ss_sw");

    // // === Plot Extracted Signal ===
    // TCanvas* canvasX = new TCanvas("canvas", ("sWeighted Signal " + branchX + " Projection").c_str(), 1000, 800);
    // RooPlot* frameX = x.frame();
    // pureSignalData->plotOn(frameX);
    // frameX->SetTitle(("sWeighted Signal " + branchX + " Projection").c_str());
    // frameX->Draw();
    // canvasX->SaveAs((outputDir + "sWeighted_signal_" + branchX + ".png").c_str());

    // TCanvas* canvasY = new TCanvas("canvas", ("sWeighted Signal " + branchY + " Projection").c_str(), 1000, 800);
    // RooPlot* frameY = y.frame();
    // pureSignalData->plotOn(frameY);
    // frameY->SetTitle(("sWeighted Signal " + branchY + " Projection").c_str());
    // frameY->Draw();
    // canvasY->SaveAs((outputDir + "sWeighted_signal_" + branchY + ".png").c_str());
    // delete canvasX; delete canvasY;

    // RooRealVar kp("K_P", "K_P", tree->GetMinimum("K_P"), tree->GetMaximum("K_P"));
    // RooRealVar pi1p("pi1_P", "pi1_P", tree->GetMinimum("pi1_P"), tree->GetMaximum("pi1_P"));
    // RooRealVar pi2p("pi2_P", "pi2_P", tree->GetMinimum("pi2_P"), tree->GetMaximum("pi2_P"));
    // RooRealVar pi3p("pi3_P", "pi3_P", tree->GetMinimum("pi3_P"), tree->GetMaximum("pi3_P"));
    // RooRealVar pisp("pis_P", "pis_P", tree->GetMinimum("pis_P"), tree->GetMaximum("pis_P"));

    // RooDataSet dataKP("dataKP", "Unbinned Data with K_P", RooArgSet(x, y, kp), RooFit::Import(*tree));
    // RooDataSet datapiP("datapiP", "Unbinned Data with pi_P", RooArgSet(x, y, pi1p, pi2p, pi3p, pisp), RooFit::Import(*tree));
    
    // PlotWeightedHistogram(data, sData, branchX, outputDir);
    // std::cout << "Plotted sWeighted " << branchX << std::endl;
    // PlotWeightedHistogram(data, sData, branchY, outputDir);
    // std::cout << "Plotted sWeighted " << branchY << std::endl;
    // PlotWeightedHistogram(dataKP, sData, "K_P", outputDir);
    // std::cout << "Plotted sWeighted K_P" << std::endl;
    // PlotWeightedHistogram(datapiP, sData, "pi1_P", outputDir);
    // std::cout << "Plotted sWeighted pi1_P" << std::endl;
    // PlotWeightedHistogram(datapiP, sData, {"pi1_P", "pi2_P", "pi3_P", "pis_P"}, "pi_P", outputDir);  // TODO: NOT WORKING

    if (false) {
        // Create a residual heatmap
        int nEntries = tree->GetEntries();
        int nBinsX = 100; 
        int nBinsY = 100;
        RooDataHist dataHist("dataHist", "Binned Data", RooArgSet(x, y), hist2D);
        std::cout << "Creating the residual heatmap..." << std::endl;
        TH2F* residualHist = new TH2F("residualHist", "Residual Heatmap;X-axis;Y-axis", nBinsX, xMin, xMax, nBinsY, yMin, yMax);
        RooArgSet vars(x, y);
        // Loop through each bin and calculate residuals
        std::cout << "Calculating residuals..." << std::endl;
        double xBinWidth = hist2D->GetXaxis()->GetBinWidth(1); // Get the bin width of the first X bin
        double yBinWidth = hist2D->GetYaxis()->GetBinWidth(1); // Get the bin width of the first Y bin
        
        for (int iX = 1; iX <= nBinsX; ++iX) {
            // Get the bin center
            double centerX = hist2D->GetXaxis()->GetBinCenter(iX);
            x.setVal(centerX);
            for (int iY = 1; iY <= nBinsY; ++iY) {
                double centerY = hist2D->GetYaxis()->GetBinCenter(iY);

                // Set the variable values for the model
                y.setVal(centerY);

                // Get observed and predicted densities
                double observed = hist2D->GetBinContent(iX, iY); // Observed count
                double predicted = model.getVal(&vars); // Predicted density
                predicted *= nEntries * xBinWidth * yBinWidth; // Scale the density to the number of entries

                // Calculate the residual
                double residual = observed - predicted;
                residual /= sqrt(observed);

                // Fill the residual histogram
                residualHist->SetBinContent(iX, iY, residual);
            }
        }

        // Create a canvas for the residual heatmap
        std::cout << "Creating the canvas for the residual heatmap..." << std::endl;
        TCanvas* heatmapCanvas = new TCanvas("heatmapCanvas", "Residual Heatmap", 1000, 800);
        int chi2 = int(model.createChi2(dataHist)->getVal());
        residualHist->Draw("COLZ"); // Draw the residual heatmap
        residualHist->GetXaxis()->SetTitle(branchX.c_str());
        residualHist->GetYaxis()->SetTitle(branchY.c_str());
        residualHist->SetTitle(("Heatmap of Residual/sqrt(Data) (Chi2: " + std::to_string(chi2) + ")").c_str());
        heatmapCanvas->SetRightMargin(0.15);  // color bar issue
        gStyle->SetOptStat(0);
        std::string suffix = std::string(1, model1s[0]) + model1b[0] + model2s[0] + model2b[0];
        heatmapCanvas->SaveAs((outputDir + "2dfit_hm_" + suffix + ".png").c_str());
        std::cout << "Residual heatmap saved to file." << std::endl;

        // Create a canvas for the 2D Gaussian fit projections
        std::cout << "Creating the canvas for the 2D Gaussian fit projections..." << std::endl;
        TCanvas* projCanvas = new TCanvas("projCanvas", "2D Gaussian Fit (Binned)", 1000, 800);
        projCanvas->Divide(1, 2);

        // X projection
        std::cout << "Creating the X projection(" << branchX << ")..." << std::endl;
        projCanvas->cd(1);
        RooPlot* frameX = x.frame();
        dataHist.plotOn(frameX, RooFit::Name("dataHist"));  // data
        model.plotOn(frameX, RooFit::Name("model"));  // model
        model.plotOn(frameX, RooFit::Components("model_ss"), RooFit::LineStyle(kDashed), RooFit::LineColor(kRed), RooFit::Name("model_ss"));  // signal-signal
        model.plotOn(frameX, RooFit::Components("model_sb"), RooFit::LineStyle(kDashed), RooFit::LineColor(kYellow), RooFit::Name("model_sb"));  // signal-background
        // model.plotOn(frameX, RooFit::Components("model_bs"), RooFit::LineStyle(kDashed), RooFit::LineColor(kBlue), RooFit::Name("model_bs"));  // background-signal
        model.plotOn(frameX, RooFit::Components("model_bb"), RooFit::LineStyle(kDashed), RooFit::LineColor(kGreen), RooFit::Name("model_bb"));  // background-background
        frameX->SetTitle((branchX + " projection").c_str());
        // legends
        TLegend* legendX = new TLegend(0.6, 0.5, 0.9, 0.9);
        legendX->AddEntry(frameX->findObject("dataHist"), "Data", "p");
        legendX->AddEntry(frameX->findObject("model"), "Model", "l");
        legendX->AddEntry(frameX->findObject("model_ss"), ("Sig(" + model1s + ")-Sig(" + model2s + ") : " + std::to_string(int(n_ss.getVal() / nEntries * 100)) + "%").c_str(), "l");
        legendX->AddEntry(frameX->findObject("model_sb"), ("Sig(" + model1s + ")-Bkg(" + model2b + ") : " + std::to_string(int(n_sb.getVal() / nEntries * 100)) + "%").c_str(), "l");
        // legendX->AddEntry(frameX->findObject("model_bs"), ("Bkg(" + model1b + ")-Sig(" + model2s + ") : " + std::to_string(int(n_bs.getVal() / nEntries * 100)) + "%").c_str(), "l");
        legendX->AddEntry(frameX->findObject("model_bb"), ("Bkg(" + model1b + ")-Bkg(" + model2b + ") : " + std::to_string(int(n_bb.getVal() / nEntries * 100)) + "%").c_str(), "l");
        frameX->Draw();
        legendX->Draw();

        // Y projection
        std::cout << "Creating the Y projection(" << branchY << ")..." << std::endl;
        projCanvas->cd(2);
        RooPlot* frameY = y.frame();
        dataHist.plotOn(frameY, RooFit::Name("dataHist"));  // data
        model.plotOn(frameY, RooFit::Name("model"));  // model
        model.plotOn(frameY, RooFit::Components("model_ss"), RooFit::LineStyle(kDashed), RooFit::LineColor(kRed), RooFit::Name("model_ss"));  // signal-signal
        model.plotOn(frameY, RooFit::Components("model_sb"), RooFit::LineStyle(kDashed), RooFit::LineColor(kYellow), RooFit::Name("model_sb"));
        // model.plotOn(frameY, RooFit::Components("model_bs"), RooFit::LineStyle(kDashed), RooFit::LineColor(kBlue), RooFit::Name("model_bs"));
        model.plotOn(frameY, RooFit::Components("model_bb"), RooFit::LineStyle(kDashed), RooFit::LineColor(kGreen), RooFit::Name("model_bb"));
        frameY->SetTitle((branchY + " projection").c_str());
        TLegend* legendY = new TLegend(0.6, 0.5, 0.9, 0.9);
        legendY->AddEntry(frameY->findObject("dataHist"), "Data", "p");
        legendY->AddEntry(frameY->findObject("model"), "Model", "l");
        legendY->AddEntry(frameY->findObject("model_ss"), ("Sig(" + model1s + ")-Sig(" + model2s + ") : " + std::to_string(int(n_ss.getVal() / nEntries * 100)) + "%").c_str(), "l");
        legendY->AddEntry(frameY->findObject("model_sb"), ("Sig(" + model1s + ")-Bkg(" + model2b + ") : " + std::to_string(int(n_sb.getVal() / nEntries * 100)) + "%").c_str(), "l");
        // legendY->AddEntry(frameY->findObject("model_bs"), ("Bkg(" + model1b + ")-Sig(" + model2s + ") : " + std::to_string(int(n_bs.getVal() / nEntries * 100)) + "%").c_str(), "l");
        legendY->AddEntry(frameY->findObject("model_bb"), ("Bkg(" + model1b + ")-Bkg(" + model2b + ") : " + std::to_string(int(n_bb.getVal() / nEntries * 100)) + "%").c_str(), "l");
        frameY->Draw();
        legendY->Draw();
        // Save the canvas
        projCanvas->SaveAs((outputDir + "2dfit_proj_" + suffix + ".png").c_str());

    }
       
    // === Save sWeighted Data to the File ===
    // make a new tree, copy the old tree, add branches n_ss_sw, n_sb_sw, n_bs_sw, n_bb_sw
    TFile* sWeightFile = new TFile("sweight.root", "RECREATE");
    TTree* sWeightTree = tree->CloneTree(0);
    Double_t n_ss_sw, n_sb_sw, n_bb_sw;
    sWeightTree->Branch("n_ss_sw", &n_ss_sw, "n_ss_sw/D");
    sWeightTree->Branch("n_sb_sw", &n_sb_sw, "n_sb_sw/D");
    // sWeightTree->Branch("n_bs_sw", &n_bs_sw, "n_bs_sw/D");
    sWeightTree->Branch("n_bb_sw", &n_bb_sw, "n_bb_sw/D");
    for (int i = 0; i < tree->GetEntries(); ++i) {
        tree->GetEntry(i);
        n_ss_sw = sData.GetSWeight(i, "n_ss");
        n_sb_sw = sData.GetSWeight(i, "n_sb");
        // n_bs_sw = sData.GetSWeight(i, "n_bs");
        n_bb_sw = sData.GetSWeight(i, "n_bb");
        sWeightTree->Fill();
    }
    sWeightTree->Write();
    sWeightFile->Close();
    std::cout << "SPlot results saved to sweight.root" << std::endl;

    // Clean up memory; the deletion should be in the reverse order of creation
    delete params;
    delete backgroundY_bb;
    // delete signalY_bs;
    // delete backgroundY_sb;
    delete signalY_ss;
    delete backgroundX_bb;
    // delete backgroundX_bs;
    // delete signalX_sb;
    delete signalX_ss;
    delete projY;
    delete projX;
    delete hist2D;
}

// Perform a 2d binned fit, then fix all but n_xx to fit an unbinned fit, then do sPlot
void BinnedFitUnbinnedFitSPlot(
    TTree* tree, const std::string& branchX, const std::string& branchY, 
    const std::string& model1s = "gauss", const std::string& model1b = "linear", 
    const std::string& model2s = "gauss", const std::string& model2b = "linear",
    const std::string& outputDir = config::temp_fig_dir
) {
    if (!check::CheckTreeBranches(tree, {branchX, branchY})) {
        return;
    }
    std::cout << "Fitting the branches '" << branchX << "' and '" << branchY << "' with models '" << model1s << "', '" << model1b << "', '" << model2s << "', and '" << model2b << "'..." << std::endl;

    // Define the range for the variables
    std::cout << "Defining the range for the branches..." << std::endl;
    Double_t xMin = tree->GetMinimum(branchX.c_str());
    Double_t xMax = tree->GetMaximum(branchX.c_str());
    Double_t yMin = tree->GetMinimum(branchY.c_str());
    Double_t yMax = tree->GetMaximum(branchY.c_str());

    // Define the independent variables
    std::cout << "Defining the RooFit independent variables..." << std::endl;
    RooRealVar x(branchX.c_str(), branchX.c_str(), xMin, xMax);
    RooRealVar y(branchY.c_str(), branchY.c_str(), yMin, yMax);
    RooArgSet vars;
    vars.add(x);
    vars.add(y);

    // Create a 2D histogram (binned data)
    std::cout << "Creating the 2D histogram..." << std::endl;
    const int nBinsX = 100;
    const int nBinsY = 100;
    TH2F* hist2D = new TH2F("hist2D", "2D Histogram;X-axis;Y-axis", nBinsX, xMin, xMax, nBinsY, yMin, yMax);
    tree->Draw((branchY + ":" + branchX + ">>hist2D").c_str(), "", "goff");
    TH1D* projX = hist2D->ProjectionX("projX");
    TH1D* projY = hist2D->ProjectionY("projY");

    // Import the histogram into RooFit
    std::cout << "Importing the histogram into RooFit..." << std::endl;
    // dataHist instead of dataSets for binned data
    // RooDataHist dataHist("dataHist", "Binned Data", RooArgSet(x, y), hist2D);
    RooDataHist dataHist("dataHist", "Binned Data", vars, hist2D);

    // Generate the signal and background models for X
    std::cout << "Generating the signal and background models for X..." << std::endl;
    // PDF variables are named as: (type of model)(X/Y)_(type of X model)(type of Y model)
    RooRealVar* xPtr = &x;
    // RooAbsPdf* signalX_ss = GenerateModel(model1s, projX, x, "signalXss");
    RooAbsPdf* signalX_ss = GenerateModel(model1s, projX, xPtr, ("signal_" + branchX).c_str());
    RooAbsPdf* backgroundX_bb = GenerateModel(model1b, projX, x, ("background_" + branchX).c_str());
    if (!signalX_ss || !backgroundX_bb) {
        delete hist2D;
        return;
    }
    
    // Generate the signal and background models for Y
    std::cout << "Generating the signal and background models for Y..." << std::endl;
    RooAbsPdf* signalY_ss = GenerateModel(model2s, projY, y, ("signal_" + branchY).c_str());
    RooAbsPdf* backgroundY_bb = GenerateModel(model2b, projY, y, ("background_" + branchY).c_str());
    if (!signalY_ss || !backgroundY_bb) {
        delete hist2D;
        return;
    }

    // Combine the signal and background models for X
    std::cout << "Combining the signal and background models..." << std::endl;
    int nEntries = hist2D->GetEntries();
    RooRealVar n_ss("n_ss", "Number of signal-signal events", 0.6 * nEntries, 0.0001, nEntries);
    RooRealVar n_sb("n_sb", "Number of signal-background events", 0.15 * nEntries, 0.0001, nEntries);
    // RooRealVar n_bs("n_bs", "Number of background-signal events", 0.2 * nEntries, 0, nEntries);
    RooRealVar n_bb("n_bb", "Number of background-background events", 0.2 * nEntries, 0.0001, nEntries);

    RooProdPdf model_ss("model_ss", "Signal X + Signal Y", RooArgList(*signalX_ss, *signalY_ss));
    RooProdPdf model_sb("model_sb", "Signal X + Background Y", RooArgList(*signalX_ss, *backgroundY_bb));
    // RooProdPdf model_bs("model_bs", "Background X + Signal Y", RooArgList(*backgroundX_bs, *signalY_bs));
    RooProdPdf model_bb("model_bb", "Background X + Background Y", RooArgList(*backgroundX_bb, *backgroundY_bb));

    // RooAddPdf model("model", "Signal + Background", RooArgList(model_ss, model_sb, model_bs, model_bb), RooArgList(n_ss, n_sb, n_bs, n_bb));
    RooAddPdf model("model", "Signal + Background", RooArgList(model_ss, model_sb, model_bb), RooArgList(n_ss, n_sb, n_bb));

    std::cout << "Fitting the model to the data..." << std::endl;
    // model.fitTo(dataHist, RooFit::Extended(kTRUE), RooFit::Verbose(), RooFit::PrintLevel(10));  // extended maximum likelihood fit (the same result)
    model.fitTo(dataHist, RooFit::Verbose(kTRUE), RooFit::PrintLevel(10));

    
    std::cout << "Creating the unbinned dataset..." << std::endl;
    RooDataSet data("data", "Unbinned Data", RooArgSet(x, y), RooFit::Import(*tree));

    std::cout << "Setting all but n_xx constant..." << std::endl;
    RooArgSet* params = model.getParameters(RooArgSet());
    for (auto param : *params) {
        if (param->GetName() != n_ss.GetName() && param->GetName() != n_sb.GetName() && param->GetName() != n_bb.GetName()) {
            param->setAttribute("Constant");
        }
    }
    n_ss.setMin(0);
    n_sb.setMin(0);
    // n_bs.setMin(0);
    n_bb.setMin(0);

    std::cout << "Performing the second fit..." << std::endl;
    model.fitTo(data, RooFit::Extended(), RooFit::Verbose(kTRUE), RooFit::PrintLevel(10));

    std::cout << "Computing sWeights..." << std::endl;
    RooStats::SPlot sData("sData", "sWeights", data, &model, RooArgList(n_ss, n_sb, n_bb));
    
    std::cout << "Yield comparison:" << std::endl;
    std::cout << "Sig-Sig: " << n_ss.getVal() << " vs " << sData.GetYieldFromSWeight("n_ss") << std::endl;
    std::cout << "Sig-Bkg: " << n_sb.getVal() << " vs " << sData.GetYieldFromSWeight("n_sb") << std::endl;
    std::cout << "Bkg-Bkg: " << n_bb.getVal() << " vs " << sData.GetYieldFromSWeight("n_bb") << std::endl;

    // Save sWeighted data to the file
    TFile* sWeightFile = new TFile("sweight.root", "RECREATE");
    TTree* sWeightTree = tree->CloneTree(0);
    Double_t n_ss_sw, n_sb_sw, n_bb_sw;
    sWeightTree->Branch("n_ss_sw", &n_ss_sw, "n_ss_sw/D");
    sWeightTree->Branch("n_sb_sw", &n_sb_sw, "n_sb_sw/D");
    // sWeightTree->Branch("n_bs_sw", &n_bs_sw, "n_bs_sw/D");
    sWeightTree->Branch("n_bb_sw", &n_bb_sw, "n_bb_sw/D");
    for (int i = 0; i < tree->GetEntries(); ++i) {
        tree->GetEntry(i);
        n_ss_sw = sData.GetSWeight(i, "n_ss");
        n_sb_sw = sData.GetSWeight(i, "n_sb");
        // n_bs_sw = sData.GetSWeight(i, "n_bs");
        n_bb_sw = sData.GetSWeight(i, "n_bb");
        sWeightTree->Fill();
    }
    sWeightTree->Write();
    sWeightFile->Close();

    // Create a residual heatmap
    std::cout << "Creating the residual heatmap..." << std::endl;
    TH2F* residualHist = new TH2F("residualHist", "Residual Heatmap;X-axis;Y-axis", nBinsX, xMin, xMax, nBinsY, yMin, yMax);
    // RooArgSet vars(x, y);
    // Loop through each bin and calculate residuals
    std::cout << "Calculating residuals..." << std::endl;
    double xBinWidth = hist2D->GetXaxis()->GetBinWidth(1); // Get the bin width of the first X bin
    double yBinWidth = hist2D->GetYaxis()->GetBinWidth(1); // Get the bin width of the first Y bin
    for (int iX = 1; iX <= nBinsX; ++iX) {
        // Get the bin center
        double centerX = hist2D->GetXaxis()->GetBinCenter(iX);
        x.setVal(centerX);
        for (int iY = 1; iY <= nBinsY; ++iY) {
            double centerY = hist2D->GetYaxis()->GetBinCenter(iY);

            // Set the variable values for the model
            y.setVal(centerY);

            // Get observed and predicted densities
            double observed = hist2D->GetBinContent(iX, iY); // Observed count
            double predicted = model.getVal(&vars); // Predicted density
            predicted *= nEntries * xBinWidth * yBinWidth; // Scale the density to the number of entries

            // Calculate the residual
            double residual = observed - predicted;
            residual /= sqrt(observed);

            // Fill the residual histogram
            residualHist->SetBinContent(iX, iY, residual);
        }
    }
    // Create a canvas for the residual heatmap
    std::cout << "Creating the canvas for the residual heatmap..." << std::endl;
    TCanvas* heatmapCanvas = new TCanvas("heatmapCanvas", "Residual Heatmap", 1000, 800);
    int chi2 = int(model.createChi2(dataHist)->getVal());
    residualHist->Draw("COLZ"); // Draw the residual heatmap
    residualHist->GetXaxis()->SetTitle(branchX.c_str());
    residualHist->GetYaxis()->SetTitle(branchY.c_str());
    residualHist->SetTitle(("Heatmap of Residual/sqrt(Data) (Chi2: " + std::to_string(chi2) + ")").c_str());
    heatmapCanvas->SetRightMargin(0.15);  // color bar issue
    gStyle->SetOptStat(0);
    std::string suffix = std::string(1, model1s[0]) + model1b[0] + model2s[0] + model2b[0];
    heatmapCanvas->SaveAs((outputDir + "2dfit_hm_" + suffix + ".png").c_str());
    std::cout << "Residual heatmap saved to file." << std::endl;

    // Create a canvas for the 2D Gaussian fit projections
    std::cout << "Creating the canvas for the 2D Gaussian fit projections..." << std::endl;
    TCanvas* projCanvas = new TCanvas("projCanvas", "2D Gaussian Fit (Binned)", 1000, 800);
    projCanvas->Divide(1, 2);

    // X projection
    std::cout << "Creating the X projection(" << branchX << ")..." << std::endl;
    projCanvas->cd(1);
    RooPlot* frameX = x.frame();
    dataHist.plotOn(frameX, RooFit::Name("dataHist"));  // data
    model.plotOn(frameX, RooFit::Name("model"));  // model
    model.plotOn(frameX, RooFit::Components("model_ss"), RooFit::LineStyle(kDashed), RooFit::LineColor(kRed), RooFit::Name("model_ss"));  // signal-signal
    model.plotOn(frameX, RooFit::Components("model_sb"), RooFit::LineStyle(kDashed), RooFit::LineColor(kYellow), RooFit::Name("model_sb"));  // signal-background
    // model.plotOn(frameX, RooFit::Components("model_bs"), RooFit::LineStyle(kDashed), RooFit::LineColor(kBlue), RooFit::Name("model_bs"));  // background-signal
    model.plotOn(frameX, RooFit::Components("model_bb"), RooFit::LineStyle(kDashed), RooFit::LineColor(kGreen), RooFit::Name("model_bb"));  // background-background
    frameX->SetTitle((branchX + " projection").c_str());
    // legends
    TLegend* legendX = new TLegend(0.6, 0.5, 0.9, 0.9);
    legendX->AddEntry(frameX->findObject("dataHist"), "Data", "p");
    legendX->AddEntry(frameX->findObject("model"), "Model", "l");
    legendX->AddEntry(frameX->findObject("model_ss"), ("Sig(" + model1s + ")-Sig(" + model2s + ") : " + std::to_string(int(n_ss.getVal() / nEntries * 100)) + "%").c_str(), "l");
    legendX->AddEntry(frameX->findObject("model_sb"), ("Sig(" + model1s + ")-Bkg(" + model2b + ") : " + std::to_string(int(n_sb.getVal() / nEntries * 100)) + "%").c_str(), "l");
    // legendX->AddEntry(frameX->findObject("model_bs"), ("Bkg(" + model1b + ")-Sig(" + model2s + ") : " + std::to_string(int(n_bs.getVal() / nEntries * 100)) + "%").c_str(), "l");
    legendX->AddEntry(frameX->findObject("model_bb"), ("Bkg(" + model1b + ")-Bkg(" + model2b + ") : " + std::to_string(int(n_bb.getVal() / nEntries * 100)) + "%").c_str(), "l");
    frameX->Draw();
    legendX->Draw();

    // Y projection
    std::cout << "Creating the Y projection(" << branchY << ")..." << std::endl;
    projCanvas->cd(2);
    RooPlot* frameY = y.frame();
    dataHist.plotOn(frameY, RooFit::Name("dataHist"));  // data
    model.plotOn(frameY, RooFit::Name("model"));  // model
    model.plotOn(frameY, RooFit::Components("model_ss"), RooFit::LineStyle(kDashed), RooFit::LineColor(kRed), RooFit::Name("model_ss"));  // signal-signal
    model.plotOn(frameY, RooFit::Components("model_sb"), RooFit::LineStyle(kDashed), RooFit::LineColor(kYellow), RooFit::Name("model_sb"));
    // model.plotOn(frameY, RooFit::Components("model_bs"), RooFit::LineStyle(kDashed), RooFit::LineColor(kBlue), RooFit::Name("model_bs"));
    model.plotOn(frameY, RooFit::Components("model_bb"), RooFit::LineStyle(kDashed), RooFit::LineColor(kGreen), RooFit::Name("model_bb"));
    frameY->SetTitle((branchY + " projection").c_str());
    TLegend* legendY = new TLegend(0.6, 0.5, 0.9, 0.9);
    legendY->AddEntry(frameY->findObject("dataHist"), "Data", "p");
    legendY->AddEntry(frameY->findObject("model"), "Model", "l");
    legendY->AddEntry(frameY->findObject("model_ss"), ("Sig(" + model1s + ")-Sig(" + model2s + ") : " + std::to_string(int(n_ss.getVal() / nEntries * 100)) + "%").c_str(), "l");
    legendY->AddEntry(frameY->findObject("model_sb"), ("Sig(" + model1s + ")-Bkg(" + model2b + ") : " + std::to_string(int(n_sb.getVal() / nEntries * 100)) + "%").c_str(), "l");
    // legendY->AddEntry(frameY->findObject("model_bs"), ("Bkg(" + model1b + ")-Sig(" + model2s + ") : " + std::to_string(int(n_bs.getVal() / nEntries * 100)) + "%").c_str(), "l");
    legendY->AddEntry(frameY->findObject("model_bb"), ("Bkg(" + model1b + ")-Bkg(" + model2b + ") : " + std::to_string(int(n_bb.getVal() / nEntries * 100)) + "%").c_str(), "l");
    frameY->Draw();
    legendY->Draw();
    // Save the canvas
    projCanvas->SaveAs((outputDir + "2dfit_proj_" + suffix + ".png").c_str());

    std::cout << "Cleaning up..." << std::endl;
    delete params;
    std::cout << "Deleted params" << std::endl;
    // delete backgroundY;
    // std::cout << "Deleted backgroundY" << std::endl;
    // delete signalY;
    // std::cout << "Deleted signalY" << std::endl;
    // delete backgroundX;
    // std::cout << "Deleted backgroundX" << std::endl;
    // delete signalX;
    // std::cout << "Deleted signalX" << std::endl;
    delete projY;
    std::cout << "Deleted projY" << std::endl;
    delete projX;
    std::cout << "Deleted projX" << std::endl;
    delete hist2D;
    std::cout << "Deleted hist2D" << std::endl;
}

void CutAndPlot(TTree* tree, float cut) {
    // TODO: apply cuts
    if (fs::exists("temp.root")) {
        std::cerr << "Error: temp.root already exists! Deleting..." << std::endl;
        if (std::remove("temp.root") != 0) {
            std::cerr << "Error: Failed to delete temp.root" << std::endl;
        } else {
            std::cout << "Deleted temp.root" << std::endl;
        }
    }
    TFile* tempFile = new TFile("temp.root", "RECREATE");
    std::cout << "Applying cuts (bdt_all > " << cut << ") on masses and saving to temp.root..." << std::endl;
    TTree* cut_tree = tree->CopyTree(("bdt_all > " + std::to_string(cut)).c_str());

    // Perform 2D fit
    // Fit(cut_tree, "D_M", "delta_M", "cb", "exp", "cb", "dstd0bg", plots_dir + "4_massfit_cpp/2d/");
    // PerformSPlot2D(cut_tree, "D_M", "delta_M", "cb", "exp", "cb", "dstd0bg", plots_dir + "4_massfit_cpp/2d/");
    BinnedFitUnbinnedFitSPlot(cut_tree, "D_M", "delta_M", "cb", "exp", "cb", "dstd0bg", config::plots_dir + "4_massfit_cpp/2d/");

    // Cleanup
    std::cout << "Closing temporary file..." << std::endl;
    tempFile->Close();
    if (std::remove("temp.root") != 0) {
        std::cerr << "Error: Failed to delete temp.root" << std::endl;
    } else {
        std::cout << "Deleted temp.root" << std::endl;
    }
}

void main04_2() {
    std::cout << "Loading TTree from file..." << std::endl;
    // check if masscut_root_file exists
    TFile* file;
    TTree* tree;
    if (std::filesystem::exists(config::masscut_rootwithprob_file)) {
        std::cout << "Reading " << config::masscut_rootwithprob_file << "..." << std::endl;
        auto [cutfile, cuttree] = tman::loadFileAndTree(config::masscut_rootwithprob_file);
        file = cutfile;
        tree = cuttree;
    } else if (std::filesystem::exists(config::full_rootwithprob_file)) {
        std::cout << "File " << config::masscut_rootwithprob_file << " does not exist. Loading " << config::full_rootwithprob_file << "..." << std::endl;
        auto [fullfile, fulltree] = tman::loadFileAndTree(config::full_rootwithprob_file);
        bool debug_mode = false;
        if (debug_mode) {
            file = fullfile;
            tree = fulltree;
        } else {
            std::cout << "Applying cuts on masses..." << std::endl;
            auto [tempfile, temptree] = tman::massCutFullTree(fulltree);
            file = tempfile;
            tree = temptree;
            delete fulltree;
            delete fullfile;

            // delete temp.root
            if (std::remove("temp.root") != 0) {
                std::cerr << "Error: Failed to delete temp.root" << std::endl;
            } else {
                std::cout << "Deleted temp.root" << std::endl;
            }
        }
    } else {
        std::cerr << "Error: Files " << config::masscut_rootwithprob_file << " and " << config::full_rootwithprob_file << " do not exist!" << std::endl;
    }
    std::cout << "Loaded TTree from file" << std::endl;
    if (!check::CheckTreeBranches(tree, {"D_M", "delta_M"})) {
        return;
    }

    // Plot the contour heatmap
    // plotter::PlotContourHeatmap(tree, "D_M", "delta_M", plots_dir + "4_massfit_cpp/contour_heatmap.png");

    // Fit the 2D models
    CutAndPlot(tree, 0.5);
    // NewCutAndPlot(tree, 0.5);

    // Cleanup
    std::cout << "Closing file..." << std::endl;
    file->Close();
}