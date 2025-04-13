#include <iostream>
#include <string>
#include <fstream>
#include <TFile.h>
#include <TTree.h>
#include <TH2F.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <RooRealVar.h>
#include <RooGaussian.h>
#include <TStyle.h>  // for gStyle
#include <RooDataSet.h>
#include <RooArgList.h>
#include <RooArgSet.h>
#include <RooAbsCollection.h>
#include <RooFit.h>
#include <RooCmdArg.h>
#include <RooProdPdf.h>
#include <RooAddPdf.h>
#include <RooGenericPdf.h>
#include <RooPlot.h>
#include <RooStats/SPlot.h>
#include "../utils/config.cpp"
#include "../utils/treeops.cpp"
#include "../utils/fileio.cpp"
#include "../utils/check.cpp"
#include "models.cpp"


class ShitFitter {
    public:
        ShitFitter(TTree* inputTree, const std::string& inputBranchX, const std::string& inputBranchY, 
                const std::string& inputModel1s, const std::string& inputModel1b, 
                const std::string& inputModel2s, const std::string& inputModel2b, 
                const std::string& inputOutputDir);
        void Fit();
        void Plot();
        // void ExportSWeights(const std::string& outputFileName);
        void ExportSWeights(const std::string& modelName, const std::string& outputFileName);
        ~ShitFitter();
    private:
        // we are using unique_ptr as much as possible as it automatically deletes the object when it goes out of scope
        // and root pointers should be deleted in order which is not so intuitive
        TTree* tree;
        std::string branchX; std::string branchY;
        std::string model1s; std::string model1b; std::string model2s; std::string model2b;
        std::string outputDir;
        Double_t xMin; Double_t xMax; Double_t yMin; Double_t yMax;
        int nBinsX; int nBinsY; int nEntries;
        std::unique_ptr<RooRealVar> xPtr; std::unique_ptr<RooRealVar> yPtr;
        RooArgSet vars;
        TH2F* hist2D;  // ROOT steals ownership of the histogram hence we cannot use unique_ptr
        TH1D* projX; TH1D* projY;  // hist2D owns its projections
        std::unique_ptr<RooDataHist> dataHist;
        std::unique_ptr<RooAbsPdf> signalXPtr; std::unique_ptr<RooAbsPdf> backgroundXPtr;
        std::unique_ptr<RooAbsPdf> signalYPtr; std::unique_ptr<RooAbsPdf> backgroundYPtr;
        std::unique_ptr<RooRealVar> n_ssPtr; std::unique_ptr<RooRealVar> n_sbPtr;
        std::unique_ptr<RooRealVar> n_bsPtr; std::unique_ptr<RooRealVar> n_bbPtr;
        std::unique_ptr<RooProdPdf> model_ssPtr; std::unique_ptr<RooProdPdf> model_sbPtr;
        std::unique_ptr<RooProdPdf> model_bsPtr; std::unique_ptr<RooProdPdf> model_bbPtr;
        std::unique_ptr<RooAddPdf> modelPtr; 
        std::unique_ptr<RooGenericPdf> ellipseMask; std::unique_ptr<RooProdPdf> maskedModelPtr;
        std::unique_ptr<RooDataSet> unbinnedData;
        std::unique_ptr<RooStats::SPlot> sData;
        TH2F* residualHist;  // ROOT steals ownership of the histogram hence we cannot use unique_ptr
        std::unique_ptr<TCanvas> heatmapCanvas;
        std::unique_ptr<TCanvas> projCanvas;
        std::string suffix;
        void GenerateModels();
        void FitModel();
        void CompareYields();
        void CreateResidualsHeatmap();
        void CreateProjections();
        void SavePlots();
};
ShitFitter::ShitFitter(
    TTree* inputTree, const std::string& inputBranchX, const std::string& inputBranchY, 
    const std::string& inputModel1s, const std::string& inputModel1b, 
    const std::string& inputModel2s, const std::string& inputModel2b, 
    const std::string& inputOutputDir
) : tree(inputTree), branchX(inputBranchX), branchY(inputBranchY), 
    model1s(inputModel1s), model1b(inputModel1b), 
    model2s(inputModel2s), model2b(inputModel2b), outputDir(inputOutputDir) 
{
    std::cout << "Fitting the branches '" << branchX << "' and '" << branchY << "' with models '" << model1s << "', '" << model1b << "', '" << model2s << "', and '" << model2b << "'..." << std::endl;

    std::cout << "Defining the range for the branches..." << std::endl;
    if (!check::CheckTreeBranches(tree, {branchX, branchY})) {
        std::cerr << "Error: One or more branches do not exist in the tree!" << std::endl;
        return;
    }
    xMin = tree->GetMinimum(branchX.c_str());
    xMax = tree->GetMaximum(branchX.c_str());
    yMin = tree->GetMinimum(branchY.c_str());
    yMax = tree->GetMaximum(branchY.c_str());
    std::cout << "Finished defining the range for the branches." << std::endl;

    std::cout << "Defining the RooFit independent variables..." << std::endl;
    xPtr = std::make_unique<RooRealVar>(branchX.c_str(), branchX.c_str(), xMin, xMax);
    yPtr = std::make_unique<RooRealVar>(branchY.c_str(), branchY.c_str(), yMin, yMax);
    // vars = RooArgSet(*xPtr, *yPtr);  // does not work
    vars.add(*xPtr);
    vars.add(*yPtr);
    std::cout << "Finished defining the RooFit independent variables." << std::endl;

    std::cout << "Creating the 2D histogram and its projections..." << std::endl;
    nBinsX = 100;
    nBinsY = 100;
    hist2D = new TH2F("hist2D", "2D Histogram;X-axis;Y-axis", nBinsX, xMin, xMax, nBinsY, yMin, yMax);
    tree->Draw((branchY + ":" + branchX + ">>hist2D").c_str(), "", "goff");
    nEntries = hist2D->GetEntries();
    projX = hist2D->ProjectionX("projX");
    projY = hist2D->ProjectionY("projY");
    std::cout << "Finished creating the 2D histogram." << std::endl;

    std::cout << "Importing the binned data..." << std::endl;
    dataHist = std::make_unique<RooDataHist>("dataHist", "Binned Data", vars, hist2D);
    std::cout << "Finished importing the binned data." << std::endl;

    std::cout << "Importing the unbinned data..." << std::endl;
    unbinnedData = std::make_unique<RooDataSet>("unbinnedData", "Unbinned Data", vars, RooFit::Import(*tree));
    std::cout << "Finished importing the unbinned data." << std::endl;

    std::cout << "Generating the suffix..." << std::endl;
    char c1s = model1s[0], c1b = model1b[0], c2s = model2s[0], c2b = model2b[0];  // cannot understand why this is needed
    suffix = std::string(1, model1s[0]) + model1b[0] + model2s[0] + model2b[0];
    std::cout << "Suffix: " << suffix << std::endl;
}
// Generate joint PDFs for signal and background. background-X-signal-Y is not generated.
void ShitFitter::GenerateModels() {
    std::cout << "Generating the signal and background models for X..." << std::endl;
    RooAbsPdf* signalXPtrTemp = GenerateModel(model1s, projX, xPtr.get(), "signalXss");
    signalXPtr = std::unique_ptr<RooAbsPdf>(dynamic_cast<RooAbsPdf*>(signalXPtrTemp->clone()));
    RooAbsPdf* backgroundXPtrTemp = GenerateModel(model1b, projX, xPtr.get(), "backgroundXbb");
    backgroundXPtr = std::unique_ptr<RooAbsPdf>(dynamic_cast<RooAbsPdf*>(backgroundXPtrTemp->clone()));
    if (!signalXPtr || !backgroundXPtr) {
        std::cerr << "Error: Failed to generate models for X" << std::endl;
        return;
    }
    std::cout << "Finished generating the signal and background models for X." << std::endl;

    // Generate the signal and background models for Y
    std::cout << "Generating the signal and background models for Y..." << std::endl;
    RooAbsPdf* signalYPtrTemp = GenerateModel(model2s, projY, yPtr.get(), "signalYss");
    signalYPtr = std::unique_ptr<RooAbsPdf>(dynamic_cast<RooAbsPdf*>(signalYPtrTemp->clone()));
    RooAbsPdf* backgroundYPtrTemp = GenerateModel(model2b, projY, yPtr.get(), "backgroundYbb");
    backgroundYPtr = std::unique_ptr<RooAbsPdf>(dynamic_cast<RooAbsPdf*>(backgroundYPtrTemp->clone()));
    if (!signalYPtr || !backgroundYPtr) {
        std::cerr << "Error: Failed to generate models for Y" << std::endl;
        return;
    }
    std::cout << "Finished generating the signal and background models for Y." << std::endl;

    std::cout << "Deleting the temporary models..." << std::endl;
    delete signalXPtrTemp;
    delete backgroundXPtrTemp;
    delete signalYPtrTemp;
    delete backgroundYPtrTemp;
    std::cout << "Finished deleting the temporary models." << std::endl;

    std::cout << "Defining the yields..." << std::endl;
    n_ssPtr = std::make_unique<RooRealVar>("n_ss", "Number of signal-signal events", 0.6 * nEntries, 0.001, nEntries);
    n_sbPtr = std::make_unique<RooRealVar>("n_sb", "Number of signal-background events", 0.15 * nEntries, 0.001, nEntries);
    // n_bsPtr = new RooRealVar("n_bs", "Number of background-signal events", 0.2 * nEntries, 0.001, nEntries);
    n_bbPtr = std::make_unique<RooRealVar>("n_bb", "Number of background-background events", 0.2 * nEntries, 0.001, nEntries);
    std::cout << "Finished defining the yields." << std::endl;

    std::cout << "Combining the signal and background models..." << std::endl;
    model_ssPtr = std::make_unique<RooProdPdf>("model_ss", "Signal X + Signal Y", RooArgList(*signalXPtr, *signalYPtr));
    model_sbPtr = std::make_unique<RooProdPdf>("model_sb", "Signal X + Background Y", RooArgList(*signalXPtr, *backgroundYPtr));
    // model_bsPtr = new RooProdPdf("model_bs", "Background X + Signal Y", RooArgList(*backgroundXPtr, *signalYPtr));
    model_bbPtr = std::make_unique<RooProdPdf>("model_bb", "Background X + Background Y", RooArgList(*backgroundXPtr, *backgroundYPtr));

    modelPtr = std::make_unique<RooAddPdf>(
        "model", "Signal + Background", 
        RooArgList(*model_ssPtr, *model_sbPtr, *model_bbPtr),
        RooArgList(*n_ssPtr, *n_sbPtr, *n_bbPtr)
    );
    std::cout << "Finished combining the signal and background models." << std::endl;
}
// Fit the model to the binned data, fit to the unbinned data after setting all but n_xx constant, then compute sWeights
void ShitFitter::FitModel() {
    std::cout << "Fitting the model to the binned data..." << std::endl;
    modelPtr->fitTo(*dataHist, RooFit::Verbose(kTRUE), RooFit::PrintLevel(10));
    std::cout << "Finished fitting the model to the binned data." << std::endl;

    // Set all but n_xx constant
    std::cout << "Setting parameters to constant adjusting yields for sWeights..." << std::endl;
    RooArgSet* params = modelPtr->getParameters(RooArgSet());  // All parameters in the model
    for (auto param : *params) {
        if (param->GetName() != n_ssPtr->GetName() && param->GetName() != n_sbPtr->GetName() && param->GetName() != n_bbPtr->GetName()) {
            param->setAttribute("Constant");
        }
    }
    n_ssPtr->setMin(0);
    n_sbPtr->setMin(0);
    // n_bsPtr->setMin(0);
    n_bbPtr->setMin(0);
    std::cout << "Finished setting parameters to constant for sWeights." << std::endl;

    std::cout << "Fitting the model to the unbinned data..." << std::endl;
    modelPtr->fitTo(*unbinnedData, RooFit::Extended(), RooFit::Verbose(kTRUE), RooFit::PrintLevel(10));
    std::cout << "Finished fitting the model to the unbinned data." << std::endl;

    std::cout << "Computing sWeights..." << std::endl;
    sData = std::make_unique<RooStats::SPlot>(
        "sData", "sWeights", *unbinnedData, modelPtr.get(), 
        RooArgList(*n_ssPtr, *n_sbPtr, *n_bbPtr)
    );
    std::cout << "Finished computing sWeights." << std::endl;
    CompareYields();
}
// Compare yields calculated from fit and sWeights to validate sWeights. They should be the same.
void ShitFitter::CompareYields() {
    std::cout << "Yield comparison: (yield calculated from fitting) vs (yield calculated by summing sWeights)" << std::endl;
    std::cout << "Sig-Sig: " << n_ssPtr->getVal() << " vs " << sData->GetYieldFromSWeight("n_ss") << std::endl;
    std::cout << "Sig-Bkg: " << n_sbPtr->getVal() << " vs " << sData->GetYieldFromSWeight("n_sb") << std::endl;
    std::cout << "Bkg-Bkg: " << n_bbPtr->getVal() << " vs " << sData->GetYieldFromSWeight("n_bb") << std::endl;
}
// Create a residual heatmap. Residual is defined as (observed - predicted) / sqrt(observed).
void ShitFitter::CreateResidualsHeatmap() {
    std::cout << "Creating the residual heatmap..." << std::endl;
    std::cout << "Defining the residual histogram..." << std::endl;
    residualHist = new TH2F("residualHist", "Residual Heatmap;X-axis;Y-axis", nBinsX, xMin, xMax, nBinsY, yMin, yMax);
    std::cout << "Finished defining the residual histogram." << std::endl;

    std::cout << "Filling the residual histogram..." << std::endl;
    double xBinWidth = hist2D->GetXaxis()->GetBinWidth(1);
    double yBinWidth = hist2D->GetYaxis()->GetBinWidth(1);
    for (int iX = 1; iX <= nBinsX; ++iX) {
        double centerX = hist2D->GetXaxis()->GetBinCenter(iX);
        xPtr->setVal(centerX);
        for (int iY = 1; iY <= nBinsY; ++iY) {
            double centerY = hist2D->GetYaxis()->GetBinCenter(iY);
            yPtr->setVal(centerY);
            
            double observed = hist2D->GetBinContent(iX, iY);
            double predicted = modelPtr->getVal(&vars);
            predicted *= nEntries * xBinWidth * yBinWidth;  // Scale the density to the number of entries
            
            double residual = observed - predicted;
            residual /= sqrt(observed);
            residualHist->SetBinContent(iX, iY, residual);
            
            std::cout << residual;
        }
    }
    std::cout << "Finished filling the residual histogram." << std::endl;

    std::cout << "Creating the canvas and drawing the residual heatmap..." << std::endl;
    heatmapCanvas = std::make_unique<TCanvas>("heatmapCanvas", "Residual Heatmap", 1000, 800);
    residualHist->Draw("COLZ");
    residualHist->GetXaxis()->SetTitle(branchX.c_str());
    residualHist->GetYaxis()->SetTitle(branchY.c_str());
    residualHist->SetTitle("Residual/sqrt(Data) Heatmap");
    heatmapCanvas->SetRightMargin(0.15);
    gStyle->SetOptStat(0);
    std::cout << "Finished drawing the residual heatmap." << std::endl;
}
// Create projections for X and Y, as a single canvas. Data, model, and components are drawn.
void ShitFitter::CreateProjections() {
    std::cout << "Creating canvas for the projections..." << std::endl;
    projCanvas = std::make_unique<TCanvas>("projCanvas", "2D Gaussian Fit (Binned)", 1000, 800);
    projCanvas->Divide(1, 2);

    std::cout << "Creating the X projection(" << branchX << ")..." << std::endl;
    projCanvas->cd(1);
    RooPlot* frameX = xPtr->frame();
    dataHist->plotOn(frameX, RooFit::Name("data"));
    modelPtr->plotOn(frameX, RooFit::Name("model"));
    modelPtr->plotOn(frameX, RooFit::Components("model_ss"), RooFit::LineStyle(kDashed), RooFit::LineColor(kRed), RooFit::Name("model_ss"));
    modelPtr->plotOn(frameX, RooFit::Components("model_sb"), RooFit::LineStyle(kDashed), RooFit::LineColor(kGreen), RooFit::Name("model_sb"));
    // modelPtr->plotOn(frameX, RooFit::Components("model_bs"), RooFit::LineStyle(kDashed), RooFit::LineColor(kBlue), RooFit::Name("model_bs"));
    modelPtr->plotOn(frameX, RooFit::Components("model_bb"), RooFit::LineStyle(kDashed), RooFit::LineColor(kBlue), RooFit::Name("model_bb"));
    frameX->SetTitle((branchX + " Projection").c_str());
    TLegend* legendX = new TLegend(0.7, 0.7, 0.9, 0.9);
    legendX->AddEntry(frameX->findObject("data"), "Data", "p");
    legendX->AddEntry(frameX->findObject("model"), "Model", "l");
    legendX->AddEntry(frameX->findObject("model_ss"), ("Sig(" + model1s + ") + Sig(" + model2s + ")").c_str(), "l");
    legendX->AddEntry(frameX->findObject("model_sb"), ("Sig(" + model1s + ") + Bkg(" + model2b + ")").c_str(), "l");
    // legendX->AddEntry(frameX->findObject("model_bs"), ("Bkg(" + model1b + ") + Sig(" + model2s + ")").c_str(), "l");
    legendX->AddEntry(frameX->findObject("model_bb"), ("Bkg(" + model1b + ") + Bkg(" + model2b + ")").c_str(), "l");
    frameX->Draw();
    legendX->Draw();
    std::cout << "Finished creating the X projection." << std::endl;

    // Y projection
    std::cout << "Creating the Y projection(" << branchY << ")..." << std::endl;
    projCanvas->cd(2);
    RooPlot* frameY = yPtr->frame();
    dataHist->plotOn(frameY, RooFit::Name("data"));
    modelPtr->plotOn(frameY, RooFit::Name("model"));
    modelPtr->plotOn(frameY, RooFit::Components("model_ss"), RooFit::LineStyle(kDashed), RooFit::LineColor(kRed), RooFit::Name("model_ss"));
    modelPtr->plotOn(frameY, RooFit::Components("model_sb"), RooFit::LineStyle(kDashed), RooFit::LineColor(kGreen), RooFit::Name("model_sb"));
    // modelPtr->plotOn(frameY, RooFit::Components("model_bs"), RooFit::LineStyle(kDashed), RooFit::LineColor(kBlue), RooFit::Name("model_bs"));
    modelPtr->plotOn(frameY, RooFit::Components("model_bb"), RooFit::LineStyle(kDashed), RooFit::LineColor(kGreen), RooFit::Name("model_bb"));
    frameY->SetTitle((branchY + " Projection").c_str());
    TLegend* legendY = new TLegend(0.7, 0.7, 0.9, 0.9);
    legendY->AddEntry(frameY->findObject("data"), "Data", "p");
    legendY->AddEntry(frameY->findObject("model"), "Model", "l");
    legendY->AddEntry(frameY->findObject("model_ss"), ("Sig(" + model1s + ") + Sig(" + model2s + ")").c_str(), "l");
    legendY->AddEntry(frameY->findObject("model_sb"), ("Sig(" + model1s + ") + Bkg(" + model2b + ")").c_str(), "l");
    // legendY->AddEntry(frameY->findObject("model_bs"), ("Bkg(" + model1b + ") + Sig(" + model2s + ")").c_str(), "l");
    legendY->AddEntry(frameY->findObject("model_bb"), ("Bkg(" + model1b + ") + Bkg(" + model2b + ")").c_str(), "l");
    frameY->Draw();
    legendY->Draw();
    std::cout << "Finished creating the Y projection." << std::endl;
}
// Save the residual heatmap and the projections to files under the output directory specified in the constructor.  TODO: get the output directory as a parameter
void ShitFitter::SavePlots() {
    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
    }
    std::string heatmapFileName = outputDir + "heatmap_" + suffix + ".png";
    heatmapCanvas->SaveAs(heatmapFileName.c_str());
    std::cout << "Saved the residual heatmap to " << heatmapFileName << std::endl;

    std::string projFileName = outputDir + "proj_" + suffix + ".png";
    projCanvas->SaveAs(projFileName.c_str());
    std::cout << "Saved the projections to " << projFileName << std::endl;
}
// Fit the model, and compute sWeights
void ShitFitter::Fit() {
    GenerateModels();
    FitModel();
}
// Create the residual heatmap and the projections, and save them
void ShitFitter::Plot() {
    CreateResidualsHeatmap();
    CreateProjections();
    SavePlots();
}
// Export the sWeights to a ROOT file. The new file will contain the same TTree as the input file, with additional branches for the sWeights.
void ShitFitter::ExportSWeights(const std::string& modelName, const std::string& outputFileName) {
    std::cout << "Exporting sWeights to " << outputFileName << "..." << std::endl;
    
    std::cout << "Creating the sWeights file..." << std::endl;
    TFile* sWeightsFile = nullptr;
    TTree* sWeightsTree = nullptr;
    bool appendNotRecreate = false;
    if (fs::exists(outputFileName)) {
        auto [existingFile, existingTree] = treeops::loadFileAndTree(outputFileName);
        if (existingTree && (existingTree->GetEntries() == tree->GetEntries())) {
            std::cout << "File already exists and has the same number of entries. Appending to it..." << std::endl;
            appendNotRecreate = true;
        }
        existingFile->Close();
    }
    if (!appendNotRecreate) {
        sWeightsFile = new TFile(outputFileName.c_str(), "RECREATE");
        sWeightsTree = tree->CloneTree();  // Clone contents
        std::vector<std::string> models = {"bdt", "torch", "randomforest"};
        std::vector<std::string> ratios = {"1", "2", "5", "10", "20", "50", "100"};
        for (const auto& model : models) {
            for (const auto& ratio : ratios) {
                TBranch* branch = sWeightsTree->GetBranch((model + "_" + ratio).c_str());
                if (branch) {
                    sWeightsTree->GetListOfBranches()->Remove(branch);
                    delete branch;
                }
            }
        }
    } else {
        std::cout << "Appending to the existing file..." << std::endl;
        auto [existingFile, existingTree] = treeops::loadFileAndTree(outputFileName, "UPDATE");
        sWeightsFile = existingFile;
        sWeightsTree = existingTree;
        for (const auto& suffix : {"ss", "sb", "bs", "bb"}) {
            TObject* branch = sWeightsTree->GetListOfBranches()->FindObject((modelName + "_" + suffix).c_str());
            if (branch) {
                std::cout << "Removing existing branch: " << (modelName + "_" + suffix) << std::endl;
                sWeightsTree->GetListOfBranches()->Remove(branch);
                delete branch;
                std::cout << "Removed existing branch: " << (modelName + "_" + suffix) << std::endl;
            } else {
                std::cout << "Branch " << (modelName + "_" + suffix) << " does not exist." << std::endl;
            }
        }
        std::cout << "Existing file opened in update mode." << std::endl;
    }
    Double_t n_ss_sw, n_sb_sw, n_bs_sw, n_bb_sw;
    sWeightsTree->Branch((modelName + "_ss").c_str(), &n_ss_sw, (modelName + "_ss/D").c_str());
    sWeightsTree->Branch((modelName + "_sb").c_str(), &n_sb_sw, (modelName + "_sb/D").c_str());
    // sWeightsTree->Branch((modelName + "_bs").c_str(), &n_bs_sw, (modelName + "_bs/D").c_str());
    sWeightsTree->Branch((modelName + "_bb").c_str(), &n_bb_sw, (modelName + "_bb/D").c_str());
    std::cout << "Filling the sWeights tree..." << std::endl;
    for (int i = 0; i < tree->GetEntries(); ++i) {
        tree->GetEntry(i);
        n_ss_sw = sData->GetSWeight(i, "n_ss");
        n_sb_sw = sData->GetSWeight(i, "n_sb");
        // n_bs_sw = sData->GetSWeight(i, "n_bs");
        n_bb_sw = sData->GetSWeight(i, "n_bb");
        sWeightsTree->GetBranch((modelName + "_ss").c_str())->Fill();
        sWeightsTree->GetBranch((modelName + "_sb").c_str())->Fill();
        // sWeightsTree->GetBranch((modelName + "_bs").c_str())->Fill();
        sWeightsTree->GetBranch((modelName + "_bb").c_str())->Fill();
    }
    std::cout << "Filled the sWeights tree." << std::endl;
    sWeightsTree->Write("", TObject::kOverwrite);
    sWeightsFile->Close();
    std::cout << "sWeights exported to " << outputFileName << std::endl;
}
// Destructor
ShitFitter::~ShitFitter() {
    std::cout << "Destructing the ShitFitter..." << std::endl;
    std::cout << "We should not delete hist2D, projX, projY, residualHist as ROOT AUTOMAGICALLY deletes them, for whatever reason" << std::endl;
    // tho i do not want ROOT to do the magic
}

void fit2d(TTree* tree, std::string model = "bdt_all", float cut = 0.5) {
    std::string tempFileName = config::tempRootFile();
    TFile* tempFile = new TFile(tempFileName.c_str(), "RECREATE");
    std::cout << "Applying cuts (" << model << " > " << cut << ") on masses and saving to temp root file..." << std::endl;
    TTree* cut_tree = tree->CopyTree((model + " > " + std::to_string(cut)).c_str());
    if (!cut_tree) {
        std::cerr << "Error: Failed to apply cuts on the tree!" << std::endl;
        return;
    }
    if (cut_tree->GetEntries() == 0) {
        std::cerr << "Error: No entries after applying the cut!" << std::endl;
        return;
    }
    std::cout << "Finished applying cuts on the tree (" << cut_tree->GetEntries() << " entries)" << std::endl;

    // Define an instance of the ShitFitter class
    ShitFitter fit(cut_tree, "D_M", "delta_M", "cb", "exp", "cb", "dstd0bg", config::plot_dir4(2, model, cut));
    fit.Fit();
    fit.Plot();
    fit.ExportSWeights(model + "_" + std::to_string(static_cast<int>(cut * 100)), config::output_dir + "sWeights.root");

    // Cleanup
    delete cut_tree;
    std::cout << "Deleted cut_tree" << std::endl;
    tempFile->Close();
    std::cout << "Closed tempFile" << std::endl;
    fileio::deleteFile(tempFileName);
    std::cout << "Deleted temp root file" << std::endl;
}
void fit2d() {
    std::cout << "Loading TTree from file..." << std::endl;
    auto [file, tree] = treeops::loadFileAndTree(config::long_wscore_file);
    if (!tree) {
        std::cerr << "Error: Null TTree provided!" << std::endl;
        return;
    }
    std::cout << "Loaded TTree from file" << std::endl;

    // Fit the 2D models
    fit2d(tree, "bdt_100", 0.5);

    // Cleanup
    std::cout << "Closing file..." << std::endl;
    file->Close();
}
