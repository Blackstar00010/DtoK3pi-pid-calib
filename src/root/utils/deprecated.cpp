#include <TTree.h>
#include <TFile.h>
#include <filesystem>
#include <common.cpp>
#include <RooDataSet.h>
#include <RooStats/SPlot.h>
namespace fs = std::filesystem;


/*
* Load a TTree from a `.root` file
* @param filename: path to the `.root` file
* @return TTree pointer if successful, nullptr otherwise
*/
TTree* load_mc() {
    // mc data is under ./data/mc and is md1.root md2.root ... so we should load all and merge

    TList* chain = new TList();
    TFile* tempFile = TFile::Open("memory:tempFile", "RECREATE");

    for (const auto& entry : fs::directory_iterator("../../data/mc")) {
        // Check if the entry is a file (not a directory)
        if (fs::is_regular_file(entry.status())) {
            TTree* tree = tman::load_tree(entry.path().string());
            if (tree) {
                tempFile->cd();
                TTree* clonedTree = tree->CloneTree(-1, "fast");  // Use "fast" mode for a lightweight clone
                if (clonedTree) {
                    chain->Add(clonedTree);
                } else {
                    std::cerr << "Warning: Failed to clone TTree from " << entry.path().string() << std::endl;
                }
            } else {
                std::cerr << "Warning: Failed to load TTree from " << entry.path().string() << std::endl;
            }
        }
    }
    TTree* tree = TTree::MergeTrees(chain);
    tree->SetName("DecayTree");
    delete chain;
    tempFile->Close();
    delete tempFile;
    return tree;
}


void PlotWeightedHistogram(RooDataSet& data, RooStats::SPlot& sData, const std::string& varZ, const std::string& outputDir) {
    // Get variable Z (assuming Z is in the dataset)
    RooRealVar* zVar = static_cast<RooRealVar*>(data.get()->find(varZ.c_str()));
    if (!zVar) {
        std::cerr << "Error: Variable " << varZ << " not found in dataset." << std::endl;
        return;
    }

    // Define histograms
    int nbins = 100;
    TH1F* hist_weighted = new TH1F("hist_weighted", "Z Distribution (Weighted); Z; Events", nbins, zVar->getMin(), zVar->getMax());
    TH1F* hist_unweighted = new TH1F("hist_unweighted", "Z Distribution (Unweighted); Z; Events", nbins, zVar->getMin(), zVar->getMax());

    // Loop over dataset entries
    for (int i = 0; i < data.numEntries(); i++) {
        data.get(i);  // Load event
        double zValue = zVar->getVal();
        double sWeight_sig = sData.GetSWeight(i, "n_ss");  // Signal sWeight
        // double sWeight_bkg = sData.GetSWeight(i, "n_sb") + sData.GetSWeight(i, "n_bb");  // Background sWeight

        // Fill histograms
        hist_weighted->Fill(zValue, sWeight_sig);
        hist_unweighted->Fill(zValue);
    }

    // Normalize histograms to compare shapes
    // hist_weighted->Scale(1.0 / hist_weighted->Integral());
    // hist_unweighted->Scale(1.0 / hist_unweighted->Integral());

    // Plot histograms
    TCanvas* canvas = new TCanvas(("canvas_" + varZ).c_str(), "Weighted vs Unweighted Histograms", 1000, 800);
    hist_weighted->SetLineColor(kRed);
    hist_unweighted->SetLineColor(kBlack);
    hist_weighted->Draw("HIST");
    hist_unweighted->Draw("HIST SAME");

    // title
    hist_weighted->SetTitle(("Weighted vs Unweighted " + varZ + " Distribution").c_str());

    // axes labels
    hist_weighted->GetXaxis()->SetTitle(varZ.c_str());
    hist_weighted->GetYaxis()->SetTitle("Events");

    // Add legend
    gStyle->SetOptStat(0);
    TLegend* legend = new TLegend(0.7, 0.7, 0.9, 0.9);
    legend->AddEntry(hist_weighted, "Weighted Data", "l");
    legend->AddEntry(hist_unweighted, "Unweighted Data", "l");
    legend->Draw();

    // Save plot
    canvas->SaveAs((outputDir + "sWeighted_" + varZ + ".png").c_str());

    // Cleanup
    delete legend;
    delete canvas;
    delete hist_unweighted;
    delete hist_weighted;
    // delete zVar;  // we don't own this
}

void PlotWeightedHistogram(RooDataSet& data, RooStats::SPlot& sData, const std::vector<std::string>& varZs, const std::string& varName, const std::string& outputDir) {
    std::cout << "Plotting weighted histogram for variable " << varName << "..." << std::endl;

    // Get variable Z (assuming Z is in the dataset)
    std::vector<RooRealVar*> zVars;
    std::cout << "hehe" << std::endl;
    double min = zVars[0]->getMin();
    std::cout << "hehe" << std::endl;
    double max = zVars[0]->getMax();
    std::cout << "hehe" << std::endl;
    for (const auto& varZ : varZs) {
        RooRealVar* zVar = (RooRealVar*)data.get()->find(varZ.c_str());
        std::cout << "hehe" << std::endl;
        if (!zVar) {
            std::cerr << "Error: Variable " << varZ << " not found in dataset." << std::endl;
            return;
        }
        zVars.push_back(zVar);
        std::cout << "hehe" << std::endl;
        if (zVar->getMin() < min) {
            min = zVar->getMin();
        }
        std::cout << "hehe" << std::endl;
        if (zVar->getMax() > max) {
            max = zVar->getMax();
        }
        std::cout << "hehe" << std::endl;
    }

    // Define histograms
    std::cout << "Creating histograms..." << std::endl;
    int nbins = 100;
    TH1F* hist_unweighted = new TH1F("hist_unweighted", "Z Distribution (Unweighted); Z; Events", nbins, min, max);
    TH1F* hist_weighted = new TH1F("hist_weighted", "Z Distribution (Weighted); Z; Events", nbins, min, max);

    // Loop over dataset entries
    std::cout << "Filling histograms..." << std::endl;
    for (int i = 0; i < data.numEntries(); i++) {
        for (const auto& zVar : zVars) {
            data.get(i);  // Load event
            double zValue = zVar->getVal();
            double sWeight_sig = sData.GetSWeight(i, "n_ss");  // Signal sWeight
            // double sWeight_bkg = sData.GetSWeight(i, "n_sb") + sData.GetSWeight(i, "n_bs") + sData.GetSWeight(i, "n_bb");  // Background sWeight
            double sWeight_bkg = sData.GetSWeight(i, "n_sb") + sData.GetSWeight(i, "n_bb");  // Background sWeight

            // Fill histograms
            hist_unweighted->Fill(zValue);
            // hist_weighted->Fill(zValue, sWeight_sig / (sWeight_sig + sWeight_bkg));
            hist_weighted->Fill(zValue, sWeight_sig);
        }
    }

    // Normalize histograms to compare shapes
    std::cout << "Normalizing histograms..." << std::endl;
    hist_unweighted->Scale(1.0 / hist_unweighted->Integral());
    hist_weighted->Scale(-1.0 / hist_weighted->Integral());

    // Plot histograms
    TCanvas* canvas = new TCanvas(("canvas_" + varName).c_str(), "sWeighted Histograms", 1000, 800);
    hist_unweighted->SetLineColor(kBlack);
    hist_weighted->SetLineColor(kRed);
    hist_unweighted->Draw("HIST");
    hist_weighted->Draw("HIST SAME");

    // axes labels
    hist_unweighted->GetXaxis()->SetTitle(varName.c_str());
    hist_unweighted->GetYaxis()->SetTitle("Events");

    // Add legend
    gStyle->SetOptStat(0);
    TLegend* legend = new TLegend(0.7, 0.7, 0.9, 0.9);
    legend->AddEntry(hist_unweighted, "Unweighted Data", "l");
    legend->AddEntry(hist_weighted, "Weighted Data", "l");
    legend->Draw();

    // Save plot
    canvas->SaveAs((outputDir + "sWeighted_" + varName + ".png").c_str());

    // Cleanup
    std::cout << "pi1" << std::endl;
    delete legend;
    std::cout << "pi2" << std::endl;
    delete canvas;
    std::cout << "pi3" << std::endl;
    delete hist_unweighted;
    std::cout << "pi4" << std::endl;
    delete hist_weighted;
    for (const auto& zVar : zVars) {
        delete zVar;
    }
    std::cout << "pi5" << std::endl;
}
