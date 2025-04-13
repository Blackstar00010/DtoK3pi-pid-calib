#include <iostream>
#include <vector>
#include <string>
#include <TTree.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TGraph.h>
#include <TLegend.h>
#include "../utils/common.cpp"
#include "../utils/config.cpp"


namespace main3 {
    TTree* plotProbs(bool plot_sig, bool plot_bkg);
    void plotProbs(TTree* prob_tree, bool plot_sig, bool plot_bkg);
    void plotPuriSigni(TTree* tree, const std::string& ansBranch, const std::string& probBranch, 
                       float exp1, float exp2, const std::string& filename);
    void plotROC(TTree* tree, const std::string& ansBranch, const std::string& probBranch, const std::string& filename);
    void main();
}

void print_branch(TTree* target_tree, std::string target_branch){
    // Print the first 10 values of the target branch
    std::cout << "Printing the first 10 values of the branch " << target_branch << std::endl;
    TBranch* branch = target_tree->GetBranch(target_branch.c_str());
    if (!branch) {
        std::cerr << "Error: Failed to get branch " << target_branch << std::endl;
        return;
    }

    float value;
    branch->SetAddress(&value);
    for (int i = 0; i < 10; i++) {
        branch->GetEntry(i);
        std::cout << value << std::endl;
    }
}

void plotMasses(TTree* tree) {
    int dmmin = 1800;
    int dmmax = 19;
}

TTree* main3::plotProbs(bool plot_sig=true, bool plot_bkg=true) {
    auto [prob_file, prob_tree] = tman::loadFileAndTree(config::proba_tt_file);
    if (!prob_tree) {
        std::cerr << "Error: Failed to load the probability tree" << std::endl;
        return nullptr;
    }
    main3::plotProbs(prob_tree, plot_sig, plot_bkg);
    return prob_tree;
}

void main3::plotProbs(TTree* prob_tree, bool plot_sig=true, bool plot_bkg=true) {
    if (!plot_sig && !plot_bkg) {
        std::cerr << "Error: Both plot_sig and plot_bkg are false" << std::endl;
        return;
    }
    if (!prob_tree) {
        std::cerr << "Error: Invalid probability tree" << std::endl;
        return;
    }

    // Get the list of branches in the TTree: bdt1, bdt2, bdt5, randomforest1, randomforest2, randomforest5 ...
    std::vector<std::string> branches = {};
    std::vector<int> bdt_branches = {1, 2, 5, 10};
    for (int i : bdt_branches) {
        branches.push_back("_ans" + std::to_string(i));  // signal -> 1, background -> 0
        branches.push_back("is_test_" + std::to_string(i));  // test -> 1, train -> 0
        branches.push_back("bdt" + std::to_string(i));  // float probability
        branches.push_back("randomforest" + std::to_string(i));  // float probability
    }

    // 1. plot for bdt : histograms with four entries - signal train, signal test, background train, background test
    // std::cout << "The number of entries: " << prob_tree->GetEntries() << std::endl;
    for (int i : bdt_branches) {
        TH1F* hist1 = nullptr;
        TH1F* hist2 = nullptr;
        TH1F* hist3 = nullptr;
        TH1F* hist4 = nullptr;
        double ksResult = 0.0;

        std::string col = "bdt" + std::to_string(i);
        int bins = 100;
        float min = prob_tree->GetMinimum(col.c_str());
        float max = prob_tree->GetMaximum(col.c_str());

        if (plot_sig) {
            hist1 = new TH1F("hist1", "Probability Distributions", bins, min, max);
            hist2 = new TH1F("hist2", "Probability Distributions", bins, min, max);
            prob_tree->Draw((col + " >> hist1").c_str(), 
                            ("is_test_" + std::to_string(i) + " == 0 && _ans" + std::to_string(i) + " == 1").c_str(), 
                            "goff");  // train signal
            prob_tree->Draw((col + " >> hist2").c_str(), 
                            ("is_test_" + std::to_string(i) + " == 1 && _ans" + std::to_string(i) + " == 1").c_str(), 
                            "goff");  // test signal
            hist1->Scale(1.0 / hist1->Integral());
            hist2->Scale(1.0 / hist2->Integral());
            ksResult = hist1->KolmogorovTest(hist2);
            std::cout << "KS result: " << ksResult << std::endl;
        }
        if (plot_bkg) {
            hist3 = new TH1F("hist3", "Probability Distributions", bins, min, max);
            hist4 = new TH1F("hist4", "Probability Distributions", bins, min, max);
            prob_tree->Draw((col + " >> hist3").c_str(), 
                            ("is_test_" + std::to_string(i) + " == 0 && _ans" + std::to_string(i) + " == 0").c_str(), 
                            "goff");  // train background
            prob_tree->Draw((col + " >> hist4").c_str(), 
                            ("is_test_" + std::to_string(i) + " == 1 && _ans" + std::to_string(i) + " == 0").c_str(), 
                            "goff");  // test background
            hist3->Scale(1.0 / hist3->Integral());
            hist4->Scale(1.0 / hist4->Integral());
            ksResult = hist3->KolmogorovTest(hist4);
            std::cout << "KS result: " << ksResult << std::endl;
        }

        TCanvas* canvas = new TCanvas("canvas", "canvas", 800, 600);
        if (plot_sig){
            // hist1 : train(filled) signal(red)
            // hist2 : test("+" marker) signal(red)
            hist1->SetLineColor(kRed);
            hist1->SetFillColor(kRed - 10);
            hist2->SetLineColor(kRed);
            hist2->SetMarkerColor(kRed);
            hist2->SetMarkerStyle(2);
            hist2->SetMarkerSize(0.1);
            hist1->Draw("HIST");
            hist2->Draw("SAME");
            hist1->SetStats(0);
            hist2->SetStats(0);
        }
        if (plot_bkg) {
            // hist3 : train(filled) background(blue)
            // hist4 : test("+" marker) background(blue)
            hist3->SetLineColor(kBlue);
            hist3->SetFillColor(kBlue - 10);
            hist4->SetLineColor(kBlue);
            hist4->SetMarkerColor(kBlue);
            hist4->SetMarkerStyle(2);
            hist4->SetMarkerSize(0.1);
            hist3->Draw("HIST SAME");
            hist4->Draw("SAME");
            hist3->SetStats(0);
            hist4->SetStats(0);
        }

        TLegend* legend = new TLegend(0.375, 0.75, 0.625, 0.9);
        legend->SetTextSize(0.03);
        if (plot_sig){
            // set font size
            legend->AddEntry(hist1, "Signal Train", "l");
            legend->AddEntry(hist2, "Signal Test", "l");
        }
        if (plot_bkg){
            legend->AddEntry(hist3, "Background Train", "l");
            legend->AddEntry(hist4, "Background Test", "l");
        }
        legend->Draw();

        // plot title
        std::string title = "Probability Distributions";
        if (!plot_sig || !plot_bkg) {
            title += " (KS result: " + std::to_string(ksResult) + ")";
        }
        canvas->SetTitle(title.c_str());

        std::string filename = config::plot_dir3 + col + "/probs";
        if (plot_sig && plot_bkg) {
            filename += ".png";
        } else if (plot_sig) {
            filename += "_sig.png";
        } else {
            filename += "_bkg.png";
        }
        canvas->SaveAs(filename.c_str());

        if (plot_sig){
            delete hist1;
            delete hist2;
        }
        if (plot_bkg){
            delete hist3;
            delete hist4;
        }
        delete canvas;

        std::cout << "Plotted histograms for " << col << std::endl;
    }
}

void main3::plotPuriSigni(TTree* tree, const std::string& ansBranch, const std::string& probBranch, 
                          float exp1, float exp2, const std::string& filename) {
    if (!tree) {
        std::cerr << "Error: TTree is null." << std::endl;
        return;
    }

    const int nBins = 50;    // Number of bins for the BDT score
    const double probMin = 0.0;
    const double probMax = 1.0;

    // Histograms to count cumulative signal and total events for scores above the cut
    TH1F* hSignalCumulative = new TH1F("hSignalCumulative", "Cumulative Signal Count", nBins, probMin, probMax);
    TH1F* hTotalCumulative = new TH1F("hTotalCumulative", "Cumulative Total Count", nBins, probMin, probMax);

    // Fill cumulative histograms
    tree->Draw((probBranch + " >> hSignalCumulative").c_str(), 
               (ansBranch + " == 1").c_str(), "goff");
    tree->Draw((probBranch + " >> hTotalCumulative").c_str(), "", "goff");

    // Convert histograms to cumulative distributions
    hSignalCumulative->ComputeIntegral();
    hTotalCumulative->ComputeIntegral();

    // Create a graph for purity
    TGraph* purityGraph = new TGraph();
    for (int i = 1; i <= nBins; ++i) {
        double s = hSignalCumulative->Integral(i, nBins);  // Cumulative signal count
        double b = hTotalCumulative->Integral(i, nBins) - s;  // Cumulative background count
        double purity = (s + b > 0) ? std::pow(s, exp1) / std::pow(s + b, exp2) : 0.0;
        double probCut = hSignalCumulative->GetBinLowEdge(i);

        purityGraph->SetPoint(i - 1, probCut, purity);
    }

    // Style the purity graph
    std::string yTitle;
    if (exp1 == 1 && exp2 == 1) {
        yTitle = "Purity";
    } else if (exp1 == 1 && exp2 == 0.5) {
        yTitle = "Significance";
    } else if (exp1 == 2 && exp2 == 1.5) {
        yTitle = "Purity * Significance";
    } else {
        yTitle = "S^{exp1} / (S + B)^{exp2}";
    }
    purityGraph->SetTitle((yTitle + " vs. BDT Cut;BDT Score Cut;" + yTitle).c_str());
    purityGraph->SetLineColor(kBlue);
    purityGraph->SetLineWidth(2);
    purityGraph->SetMarkerStyle(20);
    purityGraph->SetMarkerColor(kBlue);

    // Draw the graph
    TCanvas* canvas = new TCanvas("canvas", "Purity Plot", 800, 600);
    canvas->cd();
    purityGraph->Draw("ALP");

    // Save the canvas
    canvas->SaveAs(filename.c_str());

    // Clean up
    delete hSignalCumulative;
    delete hTotalCumulative;
    delete purityGraph;
    delete canvas;
}

void main3::plotROC(TTree* tree, const std::string& ansBranch, const std::string& probBranch, const std::string& filename) {
    if (!tree) {
        std::cerr << "Error: TTree is null." << std::endl;
        return;
    }

    const int nBins = 50;    // Number of bins for the BDT score
    const double probMin = 0.0;
    const double probMax = 1.0;

    // Histograms to count cumulative signal and total events for scores above the cut
    TH1F* hSignalCumulative = new TH1F("hSignalCumulative", "Cumulative Signal Count", nBins, probMin, probMax);
    TH1F* hTotalCumulative = new TH1F("hTotalCumulative", "Cumulative Total Count", nBins, probMin, probMax);

    // Fill cumulative histograms
    tree->Draw((probBranch + " >> hSignalCumulative").c_str(), 
               (ansBranch + " == 1").c_str(), "goff");
    tree->Draw((probBranch + " >> hTotalCumulative").c_str(), "", "goff");

    // Compute total signal and background counts
    double S_total = hSignalCumulative->Integral();
    double B_total = hTotalCumulative->Integral() - S_total;

    // Create a graph for the ROC curve
    TGraph* rocGraph = new TGraph();

    for (int i = 1; i <= nBins; ++i) {
        double S = hSignalCumulative->Integral(i, nBins);  // Cumulative signal count
        double B = hTotalCumulative->Integral(i, nBins) - S;  // Cumulative background count

        // True Positive Rate (Signal Efficiency) and False Positive Rate (Background Efficiency)
        double TPR = (S_total > 0) ? S / S_total : 0.0;
        double FPR = (B_total > 0) ? B / B_total : 0.0;

        // Set points for the ROC curve
        rocGraph->SetPoint(i - 1, FPR, TPR);
    }

    // Style the ROC graph
    rocGraph->SetTitle("ROC Curve;False Positive Rate (FPR);True Positive Rate (TPR)");
    rocGraph->SetLineColor(kBlue);
    rocGraph->SetLineWidth(2);

    // Draw the ROC curve
    TCanvas* canvas = new TCanvas("canvas", "ROC Curve", 800, 600);
    canvas->cd();
    rocGraph->Draw("AL");

    // legend = "AUC: " + str(rocGraph.Integral())
    TLegend* legend = new TLegend(0.8, 0.1, 0.9, 0.2);
    legend->AddEntry(rocGraph, ("AUC: " + std::to_string(rocGraph->Integral())).c_str(), "l");
    legend->Draw();
    

    // Save the canvas
    canvas->SaveAs(filename.c_str());

    // Clean up
    delete hSignalCumulative;
    delete hTotalCumulative;
    delete rocGraph;
    delete canvas;

}
    

// void stackedMassDist(TTree* tree, const std::string& probBranch, const float cut) {
//     // plots stacked mass distribution hitogram for signal(prob > cut) and background(prob < cut)
// }

void main3::main() {
    TTree* treett = main3::plotProbs();
    main3::plotProbs(treett, false, true);
    main3::plotProbs(treett, true, false);

    std::vector<int> ratios = {1, 2, 5, 10};
    std::vector<std::string> models = {"bdt", "randomforest"};
    for (int ratio : ratios) {
        for (std::string model : models) {
            std::string col = model + std::to_string(ratio);
            std::string target_dir = config::plot_dir3 + col + "/";
            main3::plotPuriSigni(treett, "_ans" + std::to_string(ratio), col, 1, 1, target_dir + "purity.png");
            main3::plotPuriSigni(treett, "_ans" + std::to_string(ratio), col, 1, 0.5, target_dir + "significance.png");
            main3::plotPuriSigni(treett, "_ans" + std::to_string(ratio), col, 2, 1.5, target_dir + "purisigni.png");
            main3::plotROC(treett, "_ans" + std::to_string(ratio), col, target_dir + "roc.png");
        }
    }
}

void main03_2() {
    main3::main();
}