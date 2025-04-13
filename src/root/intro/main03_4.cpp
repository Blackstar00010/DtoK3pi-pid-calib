#include "../utils/config.cpp"
#include "../utils/common.cpp"
#include "TH1D.h"
#include "TCanvas.h"
#include "TLegend.h"

void main03_4() {
    auto [fullerFile, fullerTree] = tman::loadFileAndTree(config::fuller_rootwithprob_file);

    std::vector<std::string> branches = {"D_M", "delta_M"};

    for (auto branch : branches) {
        std::cout << "Plotting weighted vs unweighted histograms for branch " << branch << std::endl;
        // plot distribution of D_M in 1d histogram
        double dmmin = fullerTree->GetMinimum(branch.c_str());
        double dmmax = fullerTree->GetMaximum(branch.c_str());
        // double dmmin = fullerTree->GetMinimum("D_M");
        // double dmmax = fullerTree->GetMaximum("D_M");
        int nbins = 100;
        // TH1D* hist1 = new TH1D("hist1", "D_M Distribution", nbins, dmmin, dmmax);
        TH1D* hist1 = new TH1D("hist1", (branch + " Distribution").c_str(), nbins, dmmin, dmmax);
        // fullerTree->Draw("D_M>>hist1", "", "goff");
        fullerTree->Draw((branch + ">>hist1").c_str(), "", "goff");

        // plot distribution of D_M where bdt_all > 0.5
        // TH1D* hist2 = new TH1D("hist2", "D_M Distribution (BDT > 0.5)", nbins, dmmin, dmmax);
        TH1D* hist2 = new TH1D("hist2", (branch + " Distribution (BDT > 0.5)").c_str(), nbins, dmmin, dmmax);
        // fullerTree->Draw("D_M>>hist2", "bdt_all > 0.5", "goff");
        fullerTree->Draw((branch + ">>hist2").c_str(), "bdt_all > 0.5", "goff");

        // normalise the two distributions for comparison
        hist1->Scale(1.0 / hist1->Integral());
        hist2->Scale(1.0 / hist2->Integral());

        // plot the two distributions
        TCanvas* canvas = new TCanvas("canvas", "canvas", 1000, 700);
        hist2->SetLineColor(kRed);
        hist2->SetLineWidth(2);
        hist2->Draw("HIST");
        hist1->SetLineColor(kBlue);
        hist1->SetLineWidth(2);
        hist1->Draw("HIST SAME");
        hist1->SetStats(0);
        hist2->SetStats(0);
        // set title, axes labels
        // hist2->SetTitle("D_M Distribution Comparison");
        hist2->SetTitle((branch + " Distribution Comparison").c_str());
        // hist2->GetXaxis()->SetTitle("D_M");
        hist2->GetXaxis()->SetTitle(branch.c_str());
        hist2->GetYaxis()->SetTitle("Normalised Events");
        // add legend
        TLegend* legend = new TLegend(0.6, 0.75, 0.9, 0.9);
        legend->AddEntry(hist1, "Full", "l");
        legend->AddEntry(hist2, "Signal (BDT score > 0.5)", "l");
        legend->SetTextSize(0.03);
        legend->Draw();
        // set y range minimum to 0
        hist1->SetMinimum(0);
        hist2->SetMinimum(0);
        // save plot
        canvas->Update();
        canvas->Modified();
        // canvas->SaveAs((config::plot_dir3("bdt_all", 0.5) + "dm_comparison.png").c_str());
        canvas->SaveAs((config::plot_dir3("bdt_all", 0.5) + branch + "_comparison.png").c_str());

        // cleanup
        delete canvas;
        delete hist1;
        delete hist2;
    }
    fullerFile->Close();
}