#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <vector>
#include <string>
#include "../config.cpp"

namespace main2 {
    void main();
}

TTree* load_tree(bool long_version=true) {
    TFile* file = nullptr;
    // Open the ROOT file
    if (long_version) {
        // in root console, use
        // $ root -l ./data/full.root
        // root [0] TFile *file = TFile::Open("./data/full.root");
        // root [1] DecayTree->Draw("D_M:delta_M");
        file = TFile::Open(full_root_file.c_str());
    } else {
        // in root console, use
        // $ root -l ./data/sample.root
        // root [0] TFile *file = TFile::Open("./data/sample.root");
        // root [1] DstToD0Pi_D0ToKPiPiPi->cd();
        // root [2] DecayTree->Draw("D_M:delta_M");
        file = TFile::Open(data_dir + "sample.root");
        file->cd("DstToD0Pi_D0ToKPiPiPi");
    }
    if (!file || file->IsZombie()) {
        std::cerr << "Error opening file!" << std::endl;
        return nullptr;
    }

    // Get the TTree named "DecayTree" (can be checked by .ls command in root prompt)
    TTree *tree = (TTree*)file->Get("DecayTree");
    if (!tree) {
        std::cerr << "Error: TTree 'DecayTree' not found!" << std::endl;
        file->Close();
        return nullptr;
    }
    return tree;
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
    std::cerr << "running split_data() of main02.cpp outputs errors" << std::endl;
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
    TString cond_in_t = TString(cond_in.c_str());
    TString cond_out_t = TString(cond_out.c_str());
    std::cout << "debug: " << cond_in_t << std::endl;
    std::cout << "debug: " << cond_out_t << std::endl;
    TTree* tree1 = tree->CopyTree(cond_in_t);
    std::cout << "debug: treespliting 1 done" << std::endl;
    return {nullptr, tree1};
    // TTree* tree2 = tree->CopyTree(cond_out_t);
    TTree* tree2 = nullptr;
    std::cout << "debug: np after tree splitting" << std::endl;
    // Return the two trees in a vector
    return {tree1, tree2};
}

void plot_box(double x_center, double y_center, double x_width, double y_width, char color) {
    double x1 = x_center - x_width/2;
    double x2 = x_center + x_width/2;
    double y1 = y_center - y_width/2;
    double y2 = y_center + y_width/2;
    
    TBox* box = new TBox(x1, y1, x2, y2);
    if (color == 'r'){box->SetLineColor(kRed);}
    else if (color == 'g') { box->SetLineColor(kGreen);}
    else if (color == 'b') { box->SetLineColor(kBlue);}
    else {
        if (color != ' ') {std::cout << "variable color for plot_box function should be one of 'r', 'g', 'b'. else, black box is plotted." << std::endl;}
        box->SetLineColor(kBlack);
    }
    box->SetLineWidth(2);         // Set the line width
    box->SetFillColorAlpha(kWhite, 0.2); // Set the fill color with transparency (0.2 for 20% opacity)
    box->SetFillStyle(0);

    // Draw the box on the canvas (on top of the histogram)
    box->Draw("same");
}
void plot_box(double x1, double y1, double x2, double y2) {  // black box
    plot_box(x1, y1, x2, y2, ' ');
}

void plot_2dmass(TTree* tree, std::string filename) {
    std::string mD = "D_M";
    std::string mdelta = "delta_M";

    TCanvas* canvas = new TCanvas("2dmass", "delta M vs M of D0", 800, 600);

    // nBinsX, xMin, xMax, nBinsY, yMin, yMax
    tree->Draw((mdelta+":"+mD+">>hist2D(300, 1790, 1925, 300, 135, 160)").c_str(), "", "COLZ");

    // change titles and redraw
    TH2F* hist = (TH2F*)gDirectory->Get("hist2D");
    if (hist) {
        hist->SetTitle("D0 mass vs (D0 mass - D* mass)");
        hist->GetXaxis()->SetTitle("D0 mass [MeV/c^2]");
        hist->GetYaxis()->SetTitle("(D0 mass - D* mass) [MeV/c^2]");

        // stats such as entries on the top right corner: turnoff
        hist->SetStats(0);
        // to set this globally for current session, use `gStyle->SetOptStat(0);`

        // Redraw to apply changes
        gPad->Modified();
        gPad->Update();
    }

    
    double dmmean = 1862.0;
    double deltamean = 145.35;
    double dmwidtho = 60.0;
    double deltawidtho = 5.0;
    double dmwidthi = 10.0;
    double deltawidthi = 1.0;
    plot_box(dmmean, deltamean, dmwidtho, deltawidtho, 'r');
    plot_box(dmmean, deltamean, dmwidthi, deltawidthi);

    canvas->SaveAs((plots_dir + "2_sigbkg_cpp/" + filename + ".png").c_str());
    delete canvas;
}

/* plot_2colscomp(TTree* tree, const std::string& col1, const std::string& col2, int bins, double min, double max)
 * Plots two 1D histograms of two columns of a TTree, respectively, on the same canvas to compare them.
 * Arguments:
 *  - tree: TTree pointer containing the data
 *  - col1: Name of the first column to plot
 *  - col2: Name of the second column to plot
 *  - bins: Number of bins for the histograms
 *  - min: Minimum value for the histograms
 *  - max: Maximum value for the histograms
 */
void plot_2colscomp(TTree* tree, const std::string& col1, const std::string& col2, int bins, double min, double max) {
    if (!tree) {
        std::cerr << "Error: Null TTree pointer provided!" << std::endl;
        return;
    }

    // Create histograms for the two columns
    TH1F* hist1 = new TH1F("hist1", (col1 + " Distribution").c_str(), bins, min, max);
    TH1F* hist2 = new TH1F("hist2", (col2 + " Distribution").c_str(), bins, min, max);

    // Draw the histograms separately to fill them
    tree->Draw((col1 + ">>hist1").c_str(), "", "");  // Fill histogram1 with data from col1
    tree->Draw((col2 + ">>hist2").c_str(), "", "");  // Fill histogram2 with data from col2

    // Create a canvas to draw the histograms
    TCanvas* canvas = new TCanvas("canvas", "Overlapping Histograms", 800, 600);

    // Set line colors and styles for the histograms
    hist1->SetLineColor(kRed);
    hist1->SetLineWidth(2);
    hist2->SetLineColor(kBlue);
    hist2->SetLineWidth(2);

    // Draw the first histogram
    hist1->Draw("HIST");
    
    // Draw the second histogram on top of the first one
    hist2->Draw("HIST SAME");

    // Create a legend to distinguish between the two histograms
    TLegend* legend = new TLegend(0.7, 0.7, 0.9, 0.9);
    legend->AddEntry(hist1, col1.c_str(), "l");
    legend->AddEntry(hist2, col2.c_str(), "l");
    legend->Draw();

    // Save the canvas as an image if needed
    canvas->SaveAs((plots_dir + "2_sigbkg_cpp/" + col1 + "_vs_" + col2 + ".png").c_str());

    // Clean up memory
    delete hist1;
    delete hist2;
    delete canvas;
}

/* plot_sig_v_bkg(TTree* sigtree, TTree* bkgtree, const std::string& col, int bins, double min, double max)
 * Plots two normalised 1D histograms of a column from two different TTrees (signal and background) on the same canvas to compare them.
 * Arguments:
 *  - sigtree: TTree pointer containing the signal data
 *  - bkgtree: TTree pointer containing the background data
 *  - col: Name of the column to plot
 *  - bins: Number of bins for the histograms
 *  - min: Minimum value for the histograms
 *  - max: Maximum value for the histograms
 */
void plot_sig_v_bkg(TTree* sigtree, TTree* bkgtree, const std::string& col, int bins, double min, double max) {
    if (!sigtree || !bkgtree) {
        std::cerr << "Error: Null TTree pointer provided!" << std::endl;
        return;
    }

    // Create histograms for the two columns
    TH1F* hist1 = new TH1F("hist1", (col + " Distribution (sig)").c_str(), bins, min, max);
    TH1F* hist2 = new TH1F("hist2", (col + " Distribution (bkg)").c_str(), bins, min, max);

    // Draw the histograms separately to fill them
    sigtree->Draw((col + ">>hist1").c_str(), "", "");  // Fill histogram1 with data from col1
    bkgtree->Draw((col + ">>hist2").c_str(), "", "");  // Fill histogram2 with data from col2

    // Normalize the histograms
    hist1->Scale(1.0 / hist1->Integral());
    hist2->Scale(1.0 / hist2->Integral());

    // Create a canvas to draw the histograms
    TCanvas* canvas = new TCanvas("canvas", "Overlapping Histograms", 800, 600);

    // Set line colors and styles for the histograms
    hist1->SetLineColor(kRed);
    hist1->SetLineWidth(2);
    hist2->SetLineColor(kBlue);
    hist2->SetLineWidth(2);

    // Draw the first histogram
    hist1->Draw("HIST");
    
    // Draw the second histogram on top of the first one
    hist2->Draw("HIST SAME");

    // Create a legend to distinguish between the two histograms
    TLegend* legend = new TLegend(0.7, 0.7, 0.9, 0.9);
    legend->AddEntry(hist1, "signal", "l");
    legend->AddEntry(hist2, "background", "l");
    legend->Draw();

    // set title and axes names
    hist1->SetTitle(("Normalised " + col + " Distribution").c_str());
    hist1->GetXaxis()->SetTitle(col.c_str());
    hist1->GetYaxis()->SetTitle("Frequency");

    // stats such as entries on the top right corner: turnoff
    hist1->SetStats(0);
    hist2->SetStats(0);
    // to set this globally for current session, use `gStyle->SetOptStat(0);`

    // Redraw to apply changes
    gPad->Modified();
    gPad->Update();

    // Save the canvas as an image if needed
    canvas->SaveAs((plots_dir + "2_sigbkg_cpp/" + col + ".png").c_str());

    // Clean up memory
    delete hist1;
    delete hist2;
    delete canvas;
}

void main2::main() {
    TTree* tree = load_tree();
    plot_2dmass(tree, "2d_mass");
    std::vector<TTree*> trees = split_data(tree);  // 0: signal, 1: background
    std::cout << "debug: finished splitting tree" << std::endl;
    plot_sig_v_bkg(trees[0], trees[1], "K_ETA", 300, 1.5, 6.0);
    plot_sig_v_bkg(trees[0], trees[1], "K_P", 300, 0.0, 100000.0);
    plot_sig_v_bkg(trees[0], trees[1], "K_PT", 300, 0.0, 10000.0);
    for (int i=1; i<4; i++) {
        plot_sig_v_bkg(trees[0], trees[1], "pi"+std::to_string(i)+"_ETA", 300, 1.5, 6.0);
        plot_sig_v_bkg(trees[0], trees[1], "pi"+std::to_string(i)+"_P", 300, 0.0, 100000.0);
        plot_sig_v_bkg(trees[0], trees[1], "pi"+std::to_string(i)+"_PT", 300, 0.0, 10000.0);
    }
    plot_sig_v_bkg(trees[0], trees[1], "D_M", 300, 1790, 1925);

    /*# plot p
    # plot_scatter(data[["K_P", "K_TRACK_P"]], "K_P", "K_TRACK_P")  # exactly the same
    for particle in ["K", "pi", "D"]:
        plot_range = (0, 150000) if particle == "D" else (0, 100000)
        particlep = f"{particle}_P"
        filename = f"{mode}/sb_"
        filename += particlep if nomenclature_particle_first else f"P_{particle}"
        plot_sigbkg(data_sig[particlep], data_bkg[particlep],
                    plot_range=plot_range, plot_quantity=particlep, filename=filename)
    */
    // std::vector<std::string> particles = {"K", "pi1", "pi2", "pi3", "D"};
    // for (std::string particle : particles) {
    //     std::pair<double, double> plot_range = {0.0, 150000.0};
    //     if (particle == "D") {plot_range = {0.0, 100000.0};}
    //     std::string particlep = particle + "_P";
    //     plot_sig_v_bkg(trees[0], trees[1], particlep, 300, plot_range.first, plot_range.second);
    // }

}