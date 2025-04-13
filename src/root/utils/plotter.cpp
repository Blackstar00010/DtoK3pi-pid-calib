#include <iostream>
#include <TTree.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TColor.h>
#include <TDirectory.h>
#include <TList.h>


namespace plotter {  // plotting functions
    TCanvas* sigbkg(TTree* sigtree, TTree* bkgtree, const std::string& col, 
                    int bins, bool normalize, std::vector<bool> autorange, std::vector<double> range, 
                    std::vector<std::string> labels, std::string xtitle, std::string title, std::string filename);
    TCanvas* hist(TTree* tree, const std::string& col, 
                  int bins, std::vector<bool> autorange, std::vector<double> range, 
                  std::string xtitle, std::string title, std::string filename);
    TCanvas* hists(TTree* tree, const std::vector<std::string>& cols, 
                   int bins, std::vector<bool> autorange, std::vector<double> range, bool normalise,
                   std::string xtitle, std::string title, std::string filename);
    TCanvas* hist2d(TTree* tree, const std::string& col1, const std::string& col2, 
                    int bins1, int bins2, std::vector<bool> autorange, std::vector<double> range1, std::vector<double> range2,
                    bool plotBoxes, std::string xtitle, std::string ytitle, std::string title, std::string filename);
    void PlotContourHeatmap(TTree* tree, const std::string& branchX, const std::string& branchY, const std::string& outputFile = "cntrhtmp.png");
}


/* sigbkg(TTree* sigtree, TTree* bkgtree, const std::string& col, int bins, double min, double max)
 * Plots two normalised 1D histograms of a column from two different TTrees (signal and background) on the same canvas to compare them.
 * Arguments:
 *  - sigtree: TTree pointer containing the signal data
 *  - bkgtree: TTree pointer containing the background data
 *  - col: Name of the column to plot
 *  - bins: Number of bins for the histograms. Default is 300.
 *  - normalize: Boolean value to normalize the histograms. Default is true.
 *  - autorange: Vector of boolean values to automatically set the range for each column. Default is {true, true}.
 *  - range: Vector{min, max} of double values to set the range for each column. Default is {0.0, 0.0}. If autorange is false and max<min, the range is automatically set.
 *  - labels: Vector of strings containing the labels for the two histograms. Default is {"sig", "bkg"}.
 *  - xtitle: Title for the x-axis. Default is "". If empty, the column name is used. If "NONE", no title is set.
 *  - title: Title for the histogram. Default is "". If empty, ("Normalised"+)`col`+" Distribution" is used. If "NONE", no title is set.
 *  - filename: Name of the file to save the plot. Default is "". If empty, col+".png" is used. If "NONE", no file is saved.
 */
TCanvas* plotter::sigbkg(TTree* sigtree, TTree* bkgtree, const std::string& col, int bins = 300, bool normalize = true,
                     std::vector<bool> autorange = {true, true}, std::vector<double> range = {0.0, 0.0},
                     std::vector<std::string> labels = {"sig", "bkg"}, std::string xtitle = "", std::string title = "",
                     std::string filename = "") {
    if (!sigtree || !bkgtree) {
        std::cerr << "Error: Null TTree pointer provided!" << std::endl;
        return nullptr;
    }
    // Get the minimum and maximum values for the column
    double min = std::min(sigtree->GetMinimum(col.c_str()), bkgtree->GetMinimum(col.c_str()));
    double max = std::max(sigtree->GetMaximum(col.c_str()), bkgtree->GetMaximum(col.c_str()));
    if ((!autorange[0] && !autorange[1]) && (range[1] < range[0])) {
        std::cerr << "Error: Minimum and maximum values are the same for branch " << col << std::endl;
        return nullptr;
    }
    if (autorange[0] || range[1] < range[0]) {
        range[0] = min;
    }
    if (autorange[1] || range[1] < range[0]) {
        range[1] = max;
    }

    // Create histograms for the two columns
    TH1F* hist1 = new TH1F("hist1", (col + " Distribution (sig)").c_str(), bins, min, max);
    TH1F* hist2 = new TH1F("hist2", (col + " Distribution (bkg)").c_str(), bins, min, max);

    // Draw the histograms separately to fill them
    sigtree->Draw((col + ">>hist1").c_str(), "", "");  // Fill histogram1 with data from col1
    bkgtree->Draw((col + ">>hist2").c_str(), "", "");  // Fill histogram2 with data from col2

    if (normalize) {
        hist1->Scale(1.0 / hist1->Integral());
        hist2->Scale(1.0 / hist2->Integral());
    }

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
    TLegend* legend = new TLegend(0.8, 0.8, 0.9, 0.9);
    legend->SetTextSize(0.03);
    legend->AddEntry(hist1, labels[0].c_str(), "l");
    legend->AddEntry(hist2, labels[1].c_str(), "l");
    legend->Draw();

    // set title and axes names
    if (title == "") {
        if (normalize) {title = "Normalised ";}
        title = title + col + " Distribution";
    } else if (title == "NONE") {
        title = "";
    }
    hist1->SetTitle(title.c_str());
    if (xtitle == "") {
        xtitle = col;
    } else if (xtitle == "NONE") {
        xtitle = "";
    }
    hist1->GetXaxis()->SetTitle(col.c_str());
    std::string ytitle = "Frequency";
    if (normalize) {ytitle = "Normalised Frequency";}
    hist1->GetYaxis()->SetTitle(ytitle.c_str());

    // stats such as entries on the top right corner: turnoff
    hist1->SetStats(0);
    hist2->SetStats(0);
    // to set this globally for current session, use `gStyle->SetOptStat(0);`

    // Redraw to apply changes
    gPad->Modified();
    gPad->Update();

    // Save the canvas as an image if needed
    if (filename == "") {
        filename = col + ".png";
    } else if (filename == "NONE") {
        delete hist1;
        delete hist2;
        return canvas;
    }
    // if filename doesn't have .png, add it
    if (filename.find(".png") == std::string::npos) {
        filename += ".png";
    }
    canvas->SaveAs(("../../plots/2_sigbkg_cpp/sigbkg/"+filename).c_str());

    // Clean up memory
    delete hist1;
    delete hist2;
    // delete canvas;
    return canvas;
}

/*
* Plot a 1D histogram of a column(branch) from a TTree
* @param tree: TTree pointer containing the data
* @param col: Name of the column to plot
* @param bins: Number of bins for the histogram. Default is 300.
* @param autorange: Vector of boolean values to automatically set the range for the column. Default is {true, true}.
* @param range: Vector{min, max} of double values to set the range for the column. Default is {0.0, 0.0}. If autorange is false and max<min, the range is automatically set.
* @param xtitle: Title for the x-axis. Default is "". If empty, the column name is used. If "NONE", no title is set.
* @param title: Title for the histogram. Default is "". If empty, `col`+" Distribution" is used. If "NONE", no title is set.
* @param filename: Name of the file to save the plot. Default is "". If empty, col+".png" is used. If "NONE", no file is saved.
* @return TCanvas pointer to the canvas containing the histogram
*/
TCanvas* plotter::hist(TTree* tree, const std::string& col, int bins = 300, std::vector<bool> autorange = {true, true}, std::vector<double> range = {0.0, 0.0},
                   std::string xtitle = "", std::string title = "", std::string filename = "") {
    if (!tree) {
        std::cerr << "Error: Null TTree pointer provided!" << std::endl;
        return nullptr;
    }
    // Get the minimum and maximum values for the column
    double min = tree->GetMinimum(col.c_str());
    double max = tree->GetMaximum(col.c_str());
    if ((!autorange[0] && !autorange[1]) && (range[1] < range[0])) {
        std::cerr << "Error: Minimum and maximum values are the same for branch " << col << std::endl;
        return nullptr;
    }
    if (autorange[0] || range[1] < range[0]) {
        range[0] = min;
    }
    if (autorange[1] || range[1] < range[0]) {
        range[1] = max;
    }

    // Create histograms for the two columns
    TH1F* hist = new TH1F("hist", (col + " Distribution").c_str(), bins, min, max);

    // Draw the histograms separately to fill them
    tree->Draw((col + ">>hist").c_str(), "", "");  // Fill histogram with data from col

    // Create a canvas to draw the histograms
    TCanvas* canvas = new TCanvas("canvas", "Histogram", 800, 600);

    // Set line colors and styles for the histograms
    hist->SetLineColor(kBlack);
    hist->SetLineWidth(2);

    // Draw the histogram
    hist->Draw("HIST");

    // set title and axes names
    if (title == "") {
        title = col + " Distribution";
    } else if (title == "NONE") {
        title = "";
    }
    hist->SetTitle(title.c_str());
    if (xtitle == "") {
        xtitle = col;
    } else if (xtitle == "NONE") {
        xtitle = "";
    }
    hist->GetXaxis()->SetTitle(col.c_str());
    hist->GetYaxis()->SetTitle("Frequency");

    // stats such as entries on the top right corner: turnoff
    hist->SetStats(0);
    // to set this globally for current session, use `gStyle->SetOptStat(0);`

    // Redraw to apply changes
    gPad->Modified();
    gPad->Update();

    // Save the canvas as an image if needed
    if (filename == "") {
        filename = col + ".png";
    } else if (filename == "NONE") {
        delete hist;
        return canvas;
    }
    // if filename doesn't have .png, add it
    if (filename.find(".png") == std::string::npos) {
        filename += ".png";
    }
    // if filename doesn't have "plots/", add it
    if (filename.find("plots/") == std::string::npos) {
        filename = "../../plots/2_sigbkg_cpp/" + filename;
    }
    canvas->SaveAs((filename).c_str());

    // Clean up memory
    delete hist;
    // delete canvas;
    return canvas;
}

TCanvas* plotter::hist2d(TTree* tree, const std::string& col1, const std::string& col2, 
                         int bins1 = 300, int bins2 = 300, 
                         std::vector<bool> autorange = {true, true}, std::vector<double> range1 = {0.0, 0.0}, std::vector<double> range2 = {0.0, 0.0},
                         bool plotBoxes = false,
                         std::string xtitle = "", std::string ytitle = "", std::string title = "", std::string filename = "") {
    if (!tree) {
        std::cerr << "Error: Null TTree pointer provided!" << std::endl;
        return nullptr;
    }

    TCanvas* canvas = new TCanvas("2dhist", (col1 + " vs " + col2).c_str(), 800, 600);

    if (autorange[0] || range1[1] <= range1[0]) {
        range1[0] = tree->GetMinimum(col1.c_str());
        range1[1] = tree->GetMaximum(col1.c_str());
    }
    if (autorange[1] || range2[1] <= range2[0]) {
        range2[0] = tree->GetMinimum(col2.c_str());
        range2[1] = tree->GetMaximum(col2.c_str());
    }

    // nBinsX, xMin, xMax, nBinsY, yMin, yMax
    tree->Draw((col2 + ":" + col1 + ">>hist2D(" + std::to_string(bins1) + ", " + std::to_string(range1[0]) + ", " + std::to_string(range1[1]) + ", " + std::to_string(bins2) + ", " + std::to_string(range2[0]) + ", " + std::to_string(range2[1]) + ")").c_str(), "", "COLZ");

    // change titles and redraw
    TH2F* hist = (TH2F*)gDirectory->Get("hist2D");
    if (hist) {
        hist->SetTitle((col1 + " vs " + col2).c_str());
        hist->GetXaxis()->SetTitle(col1.c_str());
        hist->GetYaxis()->SetTitle(col2.c_str());

        // stats such as entries on the top right corner: turnoff
        hist->SetStats(0);
        // to set this globally for current session, use `gStyle->SetOptStat(0);`

        // Redraw to apply changes
        gPad->Modified();
        gPad->Update();
    }

    if (plotBoxes) {
        double dmmean = 1862.0;
        double deltamean = 145.35;
        double dmwidtho = 60.0;
        double deltawidtho = 5.0;
        double dmwidthi = 10.0;
        double deltawidthi = 1.0;
        // plot_box(dmmean, deltamean, dmwidtho, deltawidtho, 'r');
        // plot_box(dmmean, deltamean, dmwidthi, deltawidthi);
    }

    // titles and axes
    if (title == "") {
        title = col1 + " vs " + col2;
    } else if (title == "NONE") {
        title = "";
    }
    if (xtitle == "") {
        xtitle = col1;
    } else if (xtitle == "NONE") {
        xtitle = "";
    }
    if (ytitle == "") {
        ytitle = col2;
    } else if (ytitle == "NONE") {
        ytitle = "";
    }
    hist->SetTitle(title.c_str());
    hist->GetXaxis()->SetTitle(xtitle.c_str());
    hist->GetYaxis()->SetTitle(ytitle.c_str());

    // Save the canvas as an image if needed
    if (filename == "") {  // default filename
        filename = col1 + "_vs_" + col2 + ".png";
    } else if (filename == "NONE") {  // don't save
        return canvas;
    } else if (filename.find(".png") == std::string::npos) {  // add .png if not present
        filename += ".png";
    }

    canvas->SaveAs(filename.c_str());
    return canvas;
}


void plotter::PlotContourHeatmap(TTree* tree, const std::string& branchX, const std::string& branchY, const std::string& outputFile) {
    if (!tree) {
        std::cerr << "Error: Invalid TTree pointer." << std::endl;
        return;
    }

    // Define the range for the branches
    Double_t xMin = tree->GetMinimum(branchX.c_str());
    Double_t xMax = tree->GetMaximum(branchX.c_str());
    Double_t yMin = tree->GetMinimum(branchY.c_str());
    Double_t yMax = tree->GetMaximum(branchY.c_str());

    // Create a 2D histogram (heatmap)
    const int nBinsX = 100; // Number of bins for X-axis
    const int nBinsY = 100; // Number of bins for Y-axis
    TH2F* hist2D = new TH2F("hist2D", "Contour + Heatmap;X-axis;Y-axis", nBinsX, xMin, xMax, nBinsY, yMin, yMax);

    // Fill the histogram from the TTree
    tree->Draw((branchY + ":" + branchX + ">>hist2D").c_str(), "", "goff");

    // Create a canvas
    TCanvas* canvas = new TCanvas("canvas", "Contour + Heatmap", 1000, 800);

    // set right margin to 0.15
    canvas->SetRightMargin(0.15);

    // Draw the heatmap
    hist2D->Draw("COLZ"); // "COLZ" draws the heatmap with a color palette

    // Draw the contour lines on top
    hist2D->SetContour(20); // Set the number of contour levels
    hist2D->Draw("CONT3 SAME"); // "CONT3" draws smooth contour lines

    // remove stats
    gStyle->SetOptStat(0);

    // Save the plot to a file
    canvas->SaveAs(outputFile.c_str());

    // Cleanup
    delete hist2D;
    delete canvas;
}