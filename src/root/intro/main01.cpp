#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TLegend.h>
#include <TGaxis.h>
#include <TCanvas.h>
#include <vector>
#include <string>
#include <iostream>

namespace main1 {
    void main();
}

TTree* load_tree() {
    // Open the ROOT file
    TFile *file = TFile::Open("sample.root");
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

void plot_all_branches(TTree* tree) {

    // Get the list of branches
    TObjArray *branches = tree->GetListOfBranches();
    int nBranches = branches->GetEntries();

    // Loop over all branches
    for (int i = 0; i < nBranches; ++i) {
        // Get the branch name
        TBranch *branch = (TBranch*)branches->At(i);
        const char* branchName = branch->GetName();

        // Create a canvas for each branch
        TCanvas *canvas = new TCanvas(branchName, branchName, 800, 600);

        // Draw the histogram for the current branch
        tree->Draw(Form("%s>>hist_%s(100)", branchName, branchName));

        // Retrieve the histogram created automatically by Draw()
        TH1F *hist = (TH1F*)gDirectory->Get(Form("hist_%s", branchName));
        if (hist) {
            hist->SetTitle(branchName);
            hist->GetXaxis()->SetTitle(branchName);
            hist->GetYaxis()->SetTitle("Entries");
            hist->Draw();
            
            // Optionally save each histogram as a PNG file
            canvas->SaveAs(Form("./all/%s.png", branchName));
        } else {
            std::cerr << "Error: Could not create histogram for " << branchName << std::endl;
        }

        // Clean up
        delete canvas;
    }
}
void plot_eta_vs_p(TTree* tree) {
    std::vector<std::string> particles = {"K", "pis", "pi1"};

    for (const auto& particle : particles) {
        // Create a canvas for the scatter plot
        TCanvas *canvas = new TCanvas("c_scatter", ("Scatter Plot of "+particle+"_eta vs "+particle+"_p").c_str(), 800, 600);
        
        // Define a 2D histogram with increased bin counts
        int nBinsX = 200; // Number of bins for the x-axis (p)
        int nBinsY = 200; // Number of bins for the y-axis (eta)
        float xMin = 0.0, xMax = 400.0; // Adjust according to your data range
        float yMin = 0.0, yMax = 6.0;   // Adjust according to your data range

        // Draw the scatter plot
        tree->Draw((particle+"_ETA:"+particle+"_P").c_str(), "", "COLZ");

        // Optionally save the canvas as a PNG file
        canvas->SaveAs(("plots/"+particle+"_eta_vs_p.png").c_str());

        // Clean up
        delete canvas;
    }
}
void plot_eta_vs_p2(TTree* tree) {
    // Define pairs of branches for the scatter plots using std::vector and std::string
    std::vector<std::pair<std::string, std::string>> particles = {
        {"K_P", "K_ETA"},      // p vs eta for K
        {"pis_P", "pis_ETA"},    // p vs eta for pi
        {"pi1_P", "pi1_ETA"}   // p vs eta for pi1
    };

    // Loop over the branch pairs and extract data into vectors
    for (int i=0; i<3; i++) {
        const std::string& p_branch = particles[i].first;
        const std::string& eta_branch = particles[i].second;

        // Create vectors to store the extracted data
        std::vector<float> p_values;
        std::vector<float> eta_values;

        // Set branch addresses
        float p_val, eta_val;
        tree->SetBranchAddress(p_branch.c_str(), &p_val);
        tree->SetBranchAddress(eta_branch.c_str(), &eta_val);

        // Loop over the entries and extract data
        Long64_t nEntries = tree->GetEntries();
        for (Long64_t i = 0; i < nEntries; ++i) {
            tree->GetEntry(i);
            p_values.push_back(p_val);
            eta_values.push_back(eta_val);
        }

        // Create a 2D histogram for the extracted data
        TCanvas *canvas = new TCanvas((eta_branch + "_vs_" + p_branch).c_str(),
                                      ("Scatter Plot of " + eta_branch + " vs " + p_branch).c_str(), 800, 600);
        int nbinsX = 300; // Number of bins for p
        int nbinsY = 300; // Number of bins for eta

        // Determine histogram range based on data (or set manually)
        float p_min = *std::min_element(p_values.begin(), p_values.end());
        float p_max = *std::max_element(p_values.begin(), p_values.end());
        float eta_min = *std::min_element(eta_values.begin(), eta_values.end());
        float eta_max = *std::max_element(eta_values.begin(), eta_values.end());

        TH2F *hist = new TH2F("hist", ("Scatter Plot of " + eta_branch + " vs " + p_branch).c_str(),
                              nbinsX, p_min, p_max, nbinsY, eta_min, eta_max);

        // Fill the histogram with data
        for (size_t i = 0; i < p_values.size(); ++i) {
            hist->Fill(p_values[i], eta_values[i]);
        }

        // Draw the histogram with a color map
        hist->Draw("COLZ");

        // Optionally save the canvas as a PNG file
        canvas->SaveAs((std::string("./plots/scatter_") + eta_branch + "_vs_" + p_branch + ".png").c_str());

        // Clean up
        delete hist;
        delete canvas;
    }
}
void plot_pid_difference(TTree* tree) {

    // Create a canvas to draw the histogram
    TCanvas *canvas = new TCanvas("canvas", "Histogram of K_PID_K - K_PID_PI", 800, 600);

    // Draw the histogram of K_PID_K - K_PID_PI
    tree->Draw("(K_PID_K - K_PID_PI) >> h_diff(100, -100, 100)");

    // Retrieve the histogram
    TH1F *h_diff = (TH1F*)gDirectory->Get("h_diff");
    if (h_diff) {
        h_diff->SetTitle("K_PID_K - K_PID_PI");
        h_diff->GetXaxis()->SetTitle("K_PID_K - K_PID_PI");
        h_diff->GetYaxis()->SetTitle("Entries");
        h_diff->Draw();
        
        // Optionally save the canvas as an image
        canvas->SaveAs("./plots/histogram_K_PID_diff.png");
    } else {
        std::cerr << "Error: Could not create histogram." << std::endl;
    }

    // Clean up
    delete canvas;
}

TH1F* createCDF(TH1F* hist, const char* cdfName) {
    // Clone the original histogram to create a CDF histogram
    TH1F* cdf = (TH1F*)hist->Clone(cdfName);
    cdf->Reset(); // Clear the contents of the cloned histogram

    // Calculate the cumulative sum
    double cumulative = 0;
    for (int i = 1; i <= hist->GetNbinsX(); i++) {
        cumulative += hist->GetBinContent(i);
        cdf->SetBinContent(i, cumulative);
    }

    // Normalize the CDF to a maximum of 1
    double total = hist->Integral();
    if (total > 0) {
        cdf->Scale(1.0 / total);
    }

    return cdf;
}

// does not work
void plot_cdf_with_difference(TTree* tree) {

    // Create histograms for the two columns
    TH1F *hist1 = new TH1F("hist1", "K_PID_K", 100, -10, 10);
    TH1F *hist2 = new TH1F("hist2", "K_PID_PI", 100, -10, 10);

    // Fill the histograms with data from the columns
    tree->Draw("K_PID_K >> hist1");
    tree->Draw("K_PID_PI >> hist2");

    // Create CDFs for each histogram
    TH1F *cdf1 = createCDF(hist1, "cdf1");
    TH1F *cdf2 = createCDF(hist2, "cdf2");

    // Create a histogram for the difference (cdf1 - cdf2)
    TH1F *cdf_diff = (TH1F*)cdf1->Clone("cdf_diff");
    cdf_diff->Add(cdf2, -1); // Subtract CDF2 from CDF1

    // Create a canvas to draw the CDFs and their difference
    TCanvas *canvas = new TCanvas("canvas", "CDF of K_PID_K and K_PID_PI with Difference", 800, 600);
    canvas->Divide(1, 1);

    // Draw the CDFs on the same plot
    cdf1->SetLineColor(kRed);
    cdf1->SetLineWidth(2);
    cdf2->SetLineColor(kBlue);
    cdf2->SetLineWidth(2);

    // Draw CDF1 and CDF2 on the left y-axis
    cdf1->Draw("HIST");
    cdf2->Draw("HIST SAME");

    // Create a right y-axis for the difference plot
    TGaxis *axis = new TGaxis(gPad->GetUxmax(), gPad->GetUymin(),
                              gPad->GetUxmax(), gPad->GetUymax(),
                              cdf_diff->GetMinimum(), cdf_diff->GetMaximum(),
                              510, "+L");
    axis->SetTitle("CDF Difference (CDF1 - CDF2)");
    axis->SetLineColor(kGreen + 2);
    axis->SetLabelColor(kGreen + 2);
    axis->SetTitleColor(kGreen + 2);
    axis->Draw();

    // Draw the difference plot using a different scale on the same canvas
    cdf_diff->SetLineColor(kGreen + 2);
    cdf_diff->SetLineWidth(2);
    cdf_diff->Draw("HIST SAME");

    // Add a legend
    TLegend *legend = new TLegend(0.1, 0.1, 0.3, 0.3);
    legend->AddEntry(cdf1, "CDF1: K_PID_K", "l");
    legend->AddEntry(cdf2, "CDF2: K_PID_PI", "l");
    legend->AddEntry(cdf_diff, "Difference (CDF1 - CDF2)", "l");
    legend->Draw();

    // Optionally save the canvas as an image
    canvas->SaveAs("./plots/cdf_with_difference.png");

    // Clean up
    delete hist1;
    delete hist2;
    delete cdf1;
    delete cdf2;
    delete cdf_diff;
    delete canvas;
}

void main1::main() {
    TTree* tree = load_tree();
    // plot_all_branches(tree);
    // plot_eta_vs_p(tree);  // low res version
    // plot_eta_vs_p2(tree);  // high res but much longer time
    // plot_pid_difference(tree);
    // plot_cdf_with_difference(tree);
    tree->GetCurrentFile()->Close();
}