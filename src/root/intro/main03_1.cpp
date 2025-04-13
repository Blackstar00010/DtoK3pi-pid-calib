#include <TFile.h>
#include <TTree.h>
#include <TMVA/Factory.h>
#include <TMVA/DataLoader.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TMath.h>
#include <TROOT.h>
#include <iostream>
#include <regex>
#include <vector>
#include <string>
#include <cmath>  // For isnan and isinf
#include "../utils/common.cpp"
#include "../utils/config.cpp"


namespace main3 {
    void main();
    void temp();
}
void plotProb(const std::string& tmvaOutputFile="tmva/bdt_output.root");

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

    if (gDirectory != gROOT) {
        gROOT->cd();
    }
    TTree* tree1 = tree->CopyTree(cond_in.c_str());
    TTree* tree2 = tree->CopyTree(cond_out.c_str());
    return {tree1, tree2};
}

TTree* extractBkg(TTree* tree, Long64_t nEvents) {
    if (!tree) {
        std::cerr << "Error: Background TTree is null." << std::endl;
        return nullptr;
    }

    struct EventInfo {
        Long64_t entry;  // Entry index in the original tree
        double distance; // Distance from the origin
    };
    double originDM = 1862.0, originDeltaM = 145.35;

    // Variables to hold branch values
    double D_M, delta_M;
    tree->SetBranchAddress("D_M", &D_M);
    tree->SetBranchAddress("delta_M", &delta_M);

    // Compute distances and store event info
    std::vector<EventInfo> events;
    Long64_t nEntries = tree->GetEntries();
    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        double distance = TMath::Sqrt(TMath::Power(D_M - originDM, 2) + TMath::Power(delta_M - originDeltaM, 2));
        events.push_back({i, distance});
    }

    // Sort events by distance (descending order)
    std::sort(events.begin(), events.end(), [](const EventInfo& a, const EventInfo& b) {
        return a.distance > b.distance;
    });

    // Select the top nSignal events
    TTree* selectedTree = tree->CloneTree(0); // Clone structure but no entries
    for (Long64_t i = 0; i < nEvents && i < events.size(); ++i) {
        tree->GetEntry(events[i].entry);
        selectedTree->Fill();
    }

    return selectedTree;
}

void main3::temp() {
    // TTree* tree = tman::load_tree("MC");
    auto [mcfile, mctree] = tman::loadFileAndTree((config::mc_dir + "md1.root").c_str());
    auto [fullfile, fulltree] = tman::loadFileAndTree(config::full_root_file.c_str());
    if (!mctree || !fulltree) {
        std::cerr << "Error: No TTree loaded." << std::endl;
        return;
    }

    // Create a TMVA output file
    TFile* outputFile = TFile::Open("tmva/bdt_output.root", "RECREATE");

    // Initialize TMVA Factory and DataLoader
    TMVA::Factory* factory = new TMVA::Factory("TMVAClassification", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification");
    TMVA::DataLoader* loader = new TMVA::DataLoader("dataset");

    std::vector<std::string> whitelist = {"pis_BPVIP", "pis_BPVIPCHI2", "pis_CHI2", "pis_MINIP", "pis_MINIPCHI2", "K_BPVIP", "K_BPVIPCHI2", "K_CHI2", "K_MINIP", "K_MINIPCHI2", "pi1_BPVIP", "pi1_BPVIPCHI2", "pi1_CHI2", "pi1_MINIP", "pi1_MINIPCHI2", "pi2_BPVIP", "pi2_BPVIPCHI2", "pi2_CHI2", "pi2_MINIP", "pi2_MINIPCHI2", "pi3_BPVIP", "pi3_BPVIPCHI2", "pi3_CHI2", "pi3_MINIP", "pi3_MINIPCHI2"};
    for (const auto& branchName : whitelist) {
        if (!mctree->GetBranch(branchName.c_str()) || !fulltree->GetBranch(branchName.c_str())) {
            std::cerr << "Warning: Branch " << branchName << " not found in TTree." << std::endl;
        } else {
            loader->AddVariable(branchName);
        }
    }

    // Step 2: Set up the signal selection
    // std::vector<std::string> particles = {"K", "pi1", "pi2", "pi3", "pis", "D", "Dst"};
    // std::vector<int> trueids = {321, 211, 211, 211, 211, 421, 413};
    // std::vector<int> mtrueids = {421, 421, 421, 421, 413, 413, 0};
    // TTree* tempTree;
    // for (size_t i = 0; i < particles.size(); ++i) {
    //     std::string particle = particles[i];
    //     int pid = trueids[i];
    //     int mpid = mtrueids[i];
    //     TCut signalCut = Form("%s_TRUEID == %d || %s_TRUEID == -%d", particle.c_str(), pid, particle.c_str(), pid);
    //     tempTree = mctree->CopyTree(signalCut);
    //     if (mpid != 0) {
    //         signalCut = Form("%s_MC_MOTHER_ID == %d || %s_MC_MOTHER_ID == -%d", particle.c_str(), mpid, particle.c_str(), mpid);
    //         delete mctree;
    //         mctree = tempTree->CopyTree(signalCut);
    //     }
    //     else {
    //         delete mctree;
    //         mctree = tempTree;
    //     }
    //     delete tempTree;
    // }
    TCut signalCut = "K_TRUEID == 321 || K_TRUEID == -321";
    TTree* signalTree = mctree->CopyTree(signalCut);
    
    int sbratio = 10;
    Long64_t nSignal = mctree->GetEntries();
    fulltree = extractBkg(fulltree, nSignal * sbratio);
    Long64_t nBackground = fulltree->GetEntries();

    // Prepare the dataset for training and testing then apply signal and background cuts
    // give weights according to the lengths
    loader->AddSignalTree(mctree, 1.0);
    loader->AddBackgroundTree(fulltree, nSignal / nBackground);
    std::string traintestString = Form("nTrain_Signal=%d:nTest_Background=%d:nTest_Signal=%d:NTest_Background=%d:SplitMode=Random:NormMode=NumEvents:!V", int(nSignal * 0.9), int(nBackground * 0.9), int(nSignal * 0.1), int(nBackground * 0.1));
    loader->PrepareTrainingAndTestTree("", "", traintestString.c_str());
    loader->SetWeightExpression("1", "Signal");

    // Step 3: Book the BDT Method
    factory->BookMethod(loader, TMVA::Types::kBDT, "BDT",
                        "NTrees=500:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20");

    // Step 4: Train, Test, and Evaluate
    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    // Clean up
    outputFile->Close();
    delete factory;
    delete loader;
    // // print all content of pis_BPVIP
    // for (int i=0; i<10; i++) {
    //     tree->GetEntry(i);
    //     // std::cout << tree->GetLeaf("pis_BPVIP")->GetValue() << std::endl;
    // }
    // return;
    plotProb();
}

void main3::main() {
    auto [main_file, main_tree] = tman::loadFileAndTree(config::full_root_file);
    if (!main_tree) {
        std::cerr << "Error: No TTree loaded." << std::endl;
        return;
    }
    std::vector<TTree*> trees = split_data(main_tree);
    main_tree = trees[1];
    // main_tree = extractBkg(main_tree, 1000);
    main_tree->Print();
    // std::cout << main_tree->Print() << std::endl;
    std::string target_col = "D_BPVFDCHI2";
    for (int i = 0; i < 999; i++) {
        main_tree->GetEntry(i);
        float val = main_tree->GetLeaf(target_col.c_str())->GetValue();
        std::cout << i << " : " << val << std::endl;
        // std::cout << main_tree->GetLeaf(target_col.c_str())->GetValue() << std::endl;
    }
}

void plotProb(const std::string& tmvaOutputFile="tmva/bdt_output.root") {
    // Open the TMVA output file
    TFile* file = TFile::Open(tmvaOutputFile.c_str());
    if (!file || file->IsZombie()) {
        std::cerr << "Error: Unable to open file " << tmvaOutputFile << std::endl;
        return;
    }

    // Retrieve histograms for classifier responses
    TH1* hSignal = (TH1*)file->Get("dataset/Method_BDT/BDT/MVA_BDT_S");
    TH1* hBackground = (TH1*)file->Get("dataset/Method_BDT/BDT/MVA_BDT_B");

    if (!hSignal || !hBackground) {
        std::cerr << "Error: Unable to retrieve histograms from " << tmvaOutputFile << std::endl;
        file->Close();
        return;
    }

    // Create a canvas for plotting
    TCanvas* canvas = new TCanvas("canvas", "Probability Distribution", 800, 600);
    canvas->cd();

    // Normalize the histograms
    hSignal->Scale(1.0 / hSignal->Integral());
    hBackground->Scale(1.0 / hBackground->Integral());

    // Set histogram styles
    hSignal->SetLineColor(kRed);
    hSignal->SetLineWidth(2);
    hSignal->SetTitle("Classifier Response;Response;Probability Density");
    
    hBackground->SetLineColor(kBlue);
    hBackground->SetLineWidth(2);

    // Draw the histograms
    hSignal->Draw("HIST");
    hBackground->Draw("HIST SAME");

    // Add a legend
    TLegend* legend = new TLegend(0.7, 0.7, 0.9, 0.9);
    legend->AddEntry(hSignal, "Signal", "l");
    legend->AddEntry(hBackground, "Background", "l");
    legend->Draw();

    // Save the canvas
    canvas->SaveAs((config::plot_dir3 + "prob_hist.png").c_str());

    // Clean up
    file->Close();
    delete canvas;
}