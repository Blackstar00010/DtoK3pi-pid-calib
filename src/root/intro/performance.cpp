#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <iostream>
#include <TFile.h>
#include <TTree.h>
#include "../utils/common.cpp"
#include "../utils/config.cpp"

// Function 1: Some operation (e.g., summing numbers)
void function1(TTree* tree) {
    auto [file, newtree] = tman::massCutFullTree(tree);
    if (!remove("temp.root")) {
        std::cerr << "Error: Failed to delete temp.root" << std::endl;
    }
    delete newtree;
    delete file;
}

// Function 2: Another operation (e.g., multiplying numbers)
void function2(TTree* tree) {
    std::cout << "Running function2..." << std::endl;
    // cut mass by running for loop
    Long64_t nEntries = tree->GetEntries();
    TFile* tempFile = new TFile("temp.root", "RECREATE");
    TTree* cut_tree = tree->CloneTree(0);  // clone tree structure not entries
    Double_t D_M, delta_M;
    tree->SetBranchAddress("D_M", &D_M);
    tree->SetBranchAddress("delta_M", &delta_M);

    std::cout << "Filtering the tree in function2..." << std::endl;
    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        if (D_M > 1810 && D_M < 1920 && delta_M > 139.5 && delta_M < 152) {
            cut_tree->Fill();
        }
    }
    // cut_tree->Write();
    // tempFile->Close();
    std::cout << "Filtered the tree in function2" << std::endl;
    std::cout << "Cleaning up memory in function2..." << std::endl;
    delete cut_tree;
    delete tempFile;
    if (!remove("temp.root")) {
        std::cerr << "Error: Failed to delete temp.root" << std::endl;
    }
    std::cout << "Cleaned up memory in function2" << std::endl;
}

// Function to measure execution time with arguments
template <typename Func, typename... Args>
double measureTime(Func func, int iterations, Args&&... args) {
    std::vector<double> times;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func(std::forward<Args>(args)...);  // Call the function with arguments
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
        times.push_back(duration.count());
    }

    // Calculate average execution time
    double average_time = std::accumulate(times.begin(), times.end(), 0.0) / iterations;
    return average_time;
}

void performance() {
    int iterations = 1;
    auto [file, tree] = tman::loadFileAndTree(config::full_root_file);

    double avg_time2 = measureTime(function2, iterations, tree);
    double avg_time1 = measureTime(function1, iterations, tree);

    std::cout << "Average execution time of function1: " << avg_time1 << " ms\n";
    std::cout << "Average execution time of function2: " << avg_time2 << " ms\n";

    // delete tree;
    // std::cout << "Tree deleted" << std::endl;
    file->Close();
    std::cout << "File closed" << std::endl;
}