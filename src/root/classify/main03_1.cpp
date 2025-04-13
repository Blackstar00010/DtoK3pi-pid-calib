#include "../utils/treeops.cpp"
#include "../utils/config.cpp"
#include "../utils/strops.cpp"
#include "../utils/fileio.cpp"
#include <iostream>
#include <string>
#include <vector>
#include <TTree.h>
#include <TFile.h>

void main03_1() {
    TTree* tree = treeops::loadTree(config::long_root_file);
    if (!tree) {
        std::cerr << "Error: TTree pointer is null!" << std::endl;
        return;
    }
    auto [nullPointer, bkgTree] = treeops::applyMassCut(tree, false);
    TTree* sigTree = treeops::loadTree(config::mc_file(-1));
    if (!sigTree) {
        std::cerr << "Error: TTree pointer is null!" << std::endl;
        return;
    }

    std::vector<std::string> branches = fileio::readCSVcolumns(config::scalers_dir + "std.csv");
    // std::vector<std::string> branches = treeops::getBranches(bkgTree);
    // std::vector<std::string> selectedBranches;
    // for (const auto& branch : branches) {
    //     if (strops::strin(branch, "PID")) {
    //         continue;
    //     }
    //     if (strops::strin(branch, "PROBNN")) {
    //         continue;
    //     }
    //     selectedBranches.push_back(branch);
    // }

    // use tmva to classify the data

}