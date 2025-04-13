#include <TFile.h>
#include <TTree.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>  // C++17 filesystem library
namespace fs = std::filesystem;

// THIS FILE IS DEPRECATED

// namespace tman {  // tree management
//     TFile* load_file(const std::string& filename);
//     TTree* load_tree(TFile* file, const std::string& dirname = "DstToD0Pi_D0ToKPiPiPi", const std::string& treename = "DecayTree");
//     std::tuple<TFile*, TTree*> loadFileAndTree(const std::string& filename);
//     std::vector<std::string> getBranches(TTree* tree);
//     void DeleteTempRoot();
//     std::tuple<TFile*, TTree*> massCutFullTree(TTree* tree);
// }


/*
* Load a TFile from a `.root` file
* @param filename: path to the `.root` file
* @return TFile pointer if successful, nullptr otherwise
*/
TFile* tman::load_file(const std::string& filename) {
    TFile* file = nullptr;
    file = TFile::Open(filename.c_str());
    if (!file || file->IsZombie()) {
        std::cerr << "Error opening file " << filename << " !" << std::endl;
        file->Close();
        return nullptr;
    }
    return file;
}

/*
* Load a TTree from a TFile*
* @param file: TFile pointer
* @return TTree pointer if successful, nullptr otherwise
*/
TTree* tman::load_tree(TFile* file, const std::string& dirname, const std::string& treename) {
    if (!file) {
        std::cerr << "Error: TFile pointer is null!" << std::endl;
        return nullptr;
    }
    // if DstToD0Pi_D0ToKPiPiPi in the file, cd to it
    TDirectory* dir = nullptr;
    dir = (TDirectory*)file->Get(dirname.c_str());
    TTree *tree = nullptr;
    if (dir) {
        dir->cd();
        tree = (TTree*)dir->Get(treename.c_str());
    }
    else {
        tree = (TTree*)file->Get(treename.c_str());
    }

    // Get the TTree named "DecayTree" (can be checked by .ls command in root prompt)
    if (!tree) {
        std::cerr << "Error: TTree 'DecayTree' not found!" << std::endl;
        return nullptr;
    }
    return tree;
}

/*
* Load a TFile and TTree from a `.root` file
* @param filename: path to the `.root` file
* @return tuple containing TFile* and TTree* pointers
*/
std::tuple<TFile*, TTree*> tman::loadFileAndTree(const std::string& filename) {
    TFile* file = load_file(filename);
    if (!file) {
        std::cout << "Failed to load file " << filename << std::endl;
        return {nullptr, nullptr};
    }
    TTree* tree = load_tree(file);
    return {file, tree};
}


/*
* Get the vector of branches in a TTree
* @param tree: TTree pointer
* @return vector of strings containing branch names
*/
std::vector<std::string> tman::getBranches(TTree* tree) {
    if (!tree) {
        std::cerr << "Error: TTree pointer is null!" << std::endl;
        return {};
    }
    
    // Get the branch names
    std::vector<std::string> branches;
    TObjArray* branchList = tree->GetListOfBranches();
    for (int i = 0; i < branchList->GetEntries(); ++i) {
        TBranch* branch = (TBranch*)branchList->At(i);
        if (branch) {
            branches.push_back(branch->GetName());
        }
    }
    return branches;
}

/*
* Delete the temp.root file
*/
void tman::DeleteTempRoot() {
    if (std::remove("temp.root") != 0) {
        std::cerr << "Error: Failed to delete temp.root" << std::endl;
    } else {
        std::cout << "Deleted temp.root" << std::endl;
    }
}

/*
* Apply cuts on masses and save the cut tree to a new file
* @param tree: TTree pointer
* @return tuple containing TFile* and TTree* pointers
*/
std::tuple<TFile*, TTree*> tman::massCutFullTree(TTree* tree) {
    std::cout << "Applying cuts on masses..." << std::endl;
    // check if temp.root exists
    if (fs::exists("temp.root")) {
        std::cerr << "Error: temp.root already exists! Deleting..." << std::endl;
        tman::DeleteTempRoot();
    }
    TFile* tempFile = new TFile("temp.root", "RECREATE");
    TTree* cut_tree = tree->CopyTree("D_M > 1810 && D_M < 1920 && delta_M > 139.5 && delta_M < 152");
    std::cout << "Applied cuts on masses and saved to temp.root" << std::endl;
    std::cout << "To remove the temp.root file, run the following command:" << std::endl;
    std::cout << "    $ rm temp.root" << std::endl;
    std::cout << "or use the following C++ code:" << std::endl;
    std::cout << "    if (std::remove(\"temp.root\") != 0) {" << std::endl;
    std::cout << "        std::cerr << \"Error: Failed to delete temp.root\" << std::endl;" << std::endl;
    std::cout << "    }" << std::endl;
    return std::make_tuple(tempFile, cut_tree);
}