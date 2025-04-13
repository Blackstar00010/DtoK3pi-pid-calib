#include <TFile.h>
#include <TTree.h>
#include <string>
#include <vector>
#include <tuple>
#include <set>
#include <iostream>
#include <fstream>
#include <filesystem>  // C++17 filesystem library
namespace fs = std::filesystem;

namespace treeops {  // tree management
    TFile* loadFile(const std::string& filename, const std::string& mode = "READ");
    TTree* loadTree(TFile* file, const std::string& dirname = "DstToD0Pi_D0ToKPiPiPi", const std::string& treename = "DecayTree", bool setDir = false);
    TTree* loadTree(const std::string& filename, const std::string& treename = "DecayTree");
    TTree* loadTree(const std::string& filename, const std::string& dirname, const std::string& treename);
    std::tuple<TFile*, TTree*> loadFileAndTree(const std::string& filename, const std::string& mode = "READ");
    std::vector<std::string> getBranches(TTree* tree);
    std::tuple<TFile*, TTree*> applyMassCut(TTree* tree, bool in = true);
    std::set<std::string> GetUniqueBranchValues(TTree* tree, const std::string& branchName);
    std::string GetValue(TTree* tree, const std::string& branchName, Long64_t entry);
    void Print(TTree* tree, long numRows = 3, long numCols = 3, bool printStats = true);
}


/**
 * @brief Load a TFile from a `.root` file
 * 
 * @param filename Path to the `.root` file
 * @return TFile* Pointer to the TFile object
 */
TFile* treeops::loadFile(const std::string& filename, const std::string& mode) {
    TFile* file = nullptr;
    file = TFile::Open(filename.c_str(), mode.c_str());
    if (!file || file->IsZombie()) {
        std::cerr << "Error opening file " << filename << " !" << std::endl;
        // file->Close();
        return nullptr;
    }
    return file;
}

/**
 * @brief Load a TTree from a TFile
 * 
 * @param file Pointer to the TFile object
 * @param dirname Directory name. Default is "DstToD0Pi_D0ToKPiPiPi"
 * @param treename TTree name. Default is "DecayTree"
 * @param setDir Set the directory to 0. If true, the directory is set to 0 but the file is not closed. Default is false
 * @return TTree* Pointer to the TTree object
 */
TTree* treeops::loadTree(TFile* file, const std::string& dirname, const std::string& treename, bool setDir) {
    if (setDir) {
        std::cout << "WARNING: loadTree for large trees may not work as expected due to memory limitations! Use loadFileAndTree instead!" << std::endl;
    }
    if (!file) {
        std::cerr << "Error: TFile pointer is null!" << std::endl;
        return nullptr;
    }
    // if DstToD0Pi_D0ToKPiPiPi in the file, cd to it
    TDirectory* dir = nullptr;
    dir = (TDirectory*)file->Get(dirname.c_str());
    TTree *tree = nullptr;
    if (dir) {
        std::cout << "Directory " << dirname << " found inside " << file->GetName() << std::endl;
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

    // Set the directory to 0 so that we can close the file
    if (setDir) {
        tree->SetDirectory(0);
    }
    return tree;
}

/**
 * @brief Load a TFile and TTree from a `.root` file
 * 
 * @param filename Path to the `.root` file
 * @param mode Mode to open the file. Default is "READ"
 * @return std::tuple<TFile*, TTree*> Tuple containing TFile* and TTree* pointers
 */
std::tuple<TFile*, TTree*> treeops::loadFileAndTree(const std::string& filename, const std::string& mode) {
    TFile* file = loadFile(filename, mode);
    if (!file) {
        std::cout << "Failed to load file " << filename << std::endl;
        return {nullptr, nullptr};
    }
    TTree* tree = loadTree(file);
    return {file, tree};
}

/**
 * @brief Load a TTree from a `.root` file. The tree is detached from the file.
 * 
 * @param filename Path to the `.root` file
 * @param treename Name of the TTree. Default is "DecayTree"
 * @return TTree* Pointer to the TTree object
 */
TTree* treeops::loadTree(const std::string& filename, const std::string& treename) {
    std::cout << "WARNING: loadTree for large trees may not work as expected due to memory limitations! Use loadFileAndTree instead!" << std::endl;
    TFile* file = loadFile(filename);
    if (!file) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        return nullptr;
    }
    TTree* tree = dynamic_cast<TTree*>(file->Get(treename.c_str()));
    if (!tree) {
        std::cerr << "Error: cannot find tree " << treename << " in file " << filename << std::endl;
        return nullptr;
    }
    // tree->SetDirectory(0);  // this doesn't work
    gROOT->cd();
    tree = (TTree*)tree->CloneTree(-1);
    file->Close();
    return tree;
}

/**
 * @brief Load a TTree from a `.root` file. The tree is detached from the file.
 * 
 * @param filename Path to the `.root` file
 * @param dirname Directory name. Default is "DstToD0Pi_D0ToKPiPiPi"
 * @param treename Name of the TTree. Default is "DecayTree"
 * @return TTree* Pointer to the TTree object
 */
TTree* treeops::loadTree(const std::string& filename, const std::string& dirname, const std::string& treename) {
    std::cout << "WARNING: loadTree for large trees may not work as expected due to memory limitations! Use loadFileAndTree instead!" << std::endl;
    TFile* file = loadFile(filename);
    if (!file) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        return nullptr;
    }

    // this is to avoid overloading the same signatures as loadTree(const std::string& filename, const std::string& treename)
    std::string localDirName = dirname;
    if (dirname.empty()) {
        localDirName = "DstToD0Pi_D0ToKPiPiPi";
        std::cout << "Directory name not provided. Using default directory name " << localDirName << std::endl;
    }
    std::string localTreeName = treename;
    if (treename.empty()) {
        localTreeName = "DecayTree";
        std::cout << "Tree name not provided. Using default tree name " << localTreeName << std::endl;
    }

    TTree* tree = loadTree(file, localDirName, treename, true);
    if (!tree) {
        std::cerr << "Error: cannot find tree " << treename << " in file " << filename << std::endl;
        return nullptr;
    }
    tree->SetDirectory(0);
    file->Close();
    return tree;
}

/**
 * @brief Get the vector of branches in a TTree
 * 
 * @param tree Pointer to the TTree object
 * @return std::vector<std::string> Vector of strings containing branch names
 */
std::vector<std::string> treeops::getBranches(TTree* tree) {
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

/**
 * @brief Apply cuts on masses and save the cut tree to a new file
 * 
 * @param tree Pointer to the TTree object
 * @return std::tuple<TFile*, TTree*> Tuple containing TFile* and TTree* pointers
 */
std::tuple<TFile*, TTree*> treeops::applyMassCut(TTree* tree, bool in) {
    if (!tree) {
        std::cerr << "Error: TTree pointer is null!" << std::endl;
        return {nullptr, nullptr};
    }
    std::cout << "Applying cuts on masses..." << std::endl;
    std::string cut;
    if (in) {
        cut = "D_M > 1810 && D_M < 1920 && delta_M > 139.5 && delta_M < 152";
    }
    else {
        cut = "D_M < 1810 || D_M > 1920 || delta_M < 139.5 || delta_M > 152";
    }
    // check if the tree is in directory 0
    TFile* tempFile = nullptr;
    TTree* cutTree = nullptr;
    if (tree->GetDirectory() != 0) {
        // make temp.root in the current directory and save the cut tree inside it
        tempFile = new TFile("temp.root", "RECREATE");
        cutTree = tree->CopyTree(cut.c_str());
        std::cout << "Applied cuts on masses and saved to temp.root" << std::endl;
    }
    else {
        // if the tree is in directory 0, we can save the cut tree in memory
        cutTree = tree->CopyTree(cut.c_str());
        std::cout << "Applied cuts on masses" << std::endl;
    }
    if (tempFile) {
        std::cout
        << "To remove the temp.root file, run the following command:\n"
        << "    $ rm temp.root\n"
        << "or use the following C++ code:\n"
        << "    if (std::remove(\"temp.root\") != 0) {\n"
        << "        std::cerr << \"Error: Failed to delete temp.root\" << std::endl;\n"
        << "    }\n"
        << "or use the following function:\n"
        << "    fileio::deleteFile(\"temp.root\");\n";
    }
    return std::make_tuple(tempFile, cutTree);
}

std::set<std::string> treeops::GetUniqueBranchValues(TTree* tree, const std::string& branchName) {
    std::set<std::string> uniqueValues;

    // Try to guess the type: first try string
    Char_t* strVal = nullptr;
    bool isString = tree->SetBranchAddress(branchName.c_str(), &strVal) == 0;
    std::cout << "isString: " << isString << std::endl;

    double numVal = 0.0;
    if (!isString) {
        // Try numeric fallback
        if (tree->SetBranchAddress(branchName.c_str(), &numVal) != 0) {
            std::cerr << "Could not bind branch: " << branchName << std::endl;
            return uniqueValues;
        }
        std::cout << "Set branch address for " << branchName << " to a numeric value" << std::endl;
    }

    Long64_t nEntries = tree->GetEntries();
    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        std::cout << "Entry " << i << " : " << strVal << std::endl;
        // if (isString) {
        //     if (strVal) uniqueValues.insert(strVal);
        // } else {
        //     uniqueValues.insert(std::to_string(numVal));
        // }
    }

    return uniqueValues;
}

/**
 * @brief Get the value of an entry in a branch of a TTree as a string
 * 
 * @param tree Pointer to the TTree object
 * @param branchName Name of the branch
 * @param entry Entry number
 * @return std::string Value of the branch as a string
 */
std::string treeops::GetValue(TTree* tree, const std::string& branchName, Long64_t entry) {
    if (!tree) {
        std::cerr << "Error: TTree pointer is null!" << std::endl;
        return "";
    }
    if (branchName.empty()) {
        std::cerr << "Error: Branch name is empty!" << std::endl;
        return "";
    }
    if (entry < 0 || entry >= tree->GetEntries()) {
        std::cerr << "Error: Entry number out of range!" << std::endl;
        return "";
    }
    // check type
    TBranch* branch = tree->GetBranch(branchName.c_str());
    if (!branch) {
        std::cerr << "Error: Branch " << branchName << " not found!" << std::endl;
        return "";
    }
    std::string nameWithType = branch->GetTitle();
    std::string type = nameWithType.substr(nameWithType.find_last_of('/') + 1);
    tree->ResetBranchAddresses();
    if (type == "F") {
        float value = 0.0;
        tree->SetBranchAddress(branchName.c_str(), &value);
        tree->GetEntry(entry);
        tree->ResetBranchAddresses();
        return std::to_string(value);
    }
    else if (type == "D") {
        double value = 0.0;
        tree->SetBranchAddress(branchName.c_str(), &value);
        tree->GetEntry(entry);
        tree->ResetBranchAddresses();
        return std::to_string(value);
    }
    else if (type == "I" || type == "i") {  // int
        int value = 0;
        tree->SetBranchAddress(branchName.c_str(), &value);
        tree->GetEntry(entry);
        tree->ResetBranchAddresses();
        return std::to_string(value);
    }
    else if (type == "L" || type == "l") {  // long
        Long64_t value = 0;
        tree->SetBranchAddress(branchName.c_str(), &value);
        tree->GetEntry(entry);
        tree->ResetBranchAddresses();
        return std::to_string(value);
    }
    else if (type == "C") {  // char*
        char* value = new char[100];  // allocate memory for the string
        tree->SetBranchAddress(branchName.c_str(), value);
        tree->GetEntry(entry);
        tree->ResetBranchAddresses();
        std::string result = value;
        delete[] value;  // free the memory allocated for the string
        return result;
    }
    else if (type == "O") {  // bool
        bool value = false;
        tree->SetBranchAddress(branchName.c_str(), &value);
        tree->GetEntry(entry);
        tree->ResetBranchAddresses();
        if (value) {
            return "true";
        }
        else {
            return "false";
        }
    }
    else {
        std::cerr << "Error: Branch " << branchName << " is not a float, double, int, long, char* or bool!" << std::endl;
        return "";
    }
}

/**
 * @brief Print the first and last few rows and columns of a TTree
 * 
 * @param tree Pointer to the TTree object
 * @param numRows Half of the number of rows to print. Default is 3
 * @param numCols Half of the number of columns to print. Default is 3
 * @param printStats Print the number of rows and columns. Default is true
 */
void treeops::Print(TTree* tree, long numRows, long numCols, bool printStats) {
    if (!tree) {
        std::cerr << "Error: TTree pointer is null!" << std::endl;
        return;
    }
    // std::cout << "TTree name: " << tree->GetName() << std::endl << std::endl;
    long nEntries = tree->GetEntries();
    auto branches = tree->GetListOfBranches();
    long nBranches = branches->GetEntries();

    if (nBranches == 0) {
        std::cerr << "Error: No branches found in the TTree!" << std::endl;
        return;
    }
    if (numRows < 0 || 2*numRows > nEntries) {
        numRows = nEntries / 2;
    }
    if (numCols < 0 || 2*numCols > nBranches) {
        numCols = nBranches / 2;
    }

    // column titles, first N rows, "..." and last N rows, so 2N+2
    std::vector<std::vector<std::string>> data(2*numRows+2, std::vector<std::string>(2*numCols+2, ""));

    // fill the dots 
    for (long i = 0; i < 2*numRows+2; ++i) {
        data[i][numCols+1] = "...";
    }
    for (long i = 0; i < 2*numCols+2; ++i) {
        data[numRows+1][i] = "...";
    }
    // fill the first row with the branch names
    for (long i = 0; i < numRows; ++i) {
        TBranch* firstIthBranch = (TBranch*)branches->At(i);
        if (firstIthBranch) {
            data[0][i+1] = firstIthBranch->GetName();
        }
        TBranch* lastIthBranch = (TBranch*)branches->At(nBranches - numRows + i);
        if (lastIthBranch) {
            data[0][numCols + i+2] = lastIthBranch->GetName();
        }
    }
    // fill the first column with the row numbers
    for (long i = 0; i < numRows; ++i) {
        data[i+1][0] = std::to_string(i);
        data[numRows+2+i][0] = std::to_string(nEntries - numRows + i);
    }
    // prepare to fill the data
    std::vector<std::string> stringValues(2*numCols, "");
    for (long i = 0; i < numCols; ++i) {
        TBranch* firstIthBranch = (TBranch*)branches->At(i);
        TBranch* lastIthBranch = (TBranch*)branches->At(nBranches - numCols + i);
        if (firstIthBranch) {
            data[0][i+1] = firstIthBranch->GetName();
        }
        if (lastIthBranch) {
            data[0][numCols + i+2] = lastIthBranch->GetName();
        }
    }
    // fill values
    for (long i = 0; i < numRows; ++i) {
        tree->GetEntry(i);
        for (long j = 0; j < numCols; ++j) {
            data[i+1][j+1] = GetValue(tree, data[0][j+1], i);
            data[i+1][numCols + j + 2] = GetValue(tree, data[0][numCols + j + 2], i);
        }
        tree->GetEntry(nEntries - numRows + i);
        for (long j = 0; j < numCols; ++j) {
            data[numRows+2+i][j+1] = GetValue(tree, data[0][j+1], nEntries - numRows + i);
            data[numRows+2+i][numCols + j + 2] = GetValue(tree, data[0][numCols + j + 2], nEntries - numRows + i);
        }
    }
    // add paddings to the data to make it look nice
    for (long i = 0; i < 2*numCols+2; ++i) {
        int maxLength = 0;
        for (long j = 0; j < 2*numRows+2; ++j) {
            int newLength = data[j][i].length();
            if (newLength > maxLength) {
                maxLength = newLength;
            }
        }
        for (long j = 0; j < 2*numRows+2; ++j) {
            int newLength = data[j][i].length();
            if (newLength < maxLength) {
                data[j][i] = std::string(maxLength - newLength, ' ') + data[j][i];
            }
        }
    }
    // print the data
    for (const auto& row : data) {
        for (const auto& col : row) {
            std::cout << col << "\t";
        }
        std::cout << std::endl;
    }
    if (printStats) {
        std::cout << "\n" << tree->GetName() << " [" << nEntries << " rows x " << nBranches << " columns]" << std::endl;
    }
    tree->ResetBranchAddresses();
}
