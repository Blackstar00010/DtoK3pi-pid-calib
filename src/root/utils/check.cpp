#include <iostream>
#include <TTree.h>
#include <string>
#include <vector>


namespace check {
    bool checkTree(TTree* tree);
    bool checkBranch(TTree* tree, const std::string& branch);
    bool CheckTreeBranches(TTree* tree, std::vector<std::string> branchNames);
}


/**
 * @brief Check if a TTree is not null and has entries
 * 
 * @param tree TTree pointer
 * @return true if the TTree is not null and has entries, false if the TTree is null or has no entries
 */
bool check::checkTree(TTree* tree) {
    if (!tree) {
        std::cerr << "Error: TTree pointer is null!" << std::endl;
        return false;
    }
    if (tree->GetEntries() == 0) {
        std::cerr << "Error: TTree is empty!" << std::endl;
        return false;
    }
    return true;
}

/**
 * @brief Check if a branch exists in a TTree, assuming the TTree is not null
 * 
 * @param tree TTree pointer
 * @param branch branch name
 * @return true if the branch exists, false if the branch does not exist
 */
bool check::checkBranch(TTree* tree, const std::string& branch) {
    if (!tree->GetBranch(branch.c_str())) {
        std::cerr << "Error: Branch " << branch << " not found!" << std::endl;
        return false;
    }
    return true;
}

/**
 * @brief Check if a TTree is not null and has branches
 * 
 * @param tree TTree pointer
 * @param branchNames vector of branch names to check
 * @return true if the TTree is not null and has entries, false if the TTree is null or has no entries
 */
bool check::CheckTreeBranches(TTree* tree, std::vector<std::string> branchNames) {
    if (!check::checkTree(tree)) {
        return false;
    }
    for (const std::string& branchName : branchNames) {
        if (!check::checkBranch(tree, branchName)) {
            return false;
        }
    }
    return true;
}