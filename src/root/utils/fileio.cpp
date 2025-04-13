#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <sstream>
#include <string>

namespace fileio {
    std::vector<std::string> ls(const std::string& path);
    bool exists(const std::string& filename);
    std::vector<std::vector<double>> readCSV(const std::string& filename, bool column = true, bool index = true);
    std::vector<std::string> readCSVcolumns(const std::string& filename);
    std::vector<std::string> readCSVindex(const std::string& filename);
    void deleteFile(const std::string& filename);
    void mv(const std::string& src, const std::string& dest);
    void rmdir(const std::string& path);
    void mkdir(const std::string& path);
}

/**
 * @brief List files in a directory similar to the `ls` command in Unix
 * 
 * @param path Directory path
 * @return std::vector<std::string> Vector of strings containing file names
 */
std::vector<std::string> fileio::ls(const std::string& path) {
    std::vector<std::string> files;
    try {
        // Check if the path exists and is a directory
        if (!std::filesystem::exists(path)) {
            std::cerr << path << " does not exist.\n";
            return files;
        }
        else if (!std::filesystem::is_directory(path)) {
            std::cerr << path << " is not a directory.\n";
            return files;
        }
        // Iterate over the directory contents
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            // Check if the entry is a file (not a directory)
            if (std::filesystem::is_regular_file(entry.status())) {
                files.push_back(entry.path().filename());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
    return files;
}

/**
 * @brief Check if a file exists
 * 
 * @param filename Path to the file
 * @return true if the file exists, false otherwise
 */
bool fileio::exists(const std::string& filename) {
    return std::filesystem::exists(filename);
}

/**
 * @brief Read CSV file and return a 2D vector of doubles
 * 
 * @param filename Path to the CSV file
 * @param column True if the CSV file has a column, omitting the first row
 * @param index True if the CSV file has an index, omitting the first column
 * @return std::vector<std::vector<double>> 2D vector of doubles where each vector represents a row
 */
std::vector<std::vector<double>> fileio::readCSV(const std::string& filename, bool column = true, bool index = true) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file '" + filename + "'");
    }
    std::string line;

    std::vector<std::vector<double>> data;
    
    // Skip header
    if (column) {
        std::getline(file, line);
    }
    while (std::getline(file, line)) {
        // for each line
        // std::vector<std::string> row;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
     
        // skip index
        if (index) {
            std::getline(ss, cell, ',');
        }
        // split by comma
        while (std::getline(ss, cell, ',')) {
            // row.push_back(cell);
            row.push_back(std::stod(cell));
        }

        data.push_back(row);
    }

    return data;
}

std::vector<std::string> fileio::readCSVcolumns(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file '" + filename + "'");
    }
    std::string line;

    std::vector<std::string> columns;
    std::getline(file, line);
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        columns.push_back(cell);
    }

    return columns;
}

std::vector<std::string> fileio::readCSVindex(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file '" + filename + "'");
    }
    std::string line;

    std::vector<std::string> index;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');
        index.push_back(cell);
    }

    return index;
}

void fileio::deleteFile(const std::string& filename) {
    // check if file exists
    if (!std::filesystem::exists(filename)) {
        std::cerr << "Error: " << filename << " does not exist." << std::endl;
        return;
    }
    // delete the file
    if (std::remove(filename.c_str()) != 0) {
        std::cerr << "Error: Failed to delete " << filename << std::endl;
    } else {
        std::cout << "Deleted " << filename << std::endl;
    }
}

// void fileio() {
//     auto data = fileio::readCSV("../../plots/5_sweighted_cpp/_samples/K_p_vs_eta_fine_binning_raw_yields.csv");
//     for (auto row : data) {
//         for (auto cell : row) {
//             std::cout << cell << " ";
//         }
//         std::cout << std::endl;
//     }
// }

void fileio::mv(const std::string& src, const std::string& dest) {
    // check if source file exists
    if (!std::filesystem::exists(src)) {
        std::cerr << "Error: " << src << " does not exist." << std::endl;
        return;
    }
    // if dest ends with "/", append the file name
    std::string new_dest = dest;
    if (dest.back() == '/') {
        std::filesystem::path src_path(src);
        std::filesystem::path dest_path(dest);
        dest_path /= src_path.filename();
        new_dest = dest_path.string();
    }
    // move the file
    try {
        std::filesystem::rename(src, new_dest);
        std::cout << "Moved " << src << " to " << new_dest << std::endl;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: Failed to move " << src << " to " << new_dest << ": " << e.what() << std::endl;
    }
}

void fileio::rmdir(const std::string& path) {
    // check if directory exists
    if (!std::filesystem::exists(path)) {
        std::cerr << "Error: " << path << " does not exist." << std::endl;
        return;
    }
    // remove the directory
    try {
        std::filesystem::remove_all(path);
        std::cout << "Removed directory " << path << std::endl;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: Failed to remove directory " << path << ": " << e.what() << std::endl;
    }
}

void fileio::mkdir(const std::string& path) {
    // check if directory exists
    if (std::filesystem::exists(path)) {
        std::cerr << "Error: " << path << " already exists." << std::endl;
        return;
    }
    // create the directory
    try {
        std::filesystem::create_directories(path);
        std::cout << "Created directory " << path << std::endl;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: Failed to create directory " << path << ": " << e.what() << std::endl;
    }
}
