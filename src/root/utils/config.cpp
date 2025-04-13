#include <string>
#include <filesystem> // C++17
#include <cmath> // for round
#include <time.h> // for time

namespace config{
    // Get the directory where this file is located
    const std::string utils_dir = std::filesystem::path(__FILE__).parent_path().string() + "/";
    const std::string proj_root = utils_dir + "../../../";
    const std::string src_dir = proj_root + "src/";
    const std::string root_dir = proj_root + "root/";
    const std::string data_dir = proj_root + "data/";
    const std::string plots_dir = proj_root + "plots/";

    const std::string temp_dir = data_dir + "temp/";
    std::string tempRootFile();

    std::string plot_dir3(const std::string& model_name, float cut = 0.5);
    std::string plot_dir4(int dimension, const std::string& model_name, float cut);
    std::string plot_dir5(const std::string& model_name, float cut = 0.5, const std::string& type = "");
    std::string plot_dir6 = plots_dir + "6_final/";
    
    // directories inside the data dir
    const std::string input_dir = data_dir + "input/";
    const std::string output_dir = data_dir + "output/";
    const std::string score_dir = data_dir + "score/";
    const std::string mc_dir = input_dir + "mc/";

    // input files
    const std::string long_root_file = input_dir + "long.root";
    const std::string short_root_file = input_dir + "short.root";
    const std::string wide_root_file = input_dir + "wide.root";
    const std::string long_core_file = input_dir + "long_core.root";
    std::string mc_file(int ratio);

    // score files
    const std::string long_score_file = score_dir + "proba_long.root";
    const std::string short_score_file = score_dir + "proba_short.root";
    const std::string tt_score_file = score_dir + "proba_tt.root";
    const std::string long_wscore_file = score_dir + "long_with_prob.root";
    const std::string short_wscore_file = score_dir + "short_with_prob.root";

    // output files
    const std::string long_sweight_file = output_dir + "sWeights_sorted.root";

    // models
    const std::string models_dir = proj_root + "models/";
    const std::string scalers_dir = models_dir + "scalers/";
    const std::string classifiers_dir = models_dir + "classifiers/";

    const std::string tree_name = "DecayTree";
}

/**
 * @brief Get the path to the temporary root file
 * 
 * @return std::string 
 */
std::string config::tempRootFile() {
    // return temp_dir + "temp.root";
    // add YYYYMMDD_HHMMSS to the filename
    time_t now = time(0);
    tm* ltm = localtime(&now);
    char buffer[80];
    strftime(buffer, 80, "%Y%m%d_%H%M%S", ltm);
    return temp_dir + "temp_" + std::string(buffer) + ".root";
}

std::string config::plot_dir3(const std::string& model_name, float cut) {
    return plots_dir + "3_nn_cpp/" + model_name + "/" + std::to_string(int(round(cut * 10)*10)) + "/";
}

std::string config::plot_dir4(int dimension, const std::string& model_name, float cut) {
    return plots_dir + "4_massfit_cpp/" + std::to_string(dimension) + "d/" + model_name + "/" + std::to_string(int(round(cut * 10)*10)) + "_";
}

std::string config::plot_dir5(const std::string& model_name, float cut, const std::string& type) {
    std::string ret;
    if (model_name == "_samples") {
        // return plots_dir + "5_sweighted_cpp/" + model_name + "/";
        ret = plots_dir + "5_sweighted_cpp/" + model_name + "/";
    }
    else if (type == "") {
        // return plots_dir + "5_sweighted_cpp/" + model_name + "/" + std::to_string(int(round(cut * 10)*10)) + "/";
        ret = plots_dir + "5_sweighted_cpp/" + model_name + "/" + std::to_string(int(round(cut * 10)*10)) + "/";
    }
    else {
        // return plots_dir + "5_sweighted_cpp/" + model_name + "/" + std::to_string(int(round(cut * 10)*10)) + "/" + type + "/";
        ret = plots_dir + "5_sweighted_cpp/" + model_name + "/" + std::to_string(int(round(cut * 10)*10)) + "/" + type + "/";
    }
    if (!std::filesystem::exists(ret)) {
        std::cout << "Creating directory: " << ret << std::endl;
        std::filesystem::create_directories(ret);
    }
    return ret;
}

std::string config::mc_file(int ratio) {
    if (ratio == 0) {
        return mc_dir + "mc.root";
    }
    if (ratio == -1 || ratio > 100) {
        return mc_dir + "mc_proced_all.root";
    }
    std::string ret = mc_dir + "mc_proced_" + std::to_string(ratio) + ".root";
    if (std::filesystem::exists(ret)) {
        return ret;
    }
    // if the file does not exist, return the one with all events
    std::cout << "File " << ret << " does not exist. Returning mc_proced_all.root" << std::endl;
    return mc_dir + "mc_proced_all.root";
}
