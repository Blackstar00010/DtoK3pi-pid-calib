#include <string>
#include <vector>


// this file contains several useful string operations, usually inspired by Python stuff

namespace strops {
    bool strin(const std::string& str, const std::string& sub);

    std::string replace(const std::string& str, const std::string& old_sub, const std::string& new_sub);

    std::string randstr(int len);

    bool startswith(const std::string&str, const std::string& sub);
    bool endswith(const std::string& str, const std::string& sub);

    std::vector<std::string> split(const std::string& str, const std::string& delim);
}

/**
 * @brief Check if a string contains a substring
 * 
 * @param str The string to search
 * @param sub The substring to search for
 * @return true if the substring is found, false otherwise
 */
bool strops::strin(const std::string& str, const std::string& sub) {
    return str.find(sub) != std::string::npos;
}

/**
 * @brief Replace a substring in a string
 * 
 * @param str The string to search
 * @param old_sub The substring to search for
 * @param new_sub The substring to replace with
 * @return std::string The new string with the replaced substring
 */
std::string strops::replace(const std::string& str, const std::string& old_sub, const std::string& new_sub) {
    std::string new_str = str;
    size_t pos = 0;
    while ((pos = new_str.find(old_sub, pos)) != std::string::npos) {
        new_str.replace(pos, old_sub.length(), new_sub);
        pos += new_sub.length();
    }
    return new_str;
}


/**
 * @brief Generate a random string of a given length
 * 
 * @param len The length of the string to generate
 * @return std::string The generated random string
 */
std::string strops::randstr(int len) {
    std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
    std::string new_str;
    for (int i = 0; i < len; i++) {
        new_str += str[rand() % str.size()];
    }
    return new_str;
}

/**
 * @brief Check if a string starts with a substring
 * 
 * @param str The string to search
 * @param sub The substring to search for
 * @return true if the string starts with the substring, false otherwise
 */
bool strops::startswith(const std::string& str, const std::string& sub) {
    return str.find(sub) == 0;
}

/**
 * @brief Check if a string ends with a substring
 * 
 * @param str The string to search
 * @param sub The substring to search for
 * @return true if the string ends with the substring, false otherwise
 */
bool strops::endswith(const std::string& str, const std::string& sub) {
    return str.rfind(sub) == str.length() - sub.length();
}


/**
 * @brief Split a string by a delimiter
 * 
 * @param str The string to split
 * @param delim The delimiter to split by
 * @return std::vector<std::string> The vector of strings after splitting
 */
std::vector<std::string> strops::split(const std::string& str, const std::string& delim) {
    std::vector<std::string> tokens;
    size_t start = 0, end = 0;
    while ((end = str.find(delim, start)) != std::string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + delim.length();
    }
    tokens.push_back(str.substr(start));
    return tokens;
}
