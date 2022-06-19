#ifndef CSVFILE_H
#define CSVFILE_H

#include <iostream>
#include <fstream>
#include <vector>

class csvfile;

inline static csvfile &endrow(csvfile &file);
inline static csvfile &flush(csvfile &file);

class csvfile
{
    std::ofstream fs_;
    const std::string separator_;

public:
    csvfile(const std::string filename, const std::string separator = ",")
        : fs_(), separator_(separator)
    {
        fs_.exceptions(std::ios::failbit | std::ios::badbit);
        fs_.open(filename);
    }

    ~csvfile()
    {
        flush();
        fs_.close();
    }

    void flush()
    {
        fs_.flush();
    }

    void endrow()
    {
        fs_ << std::endl;
    }

    csvfile &operator<<(csvfile &(*val)(csvfile &))
    {
        return val(*this);
    }

    csvfile &operator<<(const char *val)
    {
        fs_ << '"' << val << '"' << separator_;
        return *this;
    }

    csvfile &operator<<(const std::string &val)
    {
        fs_ << '"' << val << '"' << separator_;
        return *this;
    }

    template <typename T>
    csvfile &operator<<(const T &val)
    {
        fs_ << val << separator_;
        return *this;
    }
};

inline static csvfile &endrow(csvfile &file)
{
    file.endrow();
    return file;
}

inline static csvfile &flush(csvfile &file)
{
    file.flush();
    return file;
}

inline static void writeResults(csvfile &csv, int epochs, double lr, double accuracy, std::vector<std::pair<uint32_t, uint32_t>> digitAccuracyVec)
{
    csv << ""
        << "EPOCHS"
        << "LEARNING RATE"
        << "ACCURACY" << endrow;
    csv << "" << epochs << lr << 100 * accuracy << endrow << "" << endrow;

    csv << ""
        << "OCCURRED"
        << "CORRECT"
        << "INCORRECT"
        << "ACCURACY" << endrow;
    int counter = 0;
    for (auto &_pair : digitAccuracyVec)
    {

        csv << counter
            << _pair.first + _pair.second
            << _pair.first
            << _pair.second
            << std::to_string((int)(100 * (double)_pair.first / ((double)_pair.first + _pair.second))).append("%")
            << endrow;
        counter++;
    }
}
#endif