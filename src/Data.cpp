#include "Data.hpp"

Data::Data()
{
}

void Data::readFromFile(const std::string filename)
{
    //std::cout << "ELko\n";
    std::ifstream fileData(filename);
    std::string line;
    //std::cout << "ELko\n";
    int counter = 0;
    while (std::getline(fileData, line))
    {
        int index = 0;
        Eigen::VectorXd image(28 * 28);

        std::stringstream lineStream(line);
        std::string value;

        while (std::getline(lineStream, value, ','))
        {
            if (index == 0)
            {
                this->expectedDigit.push_back(stoi(value));
                // std::cout << "Value nr " << counter << " " << value << "\n";
                counter++;
            }
            else
            {
                double greyScaleValue = stod(value) / 255.0;
                image[index - 1] = greyScaleValue;
            }
            index += 1;
        }
        this->images.push_back(image);
    }

    //fileData.close();
    // std::cout << "Read" << this->expectedDigit.size() << "\n";
}

Data::~Data()
{
}