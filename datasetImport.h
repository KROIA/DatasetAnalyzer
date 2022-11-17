#pragma once

#include <string>
#include <vector>

class DatasetImport
{
public:
    DatasetImport();

    bool loadFile(const std::string &filePath,
                  const std::string &valueSplitter);
    void getLabels(std::vector<std::string> &inputLabels,
                   std::vector<std::string> &outputLabels);
    void getDataset(std::vector<std::vector<float> > &inputSet,
                    std::vector<std::vector<float> > &outputSet);
private:
    void readData(const std::vector<std::string> &lines);
    bool isEmpty(const std::string &str);

    std::string m_valueSplitter;

    std::vector<std::string> m_inputLabels;
    std::vector<std::string> m_outputLabels;

    std::vector<std::vector<float> > m_inputSet;
    std::vector<std::vector<float> > m_outputSet;
};
