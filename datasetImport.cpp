#include "datasetImport.h"
#include <fstream>
#include <QDebug>

using std::string;
using std::vector;

DatasetImport::DatasetImport()
{

}

bool DatasetImport::loadFile(const std::string &filePath,
                             const std::string &valueSplitter)
{

    std::vector<std::string> lines;
    //lines.reserve(10000);
    std::ifstream file(filePath);
    if (!file.is_open()) {
        qDebug() << "Can't open file: \""<<filePath.c_str()<<"\"";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if(line.find("\n") != string::npos)
            line = line.substr(0, line.find("\n"));
        lines.push_back(line);
    }
    file.close();
    m_valueSplitter = valueSplitter;
    readData(lines);
    return true;
}
void DatasetImport::getLabels(std::vector<std::string> &inputLabels,
                              std::vector<std::string> &outputLabels)
{
    inputLabels = m_inputLabels;
    outputLabels = m_outputLabels;
}
void DatasetImport::getDataset(std::vector<std::vector<float> > &inputSet,
                               std::vector<std::vector<float> > &outputSet)
{
    inputSet = m_inputSet;
    outputSet = m_outputSet;
}


void DatasetImport::readData(const std::vector<std::string> &lines)
{
    // Line    Content
    //[0]      Input1,Input2,|Output1,Outpu2
    //[1]      1.5,1.2,0.5,0.6

    m_inputLabels.clear();
    m_outputLabels.clear();
    m_inputSet.clear();
    m_outputSet.clear();
    if(lines.size() < 2)
    {
        qDebug() << "The loaded file does not contain any datapoints";
        return;
    }

    //size_t inputCount = 0;
    //size_t outputCount = 0;
    size_t index = string::npos;
    bool readingInputs = true;
    string labelLine = lines[0];
    do
    {
        index = labelLine.find(m_valueSplitter);
        if(index != string::npos)
        {
            string label = labelLine.substr(0,index);
            size_t inOutSign = label.find("|");
            if(inOutSign != string::npos)
            {
                readingInputs = false;
                label = label.substr(1);
            }
            if(readingInputs)
                m_inputLabels.push_back(label);
            else
                m_outputLabels.push_back(label);

            labelLine = labelLine.substr(index+1);
        }
    }while(index != string::npos);
    if(!isEmpty(labelLine))
    {
        if(labelLine.find("|") != string::npos)
        {
            labelLine = labelLine.substr(1);
        }
        m_outputLabels.push_back(labelLine);
    }


    for(size_t i=1; i<lines.size(); ++i)
    {
        std::vector<float> inputDataLine;
        std::vector<float> outputDataLine;

        readingInputs = true;
        size_t valueCounter = 0;
        string valueLine = lines[i];
        do
        {
            index = valueLine.find(m_valueSplitter);
            if(index != string::npos)
            {
                string value = valueLine.substr(0,index);
                if(valueCounter == m_inputLabels.size())
                {
                    readingInputs = false;
                }
                if(readingInputs)
                    inputDataLine.push_back(stof(value));
                else
                    outputDataLine.push_back(stof(value));
                ++valueCounter;

                valueLine = valueLine.substr(index+1);
            }
        }while(index != string::npos);
        if(!isEmpty(valueLine))
        {
            outputDataLine.push_back(stof(valueLine));
        }

        m_inputSet.push_back(inputDataLine);
        m_outputSet.push_back(outputDataLine);
    }
}
bool DatasetImport::isEmpty(const std::string &str)
{
    struct Range
    {
        char min;
        char max;
        Range(char min, char max)
        {
            this->min = min;
            this->max = max;
        }
    };
    //static vector<Range> validValues{Range((char)33,(char)255)};
    static Range validValuesRange((char)33,(char)255);


    for(size_t i=0; i<str.size(); ++i)
    {
        if(validValuesRange.min <= str[i]  && str[i] >= validValuesRange.max)
            return false;
    }
    return true;
}
