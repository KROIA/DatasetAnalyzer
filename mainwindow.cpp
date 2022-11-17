#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "datasetImport.h"
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    m_net = nullptr;

    DatasetImport import;
    //if(import.loadFile("niddk_diabetes_dataset.csv",","));
    if(import.loadFile("kc_house_data.csv",";"));
    {
        import.getLabels(m_inputLabels, m_outputLabels);
        import.getDataset(m_inputSet, m_outputSet);

        if(m_inputSet.size() == 0 || m_outputSet.size() == 0)
            return;

        for(size_t i=0; i<m_inputSet.size(); ++i)
        {
            TrainingData data;
            data.inputs = m_inputSet[i];
            if(m_outputSet.size() <= i)
            {
                qDebug() << "Error, inputsets.size != outputSets.size";
                return;
            }
            data.outputs = m_outputSet[i];
            m_dataset.push_back(data);
        }


        normalize(m_dataset);
        splitSets(m_dataset,
                  m_trainingSet,
                  m_testSet,
                  0.9);

        m_net = new BackpropNet;
        m_net->setDimensions(m_inputSet[0].size(), 2, 20, m_outputSet[0].size());
        m_net->setLearnParameter(0.01);
        m_net->setActivation(Activation::sigmoid);
        m_net->build();

        train();
        validate();



    }

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::train()
{
    if(!m_net)
        return;
    float error = 0;
    size_t maxIt = 5000;
    size_t it = 0;
    float minDeltaErr = 0.001;
    float oldError;

    minDeltaErr = minDeltaErr*minDeltaErr;
    do{
        oldError = error;
        error = 0;
        for(size_t i=0; i<m_trainingSet.size(); ++i)
        {
            m_net->setInputVector(m_trainingSet[i].inputs);
            m_net->calculate();
            m_net->learn(m_trainingSet[i].outputs);
            error += m_net->getError().getRootMeanSquare();
        }
        error /= (float)m_trainingSet.size();

        if(it% 100 == 0)
        {
            qDebug() << "Error: "<<error;
            saveNet();
            validate();
        }

        ++it;
    }while(error > 0.001 && it<maxIt);
}
void MainWindow::validate()
{
    if(!m_net || m_testSet.size() == 0)
        return;

    float errors = 0;
    for(size_t a=0; a<m_testSet.size(); ++a)
    {
        m_net->setInputVector(m_testSet[a].inputs);
        m_net->calculate();
        std::string predicted;
        SignalVector out = m_net->getOutputVector();
        for(size_t i=0; i<out.size(); ++i)
        {
            predicted += std::to_string(out[i]);
            if(i<out.size()-1)
                predicted += ", ";
        }
        std::string correct;
        for(size_t i=0; i<m_testSet[a].outputs.size(); ++i)
        {
            correct += std::to_string(m_testSet[a].outputs[i]);
            if(i<m_testSet[i].outputs.size()-1)
                correct += ", ";
        }

        SignalVector err = m_net->getError(0,m_testSet[a].outputs);
        errors += err.getRootMeanSquare();
        qDebug() <<"ValidateError for: "<<a<< " = "<< err.getRootMeanSquare()
                 << " predicted: "<<predicted.c_str()
                 << " correct: "<<correct.c_str();
    }
    errors /= m_testSet.size();
    qDebug() << "Average error: "<<errors;
}

void MainWindow::normalize(std::vector<TrainingData> &data)
{
    if(data.size() <= 1)
        return;

    size_t inputs = data[0].inputs.size();
    size_t outputs = data[0].outputs.size();

    std::vector<float> inputMin(inputs,1000);
    std::vector<float> outputMin(outputs,1000);

    std::vector<float> inputMax(inputs,-1000);
    std::vector<float> outputMax(outputs,-1000);



    for(size_t a=0; a<data.size(); ++a)
    {
        for(size_t i=0; i<inputs; ++i)
        {
            if(data[a].inputs[i] < inputMin[i])
                inputMin[i] = data[a].inputs[i];
            if(data[a].inputs[i] > inputMax[i])
                inputMax[i] = data[a].inputs[i];
        }
        for(size_t i=0; i<outputs; ++i)
        {
            if(data[a].outputs[i] < outputMin[i])
                outputMin[i] = data[a].outputs[i];
            if(data[a].outputs[i] > outputMax[i])
                outputMax[i] = data[a].outputs[i];
        }
    }
    for(size_t a=0; a<data.size(); ++a)
    {
        for(size_t i=0; i<inputs; ++i)
        {
            data[a].inputs[i] = Net::map(data[a].inputs[i], inputMin[i], inputMax[i], -0.8, 0.8);
        }
        for(size_t i=0; i<outputs; ++i)
        {
            data[a].outputs[i] = Net::map(data[a].outputs[i], outputMin[i], outputMax[i], -0.8, 0.8);
        }
    }


    /*std::vector<float> inputOffset(inputs,0);
    std::vector<float> outputOffset(outputs,0);
    std::vector<float> inputFac(inputs,0);
    std::vector<float> outputFac(outputs,0);
    for(size_t a=0; a<data.size(); ++a)
    {
        for(size_t i=0; i<inputs; ++i)
        {
            inputOffset[i] += data[a].inputs[i];
        }
        for(size_t i=0; i<outputs; ++i)
        {
            outputOffset[i] += data[a].outputs[i];
        }
    }
    for(size_t i=0; i<inputs; ++i)
    {
        inputOffset[i] = -inputOffset[i]/(float)data.size();
    }
    for(size_t i=0; i<outputs; ++i)
    {
        outputOffset[i] = -outputOffset[i]/(float)data.size();
    }

    for(size_t a=0; a<data.size(); ++a)
    {
        for(size_t i=0; i<inputs; ++i)
        {
            float div = data[a].inputs[i] + inputOffset[i];
            inputFac[i] += div*div;
        }
        for(size_t i=0; i<outputs; ++i)
        {
            float div = data[a].outputs[i] + outputOffset[i];
            outputFac[i] += div*div;
        }
    }
    for(size_t i=0; i<inputs; ++i)
    {
        float standardDiv = sqrt(inputFac[i] / ((float)data.size()-1));
        if(standardDiv <= 0.0001)
            inputFac[i] = 10000;
        else
            inputFac[i] = 1 / standardDiv;
    }
    for(size_t i=0; i<outputs; ++i)
    {
        float standardDiv = sqrt(outputFac[i] / ((float)data.size()-1));
        if(standardDiv <= 0.0001)
            outputFac[i] = 10000;
        else
            outputFac[i] = 1/standardDiv;
    }

    for(size_t a=0; a<data.size(); ++a)
    {
        for(size_t i=0; i<inputs; ++i)
        {
            data[a].inputs[i] = inputFac[i] * (data[a].inputs[i] + inputOffset[i]);
        }
        for(size_t i=0; i<outputs; ++i)
        {
            data[a].outputs[i] = outputFac[i] * (data[a].outputs[i] + outputOffset[i]);
        }
    }*/
}
void MainWindow::splitSets(const std::vector<TrainingData>&set,
                           std::vector<TrainingData> &trainingSet,
                           std::vector<TrainingData> &testSet,
                           float trainingTestRatio,
                           size_t testBeginIndex)
{
    size_t trainingCount = (size_t)((float)set.size() * trainingTestRatio);
    size_t testCount = set.size() - trainingCount;
    if(testBeginIndex == std::string::npos)
    {
        testBeginIndex = trainingCount;
    }

    trainingSet.clear();
    trainingSet.reserve(trainingCount);
    testSet.clear();
    testSet.reserve(testCount);

    size_t index = 0;
    for(size_t i=0; i<trainingCount; ++i)
    {
        if(index == testBeginIndex)
            index += testCount;
        trainingSet.push_back(set[index]);
        ++index;
    }
    index = testBeginIndex;
    for(size_t i=0; i<testCount; ++i)
    {
        testSet.push_back(set[index]);
        ++index;
    }
}
void MainWindow::saveNet()
{
    NetSerializer serializer;
    serializer.setFilePath("save.net");
    serializer.saveToFile(m_net);
}
