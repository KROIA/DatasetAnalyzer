#include "AI_agent.h"
#include "datasetImport.h"
#include <QDebug>

using namespace NeuronalNet;

std::vector<std::string> AI_agent::m_inputLabels;
std::vector<std::string> AI_agent::m_outputLabels;

std::vector<AI_agent::ScalingData> AI_agent::m_inputScaling;
std::vector<AI_agent::ScalingData> AI_agent::m_outputScaling;
std::vector<std::vector<float> > AI_agent::m_inputSet;
std::vector<std::vector<float> > AI_agent::m_outputSet;
std::vector<AI_agent::TrainingData> AI_agent::AI_agent::m_dataset;
std::vector<AI_agent::TrainingData> AI_agent::AI_agent::m_trainingSet;
std::vector<AI_agent::TrainingData> AI_agent::AI_agent::m_testSet;

AI_agent::AI_agent(const std::string &filePath,
                   const std::string &valueSplitter,
                   size_t hiddenX, size_t hiddenY,
                   const std::string &name,
                   CanvasObject *parent)
    : QSFML::Objects::CanvasObject(name, parent)
{
    m_net = nullptr;
    m_scalingBoundRadius = 0.8;
    m_trainingBatchIndex = 0;
    m_iterationCounterScaled = 1;
    m_lastCalculatedError = 0;

    m_chart = new QSFML::Objects::LineChart("ErrorChart");
    m_chart->setOrigin(sf::Vector2f(0,300));
    m_chart->setSize(sf::Vector2f(250,200));
    m_chart->setColor(sf::Color((rand()%155)+100, (rand()%155)+100, (rand()%155)+100));
    addChild(m_chart);



    if(m_dataset.size() == 0)
    {
        DatasetImport import;
        //if(import.loadFile("niddk_diabetes_dataset.csv",","));
        if(import.loadFile(filePath,valueSplitter))
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



            // train();
            // validate();



        }
    }

    m_net = new BackpropNet;
    m_net->setDimensions(m_inputSet[0].size(), hiddenX, hiddenY, m_outputSet[0].size());
    m_net->setLearnParameter(0.01);
    m_net->setActivation(Activation::gauss);
    m_net->build();

    m_netModel = new  NeuronalNet::Graphics::NetModel(m_net);
    m_netModel->setPos(sf::Vector2f(300,300));
    m_netModel->setNeuronSize(5);
    m_netModel->setNeuronSpacing(sf::Vector2f(100,3));
    m_netModel->setSignalWidth(0.6);
    m_netModel->setConnectionWidth(1.2);
    m_netModel->setVisualConfiguration(NeuronalNet::Graphics::NeuronPainter::getStandardVisualConfiguration() |
                                       NeuronalNet::Graphics::ConnectionPainter::getStandardVisualConfiguration());
    addChild(m_netModel);
}
AI_agent::~AI_agent()
{
    delete m_net;
}
void AI_agent::displayNet(bool enable)
{
    m_netModel->setEnabled(enable);
}
void AI_agent::displayChart(bool enable)
{
    m_chart->setEnabled(enable);
}
void AI_agent::addErrorPoint(float err)
{
    m_chart->addDataPoint(err);
}
void AI_agent::setChartPos(const sf::Vector2f &pos)
{
    m_chart->setOrigin(pos);
}
void AI_agent::setChartColor(const sf::Color &col)
{
    m_chart->setColor(col);
}

void AI_agent::update()
{
   /* if(!m_net)
        return;
    float error = 0;
    size_t maxIt = 5000;
    if(m_iterationCounter = 0;<maxIt)
    {
        error = train();
        if(m_iterationCounter = 0;% (int)m_iterationCounterScaled == 0)
            m_chart->addDataPoint(error);
        ++m_iterationCounter = 0;;
        m_iterationCounterScaled+= 0.01;

        if(m_iterationCounter = 0;% 100 == 0)
        {
            qDebug() << "Error: "<<error;
            saveNet();
            validate();
        }
    }
    m_net->setInputVector(m_trainingSet[0].inputs);
    m_net->calculate();*/
}
float AI_agent::train1(size_t iterations, size_t batchSize)
{
    if(iterations == 0)
        return 0;
    float error = 0;
    for(size_t i=0; i<iterations; ++i)
        error += train(batchSize);
    error /= (float)iterations;
    m_lastCalculatedError = error;
    return error;
}
float AI_agent::train(size_t batchSize)
{
    if(!m_net)
        return 0;
    float error = 0;
    for(size_t i=m_trainingBatchIndex; i<m_trainingBatchIndex + batchSize; ++i)
    {
        if(m_trainingSet.size() <= i)
            break;
        m_net->setInputVector(m_trainingSet[i].inputs);
        m_net->calculate();
        m_net->learn(m_trainingSet[i].outputs);
        error += m_net->getError().getRootMeanSquare();
    }
    m_trainingBatchIndex += batchSize;
    if(m_trainingBatchIndex >= m_trainingSet.size())
        m_trainingBatchIndex = 0;
    error /= (float)batchSize;

    m_lastCalculatedError = error;
    return error;

}
void AI_agent::validate()
{
    if(!m_net || m_testSet.size() == 0)
        return;

    float errors = 0;
    float naturalError = 0;
    for(size_t a=0; a<m_testSet.size(); ++a)
    {
        m_net->setInputVector(m_testSet[a].inputs);
        m_net->calculate();
        std::string predicted;
        SignalVector out = m_net->getOutputVector();
        std::vector<float> outVec(out.begin(), out.end());
        std::vector<float> natualOut = getNatural(outVec, m_outputScaling);
        for(size_t i=0; i<natualOut.size(); ++i)
        {
            predicted += std::to_string(natualOut[i]);
            if(i<out.size()-1)
                predicted += ", ";
        }
        std::string correct;
        std::vector<float> natualCorrectOut = getNatural(m_testSet[a].outputs, m_outputScaling);
        for(size_t i=0; i<natualCorrectOut.size(); ++i)
        {
            correct += std::to_string(natualCorrectOut[i]);
            if(i<m_testSet[i].outputs.size()-1)
                correct += ", ";
        }

        SignalVector err = m_net->getError(0,m_testSet[a].outputs);
        std::vector<float> naturalOut(err.begin(), err.end());
        naturalOut = getNatural(naturalOut, m_outputScaling);
        SignalVector naturalOutSignals(naturalOut);
        float naturalRootMeanSquare = naturalOutSignals.getRootMeanSquare();
        float rootMeanSquare = err.getRootMeanSquare();
        errors += rootMeanSquare;
        naturalError += naturalRootMeanSquare;


        qDebug() <<"ValidateError for: "<<a<< " = "<< rootMeanSquare
                 << " NarutalError: "<<naturalRootMeanSquare
                 << " predicted: "<<predicted.c_str()
                 << " correct: "<<correct.c_str();
    }
    errors /= m_testSet.size();
    naturalError /= m_testSet.size();
    m_lastCalculatedError = errors;
    qDebug() << "Average error: "<<errors << " Natural average error: "<<naturalError;
}
float AI_agent::getError() const
{
    return m_lastCalculatedError;
}

void AI_agent::normalize(std::vector<TrainingData> &data)
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

    m_inputScaling.clear();
    m_outputScaling.clear();
    m_inputScaling.reserve(inputs);
    m_outputScaling.reserve(outputs);
    for(size_t i=0; i<inputs; ++i)
    {
        m_inputScaling.push_back(ScalingData{.min=inputMin[i], .max=inputMax[i]});
    }
    for(size_t i=0; i<outputs; ++i)
    {
        m_outputScaling.push_back(ScalingData{.min=outputMin[i], .max=outputMax[i]});
    }
    for(size_t a=0; a<data.size(); ++a)
    {
        for(size_t i=0; i<inputs; ++i)
        {
            data[a].inputs[i] = Net::map(data[a].inputs[i], inputMin[i], inputMax[i], -m_scalingBoundRadius, m_scalingBoundRadius);
        }
        for(size_t i=0; i<outputs; ++i)
        {
            data[a].outputs[i] = Net::map(data[a].outputs[i], outputMin[i], outputMax[i], -m_scalingBoundRadius, m_scalingBoundRadius);
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

std::vector<float> AI_agent::getNormalized(const std::vector<float> &natural,
                                           const std::vector<ScalingData> &scaling)
{
    std::vector<float> data(natural.size(), 0);
    for(size_t i=0; i<natural.size(); ++i)
    {
        if(i >= scaling.size())
            break;
        data[i] = Net::map(natural[i], scaling[i].min, scaling[i].max, -m_scalingBoundRadius, m_scalingBoundRadius);
    }
    return data;
}
std::vector<float> AI_agent::getNatural(const std::vector<float> &normalized,
                                        const std::vector<ScalingData> &scaling)
{
    std::vector<float> data(normalized.size(), 0);
    for(size_t i=0; i<normalized.size(); ++i)
    {
        if(i >= scaling.size())
            break;
        data[i] = Net::map(normalized[i], -m_scalingBoundRadius, m_scalingBoundRadius, scaling[i].min, scaling[i].max);
    }
    return data;
}
const std::vector<AI_agent::ScalingData> &AI_agent::getInputScaling() const
{
    return m_inputScaling;
}
const std::vector<AI_agent::ScalingData> &AI_agent::getOutputScaling() const
{
    return m_outputScaling;
}
const std::vector<std::string> &AI_agent::getInputLabels()
{
    return m_inputLabels;
}
const std::vector<std::string> &AI_agent::getOutputLabels()
{
    return m_outputLabels;
}

void AI_agent::splitSets(const std::vector<TrainingData>&set,
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
bool AI_agent::saveNet(const std::string &path) const
{
    NetSerializer serializer;
    serializer.setFilePath(path);
    return serializer.saveToFile(m_net);
}
bool AI_agent::loadNet(const std::string &path) const
{
    NetSerializer serializer;
    serializer.setFilePath(path);
    return serializer.readFromFile(m_net);
}
NeuronalNet::BackpropNet *AI_agent::tryLoadNet(const std::string &path)
{
    BackpropNet *net = new BackpropNet();
    NetSerializer ser;
    ser.setFilePath(path);
    if(ser.readFromFile(net))
        return net;
    delete net;
    return nullptr;
}
Net* AI_agent::getNet() const
{
    return m_net;
}
