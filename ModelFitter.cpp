#include "ModelFitter.h"
#include <utility>
#include <thread>
#include <chrono>

using namespace NeuronalNet;

ModelFitter::ModelFitter(const std::string &filePath,
                         const std::string &valueSplitter,
                         const std::string &name,
                         CanvasObject *parent)
    : CanvasObject(name, parent)
{
    m_fittest.agent = nullptr;
    m_fitted = false;
    m_fitIterations = 500;
    m_blinkToggle = false;
    m_trainingEnabled = true;
    m_datasetPlot_gridStepSize = 0.2;
    m_datasetPlotSpaceMin = sf::Vector2f(0,0);
    m_datasetPlotSpaceMax = sf::Vector2f( 20, 20);

    size_t maxX = 3;
    size_t maxY = 20;

    AI_agent *agent = nullptr;
    /*agent = new AI_agent(filePath,valueSplitter, 0, 0);
    if(agent->loadNet())
    {
        m_fittest.agent = agent;
        m_fittest.error = 0;
        m_fitted = true;
        addChild(agent);
        setupDatasetPlot();
        return;
    }
    else
        delete agent;*/

    m_netDimPainter = new NeuronalNet::Graphics::PixelPainter();
    m_netDimPainter->setDimenstions(sf::Vector2u(maxX+1, maxY+1));
    m_netDimPainter->setDisplaySize(sf::Vector2f(2, 2));
    m_netDimPainter->setPos(sf::Vector2f(0,0));
    addComponent(m_netDimPainter);

    m_blinkTimer = new QTimer(this);
    connect(m_blinkTimer, &QTimer::timeout, this, &ModelFitter::onBlinkTimer);
    m_blinkTimer->setInterval(1000);

    for(size_t hiddenX = 0; hiddenX<=maxX; hiddenX++)
    {

        for(size_t hiddenY = 2; hiddenY<=maxY; hiddenY++)
        {
            agent = new AI_agent(filePath,valueSplitter,
                                           hiddenX, hiddenY);
            sf::Color col((char)Net::map(hiddenX, 0, maxX, 100,255),
                          (char)Net::map(hiddenY, 0, maxY, 100,255),
                          (char)0);
            agent->setChartColor(col);
            agent->displayNet(false);
            agent->displayChart(false);
            addChild(agent);
            m_agents.push_back(ModelData{.error=0, .agent=agent});
            if(hiddenX == 0)
                break;
        }
    }


}
ModelFitter::~ModelFitter()
{

}
void ModelFitter::setupDatasetPlot()
{
    AI_agent *agent = m_fittest.agent;
    if(!agent && m_agents.size() > 0)
        agent = m_agents[0].agent;
    if(!agent)
        return;
    if(agent->getNet()->getInputCount() == 2)
    {
        sf::Vector2f pos(50,0);
        for(size_t i=0; i<agent->getNet()->getOutputCount(); ++i)
        {
            NeuronalNet::Graphics::PixelPainter *plot= new NeuronalNet::Graphics::PixelPainter();

            unsigned int width = (unsigned int)(m_datasetPlotSpaceMax.x-m_datasetPlotSpaceMin.x)/m_datasetPlot_gridStepSize;
            unsigned int height = (unsigned int)(m_datasetPlotSpaceMax.y-m_datasetPlotSpaceMin.y)/m_datasetPlot_gridStepSize;
            plot->setDimenstions(sf::Vector2u(width,height));
            float pixelSize = 200/(float)height;
            plot->setDisplaySize(pixelSize, pixelSize);
            plot->setPos(pos);
            pos.y += pixelSize*(float)height + 10;
            m_datasetPlots.push_back(plot);
            addComponent(plot);
        }

    }
}


void ModelFitter::update()
{
    if(!m_trainingEnabled)
        return;
    if(m_fitted)
        trainModel();
    else
        fitModel();

    static int updateCount = 0;
    if(++updateCount % 10 == 0)
        updateDatasetPlot();
}
std::vector<std::string> ModelFitter::getInputLabels() const
{
    return AI_agent::getInputLabels();
}
std::vector<std::string> ModelFitter::getOutputLabels() const
{
    return AI_agent::getOutputLabels();
}
std::vector<float> ModelFitter::calculateInNet(const std::vector<float> &inputs)
{
    if(!m_fittest.agent)
        return {};
    Net *net = m_fittest.agent->getNet();
    std::vector<float> normalized = m_fittest.agent->getNormalized(inputs, m_fittest.agent->getInputScaling());
    net->setInputVector(normalized);
    net->calculate();
    SignalVector outSignals = net->getOutputVector();
    std::vector<float> out(outSignals.begin(), outSignals.end());
    return m_fittest.agent->getNatural(out, m_fittest.agent->getOutputScaling());
}
void ModelFitter::enableTraining(bool enable)
{
    m_trainingEnabled = enable;
}
bool ModelFitter::trainingIsEnabled() const
{
    return m_trainingEnabled;
}
void ModelFitter::onBlinkTimer()
{
    if(!m_fittest.agent)
        return;
    m_blinkToggle = !m_blinkToggle;
    if(m_blinkToggle)
    {
        m_netDimPainter->setPixel(sf::Vector2u(m_fittest.agent->getNet()->getHiddenXCount(),
                                               m_fittest.agent->getNet()->getHiddenYCount()),
                                  sf::Color(100,100,255));
    }
    else
    {
        m_netDimPainter->setPixel(sf::Vector2u(m_fittest.agent->getNet()->getHiddenXCount(),
                                               m_fittest.agent->getNet()->getHiddenYCount()),
                                  m_blinkDefColor);
    }
}

void ModelFitter::fitModel()
{
    static size_t it = 0;
    size_t threadCount = 12;
    size_t count = 1 + m_agents.size()/threadCount;

    std::vector<std::thread*> threads(threadCount, nullptr);
    size_t currentStart = 0;
    for(size_t i=0; i<threadCount; ++i)
    {
       threads[i] = new std::thread(&ModelFitter::fitModelThreaded, this ,currentStart, count);
       currentStart += count;
    }


    for(size_t i=0; i<threadCount; ++i)
    {
        threads[i]->join();
        delete threads[i];

    }
    updateNetDimPainter();
    ++it;
    if(it >= m_fitIterations)
    {

        evaluateBestModel();
    }
}
void ModelFitter::fitModelThreaded(size_t begin, size_t count)
{
    size_t s = m_agents.size();
    if(begin >= s)
        return;
    if(begin+count > s)
        count = s-begin;
    float filterVal = 0.3;
    for(size_t i=begin; i<begin+count; ++i)
    {
        m_agents[i].agent->train1(1,5);
        size_t hx = m_agents[i].agent->getNet()->getHiddenXCount();
        size_t hy = m_agents[i].agent->getNet()->getHiddenYCount();
        size_t hCount = hx * hy;
        float configFactor = 1;
        if(hx > 0 && hy > 0)
            configFactor = (float) hCount*0.1;
        m_agents[i].error = filterVal * (m_agents[i].agent->getError()*configFactor) + (1-filterVal)*(m_agents[i].error);
        //qDebug() << "calc: "<<i;
    }
}
void ModelFitter::evaluateBestModel()
{
    if(m_agents.size() == 0)
        return;
    m_fittest = m_agents[0];
    for(size_t i=0; i<m_agents.size(); ++i)
    {
        if(m_fittest.error > m_agents[i].error)
        {
            m_fittest = m_agents[i];
        }
    }
    setupDatasetPlot();
    qDebug() << "The best fit has:"
             << " hiddenX = "<<m_fittest.agent->getNet()->getHiddenXCount()
             << " hiddenY = "<<m_fittest.agent->getNet()->getHiddenYCount();

    m_fittest.agent->setChartPos(sf::Vector2f(0,450));
    m_fittest.agent->displayNet(true);
    m_fittest.agent->displayChart(true);
    m_fitted = true;

    m_blinkDefColor = m_netDimPainter->getPixel(sf::Vector2u(m_fittest.agent->getNet()->getHiddenXCount(),
                                                             m_fittest.agent->getNet()->getHiddenYCount()));

    m_blinkTimer->start();
}
void ModelFitter::trainModel()
{
    static size_t it = 0;
    static float iterationCounterScaled = 1;
    if(!m_fittest.agent)
        return;
    m_fittest.error = m_fittest.agent->train();
    if(it% (int)iterationCounterScaled == 0)
        m_fittest.agent->addErrorPoint(m_fittest.error);
    ++it;
    iterationCounterScaled+= 0.01;

    if(it% 100 == 0)
    {
        qDebug() << "Error: "<<m_fittest.error;
        m_fittest.agent->saveNet();
        m_fittest.agent->validate();
    }
}
void ModelFitter::getMinMaxErrors(float &min, float &average, float &max)
{
    if(m_agents.size() == 0)
        return;
    min = 99999;
    max = -min;
    float av = 0;
    for(size_t i=0; i<m_agents.size(); ++i)
    {
        if(m_agents[i].error < min)
            min = m_agents[i].error;
        else if(m_agents[i].error > max)
            max = m_agents[i].error;
        av += m_agents[i].error;
    }
    average = av/(float)m_agents.size();
}
void ModelFitter::updateNetDimPainter()
{
    float min, average, max;
    getMinMaxErrors(min, average, max);

    for(size_t i=0; i<m_agents.size(); ++i)
    {
        ModelData agent = m_agents[i];
        float colorVal = Net::map(agent.error, min, max, 255, 0);
        sf::Color col(0,colorVal,0);
        m_netDimPainter->setPixel(sf::Vector2u(agent.agent->getNet()->getHiddenXCount(),
                                               agent.agent->getNet()->getHiddenYCount()),
                                  col);
    }
}
void ModelFitter::updateDatasetPlot()
{
    if(m_datasetPlots.size() == 0)
        return;
    if(!m_fittest.agent)
        return;
    NeuronalNet::Net *net = m_fittest.agent->getNet();

    sf::Vector2u pixelCoord(0,0);
    for(float x=m_datasetPlotSpaceMin.x; x<m_datasetPlotSpaceMax.x; x+=m_datasetPlot_gridStepSize)
    {
        for(float y=m_datasetPlotSpaceMin.y; y<m_datasetPlotSpaceMax.y; y+=m_datasetPlot_gridStepSize)
        {
            net->setInputVector(m_fittest.agent->getNormalized(std::vector<float>{x,y}, m_fittest.agent->getInputScaling()));
            net->calculate();
            SignalVector output = net->getOutputVector();
            std::vector<float> naturalOutput = m_fittest.agent->getNatural(std::vector<float>(output.begin(), output.end()), m_fittest.agent->getOutputScaling());
            for(size_t i=0; i<net->getOutputCount(); ++i)
            {
                m_datasetPlots[i]->setPixel(sf::Vector2u(pixelCoord.x,m_datasetPlots[i]->getDimensions().y-pixelCoord.y-1),
                                            mapOutputToColor(naturalOutput[i]));
              // m_datasetPlots[i]->setPixel(pixelCoord,
              //                             mapOutputToColor(output[i]));
            }
            ++pixelCoord.y;
        }
        ++pixelCoord.x;
        pixelCoord.y = 0;
    }
}
sf::Color ModelFitter::mapOutputToColor(float value)
{
    value -= 0.5;
    if(value < 0)
    {
        float maped = NeuronalNet::Net::map(value,-0.6,0,0,255);
        if(maped < 0)
            maped = 0;
        if(maped > 255)
            maped = 255;
        return sf::Color(255, maped, maped);
    }
    float maped = NeuronalNet::Net::map(value,0,0.6,255,0);
    if(maped < 0)
        maped = 0;
    if(maped > 255)
        maped = 255;
    return sf::Color(maped, 255, maped);
}
