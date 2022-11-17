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

    size_t maxX = 5;
    size_t maxY = 100;

    AI_agent *loaded = new AI_agent(filePath,valueSplitter, 0, 0);
    if(loaded->loadNet())
    {
        m_fittest.agent = loaded;
        m_fittest.error = 0;
        m_fitted = true;
        addChild(loaded);
        return;
    }
    else
        delete loaded;

    m_netDimPainter = new NeuronalNet::Graphics::PixelPainter();
    m_netDimPainter->setDimenstions(sf::Vector2u(maxX, maxY));
    m_netDimPainter->setDisplaySize(sf::Vector2f(2, 2));
    m_netDimPainter->setPos(sf::Vector2f(0,0));
    addComponent(m_netDimPainter);

    m_blinkTimer = new QTimer(this);
    connect(m_blinkTimer, &QTimer::timeout, this, &ModelFitter::onBlinkTimer);
    m_blinkTimer->setInterval(1000);

    for(size_t hiddenX = 1; hiddenX<=maxX; hiddenX++)
    {
        for(size_t hiddenY = 1; hiddenY<=maxY; hiddenY++)
        {
            AI_agent *agent = new AI_agent(filePath,valueSplitter,
                                           hiddenX, hiddenY);
            sf::Color col((char)Net::map(hiddenX, 0, maxX, 100,255),
                          (char)Net::map(hiddenY, 0, maxY, 100,255),
                          (char)0);
            agent->setChartColor(col);
            agent->displayNet(false);
            agent->displayChart(false);
            addChild(agent);
            m_agents.push_back(ModelData{.error=0, .agent=agent});
        }
    }

}
ModelFitter::~ModelFitter()
{

}

void ModelFitter::update()
{
    if(!m_trainingEnabled)
        return;
    if(m_fitted)
        trainModel();
    else
        fitModel();
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
