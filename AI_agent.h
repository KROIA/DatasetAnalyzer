#pragma once

#include "neuronalNet.h"
#include "netModel.h"
#include "QSFML_EditorWidget.h"

class AI_agent : public QSFML::Objects::CanvasObject
{
    public:
        struct TrainingData
        {
            std::vector<float> inputs;
            std::vector<float> outputs;

            std::string toString() const
            {
                std::string str = "Inp: ";
                for(size_t i=0; i<inputs.size(); ++i)
                {
                    str += std::to_string(inputs[i]);
                    if(i<inputs.size()-1)
                       str+= ", ";
                }
                str += "  Out: ";
                for(size_t i=0; i<outputs.size(); ++i)
                {
                    str += std::to_string(outputs[i]);
                    if(i<outputs.size()-1)
                        str+= ", ";
                }
                return str;
            }
        };
        struct ScalingData
        {
            float min;
            float max;
        };

        AI_agent(const std::string &filePath,
                 const std::string &valueSplitter,
                 size_t hiddenX, size_t hiddenY,
                 const std::string &name = "",
                 CanvasObject *parent = nullptr);
        ~AI_agent();
        void displayNet(bool enable);
        void displayChart(bool enable);
        void addErrorPoint(float err);
        void setChartPos(const sf::Vector2f &pos);
        void setChartColor(const sf::Color &col);

        void update() override;

        float train1(size_t iterations, size_t batchSize);
        float train(size_t batchSize = 1000);
        void validate();
        float getError() const;



        bool saveNet(const std::string &path = "save.net") const ;
        bool loadNet(const std::string &path = "save.net") const ;
        static NeuronalNet::BackpropNet *tryLoadNet(const std::string &path = "save.net");
        NeuronalNet::Net* getNet() const;

        std::vector<float> getNormalized(const std::vector<float> &natural,
                                         const std::vector<ScalingData> &scaling);
        std::vector<float> getNatural(const std::vector<float> &normalized,
                                      const std::vector<ScalingData> &scaling);
        const std::vector<ScalingData> &getInputScaling() const;
        const std::vector<ScalingData> &getOutputScaling() const;

        static const std::vector<std::string> &getInputLabels();
        static const std::vector<std::string> &getOutputLabels();

    private:
        void normalize(std::vector<TrainingData> &data);
        void splitSets(const std::vector<TrainingData>&set,
                       std::vector<TrainingData> &trainingSet,
                       std::vector<TrainingData> &testSet,
                       float trainingTestRatio = 0.8f,
                       size_t testBeginIndex = std::string::npos);


        float m_lastCalculatedError;
        QSFML::Objects::LineChart *m_chart;
        NeuronalNet::Graphics::NetModel *m_netModel;

        NeuronalNet::BackpropNet *m_net;
        static std::vector<std::string> m_inputLabels;
        static std::vector<std::string> m_outputLabels;

        float m_scalingBoundRadius;
        static std::vector<ScalingData> m_inputScaling;
        static std::vector<ScalingData> m_outputScaling;
        static std::vector<std::vector<float> > m_inputSet;
        static std::vector<std::vector<float> > m_outputSet;

        float m_iterationCounterScaled;
        //size_t m_iterationCounter;
        size_t m_trainingBatchIndex;
        static std::vector<TrainingData> m_dataset;
        static std::vector<TrainingData> m_trainingSet;
        static std::vector<TrainingData> m_testSet;
};
