#pragma once

#include "AI_agent.h"

class ModelFitter : public QObject, public QSFML::Objects::CanvasObject
{
        Q_OBJECT
    public:
        ModelFitter(const std::string &filePath,
                    const std::string &valueSplitter,
                    const std::string &name = "",
                    CanvasObject *parent = nullptr);
        ~ModelFitter();

        void update() override;
        std::vector<std::string> getInputLabels() const;
        std::vector<std::string> getOutputLabels() const;


        std::vector<float> calculateInNet(const std::vector<float> &inputs);

        void enableTraining(bool enable);
        bool trainingIsEnabled() const;

    private slots:
        void onBlinkTimer();
    private:
        void fitModel();
        void fitModelThreaded(size_t begin, size_t count);
        void evaluateBestModel();
        void trainModel();

        void getMinMaxErrors(float &min, float &average, float &max);
        void updateNetDimPainter();

        void setupDatasetPlot();
        void updateDatasetPlot();
        sf::Color mapOutputToColor(float value);


        bool m_fitted;
        bool m_trainingEnabled;

        struct ModelData
        {
            float error;
            AI_agent *agent;
        };

    NeuronalNet::Graphics::PixelPainter *m_netDimPainter;
    std::vector<ModelData> m_agents;
    ModelData m_fittest;
    size_t m_fitIterations;

    QTimer *m_blinkTimer;
    bool m_blinkToggle;
    sf::Color m_blinkDefColor;

    // Displays the inputs (x and y) if the dataset is a 2 input set
    // and visualizes the output as color for each given input combination.
    std::vector<NeuronalNet::Graphics::PixelPainter*> m_datasetPlots;
    float m_datasetPlot_gridStepSize;
    sf::Vector2f m_datasetPlotSpaceMin;
    sf::Vector2f m_datasetPlotSpaceMax;

};
