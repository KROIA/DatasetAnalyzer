#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "neuronalNet.h"

using namespace NeuronalNet;
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT
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

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();


    void train();
    void validate();

private:
    void normalize(std::vector<TrainingData> &data);
    void splitSets(const std::vector<TrainingData>&set,
                   std::vector<TrainingData> &trainingSet,
                   std::vector<TrainingData> &testSet,
                   float trainingTestRatio = 0.8f,
                   size_t testBeginIndex = std::string::npos);
    void saveNet();

    Ui::MainWindow *ui;

    BackpropNet *m_net;
    std::vector<std::string> m_inputLabels;
    std::vector<std::string> m_outputLabels;

    std::vector<std::vector<float> > m_inputSet;
    std::vector<std::vector<float> > m_outputSet;

    std::vector<TrainingData> m_dataset;
    std::vector<TrainingData> m_trainingSet;
    std::vector<TrainingData> m_testSet;

};
#endif // MAINWINDOW_H
