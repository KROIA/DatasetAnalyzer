#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QSFML::CanvasSettings settings;
    m_canvas = new QSFML::Canvas(ui->netWidget,settings);

    m_canvas->addObject(new QSFML::Objects::DefaultEditor());
    m_canvas->addObject(m_fitter = new ModelFitter("kc_house_data.csv",";"));


    std::vector<std::string> inpLabels = m_fitter->getInputLabels();
    std::vector<std::string> outLabels = m_fitter->getOutputLabels();

    for(size_t i=0; i<inpLabels.size(); ++i)
    {
        InputField *field = new InputField(this);
        ui->inputContent->layout()->addWidget(field);
        field->setLabel(inpLabels[i]);
        m_inputFields.push_back(field);
    }
    for(size_t i=0; i<outLabels.size(); ++i)
    {
        InputField *field = new InputField(this);
        ui->outputContent->layout()->addWidget(field);
        field->setLabel(outLabels[i]);
        field->setEnabled(false);
        m_outputFields.push_back(field);
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_predictFromInput_pushButton_clicked()
{
    vector<float> inputs(m_inputFields.size(), 0);
    for(size_t i=0; i<m_inputFields.size(); ++i)
    {
        inputs[i] = m_inputFields[i]->getValue();
    }
    vector<float> outputs = m_fitter->calculateInNet(inputs);
    for(size_t i=0; i<outputs.size(); ++i)
    {
        m_outputFields[i]->setValue(outputs[i]);
    }
}
void MainWindow::on_startStopTraining_pushButton_clicked()
{
    m_fitter->enableTraining(!m_fitter->trainingIsEnabled());
    if(m_fitter->trainingIsEnabled())
        ui->startStopTraining_pushButton->setText("Stop training");
    else
        ui->startStopTraining_pushButton->setText("Start training");
}

