#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "neuronalNet.h"
#include "SFML_EditorWidget.h"
#include "inputfield.h"
#include "ModelFitter.h"

using namespace NeuronalNet;
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT


public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();


    private slots:
    void on_predictFromInput_pushButton_clicked();

    void on_startStopTraining_pushButton_clicked();

    private:


    Ui::MainWindow *ui;
    QSFML::Canvas *m_canvas;

    std::vector<InputField*> m_inputFields;
    std::vector<InputField*> m_outputFields;
    ModelFitter *m_fitter;


};
#endif // MAINWINDOW_H
