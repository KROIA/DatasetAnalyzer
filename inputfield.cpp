#include "inputfield.h"
#include "ui_inputfield.h"

InputField::InputField(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::InputField)
{
    ui->setupUi(this);
}

InputField::~InputField()
{
    delete ui;
}

void InputField::setLabel(const std::string &label)
{
    ui->label->setText(label.c_str());
}
float InputField::getValue() const
{
    return ui->doubleSpinBox->value();
}
void InputField::setValue(float value)
{
    ui->doubleSpinBox->setValue(value);
}
