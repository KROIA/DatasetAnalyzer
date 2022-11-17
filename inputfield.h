#ifndef INPUTFIELD_H
#define INPUTFIELD_H

#include <QWidget>

namespace Ui {
class InputField;
}

class InputField : public QWidget
{
        Q_OBJECT

    public:
        explicit InputField(QWidget *parent = nullptr);
        ~InputField();

        void setLabel(const std::string &label);
        float getValue() const;
        void setValue(float value);

    private:
        Ui::InputField *ui;
};

#endif // INPUTFIELD_H
