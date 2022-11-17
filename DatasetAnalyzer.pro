QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

include(Extern/Neural-net-2-graphics/Neural-net-2-graphics.pri)

SOURCES += \
    AI_agent.cpp \
    ModelFitter.cpp \
    datasetImport.cpp \
    inputfield.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    AI_agent.h \
    ModelFitter.h \
    datasetImport.h \
    inputfield.h \
    mainwindow.h

FORMS += \
    inputfield.ui \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
