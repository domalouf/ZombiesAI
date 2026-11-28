// zombies — the ZombiesAI dashboard: record demos, start and stop training runs, and watch them.

#include <QGuiApplication>
#include <QIcon>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>
#include <QUrl>

#include "backend.h"
#include "theme.h"
#include "translator.h"

int main(int argc, char *argv[]) {
    QGuiApplication app(argc, argv);
    app.setApplicationName("zombies");
    app.setApplicationVersion("0.1.0");

    // Matches the window to zombies.desktop, so the compositor (Wayland app_id)
    // and the launcher pick up the installed icon.
    app.setDesktopFileName("zombies");
    app.setWindowIcon(QIcon::fromTheme("zombies"));

    QQuickStyle::setStyle("Material");

    // Installed before anything is translated, and every qsTr() binding is
    // re-evaluated when the Omarchy language changes.
    Translator translator;
    app.installTranslator(&translator);

    Theme theme;
    // ZOMBIESAI_DASHBOARD_URL points the app at a server on another port (a second checkout's).
    const QString url = qEnvironmentVariable("ZOMBIESAI_DASHBOARD_URL", "http://127.0.0.1:8765");
    Backend backend{QUrl(url)};

    QQmlApplicationEngine engine;
    QObject::connect(&translator, &Translator::changed, &engine, &QQmlEngine::retranslate);
    engine.rootContext()->setContextProperty("translator", &translator);
    engine.rootContext()->setContextProperty("theme", &theme);
    engine.rootContext()->setContextProperty("backend", &backend);
    engine.load(QUrl("qrc:/Main.qml"));
    if (engine.rootObjects().isEmpty())
        return 1;

    return app.exec();
}
