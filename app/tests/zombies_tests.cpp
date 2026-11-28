#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QSignalSpy>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QtTest>

#include "backend.h"
#include "theme.h"
#include "translator.h"

// A one-file HTTP server: answers each path with canned JSON and keeps every request it saw, so the tests can
// check what the app sends without the real Training Room server.
class FakeServer : public QTcpServer {
public:
    QHash<QString, QByteArray> answers;
    QList<QByteArray> requests;

    FakeServer() {
        listen(QHostAddress::LocalHost, 0);
        connect(this, &QTcpServer::newConnection, this, [this] {
            while (QTcpSocket *socket = nextPendingConnection()) {
                connect(socket, &QTcpSocket::readyRead, socket, [this, socket] {
                    const QByteArray request = socket->readAll();
                    const int headerEnd = request.indexOf("\r\n\r\n");
                    const int length = request.mid(0, headerEnd).toLower().contains("content-length")
                        ? QString::fromLatin1(request.mid(request.toLower().indexOf("content-length:") + 15).split('\r').first()).trimmed().toInt()
                        : 0;
                    if (headerEnd < 0 || request.size() < headerEnd + 4 + length)
                        return;  // the body is still on its way; wait for the rest (small bodies arrive whole)
                    requests.append(request);
                    const QString path = QString::fromLatin1(request.split(' ').value(1)).section('?', 0, 0);
                    const QByteArray body = answers.value(path, R"({"error": "not found"})");
                    socket->write("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                                  + QByteArray::number(body.size()) + "\r\n\r\n" + body);
                    socket->disconnectFromHost();
                });
            }
        });
    }
    QUrl url() const { return QUrl(QString("http://127.0.0.1:%1").arg(serverPort())); }
};

class ZombiesTests : public QObject {
    Q_OBJECT

private slots:
    void durationsReadLikeAClock()
    {
        QCOMPARE(Backend::duration(0), QString("0:00"));
        QCOMPARE(Backend::duration(75), QString("1:15"));
        QCOMPARE(Backend::duration(3725), QString("1:02:05"));
        QCOMPARE(Backend::duration(-1), QString("–"));
        QCOMPARE(Backend::duration(qQNaN()), QString("–"));
    }

    void onlyChangedSettingsBecomeFlags()
    {
        const QVariantList schema = QJsonDocument::fromJson(R"([
            {"name": "total_steps", "kind": "int", "default": 2000000},
            {"name": "lr", "kind": "float", "default": 0.0001},
            {"name": "calibrate", "kind": "bool", "default": true},
            {"name": "device", "kind": "str", "default": "cuda"},
            {"name": "frame_offsets", "kind": "list", "default": "0 2 4"},
            {"name": "seed", "kind": "int", "default": null}
        ])").toVariant().toList();
        QVariantMap values{{"total_steps", "2000000"}, {"lr", "0.0003"}, {"calibrate", false},
                           {"device", "cuda"}, {"frame_offsets", "0 2 4"}, {"seed", "7"}, {"unknown", 1}};
        const QVariantMap out = Backend::changedSettings(schema, values);
        QCOMPARE(out.keys(), (QStringList{"calibrate", "lr", "seed"}));
        QCOMPARE(out.value("lr").toDouble(), 0.0003);
        QCOMPARE(out.value("calibrate").toBool(), false);
        QCOMPARE(out.value("seed").toLongLong(), 7LL);
        // Typing a number that is not one leaves the setting at its default rather than sending junk.
        QVERIFY(Backend::changedSettings(schema, {{"total_steps", "2e"}}).isEmpty());
        QCOMPARE(Backend::changedSettings(schema, {{"frame_offsets", "0 1"}}).value("frame_offsets").toString(), QString("0 1"));
    }

    void eventsAreNeverShownTwice()
    {
        auto event = [](int seq) { return QVariantMap{{"seq", seq}, {"title", QString::number(seq)}}; };
        QVariantList have{event(1), event(2)};
        const QVariantList merged = Backend::mergeEvents(have, {event(2), event(3)});
        QCOMPARE(merged.size(), 3);
        QCOMPARE(merged.last().toMap().value("seq").toInt(), 3);
        QVariantList many;
        for (int i = 1; i <= 120; ++i)
            many.append(event(i));
        QCOMPARE(Backend::mergeEvents({}, many, 100).size(), 100);
        QCOMPARE(Backend::mergeEvents({}, many, 100).first().toMap().value("seq").toInt(), 21);
    }

    void connectsAndReadsTheFeeds()
    {
        FakeServer server;
        server.answers["/api/ping"] = R"({"app": "zombiesai-supervise"})";
        server.answers["/api/summary"] = R"({"runs": [{"name": "rl3", "status": "running"}], "trainers": []})";
        server.answers["/api/events"] = R"({"events": [{"seq": 4, "title": "rl2 stopped"}], "seq": 4})";
        Backend backend(server.url(), nullptr, true, {"/bin/false"});
        QTRY_COMPARE(backend.serverState(), QString("up"));
        QTRY_COMPARE(backend.summary().value("runs").toList().size(), 1);
        QTRY_COMPARE(backend.events().size(), 1);
    }

    void commandsCarryTheSupervisorHeader()
    {
        FakeServer server;
        server.answers["/api/record"] = R"({"ok": true, "pid": 12})";
        Backend backend(server.url(), nullptr, false);
        QSignalSpy finished(&backend, &Backend::actionFinished);
        backend.post("/api/record", {{"minutes", 20}, {"notes", "later rounds"}}, "record");
        QTRY_COMPARE(finished.size(), 1);
        QCOMPARE(finished.first().at(0).toString(), QString("record"));
        QVERIFY(finished.first().at(1).toBool());
        const QByteArray request = server.requests.first();
        QVERIFY(request.startsWith("POST /api/record "));
        QVERIFY(request.contains("X-Supervise: 1"));
        QVERIFY(request.contains(R"("notes":"later rounds")"));
    }

    void aRefusedCommandSaysWhy()
    {
        FakeServer server;
        server.answers["/api/record/stop"] = R"({"error": "no recording is running"})";
        Backend backend(server.url(), nullptr, false);
        QSignalSpy finished(&backend, &Backend::actionFinished);
        backend.post("/api/record/stop", {});
        QTRY_COMPARE(finished.size(), 1);
        QVERIFY(!finished.first().at(1).toBool());
        QCOMPARE(finished.first().at(2).toMap().value("error").toString(), QString("no recording is running"));
    }

    void aMissingLauncherIsReported()
    {
        Backend backend(QUrl("http://127.0.0.1:9"), nullptr, false, {"/nonexistent/zombiesai-dashboard"});
        backend.startServer();
        QCOMPARE(backend.serverState(), QString("failed"));
        QVERIFY(backend.serverError().contains("/nonexistent/zombiesai-dashboard"));
    }

    void themeFallsBack()
    {
        QTemporaryDir dir;
        Theme theme(dir.path());
        QCOMPARE(theme.accent(), QString("#FFD60A"));
        QCOMPARE(theme.accentForeground(), QString("black"));
    }

    void themeReadsAccent()
    {
        QTemporaryDir dir;
        QDir(dir.path()).mkpath("theme");
        write(dir.path() + "/theme/colors.toml", "accent = \"#112233\"\n");
        Theme theme(dir.path());
        QCOMPARE(theme.accent(), QString("#112233"));
        QCOMPARE(theme.accentForeground(), QString("white"));
    }

    void themeFollowsHyprlandRounding()
    {
        QTemporaryDir dir;
        const QString response = dir.filePath("response.json"), command = dir.filePath("hyprctl");
        write(command, "#!/bin/sh\ncat '" + response.toUtf8() + "'\n");
        QVERIFY(QFile::setPermissions(command, QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner));
        write(response, "{\"int\":0}");
        Theme theme(dir.path(), nullptr, command);
        QCOMPARE(theme.radius(), 0);
        write(response, "{\"int\":6}");
        QTRY_COMPARE(theme.radius(), 6);
    }

    void translatorFallsBackToEnglish()
    {
        QTemporaryDir dir;
        Translator translator(dir.path() + "/language", dir.path());
        QCOMPARE(translator.language(), QString("en"));
        QVERIFY(translator.translate("Main", "Save").isNull());
    }

    void translatorFollowsTheLanguageFile()
    {
        QTemporaryDir dir;
        write(dir.path() + "/language", "pt-BR\n");
        write(dir.path() + "/pt.json", R"({"Save": "Guardar", "Open": "Abrir"})");
        write(dir.path() + "/pt-BR.json", R"({"Save": "Salvar"})");
        Translator translator(dir.path() + "/language", dir.path());
        QCOMPARE(translator.language(), QString("pt-BR"));
        QCOMPARE(translator.translate("Main", "Save"), QString("Salvar"));
        QCOMPARE(translator.translate("Main", "Open"), QString("Abrir"));
        QVERIFY(translator.translate("Main", "Quit").isNull());
    }

    void englishCatalogGivesPluralForms()
    {
        QTemporaryDir dir;
        Translator translator(dir.path() + "/language");  // the compiled-in catalogs
        QCOMPARE(translator.translate("Main", "%n runs", nullptr, 1), QString("%n run"));
        QCOMPARE(translator.translate("Main", "%n runs", nullptr, 3), QString("%n runs"));
        QCOMPARE(translator.translate("Main", "%n settings", nullptr, 1), QString("%n setting"));
    }

private:
    static void write(const QString &path, const QByteArray &contents)
    {
        QFile file(path);
        QVERIFY(file.open(QIODevice::WriteOnly));
        file.write(contents);
    }
};

QTEST_MAIN(ZombiesTests)
#include "zombies_tests.moc"
