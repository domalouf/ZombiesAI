#pragma once

#include <QElapsedTimer>
#include <QHash>
#include <QNetworkAccessManager>
#include <QObject>
#include <QTimer>
#include <QUrl>
#include <QVariant>
#include <functional>

// The app's whole view of the project comes from the Training Room's backend (scripts/supervise.py), a Python
// server on 127.0.0.1 that owns everything that can fail: starting and stopping runs, recordings, BC training,
// the games. The app only reads its JSON and posts commands, so the browser page and this window can never
// disagree about what is running. If the server is not up, the app starts it, detached, so it outlives the
// window like the runs it starts.
class Backend : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString serverState READ serverState NOTIFY serverChanged)
    Q_PROPERTY(QString serverError READ serverError NOTIFY serverChanged)
    Q_PROPERTY(bool connected READ connected NOTIFY serverChanged)
    Q_PROPERTY(QString baseUrl READ baseUrl CONSTANT)
    Q_PROPERTY(QVariantMap summary READ summary NOTIFY summaryChanged)
    Q_PROPERTY(QVariantMap record READ record NOTIFY recordChanged)
    Q_PROPERTY(QVariantMap demos READ demos NOTIFY demosChanged)
    Q_PROPERTY(QVariantMap launch READ launch NOTIFY launchChanged)
    Q_PROPERTY(QVariantMap bc READ bc NOTIFY bcChanged)
    Q_PROPERTY(QVariantList presets READ presets NOTIFY presetsChanged)
    Q_PROPERTY(QVariantMap games READ games NOTIFY gamesChanged)
    Q_PROPERTY(QVariantMap fleet READ fleet NOTIFY fleetChanged)
    Q_PROPERTY(QVariantMap machine READ machine NOTIFY machineChanged)
    Q_PROPERTY(QVariantList events READ events NOTIFY eventsChanged)

public:
    // `autostart` false (the tests) never polls or launches anything; `launcher` overrides how the server is
    // started (a program and its arguments).
    explicit Backend(const QUrl &base = QUrl("http://127.0.0.1:8765"), QObject *parent = nullptr,
                     bool autostart = true, const QStringList &launcher = {});

    QString serverState() const { return m_serverState; }
    QString serverError() const { return m_serverError; }
    bool connected() const { return m_serverState == "up"; }
    QString baseUrl() const { return m_base.toString(); }
    QVariantMap summary() const { return m_summary; }
    QVariantMap record() const { return m_record; }
    QVariantMap demos() const { return m_demos; }
    QVariantMap launch() const { return m_launch; }
    QVariantMap bc() const { return m_bc; }
    QVariantList presets() const { return m_presets; }
    QVariantMap games() const { return m_games; }
    QVariantMap fleet() const { return m_fleet; }
    QVariantMap machine() const { return m_machine; }
    QVariantList events() const { return m_events; }

    // Re-read one feed now ("summary", "record", "demos", "launch", "bc", "presets", "games", "fleet",
    // "machine", "events"), or every feed with no argument.
    Q_INVOKABLE void refresh(const QString &what = {});
    // POST a command. `actionFinished(tag, ok, response)` follows, and the feeds it touches are re-read.
    Q_INVOKABLE void post(const QString &path, const QVariantMap &body, const QString &tag = {});
    Q_INVOKABLE void startServer();
    Q_INVOKABLE void openPath(const QString &path) const;
    Q_INVOKABLE void openTrainingRoom() const;
    Q_INVOKABLE QString formatDuration(double seconds) const { return duration(seconds); }
    Q_INVOKABLE QVariantMap buildSettings(const QVariantList &schema, const QVariantMap &values) const {
        return changedSettings(schema, values);
    }
    Q_INVOKABLE QVariant settingDefault(const QVariantMap &setting) const { return setting.value("default"); }

    // What the feeds' JSON turns into, kept pure so the tests can check it without a server.
    static QString duration(double seconds);
    // Only the settings whose value differs from the schema's default, typed as the schema says: the server
    // turns exactly these into flags, so a run started here and one started from a terminal read the same.
    static QVariantMap changedSettings(const QVariantList &schema, const QVariantMap &values);
    static QVariantList mergeEvents(const QVariantList &have, const QVariantList &incoming, int keep = 100);
    static QStringList defaultLauncher();

signals:
    void serverChanged();
    void summaryChanged();
    void recordChanged();
    void demosChanged();
    void launchChanged();
    void bcChanged();
    void presetsChanged();
    void gamesChanged();
    void fleetChanged();
    void machineChanged();
    void eventsChanged();
    void actionFinished(const QString &tag, bool ok, const QVariantMap &response);

private:
    void get(const QString &path, const std::function<void(const QVariant &)> &done);
    void tick();
    void ping();
    void setServer(const QString &state, const QString &error = {});
    void touched(const QString &path);

    QUrl m_base;
    QNetworkAccessManager m_net;
    QTimer m_timer;
    QStringList m_launcher;
    QString m_serverState = "connecting";
    QString m_serverError;
    QElapsedTimer m_startedAt;
    int m_tick = 0;
    double m_lastEvent = 0;
    QHash<QString, bool> m_inFlight;
    QVariantMap m_summary, m_record, m_demos, m_launch, m_bc, m_games, m_fleet, m_machine;
    QVariantList m_presets, m_events;
};
