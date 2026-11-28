#include "backend.h"

#include <QDesktopServices>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QProcess>
#include <QStandardPaths>
#include <cmath>

namespace {
// How long a just-launched server gets to answer before the window says it failed: it imports torch's
// neighbours and walks every run on the first request.
constexpr int StartupGraceMs = 30000;
constexpr int RequestTimeoutMs = 8000;
}

Backend::Backend(const QUrl &base, QObject *parent, bool autostart, const QStringList &launcher)
    : QObject(parent), m_base(base), m_launcher(launcher.isEmpty() ? defaultLauncher() : launcher) {
    m_timer.setInterval(1000);
    connect(&m_timer, &QTimer::timeout, this, &Backend::tick);
    if (autostart) {
        ping();
        m_timer.start();
    }
}

// The server is started the way Super+Shift+Z starts it (~/.local/bin/zombiesai-dashboard, which finds the
// checkout and its venv), without opening a browser. Without that script, the checkout named by
// $ZOMBIESAI_REPO, or ~/Projects/ZombiesAI.
QStringList Backend::defaultLauncher() {
    const QString script = QStandardPaths::findExecutable("zombiesai-dashboard");
    const QString local = QDir::homePath() + "/.local/bin/zombiesai-dashboard";
    if (!script.isEmpty())
        return {script, "--no-open"};
    if (QFileInfo(local).isExecutable())
        return {local, "--no-open"};
    const QString repo = qEnvironmentVariableIsEmpty("ZOMBIESAI_REPO") ? QDir::homePath() + "/Projects/ZombiesAI"
                                                                        : qEnvironmentVariable("ZOMBIESAI_REPO");
    return {repo + "/.venv/bin/python", repo + "/scripts/supervise.py", "--no-open"};
}

void Backend::setServer(const QString &state, const QString &error) {
    if (state == m_serverState && error == m_serverError)
        return;
    const bool cameUp = state == "up" && m_serverState != "up";
    m_serverState = state;
    m_serverError = error;
    emit serverChanged();
    if (cameUp)
        refresh();
}

void Backend::startServer() {
    if (m_launcher.isEmpty() || !QFileInfo(m_launcher.first()).isExecutable()) {
        setServer("failed", tr("Cannot start the Training Room server: %1 is missing.")
                                .arg(m_launcher.value(0, "zombiesai-dashboard")));
        return;
    }
    if (!QProcess::startDetached(m_launcher.first(), m_launcher.mid(1))) {
        setServer("failed", tr("Could not run %1.").arg(m_launcher.first()));
        return;
    }
    m_startedAt.start();
    setServer("starting");
}

void Backend::ping() {
    if (m_inFlight.value("/api/ping"))
        return;
    m_inFlight["/api/ping"] = true;
    QNetworkRequest request(m_base.resolved(QUrl("/api/ping")));
    request.setTransferTimeout(2000);
    QNetworkReply *reply = m_net.get(request);
    connect(reply, &QNetworkReply::finished, this, [this, reply] {
        reply->deleteLater();
        m_inFlight["/api/ping"] = false;
        const auto app = QJsonDocument::fromJson(reply->readAll()).toVariant().toMap().value("app").toString();
        if (reply->error() == QNetworkReply::NoError && app == "zombiesai-supervise") {
            setServer("up");
        } else if (m_serverState == "starting") {
            if (m_startedAt.elapsed() > StartupGraceMs)
                setServer("failed", tr("The Training Room server did not answer within %1 s. Its log is in "
                                       "$XDG_RUNTIME_DIR/zombiesai-supervise/server.log.")
                                        .arg(StartupGraceMs / 1000));
        } else if (m_serverState == "connecting") {
            startServer();
        } else if (m_serverState == "up") {
            setServer("lost", tr("Lost the Training Room server; trying again."));
        }
    });
}

void Backend::get(const QString &path, const std::function<void(const QVariant &)> &done) {
    const QString key = QUrl(path).path();
    if (m_inFlight.value(key))
        return;  // the last read of this feed is still on its way: never stack them up
    m_inFlight[key] = true;
    QNetworkRequest request(m_base.resolved(QUrl(path)));
    request.setTransferTimeout(RequestTimeoutMs);
    QNetworkReply *reply = m_net.get(request);
    connect(reply, &QNetworkReply::finished, this, [this, reply, key, done] {
        reply->deleteLater();
        m_inFlight[key] = false;
        if (reply->error() == QNetworkReply::ConnectionRefusedError) {
            setServer("lost", tr("Lost the Training Room server; trying again."));
            return;
        }
        const QVariant value = QJsonDocument::fromJson(reply->readAll()).toVariant();
        if (value.isValid())
            done(value);
    });
}

void Backend::refresh(const QString &what) {
    if (!connected())
        return;
    auto want = [&](const char *name) { return what.isEmpty() || what == QLatin1String(name); };
    if (want("summary"))
        get("/api/summary", [this](const QVariant &v) { m_summary = v.toMap(); emit summaryChanged(); });
    if (want("record"))
        get("/api/record", [this](const QVariant &v) { m_record = v.toMap(); emit recordChanged(); });
    if (want("demos"))
        get("/api/demos", [this](const QVariant &v) { m_demos = v.toMap(); emit demosChanged(); });
    if (want("launch"))
        get("/api/launch", [this](const QVariant &v) { m_launch = v.toMap(); emit launchChanged(); });
    if (want("bc"))
        get("/api/bc", [this](const QVariant &v) { m_bc = v.toMap(); emit bcChanged(); });
    if (want("presets"))
        get("/api/presets", [this](const QVariant &v) {
            m_presets = v.toMap().value("presets").toList();
            emit presetsChanged();
        });
    if (want("games"))
        get("/api/games", [this](const QVariant &v) { m_games = v.toMap(); emit gamesChanged(); });
    if (want("fleet"))
        get("/api/fleet", [this](const QVariant &v) { m_fleet = v.toMap(); emit fleetChanged(); });
    if (want("machine"))
        get("/api/machine", [this](const QVariant &v) { m_machine = v.toMap(); emit machineChanged(); });
    if (want("events"))
        get(QString("/api/events?since=%1").arg(m_lastEvent, 0, 'f', 0), [this](const QVariant &v) {
            const QVariantMap map = v.toMap();
            const QVariantList incoming = map.value("events").toList();
            m_lastEvent = std::max(m_lastEvent, map.value("seq").toDouble());
            if (!incoming.isEmpty()) {
                m_events = mergeEvents(m_events, incoming);
                emit eventsChanged();
            }
        });
}

// One clock for every feed, each at the pace it changes: a recording's timer every second while one runs,
// the curves every five, the demo list every ten. Settings and presets only change when someone acts, so
// they are re-read after each command and otherwise twice a minute.
void Backend::tick() {
    ++m_tick;
    if (!connected()) {
        ping();
        return;
    }
    if (m_tick % 5 == 0)
        ping();
    const bool recording = m_record.value("active").toBool();
    if (recording || m_tick % 3 == 0)
        refresh("record");
    if (m_tick % 2 == 0)
        refresh("events");
    if (m_tick % 3 == 0)
        refresh("machine");
    if (m_tick % 5 == 0) {
        refresh("summary");
        refresh("games");
        refresh("fleet");
    }
    if (m_tick % 10 == 0)
        refresh("demos");
    if (m_tick % 30 == 0) {
        refresh("launch");
        refresh("bc");
        refresh("presets");
    }
}

void Backend::touched(const QString &path) {
    // What a command can change, re-read at once rather than at the next tick.
    if (path.startsWith("/api/record") || path == "/api/desktop")
        refresh("record");
    if (path.startsWith("/api/demos"))
        refresh("demos");
    if (path.startsWith("/api/presets"))
        refresh("presets");
    if (path == "/api/start" || path == "/api/stop" || path == "/api/bc") {
        refresh("summary");
        refresh("launch");
        refresh("bc");
    }
    if (path == "/api/fleet" || path == "/api/viewers") {
        refresh("fleet");
        refresh("games");
    }
    refresh("events");
}

void Backend::post(const QString &path, const QVariantMap &body, const QString &tag) {
    QNetworkRequest request(m_base.resolved(QUrl(path)));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    // The server takes commands only with this header, which no web page can send without a preflight it
    // refuses (viz/supervise.py).
    request.setRawHeader("X-Supervise", "1");
    request.setTransferTimeout(RequestTimeoutMs * 2);
    QNetworkReply *reply = m_net.post(request, QJsonDocument::fromVariant(body).toJson(QJsonDocument::Compact));
    connect(reply, &QNetworkReply::finished, this, [this, reply, path, tag] {
        reply->deleteLater();
        QVariantMap response = QJsonDocument::fromJson(reply->readAll()).toVariant().toMap();
        const bool ok = reply->error() == QNetworkReply::NoError && !response.contains("error");
        if (!ok && !response.contains("error"))
            response["error"] = reply->errorString();
        emit actionFinished(tag.isEmpty() ? path : tag, ok, response);
        touched(path);
    });
}

void Backend::openPath(const QString &path) const {
    if (!path.isEmpty())
        QDesktopServices::openUrl(QUrl::fromLocalFile(path));
}

void Backend::openTrainingRoom() const {
    QDesktopServices::openUrl(m_base);
}

QString Backend::duration(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0)
        return QStringLiteral("–");
    const qint64 s = qint64(std::llround(seconds));
    const qint64 h = s / 3600, m = s % 3600 / 60, sec = s % 60;
    if (h > 0)
        return QString("%1:%2:%3").arg(h).arg(m, 2, 10, QChar('0')).arg(sec, 2, 10, QChar('0'));
    return QString("%1:%2").arg(m).arg(sec, 2, 10, QChar('0'));
}

QVariantMap Backend::changedSettings(const QVariantList &schema, const QVariantMap &values) {
    QVariantMap out;
    for (const QVariant &item : schema) {
        const QVariantMap setting = item.toMap();
        const QString name = setting.value("name").toString();
        if (!values.contains(name))
            continue;
        const QString kind = setting.value("kind").toString();
        const QVariant def = setting.value("default");
        QVariant value = values.value(name);
        if (kind == "bool") {
            value = value.toBool();
            if (value.toBool() == def.toBool())
                continue;
        } else if (kind == "int") {
            bool ok = false;
            const qlonglong n = value.toString().trimmed().toLongLong(&ok);
            if (!ok || (def.isValid() && !def.isNull() && n == def.toLongLong()))
                continue;
            value = n;
        } else if (kind == "float") {
            bool ok = false;
            const double x = value.toString().trimmed().toDouble(&ok);
            if (!ok || (def.isValid() && !def.isNull() && std::abs(x - def.toDouble()) <= 1e-12 * std::max(1.0, std::abs(x))))
                continue;
            value = x;
        } else {
            const QString text = value.toString().trimmed();
            if (text == def.toString())
                continue;
            value = text;
        }
        out.insert(name, value);
    }
    return out;
}

QVariantList Backend::mergeEvents(const QVariantList &have, const QVariantList &incoming, int keep) {
    QVariantList out = have;
    double last = have.isEmpty() ? -1 : have.last().toMap().value("seq").toDouble();
    for (const QVariant &event : incoming) {
        const double seq = event.toMap().value("seq").toDouble();
        if (seq > last) {  // the feed can overlap a read that crossed a tick; never show one twice
            out.append(event);
            last = seq;
        }
    }
    while (out.size() > keep)
        out.removeFirst();
    return out;
}
