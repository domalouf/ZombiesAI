import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

// Everything at a glance: the recording, the trainers, today's recording goal, the machine, and what happened.
ScrollView {
    id: page
    contentWidth: availableWidth
    clip: true

    readonly property var rec: backend.record
    readonly property var status: rec.status || ({})
    readonly property var trainers: backend.summary.trainers || []
    readonly property var runs: backend.summary.runs || []
    readonly property var demos: backend.demos
    readonly property var now: backend.machine.now || ({})

    function primaryAction() { win.go(1) }

    ColumnLayout {
        width: page.availableWidth
        spacing: 16

        Item { Layout.preferredHeight: 8 }

        RowLayout {
            Layout.leftMargin: 24; Layout.rightMargin: 24
            Text { text: qsTr("Overview"); color: ui.ink; font.pixelSize: 24; font.bold: true; Layout.fillWidth: true }
            AccentButton { text: qsTr("Record"); onClicked: win.go(1) }
            Button { Material.roundedScale: theme.radius; text: qsTr("New run"); onClicked: win.go(3) }
        }

        GridLayout {
            Layout.leftMargin: 24; Layout.rightMargin: 24
            Layout.fillWidth: true
            columns: page.width > 1100 ? 4 : 2
            columnSpacing: 16; rowSpacing: 16

            Panel {
                Layout.fillWidth: true; Layout.fillHeight: true
                title: qsTr("Recording")
                StatTile {
                    label: page.rec.active ? (page.status.playing === false ? qsTr("Not playing") : qsTr("Recording"))
                                           : qsTr("Idle")
                    value: page.rec.active ? backend.formatDuration(page.status.elapsed_s || 0) : "–"
                    valueColor: page.rec.active ? (page.status.playing === false ? ui.warn : ui.bad) : ui.dim
                    note: page.rec.active && page.status.limit_s ? qsTr("of %1").arg(backend.formatDuration(page.status.limit_s))
                        : page.rec.last ? qsTr("last: %1").arg(page.rec.last.name) : ""
                }
                Text {
                    Layout.fillWidth: true
                    color: ui.dim; font.pixelSize: 12; wrapMode: Text.Wrap
                    text: page.rec.game && page.rec.game.running ? qsTr("Desktop game is running") : qsTr("Desktop game is not running")
                }
            }

            Panel {
                Layout.fillWidth: true; Layout.fillHeight: true
                title: qsTr("Training")
                StatTile {
                    label: qsTr("%n trainers running", "", page.trainers.length)
                    value: page.trainers.length ? page.trainers.map(t => t.run || t.script).join(", ") : "–"
                    valueColor: page.trainers.length ? ui.good : ui.dim
                }
                Repeater {
                    model: page.trainers
                    delegate: Meter {
                        required property var modelData
                        Layout.fillWidth: true
                        label: modelData.run || modelData.script
                        text: modelData.progress && modelData.progress.progress_text ? modelData.progress.progress_text
                            : backend.formatDuration(modelData.elapsed_s)
                        value: modelData.progress && modelData.progress.progress ? modelData.progress.progress : 0
                        maximum: 1
                    }
                }
            }

            Panel {
                id: goalPanel
                Layout.fillWidth: true; Layout.fillHeight: true
                title: qsTr("Today's recording")
                readonly property real today: page.demos.today_min || 0
                readonly property real goal: page.demos.goal_min || 20
                StatTile {
                    label: qsTr("Minutes played")
                    value: Math.round(goalPanel.today).toLocaleString(Qt.locale(), 'f', 0) + " / " + goalPanel.goal
                    valueColor: goalPanel.today >= goalPanel.goal ? ui.good : ui.ink
                    note: qsTr("%1 h recorded in all").arg(((page.demos.total_play_min || 0) / 60).toLocaleString(Qt.locale(), 'f', 1))
                }
                WeekBars { Layout.fillWidth: true; week: page.demos.week || []; goal: page.demos.goal_min || 20 }
            }

            Panel {
                Layout.fillWidth: true; Layout.fillHeight: true
                title: qsTr("Machine")
                subtitle: (backend.machine.gpu || {}).name || ""
                Meter { Layout.fillWidth: true; label: qsTr("CPU"); value: page.now.cpu_busy || 0
                        text: qsTr("%1 % · %2 °C").arg(Math.round(page.now.cpu_busy || 0)).arg(Math.round(page.now.cpu_temp || 0))
                        barColor: ui.levelColor((backend.machine.levels || {}).cpu_temp === "good" ? "" : "warn") }
                Meter { Layout.fillWidth: true; label: qsTr("GPU"); value: page.now.gpu_busy || 0
                        text: qsTr("%1 % · %2 °C").arg(Math.round(page.now.gpu_busy || 0)).arg(Math.round(page.now.gpu_temp || 0)) }
                Meter { Layout.fillWidth: true; label: qsTr("VRAM"); value: page.now.vram || 0
                        text: qsTr("%1 %").arg(Math.round(page.now.vram || 0)) }
                Meter { Layout.fillWidth: true; label: qsTr("RAM"); value: page.now.ram || 0
                        text: qsTr("%1 %").arg(Math.round(page.now.ram || 0)) }
            }
        }

        RowLayout {
            Layout.leftMargin: 24; Layout.rightMargin: 24
            Layout.fillWidth: true
            spacing: 16

            Panel {
                Layout.fillWidth: true
                Layout.preferredWidth: 3
                Layout.alignment: Qt.AlignTop
                title: qsTr("Latest runs")
                subtitle: qsTr("%n runs", "", page.runs.length)
                Repeater {
                    model: page.runs.slice(0, 6)
                    delegate: RowLayout {
                        required property var modelData
                        Layout.fillWidth: true
                        spacing: 12
                        Rectangle { width: 8; height: 8; radius: 4; color: ui.statusColor(modelData.status) }
                        Text { text: modelData.name; color: ui.ink; font.pixelSize: 13; font.bold: true; Layout.preferredWidth: 120; elide: Text.ElideRight }
                        Text { text: modelData.status; color: ui.dim; font.pixelSize: 12; Layout.preferredWidth: 70 }
                        Text {
                            Layout.fillWidth: true
                            color: ui.dim; font.pixelSize: 12; elide: Text.ElideRight
                            text: modelData.headline ? "%1 %2".arg(modelData.headline.label).arg(Number(modelData.headline.last).toLocaleString(Qt.locale(), 'f', 3))
                                                     : modelData.progress_text || ""
                        }
                        Text { text: modelData.preset ? "◆ " + modelData.preset : ""; color: ui.accent; font.pixelSize: 11 }
                    }
                }
                Button { Material.roundedScale: theme.radius; flat: true; text: qsTr("All runs"); onClicked: win.go(2) }
            }

            Panel {
                Layout.fillWidth: true
                Layout.preferredWidth: 2
                Layout.alignment: Qt.AlignTop
                title: qsTr("Activity")
                subtitle: qsTr("Runs ending, recordings saved, renders done")
                Text {
                    visible: backend.events.length === 0
                    text: qsTr("Nothing yet. Events since the server started show here, and as desktop notifications.")
                    color: ui.faint; font.pixelSize: 12; wrapMode: Text.Wrap; Layout.fillWidth: true
                }
                Repeater {
                    model: backend.events.slice(-8).reverse()
                    delegate: RowLayout {
                        required property var modelData
                        Layout.fillWidth: true
                        spacing: 10
                        Rectangle { width: 3; Layout.fillHeight: true; color: ui.levelColor(modelData.level) }
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 1
                            Text { text: modelData.title; color: ui.ink; font.pixelSize: 13; font.bold: true; elide: Text.ElideRight; Layout.fillWidth: true }
                            Text { text: modelData.text; color: ui.dim; font.pixelSize: 11; wrapMode: Text.Wrap; Layout.fillWidth: true; maximumLineCount: 2; elide: Text.ElideRight }
                        }
                        Text {
                            text: new Date(modelData.t * 1000).toLocaleTimeString(Qt.locale(), Locale.ShortFormat)
                            color: ui.faint; font.pixelSize: 11
                        }
                    }
                }
            }
        }
        Item { Layout.preferredHeight: 16 }
    }
}
