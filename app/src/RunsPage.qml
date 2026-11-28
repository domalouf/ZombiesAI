import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

// Every run, newest first; the selected one's curves, state and controls.
Item {
    id: page
    readonly property var runs: backend.summary.runs || []
    readonly property var trainers: backend.summary.trainers || []
    property string selected: ""
    readonly property var run: runs.find(r => r.name === selected) || runs[0] || null
    readonly property var trainer: run ? trainers.find(t => t.run === run.name) || null : null

    function primaryAction() {}

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // The list.
        Rectangle {
            Layout.fillHeight: true
            Layout.preferredWidth: 300
            color: ui.window
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 8
                Text { text: qsTr("Runs"); color: ui.ink; font.pixelSize: 24; font.bold: true }
                Text { text: qsTr("%n runs", "", page.runs.length); color: ui.dim; font.pixelSize: 12 }
                ListView {
                    id: list
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    spacing: 2
                    model: page.runs
                    delegate: ItemDelegate {
                        id: row
                        required property var modelData
                        width: ListView.view.width
                        highlighted: page.run && page.run.name === modelData.name
                        onClicked: page.selected = modelData.name
                        background: Rectangle { radius: ui.small; color: row.highlighted ? ui.raised : row.hovered ? "#18181d" : "transparent" }
                        contentItem: ColumnLayout {
                            spacing: 2
                            RowLayout {
                                Rectangle { width: 8; height: 8; radius: 4; color: ui.statusColor(row.modelData.status) }
                                Text { text: row.modelData.name; color: ui.ink; font.bold: true; font.pixelSize: 13; Layout.fillWidth: true; elide: Text.ElideRight }
                                Text { text: (row.modelData.kind || "").toUpperCase(); color: ui.faint; font.pixelSize: 10 }
                            }
                            Text {
                                Layout.fillWidth: true
                                text: [row.modelData.status, row.modelData.progress_text].filter(x => !!x).join(" · ")
                                color: ui.dim; font.pixelSize: 11; elide: Text.ElideRight
                            }
                        }
                    }
                }
            }
        }

        Rectangle { Layout.fillHeight: true; width: 1; color: ui.line }

        // The selected run.
        ScrollView {
            id: detail
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentWidth: availableWidth
            clip: true

            ColumnLayout {
                width: detail.availableWidth
                spacing: 16
                visible: page.run !== null

                Item { Layout.preferredHeight: 8 }
                RowLayout {
                    Layout.leftMargin: 24; Layout.rightMargin: 24
                    spacing: 12
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 2
                        Text { text: page.run ? page.run.name : ""; color: ui.ink; font.pixelSize: 24; font.bold: true }
                        Text {
                            color: ui.dim; font.pixelSize: 12
                            text: page.run ? [page.run.tree, page.run.kind, page.run.env, page.run.preset ? qsTr("preset %1").arg(page.run.preset) : ""]
                                             .filter(x => !!x).join(" · ") : ""
                        }
                    }
                    AccentButton {
                        text: page.trainer && page.trainer.stopping_s !== null ? qsTr("Stopping…") : qsTr("Stop")
                        visible: !!page.trainer
                        enabled: !!page.trainer && page.trainer.stopping_s === null
                        tone: ui.bad
                        onClicked: stopDialog.open()
                    }
                    Button { Material.roundedScale: theme.radius;
                        text: qsTr("Force stop")
                        visible: !!page.trainer && page.trainer.can_force === true
                        onClicked: backend.post("/api/stop", { pid: page.trainer.pid, force: true }, "force-stop")
                    }
                    Button { Material.roundedScale: theme.radius; text: qsTr("Open log"); flat: true; visible: !!(page.run && page.run.log); onClicked: backend.openPath(page.run.log) }
                    Button { Material.roundedScale: theme.radius; text: qsTr("Open folder"); flat: true; visible: !!(page.run && page.run.path); onClicked: backend.openPath(page.run.path) }
                }

                GridLayout {
                    Layout.leftMargin: 24; Layout.rightMargin: 24
                    Layout.fillWidth: true
                    columns: 4
                    columnSpacing: 24
                    StatTile { label: qsTr("Status"); value: page.run ? page.run.status : "–"; valueColor: ui.statusColor(page.run ? page.run.status : "") }
                    StatTile { label: qsTr("Progress"); value: page.run && page.run.progress !== null && page.run.progress !== undefined ? Math.round(page.run.progress * 100) + " %" : "–"; note: page.run ? page.run.progress_text || "" : "" }
                    StatTile { label: qsTr("Elapsed"); value: page.run ? backend.formatDuration(page.run.elapsed_s) : "–" }
                    StatTile { label: qsTr("Left"); value: page.run && page.run.eta_s ? backend.formatDuration(page.run.eta_s) : "–" }
                }

                Repeater {
                    model: page.run ? page.run.notes || [] : []
                    delegate: Rectangle {
                        required property var modelData
                        Layout.leftMargin: 24; Layout.rightMargin: 24
                        Layout.fillWidth: true
                        implicitHeight: noteText.implicitHeight + 16
                        color: "#2e2817"; radius: ui.small
                        Text { id: noteText; anchors.fill: parent; anchors.margins: 8; text: modelData; color: ui.ink; wrapMode: Text.Wrap; font.pixelSize: 12 }
                    }
                }

                GridLayout {
                    Layout.leftMargin: 24; Layout.rightMargin: 24
                    Layout.fillWidth: true
                    columns: detail.width > 1000 ? 2 : 1
                    columnSpacing: 16; rowSpacing: 16
                    Repeater {
                        // The headline first: it is what the run is judged on.
                        model: page.run ? Object.keys(page.run.series || {}).sort((a, b) =>
                            (page.run.headline && b === page.run.headline.key) - (page.run.headline && a === page.run.headline.key)) : []
                        delegate: Panel {
                            id: seriesPanel
                            required property string modelData
                            readonly property var series: page.run.series[modelData]
                            Layout.fillWidth: true
                            LineChart {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 150
                                label: seriesPanel.series.label
                                xs: seriesPanel.series.x
                                ys: seriesPanel.series.y
                                lineColor: page.run.headline && page.run.headline.key === modelData ? ui.accent : ui.ink
                                valueText: {
                                    const y = seriesPanel.series.y
                                    return y.length ? Number(y[y.length - 1]).toLocaleString(Qt.locale(), 'f', 3) : ""
                                }
                            }
                        }
                    }
                }

                Panel {
                    Layout.leftMargin: 24; Layout.rightMargin: 24
                    Layout.fillWidth: true
                    visible: !!page.trainer
                    title: qsTr("Trainer output")
                    subtitle: page.trainer ? "pid %1 · %2".arg(page.trainer.pid).arg(page.trainer.args) : ""
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 220
                        color: ui.window; radius: ui.small
                        ScrollView {
                            anchors.fill: parent; anchors.margins: 10; clip: true
                            TextArea {
                                readOnly: true; background: null
                                text: page.trainer ? (page.trainer.tail || []).join("\n") : ""
                                color: ui.dim; font.family: "monospace"; font.pixelSize: 11; wrapMode: TextArea.Wrap
                            }
                        }
                    }
                }
                Item { Layout.preferredHeight: 16 }
            }

            Text {
                visible: page.run === null
                anchors.centerIn: parent
                text: backend.connected ? qsTr("No runs yet. Start one under New run.") : ""
                color: ui.dim
            }
        }
    }

    Dialog {
        id: stopDialog
        anchors.centerIn: parent
        modal: true
        title: qsTr("Stop %1?").arg(page.run ? page.run.name : "")
        standardButtons: Dialog.Ok | Dialog.Cancel
        Text {
            width: 380
            wrapMode: Text.Wrap
            color: ui.ink
            text: qsTr("It stops as Ctrl-C would: the games release every key and the trainer writes its checkpoint. You can force it 30 s later if it hangs.")
        }
        onAccepted: backend.post("/api/stop", { pid: page.trainer.pid }, "stop")
    }
}
