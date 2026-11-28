import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

// The games: the fleet's headless instances (up, down, watched in viewer windows) and the desktop game.
ScrollView {
    id: page
    contentWidth: availableWidth
    clip: true

    readonly property var fleets: backend.fleet.fleets || []
    readonly property var screens: backend.games.screens || []
    readonly property var viewer: backend.games.viewer || ({})
    readonly property var game: backend.record.game || ({})

    function primaryAction() {}

    ColumnLayout {
        width: page.availableWidth
        spacing: 16

        Item { Layout.preferredHeight: 8 }
        Text { Layout.leftMargin: 24; text: qsTr("Games"); color: ui.ink; font.pixelSize: 24; font.bold: true }

        RowLayout {
            Layout.leftMargin: 24; Layout.rightMargin: 24
            Layout.fillWidth: true
            spacing: 16

            Repeater {
                model: page.fleets
                delegate: Panel {
                    id: fleetPanel
                    required property var modelData
                    readonly property bool busyOp: !!(modelData.op && modelData.op.running)
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignTop
                    title: qsTr("Fleet · %1").arg(modelData.label)
                    subtitle: qsTr("%1 of %2 games up").arg(modelData.running).arg(modelData.n)
                        + ((modelData.busy || []).length ? qsTr(" · playing for %1").arg(modelData.busy.join(", ")) : "")
                    RowLayout {
                        SpinBox { Material.roundedScale: theme.radius; id: count; from: 1; to: 16; editable: true; value: modelData.n || backend.fleet.recommended || 6 }
                        AccentButton {
                            text: qsTr("Start games")
                            enabled: !fleetPanel.busyOp
                            onClicked: backend.post("/api/fleet", { fleet: fleetPanel.modelData.label, action: "up", n: count.value }, "fleet-up")
                        }
                        Button { Material.roundedScale: theme.radius;
                            text: qsTr("Stop games")
                            enabled: !fleetPanel.busyOp && fleetPanel.modelData.running > 0 && !(fleetPanel.modelData.busy || []).length
                            onClicked: backend.post("/api/fleet", { fleet: fleetPanel.modelData.label, action: "down" }, "fleet-down")
                        }
                        BusyIndicator { visible: fleetPanel.busyOp; running: visible; implicitWidth: 28; implicitHeight: 28 }
                    }
                    Text {
                        visible: !!backend.fleet.recommended
                        text: qsTr("This machine is best at about %n games", "", backend.fleet.recommended || 0)
                        color: ui.faint; font.pixelSize: 11
                    }
                }
            }

            Panel {
                Layout.preferredWidth: 360
                Layout.alignment: Qt.AlignTop
                title: qsTr("Desktop game")
                subtitle: page.game.running ? qsTr("Running, for recording demos") : qsTr("Not running")
                RowLayout {
                    Button { Material.roundedScale: theme.radius; text: qsTr("Start"); enabled: !page.game.running && !page.game.op; onClicked: backend.post("/api/desktop", { action: "up" }, "game-up") }
                    Button { Material.roundedScale: theme.radius; text: qsTr("Quit"); enabled: !!page.game.running && !page.game.op && !backend.record.active; onClicked: backend.post("/api/desktop", { action: "down" }, "game-down") }
                    BusyIndicator { visible: !!page.game.op; running: visible; implicitWidth: 28; implicitHeight: 28 }
                }
            }
        }

        Panel {
            Layout.leftMargin: 24; Layout.rightMargin: 24
            Layout.fillWidth: true
            title: qsTr("Instances")
            subtitle: page.screens.length ? qsTr("%n games", "", page.screens.length) : qsTr("No games running")
            headerExtra: [
                Text { text: qsTr("Workspace"); color: ui.dim; font.pixelSize: 12 },
                TextField { Material.roundedScale: theme.radius; id: workspace; implicitWidth: 60; text: page.viewer.workspace || "9" },
                Button { Material.roundedScale: theme.radius;
                    text: qsTr("Watch all")
                    enabled: page.screens.length > 0
                    onClicked: backend.post("/api/viewers", { displays: page.screens.map(s => s.number), workspace: workspace.text, fps: page.viewer.fps || 30, focus: true }, "viewers")
                },
                Button { Material.roundedScale: theme.radius;
                    text: qsTr("Close viewers"); flat: true
                    enabled: page.screens.some(s => s.watched)
                    onClicked: backend.post("/api/viewers", { displays: [], workspace: workspace.text }, "viewers")
                }
            ]
            GridLayout {
                id: tiles
                Layout.fillWidth: true
                columns: Math.max(1, Math.floor(width / 330))
                readonly property real cell: (width - columnSpacing * (columns - 1)) / columns
                columnSpacing: 12; rowSpacing: 12
                Repeater {
                    model: page.screens
                    delegate: ColumnLayout {
                        id: screen
                        required property var modelData
                        Layout.preferredWidth: tiles.cell
                        Layout.maximumWidth: tiles.cell
                        spacing: 4
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: width * 9 / 16
                            color: ui.window
                            radius: ui.small
                            clip: true
                            Image {
                                anchors.fill: parent
                                fillMode: Image.PreserveAspectFit
                                cache: false
                                asynchronous: true
                                source: screen.modelData.thumb_at ? "%1/thumb/%2.png?t=%3".arg(backend.baseUrl).arg(screen.modelData.number).arg(screen.modelData.thumb_at) : ""
                            }
                            Text { anchors.centerIn: parent; visible: !screen.modelData.thumb_at; text: qsTr("no picture yet"); color: ui.faint }
                        }
                        RowLayout {
                            Text { text: screen.modelData.display; color: ui.ink; font.bold: true }
                            Text { text: screen.modelData.game || ""; color: ui.dim; font.pixelSize: 11; Layout.fillWidth: true; elide: Text.ElideRight }
                            Text { text: screen.modelData.fps ? qsTr("%1 fps").arg(screen.modelData.fps) : ""; color: ui.dim; font.pixelSize: 11 }
                            Text { text: screen.modelData.watched ? "👁" : ""; color: ui.accent }
                        }
                    }
                }
            }
        }
        Item { Layout.preferredHeight: 16 }
    }
}
