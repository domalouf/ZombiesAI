import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

// Every recorded session: how much play it holds, whether it is sound, and its input video.
ScrollView {
    id: page
    contentWidth: availableWidth
    clip: true

    readonly property var info: backend.demos
    readonly property var demos: info.demos || []

    function primaryAction() { win.go(1) }

    ColumnLayout {
        width: page.availableWidth
        spacing: 16

        Item { Layout.preferredHeight: 8 }
        RowLayout {
            Layout.leftMargin: 24; Layout.rightMargin: 24
            Text { text: qsTr("Demos"); color: ui.ink; font.pixelSize: 24; font.bold: true; Layout.fillWidth: true }
            Button { Material.roundedScale: theme.radius; text: qsTr("Open folder"); flat: true; visible: !!page.info.root; onClicked: backend.openPath(page.info.root) }
            AccentButton { text: qsTr("Record"); onClicked: win.go(1) }
        }

        RowLayout {
            Layout.leftMargin: 24; Layout.rightMargin: 24
            Layout.fillWidth: true
            spacing: 16
            Panel {
                Layout.preferredWidth: 280
                Layout.fillHeight: true
                title: qsTr("Today")
                StatTile {
                    label: qsTr("Minutes of play")
                    value: Math.round(page.info.today_min || 0) + " / " + (page.info.goal_min || 20)
                    valueColor: (page.info.today_min || 0) >= (page.info.goal_min || 20) ? ui.good : ui.ink
                }
                Meter {
                    Layout.fillWidth: true
                    label: qsTr("Daily goal")
                    text: Math.round(100 * (page.info.today_min || 0) / (page.info.goal_min || 20)) + " %"
                    value: page.info.today_min || 0
                    maximum: page.info.goal_min || 20
                }
            }
            Panel {
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: qsTr("This week")
                WeekBars { Layout.fillWidth: true; week: page.info.week || []; goal: page.info.goal_min || 20 }
            }
            Panel {
                Layout.preferredWidth: 280
                Layout.fillHeight: true
                title: qsTr("In all")
                StatTile {
                    label: qsTr("Hours of play")
                    value: ((page.info.total_play_min || 0) / 60).toLocaleString(Qt.locale(), 'f', 1)
                    note: qsTr("%n demos", "", page.demos.length)
                }
            }
        }

        Panel {
            Layout.leftMargin: 24; Layout.rightMargin: 24
            Layout.fillWidth: true
            title: qsTr("Sessions")
            subtitle: qsTr("Newest first. The input video is the session with your keys and mouse drawn on it.")

            Text {
                visible: page.demos.length === 0
                text: qsTr("No demos yet.")
                color: ui.faint
            }
            Repeater {
                model: page.demos
                delegate: Rectangle {
                    id: row
                    required property var modelData
                    Layout.fillWidth: true
                    implicitHeight: rowLayout.implicitHeight + 20
                    color: hover.hovered ? ui.raised : "transparent"
                    radius: ui.small
                    HoverHandler { id: hover }

                    RowLayout {
                        id: rowLayout
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 14

                        ColumnLayout {
                            Layout.preferredWidth: 150
                            Layout.minimumWidth: 150
                            spacing: 1
                            Text { text: row.modelData.name; color: ui.ink; font.bold: true; font.pixelSize: 14 }
                            Text {
                                text: new Date(row.modelData.created * 1000).toLocaleString(Qt.locale(), Locale.ShortFormat)
                                color: ui.dim; font.pixelSize: 11
                            }
                        }
                        ColumnLayout {
                            Layout.preferredWidth: 110
                            Layout.minimumWidth: 110
                            spacing: 1
                            Text { text: qsTr("%1 min").arg(Math.round(row.modelData.play_minutes)); color: ui.ink; font.pixelSize: 14 }
                            Text {
                                text: row.modelData.not_playing_minutes > 0.05
                                      ? qsTr("+%1 not playing").arg(row.modelData.not_playing_minutes.toLocaleString(Qt.locale(), 'f', 1))
                                      : qsTr("no F8 marks")
                                color: row.modelData.not_playing_minutes > 0.05 ? ui.dim : ui.warn; font.pixelSize: 11
                            }
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 4
                            Text { text: row.modelData.notes || "–"; color: ui.ink; font.pixelSize: 13; elide: Text.ElideRight; Layout.fillWidth: true }
                            Flow {
                                Layout.fillWidth: true
                                spacing: 6
                                Repeater {
                                    model: {
                                        const d = row.modelData, chips = []
                                        if (d.status === "open") chips.push([qsTr("not closed"), "bad"])
                                        if (d.scene) chips.push([qsTr("scene"), ""])
                                        if (d.audio) chips.push([qsTr("audio"), ""])
                                        if (d.gamepad) chips.push([qsTr("gamepad"), ""])
                                        if (d.capture_lost) chips.push([qsTr("capture lost"), "warn"])
                                        if (d.stale_rate > 0.02) chips.push([qsTr("%1 % stale").arg(Math.round(100 * d.stale_rate)), "warn"])
                                        if (d.overrun_rate > 0.01) chips.push([qsTr("%1 % late").arg(Math.round(100 * d.overrun_rate)), "warn"])
                                        chips.push([qsTr("%1 MB").arg(Math.round(d.size_mb || 0)), ""])
                                        return chips
                                    }
                                    delegate: Rectangle {
                                        required property var modelData
                                        implicitWidth: chip.implicitWidth + 12
                                        implicitHeight: chip.implicitHeight + 4
                                        radius: ui.small
                                        color: "transparent"
                                        border.color: modelData[1] === "" ? ui.line : ui.levelColor(modelData[1])
                                        Text { id: chip; anchors.centerIn: parent; text: modelData[0]; font.pixelSize: 10
                                               color: modelData[1] === "" ? ui.dim : ui.levelColor(modelData[1]) }
                                    }
                                }
                            }
                        }
                        BusyIndicator { visible: row.modelData.rendering === true; running: visible; implicitWidth: 28; implicitHeight: 28 }
                        Button { Material.roundedScale: theme.radius;
                            text: row.modelData.video ? qsTr("Play video") : qsTr("Render video")
                            enabled: row.modelData.rendering !== true
                            flat: !row.modelData.video
                            onClicked: row.modelData.video ? backend.openPath(row.modelData.video)
                                                           : backend.post("/api/demos/render", { name: row.modelData.name }, "render")
                        }
                        ToolButton { Material.roundedScale: theme.radius;
                            text: "⋯"
                            onClicked: menu.popup()
                            Menu {
                                id: menu
                                MenuItem { text: qsTr("Open folder"); onTriggered: backend.openPath(row.modelData.path) }
                                MenuItem {
                                    text: qsTr("Render the video again")
                                    enabled: !!row.modelData.video && row.modelData.rendering !== true
                                    onTriggered: backend.post("/api/demos/render", { name: row.modelData.name }, "render")
                                }
                            }
                        }
                    }
                }
            }
        }
        Item { Layout.preferredHeight: 16 }
    }
}
