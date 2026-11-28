import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

// Record a demo on the fleet's own game (scripts/record_lan.sh): set it up, start it, watch it, stop it.
ScrollView {
    id: page
    contentWidth: availableWidth
    clip: true

    readonly property var rec: backend.record
    readonly property var status: rec.status || ({})
    readonly property bool active: rec.active === true
    readonly property var game: rec.game || ({})

    function primaryAction() { active ? stop() : start() }
    function start() {
        backend.post("/api/record", {
            minutes: minutes.value, notes: notes.text.trim() || qsTr("dashboard session"),
            pad: pad.currentIndex === 0 ? "" : pad.currentIndex === 1 ? "auto" : padPath.text.trim(),
            overlay: overlay.checked, audio: audio.checked
        }, "record")
    }
    function stop() { backend.post("/api/record/stop", {}, "record-stop") }

    ColumnLayout {
        width: page.availableWidth
        spacing: 16

        Item { Layout.preferredHeight: 8 }
        RowLayout {
            Layout.leftMargin: 24; Layout.rightMargin: 24
            Text { text: qsTr("Record"); color: ui.ink; font.pixelSize: 24; font.bold: true; Layout.fillWidth: true }
            AccentButton {
                text: page.active ? qsTr("Stop recording") : qsTr("Start recording")
                tone: page.active ? ui.bad : theme.accent
                enabled: backend.connected
                onClicked: page.primaryAction()
            }
        }

        RowLayout {
            Layout.leftMargin: 24; Layout.rightMargin: 24
            Layout.fillWidth: true
            spacing: 16

            ColumnLayout {
                Layout.preferredWidth: 420
                Layout.maximumWidth: 460
                Layout.alignment: Qt.AlignTop
                spacing: 16

                Panel {
                    Layout.fillWidth: true
                    title: qsTr("Session")
                    subtitle: qsTr("Plutonium LAN, the fleet's settings, the game's own audio")
                    enabled: !page.active
                    opacity: enabled ? 1 : 0.55

                    RowLayout {
                        Text { text: qsTr("Minutes"); color: ui.dim; Layout.preferredWidth: 110 }
                        SpinBox { Material.roundedScale: theme.radius; id: minutes; from: 1; to: 120; value: 20; editable: true }
                    }
                    TextField { Material.roundedScale: theme.radius;
                        id: notes
                        Layout.fillWidth: true
                        placeholderText: qsTr("What you're going for (camping, trains, later rounds…)")
                        selectByMouse: true
                    }
                    RowLayout {
                        Text { text: qsTr("Gamepad"); color: ui.dim; Layout.preferredWidth: 110 }
                        ComboBox { Material.roundedScale: theme.radius;
                            id: pad
                            Layout.fillWidth: true
                            model: [qsTr("Off: mouse and keyboard"), qsTr("First pad found"), qsTr("This device…")]
                        }
                    }
                    TextField { Material.roundedScale: theme.radius;
                        id: padPath
                        visible: pad.currentIndex === 2
                        Layout.fillWidth: true
                        placeholderText: "/dev/input/event…"
                    }
                    RowLayout {
                        Text { text: qsTr("Live overlay"); color: ui.dim; Layout.fillWidth: true }
                        Switch { Material.roundedScale: theme.radius; id: overlay; checked: true }
                    }
                    RowLayout {
                        Text { text: qsTr("Game audio"); color: ui.dim; Layout.fillWidth: true }
                        Switch { Material.roundedScale: theme.radius; id: audio; checked: true }
                    }
                }

                Panel {
                    Layout.fillWidth: true
                    title: qsTr("Desktop game")
                    subtitle: page.game.running ? qsTr("Running. In the menu press F9 to start Nacht.")
                                                : qsTr("Not running. Starting a recording starts it too.")
                    headerExtra: [
                        Rectangle { width: 10; height: 10; radius: 5; color: page.game.running ? ui.good : ui.faint }
                    ]
                    RowLayout {
                        Button { Material.roundedScale: theme.radius;
                            text: qsTr("Start game")
                            enabled: !page.game.running && !page.game.op
                            onClicked: backend.post("/api/desktop", { action: "up" }, "game-up")
                        }
                        Button { Material.roundedScale: theme.radius;
                            text: qsTr("Quit game")
                            enabled: !!page.game.running && !page.game.op && !page.active
                            onClicked: backend.post("/api/desktop", { action: "down" }, "game-down")
                        }
                        BusyIndicator { running: !!page.game.op; visible: running; implicitWidth: 28; implicitHeight: 28 }
                    }
                }

                Panel {
                    Layout.fillWidth: true
                    title: qsTr("While you play")
                    Repeater {
                        model: [
                            ["F9", qsTr("start Nacht from the menu, and again after each game over")],
                            ["F8", qsTr("going into a menu, pause, death or loading, and again coming back")],
                            [qsTr("D-pad ↓"), qsTr("the same as F8, on a gamepad")]
                        ]
                        delegate: RowLayout {
                            required property var modelData
                            Layout.fillWidth: true
                            Text { text: modelData[0]; color: ui.accent; font.family: "monospace"; Layout.preferredWidth: 70 }
                            Text { text: modelData[1]; color: ui.dim; font.pixelSize: 12; wrapMode: Text.Wrap; Layout.fillWidth: true }
                        }
                    }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignTop
                spacing: 16

                Panel {
                    Layout.fillWidth: true
                    title: qsTr("Now")
                    readonly property string state: page.active ? (page.status.state || "starting") : "idle"
                    Text {
                        color: !page.active ? ui.dim : page.status.stale ? ui.bad
                             : page.status.playing === false ? ui.warn
                             : page.status.state === "recording" ? ui.bad : ui.ink
                        font.pixelSize: 30; font.bold: true
                        text: !page.active ? qsTr("Not recording")
                            : page.status.state === "waiting" ? qsTr("Waiting for the game window")
                            : page.status.state === "countdown" ? qsTr("Counting down…")
                            : page.status.stale ? qsTr("Capture lost")
                            : page.status.playing === false ? qsTr("NOT PLAYING · press F8")
                            : page.status.state === "recording" ? qsTr("● REC  %1").arg(backend.formatDuration(page.status.elapsed_s || 0))
                            : qsTr("Starting…")
                    }
                    Meter {
                        Layout.fillWidth: true
                        visible: page.active && !!page.status.limit_s
                        label: qsTr("Session")
                        text: qsTr("%1 of %2").arg(backend.formatDuration(page.status.elapsed_s || 0)).arg(backend.formatDuration(page.status.limit_s || 0))
                        value: page.status.elapsed_s || 0
                        maximum: page.status.limit_s || 1
                        barColor: page.status.playing === false ? ui.warn : ui.bad
                    }
                    Text {
                        visible: page.active && page.status.state === "waiting"
                        text: qsTr("Switch to the game's workspace: recording starts 3 s after it is on screen.")
                        color: ui.dim; wrapMode: Text.Wrap; Layout.fillWidth: true
                    }
                    Text {
                        visible: !page.active && !!page.rec.last
                        text: page.rec.last ? qsTr("Last recording: %1").arg(page.rec.last.name) : ""
                        color: ui.dim
                    }
                    RowLayout {
                        visible: !page.active && !!page.rec.last
                        Button { Material.roundedScale: theme.radius; text: qsTr("See it in Demos"); flat: true; onClicked: win.go(4) }
                        Button { Material.roundedScale: theme.radius;
                            text: qsTr("Render input video")
                            flat: true
                            onClicked: backend.post("/api/demos/render", { name: page.rec.last.name }, "render")
                        }
                    }
                }

                Panel {
                    Layout.fillWidth: true
                    title: qsTr("Recorder output")
                    subtitle: page.rec.log || ""
                    headerExtra: [
                        Button { Material.roundedScale: theme.radius; flat: true; text: qsTr("Open log"); visible: !!page.rec.log; onClicked: backend.openPath(page.rec.log) }
                    ]
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 300
                        color: ui.window
                        radius: ui.small
                        ScrollView {
                            anchors.fill: parent
                            anchors.margins: 10
                            clip: true
                            TextArea {
                                readOnly: true
                                text: (page.rec.log_tail || []).join("\n") || qsTr("Nothing yet.")
                                color: ui.dim
                                font.family: "monospace"; font.pixelSize: 11
                                wrapMode: TextArea.Wrap
                                background: null
                            }
                        }
                    }
                }
            }
        }
        Item { Layout.preferredHeight: 16 }
    }
}
