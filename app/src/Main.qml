import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

ApplicationWindow {
    id: win
    width: 1280
    height: 820
    minimumWidth: 900
    minimumHeight: 600
    visible: true
    title: "Zombies"
    color: ui.window

    Material.theme: Material.Dark
    Material.accent: theme.accent
    Material.roundedScale: theme.radius

    property bool helpVisible: false
    property int page: 0
    readonly property var pages: [
        { key: "1", name: qsTr("Overview") },
        { key: "2", name: qsTr("Record") },
        { key: "3", name: qsTr("Runs") },
        { key: "4", name: qsTr("New run") },
        { key: "5", name: qsTr("Demos") },
        { key: "6", name: qsTr("Games") }
    ]

    // One palette for every page, so the status colours mean the same thing everywhere.
    QtObject {
        id: ui
        readonly property color window: "#0e0e10"
        readonly property color panel: "#16161a"
        readonly property color raised: "#1e1e23"
        readonly property color line: "#2a2a31"
        readonly property color ink: "#e8e8ea"
        readonly property color dim: "#8d8d96"
        readonly property color faint: "#5a5a63"
        readonly property color accent: theme.accent
        readonly property color good: theme.accent
        readonly property color warn: "#e3b341"
        readonly property color bad: "#f0625a"
        readonly property int radius: theme.radius
        readonly property int small: Math.min(4, theme.radius)
        function levelColor(level) {
            return level === "bad" || level === "critical" ? bad
                 : level === "warn" || level === "serious" ? warn
                 : level === "good" ? good : dim
        }
        function statusColor(status) {
            return status === "running" ? good : status === "crashed" || status === "failed" ? bad
                 : status === "stopped" ? warn : dim
        }
    }

    function go(index) { page = index; helpVisible = false }
    function toast(text, bad) { toaster.show(text, bad) }

    Connections {
        target: backend
        function onActionFinished(tag, ok, response) {
            if (!ok)
                win.toast(response.error || qsTr("%1 failed").arg(tag), true)
            else if (response.message)
                win.toast(response.message, false)
        }
    }

    Shortcut { sequence: "?"; context: Qt.ApplicationShortcut; onActivated: win.helpVisible = !win.helpVisible }
    Shortcut { sequence: "Escape"; context: Qt.ApplicationShortcut; onActivated: win.helpVisible = false }
    Shortcut { sequence: "Q"; context: Qt.ApplicationShortcut; enabled: !win.typing(); onActivated: Qt.quit() }
    Shortcut { sequence: "Ctrl+Q"; context: Qt.ApplicationShortcut; onActivated: Qt.quit() }
    Shortcut { sequence: "Ctrl+R"; context: Qt.ApplicationShortcut; onActivated: backend.refresh() }
    Shortcut { sequence: "Ctrl+B"; context: Qt.ApplicationShortcut; onActivated: backend.openTrainingRoom() }
    Repeater {
        model: win.pages
        delegate: Item {
            required property var modelData
            required property int index
            Shortcut {
                sequence: modelData.key
                context: Qt.ApplicationShortcut
                enabled: !win.typing()
                onActivated: win.go(index)
            }
        }
    }
    Shortcut {
        sequence: "Space"
        context: Qt.ApplicationShortcut
        enabled: !win.typing()
        onActivated: {
            const current = stack.children[stack.currentIndex]
            if (current && current.primaryAction)
                current.primaryAction()
        }
    }

    // Single-key shortcuts must not fire while a text field has the keyboard (typing "1" in the notes).
    function typing() {
        const item = win.activeFocusItem
        return !!item && (item instanceof TextInput || item instanceof TextEdit)
    }

    Item {
        anchors.fill: parent
        LayoutMirroring.enabled: Qt.application.layoutDirection === Qt.RightToLeft
        LayoutMirroring.childrenInherit: true

        RowLayout {
            anchors.fill: parent
            spacing: 0

            // The rail: the pages, with what needs attention marked on them.
            Rectangle {
                Layout.fillHeight: true
                Layout.preferredWidth: 196
                color: ui.panel

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 4

                    RowLayout {
                        Layout.bottomMargin: 18
                        spacing: 10
                        Rectangle {
                            width: 30; height: 30; radius: ui.small; color: ui.accent
                            Text { anchors.centerIn: parent; text: "Z"; color: theme.accentForeground; font.bold: true; font.pixelSize: 17 }
                        }
                        ColumnLayout {
                            spacing: 0
                            Text { text: "Zombies"; color: ui.ink; font.pixelSize: 17; font.bold: true }
                            Text { text: qsTr("ZombiesAI dashboard"); color: ui.dim; font.pixelSize: 11 }
                        }
                    }

                    Repeater {
                        model: win.pages
                        delegate: ItemDelegate {
                            id: navItem
                            required property var modelData
                            required property int index
                            Layout.fillWidth: true
                            highlighted: win.page === index
                            onClicked: win.go(index)
                            contentItem: RowLayout {
                                spacing: 10
                                Text {
                                    text: navItem.modelData.key
                                    color: navItem.highlighted ? ui.accent : ui.faint
                                    font.pixelSize: 12; font.family: "monospace"
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: navItem.modelData.name
                                    color: navItem.highlighted ? ui.ink : ui.dim
                                    font.pixelSize: 14; font.bold: navItem.highlighted
                                    elide: Text.ElideRight
                                }
                                Rectangle {  // a live dot: recording on Record, training on Runs
                                    width: 8; height: 8; radius: 4
                                    visible: (navItem.index === 1 && backend.record.active === true)
                                          || (navItem.index === 2 && (backend.summary.trainers || []).length > 0)
                                    color: navItem.index === 1 ? ui.bad : ui.good
                                }
                            }
                            background: Rectangle {
                                radius: ui.small
                                color: navItem.highlighted ? ui.raised : navItem.hovered ? "#1a1a1f" : "transparent"
                                Rectangle {
                                    visible: navItem.highlighted
                                    width: 3; height: parent.height - 12
                                    anchors.verticalCenter: parent.verticalCenter
                                    color: ui.accent
                                }
                            }
                        }
                    }

                    Item { Layout.fillHeight: true }

                    // The server's state, always in view: everything on every page comes from it.
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        Rectangle {
                            width: 8; height: 8; radius: 4
                            color: backend.connected ? ui.good : backend.serverState === "failed" ? ui.bad : ui.warn
                        }
                        Text {
                            Layout.fillWidth: true
                            color: ui.dim; font.pixelSize: 11; elide: Text.ElideRight
                            text: backend.connected ? qsTr("Connected · %1").arg(backend.baseUrl.replace("http://", ""))
                                : backend.serverState === "starting" ? qsTr("Starting the server…")
                                : backend.serverState === "failed" ? qsTr("Server failed") : qsTr("Connecting…")
                        }
                    }
                    Button { Material.roundedScale: theme.radius;
                        Layout.fillWidth: true
                        flat: true
                        text: qsTr("Training Room in browser")
                        font.pixelSize: 11
                        onClicked: backend.openTrainingRoom()
                    }
                    Text {
                        Layout.fillWidth: true
                        text: qsTr("? for shortcuts")
                        color: ui.faint; font.pixelSize: 11
                        horizontalAlignment: Text.AlignHCenter
                    }
                }
            }

            Rectangle { Layout.fillHeight: true; width: 1; color: ui.line }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0

                // Said in the window rather than worked around: without the server nothing here is real.
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: banner.implicitHeight + 20
                    visible: !backend.connected
                    color: backend.serverState === "failed" ? "#3a1d1c" : "#2e2817"
                    RowLayout {
                        id: banner
                        anchors.fill: parent
                        anchors.margins: 10
                        anchors.leftMargin: 20
                        Text {
                            Layout.fillWidth: true
                            wrapMode: Text.Wrap
                            color: ui.ink
                            text: backend.serverError !== "" ? backend.serverError
                                : backend.serverState === "starting"
                                    ? qsTr("Starting the Training Room server (scripts/supervise.py)…")
                                    : qsTr("Looking for the Training Room server at %1…").arg(backend.baseUrl)
                        }
                        Button { Material.roundedScale: theme.radius;
                            visible: backend.serverState === "failed"
                            text: qsTr("Try again")
                            onClicked: backend.startServer()
                        }
                    }
                }

                StackLayout {
                    id: stack
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    currentIndex: win.page

                    OverviewPage {}
                    RecordPage {}
                    RunsPage {}
                    LaunchPage {}
                    DemosPage {}
                    GamesPage {}
                }
            }
        }

        // What a command answered, for a few seconds, bottom centre.
        Rectangle {
            id: toaster
            property string message: ""
            property bool failed: false
            function show(text, bad) { message = text; failed = bad; opacity = 1; hide.restart() }
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 24
            width: Math.min(parent.width - 80, toastText.implicitWidth + 40)
            height: toastText.implicitHeight + 22
            radius: ui.small
            color: failed ? "#3a1d1c" : ui.raised
            border.color: failed ? ui.bad : ui.line
            opacity: 0
            visible: opacity > 0
            Behavior on opacity { NumberAnimation { duration: 180 } }
            Text {
                id: toastText
                anchors.centerIn: parent
                width: Math.min(implicitWidth, toaster.parent.width - 120)
                text: toaster.message
                color: ui.ink
                wrapMode: Text.Wrap
            }
            Timer { id: hide; interval: 5000; onTriggered: toaster.opacity = 0 }
        }

        HelpOverlay {
            anchors.fill: parent
            visible: win.helpVisible
            onClosed: win.helpVisible = false
        }
    }
}
