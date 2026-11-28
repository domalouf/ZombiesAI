import QtQuick
import QtQuick.Layouts

// Every shortcut, on `?`.
Rectangle {
    id: overlay
    signal closed()
    color: "#cc000000"
    MouseArea { anchors.fill: parent; onClicked: overlay.closed() }

    readonly property var keys: [
        ["1 – 6", qsTr("Overview, Record, Runs, New run, Demos, Games")],
        ["Space", qsTr("The page's main action: start or stop recording, start the run")],
        ["Ctrl+R", qsTr("Refresh everything now")],
        ["Ctrl+B", qsTr("Open the Training Room in the browser")],
        ["?", qsTr("Show or hide this list")],
        ["Esc", qsTr("Close this list")],
        ["Q", qsTr("Quit (runs and recordings keep going)")]
    ]

    Rectangle {
        anchors.centerIn: parent
        width: 560
        height: grid.implicitHeight + 72
        color: ui.panel
        radius: ui.radius
        border.color: ui.line
        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 24
            spacing: 16
            Text { text: qsTr("Keyboard shortcuts"); color: ui.ink; font.pixelSize: 17; font.bold: true }
            GridLayout {
                id: grid
                columns: 2
                columnSpacing: 20
                rowSpacing: 10
                Repeater {
                    model: overlay.keys
                    delegate: Text {
                        required property var modelData
                        required property int index
                        Layout.row: index
                        Layout.column: 0
                        text: modelData[0]
                        color: ui.accent; font.family: "monospace"; font.pixelSize: 13
                    }
                }
                Repeater {
                    model: overlay.keys
                    delegate: Text {
                        required property var modelData
                        required property int index
                        Layout.row: index
                        Layout.column: 1
                        Layout.fillWidth: true
                        text: modelData[1]
                        color: ui.ink; font.pixelSize: 13; wrapMode: Text.Wrap
                    }
                }
            }
        }
    }
}
