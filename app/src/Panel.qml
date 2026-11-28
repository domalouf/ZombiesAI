import QtQuick
import QtQuick.Layouts

// A titled box. Children go in `content`, laid out top to bottom.
Rectangle {
    id: panel
    property string title: ""
    property string subtitle: ""
    property alias headerExtra: extra.data
    default property alias content: body.data
    color: ui.panel
    radius: ui.radius
    border.color: ui.line
    implicitHeight: column.implicitHeight + 32
    implicitWidth: column.implicitWidth + 32

    ColumnLayout {
        id: column
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            visible: panel.title !== "" || extra.children.length > 0
            spacing: 10
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 2
                Text { text: panel.title; color: ui.ink; font.pixelSize: 14; font.bold: true; visible: text !== "" }
                Text {
                    Layout.fillWidth: true
                    text: panel.subtitle; color: ui.dim; font.pixelSize: 11
                    visible: text !== ""; elide: Text.ElideRight
                }
            }
            RowLayout { id: extra; spacing: 8 }
        }
        ColumnLayout {
            id: body
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 10
        }
    }
}
