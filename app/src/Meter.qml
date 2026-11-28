import QtQuick
import QtQuick.Layouts

// A labelled bar: `value` of `maximum`, coloured by how close to its limit it is when `limits` are given.
ColumnLayout {
    id: meter
    property string label: ""
    property string text: ""
    property real value: 0
    property real maximum: 100
    property color barColor: ui.accent
    spacing: 4
    RowLayout {
        Layout.fillWidth: true
        Text { text: meter.label; color: ui.dim; font.pixelSize: 12; Layout.fillWidth: true; elide: Text.ElideRight }
        Text { text: meter.text; color: ui.ink; font.pixelSize: 12 }
    }
    Rectangle {
        Layout.fillWidth: true
        height: 6
        radius: Math.min(3, ui.radius)
        color: ui.raised
        Rectangle {
            width: parent.width * Math.max(0, Math.min(1, meter.maximum > 0 ? meter.value / meter.maximum : 0))
            height: parent.height
            radius: parent.radius
            color: meter.barColor
            Behavior on width { NumberAnimation { duration: 250 } }
        }
    }
}
