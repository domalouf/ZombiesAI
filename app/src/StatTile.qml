import QtQuick
import QtQuick.Layouts

// A label over a big number, with an optional line under it.
ColumnLayout {
    property string label: ""
    property string value: "–"
    property string note: ""
    property color valueColor: ui.ink
    spacing: 2
    Text { text: parent.label; color: ui.dim; font.pixelSize: 11; font.capitalization: Font.AllUppercase; font.letterSpacing: 0.6 }
    Text { text: parent.value; color: parent.valueColor; font.pixelSize: 26; font.bold: true }
    Text { text: parent.note; color: ui.faint; font.pixelSize: 11; visible: text !== "" }
}
