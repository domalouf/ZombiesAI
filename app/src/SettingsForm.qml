import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

// Every setting in a schema, grouped as the server groups them, each group folding open on a click. `values`
// holds what was typed (name -> value); bump `generation` to rebuild the fields from it (a preset loaded).
ColumnLayout {
    id: form
    property var schema: []
    property var values: ({})
    property int generation: 0
    property string filter: ""
    readonly property var groups: {
        const seen = []
        for (const s of schema) if (seen.indexOf(s.group || "other") < 0) seen.push(s.group || "other")
        return seen
    }
    readonly property int changedCount: Object.keys(backend.buildSettings(schema, values)).length
    spacing: 8

    function set(name, value) {
        const next = Object.assign({}, values)
        next[name] = value
        values = next
    }

    RowLayout {
        Layout.fillWidth: true
        TextField { Material.roundedScale: theme.radius;
            id: search
            Layout.fillWidth: true
            placeholderText: qsTr("Find a setting")
            onTextEdited: form.filter = text.toLowerCase()
        }
        Text {
            text: form.changedCount ? qsTr("%1 changed").arg(form.changedCount) : qsTr("all defaults")
            color: form.changedCount ? ui.accent : ui.faint; font.pixelSize: 12
        }
    }

    Repeater {
        model: form.generation >= 0 ? form.groups : []
        delegate: ColumnLayout {
            id: group
            required property string modelData
            property bool open: modelData === "run" || modelData === "length" || form.filter !== ""
            readonly property var members: form.schema.filter(s => (s.group || "other") === modelData
                && (form.filter === "" || s.name.indexOf(form.filter.replace(/ /g, "_")) >= 0 || (s.hint || "").toLowerCase().indexOf(form.filter) >= 0))
            Layout.fillWidth: true
            visible: members.length > 0
            spacing: 6

            ItemDelegate {
                Layout.fillWidth: true
                onClicked: group.open = !group.open
                contentItem: RowLayout {
                    Text { text: group.open ? "▾" : "▸"; color: ui.dim }
                    Text { text: group.modelData; color: ui.ink; font.bold: true; font.capitalization: Font.Capitalize; Layout.fillWidth: true }
                    Text { text: qsTr("%n settings", "", group.members.length); color: ui.faint; font.pixelSize: 11 }
                }
                background: Rectangle { color: "transparent"; Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: ui.line } }
            }
            Repeater {
                model: group.open ? group.members : []
                delegate: SettingField {
                    required property var modelData
                    Layout.fillWidth: true
                    Layout.leftMargin: 18
                    setting: modelData
                    value: form.values[modelData.name] !== undefined ? form.values[modelData.name] : modelData["default"]
                    onEdited: v => form.set(modelData.name, v)
                }
            }
        }
    }
}
