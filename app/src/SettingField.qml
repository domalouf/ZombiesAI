import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

// One training setting from the server's schema ({name, flag, kind, default, hint}), edited as its kind says.
// `value` is what the field holds now; the default shows dimmed until it is changed.
RowLayout {
    id: field
    required property var setting
    property var value: setting["default"]
    readonly property bool changed: String(value) !== String(setting["default"])
    signal edited(var value)
    spacing: 12

    function reset() { value = setting["default"]; edited(value) }

    ColumnLayout {
        Layout.preferredWidth: 230
        Layout.maximumWidth: 230
        spacing: 0
        Text {
            text: field.setting.name.replace(/_/g, " ")
            color: field.changed ? ui.ink : ui.dim
            font.pixelSize: 13; font.bold: field.changed
            elide: Text.ElideRight; Layout.fillWidth: true
        }
        Text {
            text: field.setting.hint || field.setting.flag
            color: ui.faint; font.pixelSize: 10
            elide: Text.ElideRight; Layout.fillWidth: true
        }
    }
    Loader {
        Layout.fillWidth: true
        sourceComponent: field.setting.kind === "bool" ? boolEditor : textEditor
    }
    ToolButton { Material.roundedScale: theme.radius;
        text: "↺"
        visible: field.changed
        onClicked: field.reset()
        ToolTip.visible: hovered
        ToolTip.text: qsTr("Back to %1").arg(String(field.setting["default"]))
    }

    Component {
        id: boolEditor
        Switch { Material.roundedScale: theme.radius;
            checked: field.value === true || field.value === "true"
            onToggled: { field.value = checked; field.edited(checked) }
        }
    }
    Component {
        id: textEditor
        TextField { Material.roundedScale: theme.radius;
            text: field.value === null || field.value === undefined ? "" : String(field.value)
            placeholderText: field.setting["default"] === null ? qsTr("not set") : ""
            selectByMouse: true
            validator: field.setting.kind === "int" ? intCheck : field.setting.kind === "float" ? floatCheck : null
            inputMethodHints: field.setting.kind === "str" ? Qt.ImhNone : Qt.ImhFormattedNumbersOnly
            onTextEdited: { field.value = text; field.edited(text) }
            RegularExpressionValidator { id: intCheck; regularExpression: /-?\d*/ }
            RegularExpressionValidator { id: floatCheck; regularExpression: /-?(\d+\.?\d*([eE]-?\d+)?|\.\d+)?/ }
        }
    }
}
