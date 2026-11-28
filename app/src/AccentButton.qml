import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material

// A filled button in the theme's accent, its label in the colour the theme says reads on it (black on a light
// accent, white on a dark one). Material always writes white on a highlighted button, which disappears on
// Omarchy's yellow. `tone` replaces the accent for a destructive action (Stop).
Button {
    id: control
    property color tone: theme.accent
    property color label: Qt.colorEqual(tone, theme.accent) ? theme.accentForeground : "white"
    highlighted: true
    Material.accent: tone
    Material.roundedScale: theme.radius
    contentItem: Text {
        text: control.text
        font: control.font
        color: control.enabled ? control.label : ui.faint
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
}
